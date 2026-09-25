"""Inferencia y submission automatizada de Pulso TransMi (Fase 4).

Sigue el patron que exige la guia metodologica ("Cómo se entrega una
prediccion" / "Automatizaciones esperadas -> Inferencia y submission"):

  1. Consulta el reloj y el ciclo vigente. Si no hay ciclo abierto, termina
     sin error (no es una falla; es el estado normal antes de la
     activacion, y tambien el estado normal entre ciclos).
  2. Usa el `data_cutoff`, los targets (station_id + target_at) y el
     `closes_at` que devuelve la API - nunca los calcula por su cuenta.
  3. Carga el champion vigente desde Supabase Storage.
  4. Construye las features SOLO con datos <= data_cutoff (nunca con el
     futuro que se esta prediciendo).
  5. Genera el batch completo (hasta 100 predicciones, normalmente 48) y lo
     envia con una llave de idempotencia estable (mismo cycle_id + mismo
     modelo -> misma llave, para que un reintento nunca cree una entrega
     distinta).

La API key de submissions todavia no existe (el profesor no ha activado el
reloj). Sin ella, el script arma y valida el batch completo igual, pero no
lo envia - deja explicito que esta en modo "listo, esperando la key", no
fallando.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import (  # noqa: E402
    HORIZONS_MINUTES,
    aplicar_features_de_target,
    build_feature_frame,
)
from perfil import DIAS_DE_HISTORIA, PESO_PERFIL_CIERRE, PerfilAdaptativo, mezclar  # noqa: E402
from predict import get_champion, load_model_from_storage  # noqa: E402

API_URL = os.environ.get("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")
SCHEMA_VERSION = "1.0"


def supabase_headers() -> dict:
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def get_current_cycle(client: httpx.Client) -> dict | None:
    """GET /v1/forecast-cycles/current. `None` si no hay ciclo abierto.

    La guia es explicita: "Si no existe un ciclo abierto, termina sin
    error" - por eso un 404 con code=no_open_cycle no es una excepcion,
    es el camino normal.
    """
    response = client.get(f"{API_URL}/v1/forecast-cycles/current")
    if response.status_code == 404:
        detail = response.json().get("detail", {})
        if isinstance(detail, dict) and detail.get("code") == "no_open_cycle":
            return None
    response.raise_for_status()
    return response.json()


def parse_targets(cycle: dict) -> list[dict]:
    """Extrae la lista de {station_id, target_at} que pide el ciclo.

    El nombre exacto del campo no esta confirmado en produccion (el reloj
    sigue en `waiting` y esta ruta nunca ha devuelto un ciclo real todavia).
    Se intentan los nombres mas probables dado el resto del contrato
    (PredictionInput ya usa station_id/target_at) y si ninguno calza se
    falla con el payload completo a la vista, para que el ajuste sea de
    un minuto el dia de la activacion, no una adivinanza a ciegas.
    """
    for key in ("targets", "predictions_requested", "requested_targets", "items"):
        if key in cycle:
            return cycle[key]
    raise KeyError(
        "No se encontro la lista de targets en la respuesta de "
        f"/v1/forecast-cycles/current. Payload completo: {json.dumps(cycle, default=str)}"
    )


PAGE_SIZE = 1000  # Supabase/PostgREST NUNCA devuelve mas de 1000 filas por
# respuesta, sin importar el `limit` que se pida. Asumir un tamano mayor hace
# que la paginacion termine en la primera pagina: el bug que rompio la primera
# corrida contra el ciclo de practica real (solo cargaba 1 de las 12
# estaciones, porque 1000 filas ordenadas por estacion no alcanzan ni para la
# segunda). Se confirma empiricamente con la cabecera Content-Range: 0-999/N.


def fetch_all_rows(client: httpx.Client, url: str, headers: dict, params: dict) -> list[dict]:
    """Pagina una consulta de PostgREST hasta traer todas las filas."""
    rows: list[dict] = []
    start = 0
    while True:
        response = client.get(
            url, headers={**headers, "Range-Unit": "items", "Range": f"{start}-{start + PAGE_SIZE - 1}"}, params=params,
        )
        response.raise_for_status()
        page = response.json()
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            return rows
        start += PAGE_SIZE


def build_anchor_features(client: httpx.Client, supabase_url: str, data_cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruye, por estacion, la fila de features tal como se veian en
    `data_cutoff`.

    Devuelve tambien las observaciones crudas, que necesita el perfil
    adaptativo (promedia dias del mismo tipo con vida media de 14 dias y se
    queda ciego con poca historia). La ventana larga se pide una sola vez y
    la comparten el champion -por `slot_mean`- y el perfil.
    """
    window_start = (data_cutoff - pd.Timedelta(days=DIAS_DE_HISTORIA)).isoformat()
    headers = supabase_headers()

    obs_rows = fetch_all_rows(
        client, f"{supabase_url}/rest/v1/observations", headers,
        {
            "observed_at": [f"gte.{window_start}", f"lte.{data_cutoff.isoformat()}"],
            "select": "station_id,observed_at,demand",
            "order": "station_id,observed_at",
        },
    )
    context_rows = fetch_all_rows(
        client, f"{supabase_url}/rest/v1/context_readings", headers,
        {
            "observed_at": [f"gte.{window_start}", f"lte.{data_cutoff.isoformat()}"],
            # rain_forecast/temperature_forecast son features del modelo: si no
            # se piden aqui, build_feature_frame no puede calcularlas y se
            # mandaban en 0. Pedir de menos en un `select` no falla, solo
            # empobrece la prediccion en silencio.
            "select": (
                "observed_at,rain_mm,rain_forecast,temperature_c,"
                "temperature_forecast,event_intensity"
            ),
            "order": "observed_at",
        },
    )

    observations = pd.DataFrame(obs_rows)
    if observations.empty:
        raise RuntimeError(f"Supabase no devolvio observaciones entre {window_start} y {data_cutoff}")
    observations["station_id"] = observations["station_id"].astype("string")
    observations["observed_at"] = pd.to_datetime(observations["observed_at"])
    context = pd.DataFrame(context_rows)
    context["observed_at"] = pd.to_datetime(context["observed_at"])

    # El champion tambien recibe la ventana larga. `slot_mean` es un promedio
    # ACUMULADO de las semanas previas de cada franja: en entrenamiento junta
    # ~4-6 semanas, y con solo 8 dias de historia en produccion veia UNA sola
    # observacion, un valor mucho mas ruidoso que el que aprendio. Medido con
    # origen rodante en 5 ventanas, dar 28 dias sube el champion de 86.98 a
    # 87.62 de media y de 84.58 a 84.97 en el peor dia. Los roll_* y los lags
    # no cambian: ninguno mira mas de 2 dias atras.
    anchor_frame = build_feature_frame(observations, context)
    anchors = anchor_frame.sort_values("observed_at").groupby("station_id").tail(1).set_index("station_id")
    print(f"Ancla reconstruida para {len(anchors)} estaciones ({len(observations)} observaciones leidas).")
    return anchors, observations


def build_batch_predictions(
    model, feature_columns: list[str], anchor_by_station: pd.DataFrame, targets: list[dict], data_cutoff: pd.Timestamp,
    perfil: PerfilAdaptativo | None = None,
) -> list[dict]:
    """Arma las predicciones del batch. Cada target usa la MISMA fila ancla
    de su estacion (lo unico que cambia entre horizontes es `horizon_minutes`
    dentro del vector de features) - nunca reconstruye lags con datos del
    futuro que todavia no se conocen.

    El horizonte se mide desde el ANCLA REAL (la ultima observacion que
    tenemos), no desde `data_cutoff`. Normalmente son el mismo instante, pero
    si el collector va atrasado el ancla queda antes del corte: decirle al
    modelo "salta 15 minutos" cuando en realidad debe saltar 45 produce
    predicciones sistematicamente malas SIN lanzar ningun error. Se calcula
    la distancia verdadera y se avisa fuerte cuando hay atraso.
    """
    predictions = []
    for target in targets:
        station_id = str(target["station_id"])
        target_at = pd.Timestamp(target["target_at"])

        if station_id not in anchor_by_station.index:
            raise ValueError(f"No hay historico reciente para la estacion {station_id!r} en data_cutoff={data_cutoff}")

        row = anchor_by_station.loc[[station_id]].copy()
        anchor_at = pd.Timestamp(row["observed_at"].iloc[0])
        horizon_minutes = round((target_at - anchor_at).total_seconds() / 60)

        lag_minutes = round((data_cutoff - anchor_at).total_seconds() / 60)
        if lag_minutes > 0:
            print(
                f"  AVISO {station_id}: el ancla ({anchor_at}) va {lag_minutes} min por detras del "
                f"data_cutoff ({data_cutoff}). Horizonte real usado: {horizon_minutes} min."
            )
        if horizon_minutes > max(HORIZONS_MINUTES):
            print(
                f"  AVISO {station_id}: horizonte {horizon_minutes} min supera el maximo entrenado "
                f"({max(HORIZONS_MINUTES)} min). El modelo esta extrapolando; revisar el collector."
            )

        row["horizon_minutes"] = horizon_minutes

        # Hora y dia del instante objetivo, con la MISMA funcion que usa el
        # entrenamiento. Si esto falta, el relleno de abajo las pondria en 0
        # y el modelo recibiria un seno y un coseno valiendo 0 a la vez: un
        # punto imposible que nunca vio. Costaba 18.5 puntos, en silencio.
        for columna, valor in aplicar_features_de_target(target_at).items():
            row[columna] = valor

        # Solo los one-hot de OTRAS estaciones pueden faltar, y para esos 0 es
        # el valor correcto por definicion. Cualquier otra columna ausente es
        # una diferencia real entre entrenar y predecir, y rellenarla con 0
        # produce predicciones malas sin lanzar ningun error: se corta aqui.
        missing = [c for c in feature_columns if c not in row.columns]
        no_dummies = [c for c in missing if not c.startswith("station_")]
        if no_dummies:
            raise ValueError(
                f"La inferencia no sabe calcular estas features que el modelo espera: {no_dummies}. "
                "Rellenarlas con 0 daria predicciones silenciosamente malas."
            )
        for col in missing:
            row[col] = 0  # dummies de otras estaciones: 0 por definicion de one-hot

        value = float(model.predict(row[feature_columns])[0])

        # Segunda opinion: el perfil adaptativo reacciona a un cambio de
        # regimen en la hora siguiente, mientras el champion sigue creyendo
        # en el regimen con el que se entreno. Si no tiene opinion para esta
        # franja, `mezclar` devuelve el champion intacto.
        if perfil is not None:
            valor_perfil = perfil.predecir(station_id, target_at, anchor_at)
            peso = perfil.peso_de_mezcla(station_id, anchor_at)
            if valor_perfil is not None:
                aviso = "  CIERRE: manda el perfil" if peso >= PESO_PERFIL_CIERRE else ""
                print(
                    f"  {station_id} +{horizon_minutes:2d}min: champion={value:8.1f}  "
                    f"perfil={valor_perfil:8.1f}  ->  {mezclar(value, valor_perfil, peso):8.1f}{aviso}"
                )
            value = mezclar(value, valor_perfil, peso)

        value = max(0.0, min(value, 100000.0))
        predictions.append({
            "station_id": station_id,
            "target_at": target_at.isoformat(),
            "value": round(value, 2),
        })
    return predictions


def make_idempotency_key(cycle_id: str, version_id: str) -> str:
    """Estable por (ciclo, modelo): un reintento con el mismo resultado
    reutiliza la misma llave y nunca crea una segunda entrega.
    """
    key = f"{cycle_id}-{version_id}"
    return key[:128].rjust(8, "0")


def make_submission_payload(cycle: dict, champion: dict, predictions: list[dict], client_run_id: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle["cycle_id"],
        "client_run_id": client_run_id,
        "data_cutoff": pd.Timestamp(cycle["data_cutoff"]).isoformat(),
        "model": {
            "version": champion["version_id"][:64],
            "trained_at": champion.get("created_at"),
            "training_data_end": champion["data_cutoff"],
            "git_commit": champion.get("code_commit"),
        },
        "predictions": predictions,
    }


def submit(client: httpx.Client, api_key: str, idempotency_key: str, payload: dict) -> httpx.Response:
    return client.post(
        f"{API_URL}/v1/submissions",
        headers={"Authorization": f"Bearer {api_key}", "Idempotency-Key": idempotency_key},
        json=payload,
    )


def store_predictions_and_submission(
    client: httpx.Client, supabase_url: str, cycle: dict, champion: dict, predictions: list[dict], submission_id: str | None, receipt: dict,
) -> None:
    headers = {**supabase_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"}

    client.post(
        f"{supabase_url}/rest/v1/cycles",
        headers=headers,
        params={"on_conflict": "cycle_id"},
        json=[{
            "cycle_id": cycle["cycle_id"], "opens_at": cycle.get("opens_at", cycle["data_cutoff"]),
            "data_cutoff": cycle["data_cutoff"], "closes_at": cycle["closes_at"], "status": "open",
        }],
    )

    pred_rows = [
        {
            "cycle_id": cycle["cycle_id"], "station_id": p["station_id"], "target_at": p["target_at"],
            "horizon_minutes": round((pd.Timestamp(p["target_at"]) - pd.Timestamp(cycle["data_cutoff"])).total_seconds() / 60),
            "predicted_value": p["value"], "model_version_id": champion["version_id"],
        }
        for p in predictions
    ]
    client.post(
        f"{supabase_url}/rest/v1/predictions",
        headers=headers, params={"on_conflict": "cycle_id,station_id,target_at"}, json=pred_rows,
    )

    if submission_id is not None:
        client.post(
            f"{supabase_url}/rest/v1/submissions",
            headers=headers, params={"on_conflict": "submission_id"},
            json=[{
                "submission_id": submission_id, "cycle_id": cycle["cycle_id"],
                "model_version_id": champion["version_id"], "idempotency_key": receipt.pop("_idempotency_key"),
                "accepted": True, "receipt": receipt,
            }],
        )


def already_submitted(client: httpx.Client, supabase_url: str, cycle_id: str, version_id: str) -> bool:
    """¿Ya entregamos este ciclo con este mismo modelo?

    Protege contra corridas solapadas: el cron dispara varias veces dentro
    de la misma ventana de 25 minutos a proposito (GitHub descarta corridas
    programadas bajo carga, asi que se pide redundancia), y no tiene sentido
    volver a armar y reenviar un batch que la API ya acepto.
    """
    response = client.get(
        f"{supabase_url}/rest/v1/submissions",
        headers=supabase_headers(),
        params={
            "cycle_id": f"eq.{cycle_id}", "model_version_id": f"eq.{version_id}",
            "accepted": "is.true", "select": "submission_id", "limit": 1,
        },
    )
    response.raise_for_status()
    return bool(response.json())


def main() -> None:
    supabase_url = os.environ["SUPABASE_URL"].rstrip("/")

    # Una sola corrida puede cubrir varios minutos de la ventana del ciclo:
    # GitHub Actions retrasa y descarta corridas programadas bajo carga, asi
    # que el workflow arranca seguido Y cada corrida vigila un rato. La guia
    # lo anticipa: "GitHub Actions solo despierta el proceso: la decision
    # final siempre se toma consultando el estado que devuelve la API".
    poll_minutes = float(os.environ.get("PULSO_POLL_MINUTES", "0"))
    poll_interval = float(os.environ.get("PULSO_POLL_INTERVAL_SECONDS", "120"))
    deadline = time.monotonic() + poll_minutes * 60

    while True:
        try:
            status = run_once(supabase_url)
        except Exception as exc:
            # Un fallo pasajero (red, 5xx de la API) no debe quemar el resto
            # de la ventana de vigilancia: se registra y se reintenta. Si ya
            # no queda tiempo, se propaga para que la corrida quede marcada
            # como fallida en Actions en vez de fingir exito.
            if time.monotonic() >= deadline:
                raise
            print(f"Error en el intento ({type(exc).__name__}: {exc}). Se reintenta dentro de la ventana.")
            status = "error"

        if status not in ("no_cycle", "error") or time.monotonic() >= deadline:
            return
        print(f"Reintentando en {poll_interval:.0f}s (quedan {(deadline - time.monotonic()) / 60:.1f} min de vigilancia)...")
        time.sleep(poll_interval)


def run_once(supabase_url: str) -> str:
    with httpx.Client(timeout=60.0) as client:
        clock = client.get(f"{API_URL}/v1/clock").json()
        print(f"Reloj: {clock}")

        cycle = get_current_cycle(client)
        if cycle is None:
            print("Sin ciclo abierto. Termina sin error (comportamiento esperado por la guia).")
            return "no_cycle"

        data_cutoff = pd.Timestamp(cycle["data_cutoff"])
        targets = parse_targets(cycle)
        print(f"Ciclo abierto: {cycle['cycle_id']} | data_cutoff={data_cutoff} | {len(targets)} targets")

        champion = get_champion()
        if already_submitted(client, supabase_url, cycle["cycle_id"], champion["version_id"]):
            print(f"Este ciclo ya fue entregado con {champion['version_id']}. No se reenvia.")
            return "already_submitted"

        bundle = load_model_from_storage(champion["artifact_location"])
        model, feature_columns = bundle["model"], bundle["feature_columns"]

        anchor_by_station, observaciones = build_anchor_features(client, supabase_url, data_cutoff)

        # El perfil es una mejora, no un requisito: si falla por lo que sea,
        # la entrega sale igual con el champion solo. Perder un ciclo cuesta
        # mucho mas que entregarlo un punto peor.
        try:
            perfil = PerfilAdaptativo(observaciones)
        except Exception as exc:
            print(f"AVISO: perfil adaptativo no disponible ({type(exc).__name__}: {exc}). Se entrega con el champion solo.")
            perfil = None

        predictions = build_batch_predictions(model, feature_columns, anchor_by_station, targets, data_cutoff, perfil)
        print(f"Batch armado: {len(predictions)} predicciones (min={min(p['value'] for p in predictions):.1f}, "
              f"max={max(p['value'] for p in predictions):.1f})")

        client_run_id = str(uuid.uuid4())
        payload = make_submission_payload(cycle, champion, predictions, client_run_id)
        idempotency_key = make_idempotency_key(cycle["cycle_id"], champion["version_id"])

        api_key = os.environ.get("PULSO_API_KEY")
        if not api_key:
            print(
                "\nNo hay PULSO_API_KEY todavia (el profesor no la ha entregado). "
                "El batch quedo armado y validado localmente, pero NO se envia."
            )
            print(json.dumps(payload, indent=2, default=str)[:2000])
            return "no_api_key"

        response = submit(client, api_key, idempotency_key, payload)
        if response.status_code >= 300:
            print(f"Submission RECHAZADA: {response.status_code} {response.text[:500]}")
            store_predictions_and_submission(client, supabase_url, cycle, champion, predictions, None, {})
            return "rejected"

        receipt = response.json()
        submission_id = receipt.get("submission_id")
        print(f"Submission ACEPTADA: {submission_id}")
        receipt["_idempotency_key"] = idempotency_key
        store_predictions_and_submission(client, supabase_url, cycle, champion, predictions, submission_id, receipt)
        return "submitted"


if __name__ == "__main__":
    main()
