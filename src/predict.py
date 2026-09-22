"""Carga el modelo champion registrado en Supabase y genera predicciones de
humo (smoke test) para confirmar que el artefacto es reproducible.

No esta conectado a un ciclo real de la API (el reloj de competencia sigue
en `waiting` y no hay targets que la API haya solicitado todavia). En su
lugar, elige un ancla 4 pasos antes del ultimo dato conocido por estacion
(para que los 4 horizontes +15/+30/+45/+60 min todavia caigan dentro del
historico) y compara la prediccion real contra el valor real ya conocido -
una prueba honesta de pronostico, no solo de "el archivo carga".
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import httpx
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

sys.path.insert(0, str(ROOT / "src"))
from features import (  # noqa: E402
    HORIZONS_MINUTES,
    aplicar_features_de_target,
    build_feature_frame,
)


def get_champion() -> dict:
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"]
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            f"{url}/rest/v1/model_versions",
            headers=headers,
            params={"status": "eq.champion", "order": "created_at.desc", "limit": 1},
        )
        response.raise_for_status()
        rows = response.json()
    if not rows:
        raise RuntimeError("No hay ningun modelo champion registrado en model_versions")
    return rows[0]


def load_model_from_storage(artifact_location: str, use_cache: bool = True) -> dict:
    """Descarga el artefacto del champion desde Supabase Storage.

    Guarda una copia local y la reutiliza si ya existe. El champion actual es
    un ensamble de 38 MB y la inferencia corre una vez por ciclo: sin cache
    serian ~6 GB de descarga a la semana, muy por encima de lo que permite el
    plan gratuito de Supabase. Quedarse sin cuota a mitad de competencia
    significa dejar de entregar, y ninguna mejora de accuracy compensa eso.

    El nombre del artefacto incluye la version, asi que un champion nuevo
    nunca reutiliza por error el archivo del anterior.
    """
    bucket, path = artifact_location.replace("supabase-storage://", "").split("/", 1)
    local = ARTIFACTS / path

    if use_cache and local.exists():
        print(f"Modelo tomado de cache local: {local.name} ({local.stat().st_size / 1024 / 1024:.1f} MB)")
        return joblib.load(local)

    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"]
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    with httpx.Client(timeout=120.0) as client:
        response = client.get(f"{url}/storage/v1/object/{bucket}/{path}", headers=headers)
        response.raise_for_status()

    if use_cache:
        ARTIFACTS.mkdir(exist_ok=True)
        local.write_bytes(response.content)
        print(f"Modelo descargado y cacheado: {local.name} ({len(response.content) / 1024 / 1024:.1f} MB)")
    return joblib.load(io.BytesIO(response.content))


def main() -> None:
    champion = get_champion()
    print(f"Champion actual: {champion['version_id']} (metrica={champion['validation_metric']:.2f})")

    bundle = load_model_from_storage(champion["artifact_location"])
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    print(f"Modelo cargado desde Storage. Features esperadas: {feature_columns}")

    observations = pd.read_csv(
        DATA / "observations.csv", dtype={"station_id": "string"}, parse_dates=["observed_at"]
    )
    context = pd.read_csv(DATA / "context.csv", parse_dates=["observed_at"])
    anchor_frame = build_feature_frame(observations, context)

    # Ancla 4 pasos (60 min) antes del ultimo dato: asi los 4 horizontes
    # todavia tienen un valor real conocido para comparar.
    anchors = anchor_frame.sort_values("observed_at").groupby("station_id").nth(-5)
    demand_by_station_time = observations.set_index(["station_id", "observed_at"])["demand"]

    rows = []
    for idx in anchors.index:
        # .loc[[idx]] (no .iterrows(), que transpone a una Series y mezcla
        # los dtypes numericos a "object" - XGBoost los rechaza).
        anchor_row = anchors.loc[[idx]]
        station_id = anchor_row["station_id"].iloc[0]
        observed_at = pd.Timestamp(anchor_row["observed_at"].iloc[0])
        for horizon in HORIZONS_MINUTES:
            target_at = observed_at + pd.Timedelta(minutes=int(horizon))
            actual = demand_by_station_time.get((station_id, target_at))
            if actual is None:
                continue
            feature_row = anchor_row.copy()
            feature_row["horizon_minutes"] = horizon
            # Misma funcion que el entrenamiento. Sin esto, el reindex de
            # abajo las rellenaria con 0 (ver src/infer.py) y este smoke test
            # mediria un modelo peor del que realmente tenemos.
            for columna, valor in aplicar_features_de_target(target_at).items():
                feature_row[columna] = valor
            feature_row = feature_row.reindex(columns=feature_columns, fill_value=0)
            prediction = model.predict(feature_row)[0]
            rows.append({
                "station_id": station_id, "anchor_at": observed_at,
                "horizon_min": horizon, "target_at": target_at, "actual": actual, "prediction": round(prediction, 1),
            })

    result = pd.DataFrame(rows)
    print(f"\nSmoke test (ancla = ultimo dato - 60min, horizontes +15/+30/+45/+60, real vs prediccion):")
    print(result.to_string(index=False))
    wape = (result["actual"] - result["prediction"]).abs().sum() / result["actual"].abs().sum()
    print(f"\nAccuracy agregada de este smoke test: {100 * max(0.0, 1 - wape):.2f} (referencial, no reemplaza la CV de train.py)")


if __name__ == "__main__":
    main()
