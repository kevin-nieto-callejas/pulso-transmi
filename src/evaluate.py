"""Evaluacion y monitoreo: aprender del error (Fase 5).

La guia lo describe asi: *"Cuando aparece la realidad, une cada target con su
prediccion, calcula metricas locales, consulta el leaderboard y determina si
hay una senal que merezca investigacion o reentrenamiento"*.

Cuatro pasos:

  1. **Evaluar**: para cada prediccion emitida cuya observacion real ya llego,
     calcular el error absoluto y guardarlo en `prediction_evaluations`.
  2. **Medir**: accuracy por estacion y por ciclo con la formula oficial
     (WAPE por estacion, promedio NO ponderado entre las 12), en `cycle_metrics`.
  3. **Vigilar**: las tres senales que distingue la guia -degradacion de
     rendimiento, cambio en los datos de entrada, y falla operacional-.
  4. **Decidir**: mantener, investigar o reentrenar, con criterio de
     persistencia. *"El reentrenamiento no debe reaccionar a un unico periodo
     dificil"*.

Sobre los umbrales: la guia insiste en que *"deben justificarse, no copiarse
como una cifra universal"*. Los de aqui salen de mediciones propias, no de
numeros redondos elegidos al azar; cada uno explica su origen donde se define.

Corre sin datos sin fallar: antes de la competencia no hay nada que evaluar,
y eso no es un error.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import wape_accuracy  # noqa: E402

# --- Umbrales, con su justificacion ----------------------------------------

# Cuanto puede caer el accuracy antes de considerarlo degradacion real.
# Origen: el candidato `rf_no_weekly_lag` existe para medir que pasa si la
# senal de estacionalidad semanal deja de ser confiable, y esa perdida medida
# fue de 2.11 puntos. Es decir: 2 puntos es la magnitud que tendria un drift
# que nos rompa una feature central. Por debajo de eso es ruido de ciclo.
CAIDA_SIGNIFICATIVA = 2.0

# Cuantos ciclos seguidos degradados antes de actuar. Con ciclos de una hora,
# 3 ciclos son 3 horas: suficiente para descartar una hora punta rara o un
# evento puntual, y poco para no perder medio dia reaccionando tarde.
CICLOS_DE_PERSISTENCIA = 3

# Cambio en la demanda media de una estacion que amerita mirarla. Origen: en
# el historico, la desviacion tipica entre estaciones es grande, asi que se
# compara cada estacion CONTRA SI MISMA y se marca a partir de 3 desviaciones
# de su propia distribucion de entrenamiento.
DESVIACIONES_PARA_DATA_DRIFT = 3.0


@dataclass
class Senal:
    tipo: str          # 'performance' | 'data' | 'operational'
    descripcion: str
    valor: float | None
    umbral: float | None
    accion: str


def supabase_headers() -> dict:
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


# --- Paso 1: unir prediccion con realidad ----------------------------------

def emparejar_con_la_realidad(predicciones: pd.DataFrame, observaciones: pd.DataFrame) -> pd.DataFrame:
    """Une cada prediccion con la observacion real de su target.

    Solo devuelve las que ya tienen realidad conocida: un ciclo se evalua de
    forma progresiva, porque +15 min se resuelve antes que +60.
    """
    if predicciones.empty or observaciones.empty:
        return pd.DataFrame(columns=["prediction_id", "cycle_id", "station_id", "predicted_value", "actual_value"])

    unido = predicciones.merge(
        observaciones.rename(columns={"observed_at": "target_at", "demand": "actual_value"}),
        on=["station_id", "target_at"], how="inner",
    )
    unido["absolute_error"] = (unido["actual_value"] - unido["predicted_value"]).abs()
    return unido


# --- Paso 2: metricas con la formula oficial --------------------------------

def metricas_por_ciclo(evaluaciones: pd.DataFrame) -> pd.DataFrame:
    """Accuracy por estacion y total del ciclo.

    El total es el promedio NO ponderado de las 12 estaciones, como manda la
    guia: *"una estacion de gran volumen no puede ocultar el mal desempeno en
    estaciones mas pequenas"*.
    """
    if evaluaciones.empty:
        return pd.DataFrame(columns=["cycle_id", "station_id", "wape", "accuracy"])

    filas = []
    for cycle_id, ciclo in evaluaciones.groupby("cycle_id"):
        por_estacion = []
        for station_id, g in ciclo.groupby("station_id"):
            real, pred = g["actual_value"].values, g["predicted_value"].values
            wape = abs(real - pred).sum() / abs(real).sum() if abs(real).sum() else None
            accuracy = wape_accuracy(real, pred)
            filas.append({"cycle_id": cycle_id, "station_id": station_id, "wape": wape, "accuracy": accuracy})
            por_estacion.append(accuracy)
        filas.append({
            "cycle_id": cycle_id, "station_id": None,
            "wape": None, "accuracy": sum(por_estacion) / len(por_estacion),
        })
    return pd.DataFrame(filas)


def accuracy_movil(historial: pd.DataFrame, horas: int = 24) -> float | None:
    """Accuracy de las ultimas `horas`. La guia pide la lectura rolling 24 h
    porque muestra el desempeno reciente, no el diluido de toda la competencia."""
    if historial.empty or "station_id" not in historial.columns:
        return None
    corte = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=horas)
    reciente = historial[pd.to_datetime(historial["computed_at"], utc=True) >= corte]
    reciente = reciente[reciente["station_id"].isna()]
    return float(reciente["accuracy"].mean()) if not reciente.empty else None


# --- Paso 3: las tres senales ----------------------------------------------

def detectar_degradacion(historial: pd.DataFrame, metrica_esperada: float) -> Senal | None:
    """Performance drift, comparando SIEMPRE ventanas de 24 h completas.

    Corregido tras un simulacro (`src/simulate_cycle.py`) que midio ciclos
    reales a distintas horas del dia:

        09:45 mañana  85.73     21:45 noche      80.05
        15:45 tarde   85.47     03:45 madrugada  74.99
                                22:45 noche      72.17

    Trece puntos de diferencia sin que nada falle: de madrugada la demanda es
    baja y WAPE castiga mucho los errores sobre valores pequenos. La version
    anterior comparaba los ultimos 3 ciclos contra la metrica de validacion y
    habria declarado degradacion TODAS LAS NOCHES, con riesgo de disparar un
    reentrenamiento por nada.

    La ventana movil de 24 h -la que pide la guia- resuelve el problema
    porque cubre todas las horas del dia, igual que la validacion con la que
    se compara. Por eso tambien se exige que la ventana este razonablemente
    completa antes de juzgar: media ventana vuelve a ser una muestra sesgada
    por la hora.
    """
    if historial.empty or "station_id" not in historial.columns:
        return None

    totales = historial[historial["station_id"].isna()].copy()
    if totales.empty:
        return None
    totales["computed_at"] = pd.to_datetime(totales["computed_at"], utc=True)
    totales = totales.sort_values("computed_at")

    ahora = pd.Timestamp.now(tz="UTC")
    ventanas = []
    for i in range(CICLOS_DE_PERSISTENCIA):
        fin = ahora - pd.Timedelta(hours=i)
        inicio = fin - pd.Timedelta(hours=24)
        ventana = totales[(totales["computed_at"] > inicio) & (totales["computed_at"] <= fin)]
        # Con ciclos de una hora, 24 h son ~24 ciclos. Se exige al menos la
        # mitad para no comparar una franja horaria suelta contra un promedio
        # de dia completo.
        if len(ventana) < 12:
            return None
        ventanas.append(float(ventana["accuracy"].mean()))

    caidas = [metrica_esperada - v for v in ventanas]
    if all(c > CAIDA_SIGNIFICATIVA for c in caidas):
        promedio = sum(ventanas) / len(ventanas)
        return Senal(
            tipo="performance",
            descripcion=(
                f"La ventana movil de 24 h lleva {CICLOS_DE_PERSISTENCIA} lecturas seguidas por "
                f"debajo de lo esperado: {promedio:.2f} frente a {metrica_esperada:.2f} de la "
                f"validacion (caida media de {sum(caidas)/len(caidas):.2f} puntos)."
            ),
            valor=promedio, umbral=metrica_esperada - CAIDA_SIGNIFICATIVA,
            accion="Investigar antes de reentrenar: revisar si la caida se concentra en pocas estaciones u horizontes.",
        )
    return None


def detectar_cambio_en_datos(reciente: pd.DataFrame, referencia: pd.DataFrame) -> list[Senal]:
    """Data drift: la demanda de una estacion se aleja de lo que el modelo vio.

    Se compara cada estacion contra SI MISMA, no contra las demas: entre
    estaciones hay mas de 3x de diferencia en demanda media, asi que un umbral
    comun no significaria nada.
    """
    if reciente.empty or referencia.empty:
        return []

    base = referencia.groupby("station_id")["demand"].agg(["mean", "std"])
    ahora = reciente.groupby("station_id")["demand"].mean()

    senales = []
    for station_id, media_actual in ahora.items():
        if station_id not in base.index:
            continue
        media_base, desv = base.loc[station_id, "mean"], base.loc[station_id, "std"]
        if not desv or pd.isna(desv):
            continue
        z = abs(media_actual - media_base) / desv
        if z > DESVIACIONES_PARA_DATA_DRIFT:
            direccion = "por encima" if media_actual > media_base else "por debajo"
            senales.append(Senal(
                tipo="data",
                descripcion=(
                    f"Estacion {station_id}: demanda media reciente {media_actual:.0f} vs {media_base:.0f} "
                    f"del entrenamiento, {z:.1f} desviaciones {direccion}."
                ),
                valor=float(z), umbral=DESVIACIONES_PARA_DATA_DRIFT,
                accion="Revisar si es un cambio real de la ciudad o un problema de ingesta.",
            ))
    return senales


def detectar_falla_operacional(ciclos: pd.DataFrame, submissions: pd.DataFrame) -> Senal | None:
    """Ciclos que se cerraron sin que entregaramos nada.

    La guia la lista como senal propia y advierte: *"corregir la operacion
    antes de culpar al modelo"*. Un accuracy bajo por no haber entregado no
    se arregla reentrenando.
    """
    if ciclos.empty or "cycle_id" not in ciclos.columns:
        return None
    entregados = set(submissions["cycle_id"]) if ("cycle_id" in submissions.columns and not submissions.empty) else set()
    sin_entregar = [c for c in ciclos["cycle_id"] if c not in entregados]
    if not sin_entregar:
        return None
    return Senal(
        tipo="operational",
        descripcion=f"{len(sin_entregar)} ciclo(s) sin submission registrada: {', '.join(sin_entregar[:5])}.",
        valor=float(len(sin_entregar)), umbral=0.0,
        accion="Revisar los logs de GitHub Actions: el problema es del pipeline, no del modelo.",
    )


# --- Paso 4: decidir --------------------------------------------------------

def decidir(senales: list[Senal], observaciones_nuevas: int, horas_desde_entrenamiento: float) -> str:
    """Mantener, investigar o reentrenar.

    La guia: *"La decision considera persistencia, volumen de datos nuevos y
    tiempo transcurrido desde el ultimo entrenamiento"*. Reentrenar sin datos
    nuevos suficientes produce un modelo casi identico con otro nombre.
    """
    if not senales:
        return "MANTENER: ninguna senal activa."

    tipos = {s.tipo for s in senales}
    if "operational" in tipos:
        return ("INVESTIGAR (operacion): hay ciclos sin entregar. Arreglar el pipeline antes de tocar el modelo; "
                "un accuracy bajo por no haber entregado no se corrige reentrenando.")

    if "performance" in tipos:
        # Un dia entero de datos nuevos son 1152 observaciones (12 estaciones
        # x 96 periodos). Por debajo de eso, reentrenar es repetir el mismo
        # modelo con otro nombre.
        if observaciones_nuevas < 1152:
            return (f"INVESTIGAR: hay degradacion sostenida pero solo {observaciones_nuevas} observaciones nuevas "
                    "(menos de un dia). Reentrenar ahora daria practicamente el mismo modelo.")
        if horas_desde_entrenamiento < 6:
            return ("INVESTIGAR: hay degradacion pero el champion se entreno hace menos de 6 horas. "
                    "Revisar primero si la causa es un cambio real o un problema de datos.")
        return (f"REENTRENAR: degradacion sostenida, {observaciones_nuevas} observaciones nuevas y "
                f"{horas_desde_entrenamiento:.0f} h desde el ultimo entrenamiento. "
                "Ejecutar train.py; la regla de promocion decide si el candidato entra.")

    return "INVESTIGAR (datos): cambio en la distribucion de entrada sin degradacion de accuracy todavia. Vigilar."


# --- Conexion con Supabase --------------------------------------------------

def traer(client: httpx.Client, url: str, tabla: str, params: dict) -> pd.DataFrame:
    """Lee una tabla paginando: PostgREST corta en 1000 filas por respuesta."""
    filas, inicio = [], 0
    while True:
        r = client.get(
            f"{url}/rest/v1/{tabla}",
            headers={**supabase_headers(), "Range-Unit": "items", "Range": f"{inicio}-{inicio + 999}"},
            params=params,
        )
        r.raise_for_status()
        pagina = r.json()
        filas.extend(pagina)
        if len(pagina) < 1000:
            return pd.DataFrame(filas)
        inicio += 1000


def guardar(client: httpx.Client, url: str, tabla: str, filas: list[dict], on_conflict: str | None = None) -> None:
    if not filas:
        return
    params = {"on_conflict": on_conflict} if on_conflict else {}
    prefer = "resolution=merge-duplicates,return=minimal" if on_conflict else "return=minimal"
    r = client.post(f"{url}/rest/v1/{tabla}", headers={**supabase_headers(), "Prefer": prefer},
                    params=params, json=filas)
    if r.status_code >= 300:
        raise RuntimeError(f"No se pudo escribir en {tabla}: {r.status_code} {r.text[:300]}")


def main() -> None:
    url = os.environ["SUPABASE_URL"].rstrip("/")
    with httpx.Client(timeout=90.0) as client:
        predicciones = traer(client, url, "predictions",
                             {"select": "id,cycle_id,station_id,target_at,predicted_value,model_version_id"})
        if predicciones.empty:
            print("Todavia no hay predicciones emitidas. Nada que evaluar (esperado antes de la competencia).")
            return

        ya_evaluadas = traer(client, url, "prediction_evaluations", {"select": "prediction_id"})
        pendientes = predicciones[~predicciones["id"].isin(
            ya_evaluadas["prediction_id"] if not ya_evaluadas.empty else [])]
        print(f"Predicciones emitidas: {len(predicciones)} | pendientes de evaluar: {len(pendientes)}")

        observaciones = traer(client, url, "observations", {"select": "station_id,observed_at,demand"})
        nuevas = emparejar_con_la_realidad(pendientes, observaciones)
        print(f"Con realidad ya conocida: {len(nuevas)}")

        if not nuevas.empty:
            guardar(client, url, "prediction_evaluations", [
                {"prediction_id": int(r.id), "actual_value": float(r.actual_value),
                 "absolute_error": float(r.absolute_error)} for r in nuevas.itertuples()
            ], on_conflict="prediction_id")

            metricas = metricas_por_ciclo(nuevas)
            guardar(client, url, "cycle_metrics", [
                {"cycle_id": r["cycle_id"], "station_id": r["station_id"],
                 "wape": None if pd.isna(r["wape"]) else float(r["wape"]), "accuracy": float(r["accuracy"])}
                for _, r in metricas.iterrows()
            ])
            print(f"Metricas calculadas para {metricas['cycle_id'].nunique()} ciclo(s).")

        historial = traer(client, url, "cycle_metrics",
                          {"select": "cycle_id,station_id,accuracy,computed_at", "order": "computed_at.desc"})
        champion = traer(client, url, "model_versions",
                         {"select": "version_id,validation_metric,created_at,data_cutoff", "status": "eq.champion"})
        if champion.empty:
            print("No hay champion registrado; se omite el monitoreo.")
            return
        champion = champion.iloc[0]

        movil = accuracy_movil(historial)
        print(f"\nAccuracy movil 24 h: {f'{movil:.2f}' if movil is not None else 'sin datos todavia'}"
              f"  (validacion del champion: {champion['validation_metric']:.2f})")

        corte = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
        reciente = observaciones[pd.to_datetime(observaciones["observed_at"], utc=True) >= corte]
        referencia = observaciones[pd.to_datetime(observaciones["observed_at"], utc=True)
                                   <= pd.Timestamp(champion["data_cutoff"])]

        senales: list[Senal] = []
        degradacion = detectar_degradacion(historial, float(champion["validation_metric"]))
        if degradacion:
            senales.append(degradacion)
        senales += detectar_cambio_en_datos(reciente, referencia)
        falla = detectar_falla_operacional(
            traer(client, url, "cycles", {"select": "cycle_id,status"}),
            traer(client, url, "submissions", {"select": "cycle_id"}),
        )
        if falla:
            senales.append(falla)

        print(f"\n=== Senales activas: {len(senales)} ===")
        for s in senales:
            print(f"  [{s.tipo}] {s.descripcion}")
            print(f"     -> {s.accion}")

        horas = (datetime.now(timezone.utc) - pd.Timestamp(champion["created_at"]).to_pydatetime()).total_seconds() / 3600
        nuevas_obs = int((pd.to_datetime(observaciones["observed_at"], utc=True)
                          > pd.Timestamp(champion["data_cutoff"])).sum())
        decision = decidir(senales, nuevas_obs, horas)
        print(f"\n=== DECISION ===\n  {decision}")

        if senales:
            guardar(client, url, "drift_signals", [
                {"signal_type": s.tipo, "description": s.descripcion, "metric_value": s.valor,
                 "threshold_value": s.umbral, "action_taken": decision[:400]} for s in senales
            ])
            print("\nSenales registradas en drift_signals.")


if __name__ == "__main__":
    main()
