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

# Cuantas desviaciones tipicas por debajo de su propio historial debe caer el
# accuracy para considerarlo degradacion.
#
# Medido sobre 2.229 ciclos evaluados fuera de muestra:
#   - accuracy por ciclo suelto: media 85.74, desviacion 2.60 (min 57.79)
#   - promedio movil de 24 ciclos: desviacion 1.51
#
# Un umbral fijo en puntos no sirve: 2 puntos sobre ventanas de 24 ciclos son
# apenas 1.3 desviaciones, es decir alarma por azar una de cada diez veces.
# Por eso el umbral se expresa en desviaciones y se calcula contra la
# variabilidad REAL observada, no contra una cifra elegida de antemano.
SIGMAS_PARA_ALARMA = 3.0

# Ciclos por ventana. Con ciclos de una hora, 24 cubren un dia completo -la
# lectura rolling 24 h que pide la guia- y reducen el ruido de 2.60 a 1.51.
CICLOS_POR_VENTANA = 24

# Ventanas necesarias para tener un historial de referencia. Sin base propia
# no se puede hablar de degradacion: haria falta saber contra que.
VENTANAS_DE_REFERENCIA = 24

# Se conserva como referencia documental: es la caida medida al quitarle al
# modelo la estacionalidad semanal, o sea la magnitud de un drift que rompe
# una feature central. Queda por DEBAJO del ruido de una ventana de 24 h
# (3 sigmas = 4.5 puntos), asi que este detector no la vera; hacen falta
# ventanas mas largas para eso. Es una limitacion conocida, no un descuido.
CAIDA_POR_PERDER_LAG_SEMANAL = 2.11

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

def detectar_degradacion(historial: pd.DataFrame, metrica_esperada: float | None = None) -> Senal | None:
    """Performance drift: el accuracy cae respecto a SU PROPIO historial.

    Dos correcciones sobre la version anterior, ambas nacidas de medir en vez
    de suponer (2.229 ciclos evaluados fuera de muestra):

    1. **La referencia no puede ser la metrica de validacion.** Esa cifra
       (86.61) se calcula agrupando todas las predicciones de la partición;
       el accuracy de un ciclo suelto promedia 85.74. Son formas distintas de
       agregar lo mismo, asi que comparar una contra otra mete un sesgo de
       0.87 puntos que hace sobre-disparar la alarma. Se compara la ventana
       reciente contra la mediana de las ventanas anteriores: drift es
       "cambio respecto a como venia", no "difiere del numero de
       entrenamiento".

    2. **El umbral debe salir de la variabilidad real.** Un ciclo suelto
       tiene desviacion 2.60 (llega a bajar a 57.79 sin que nada falle);
       promediar 24 la baja a 1.51. Un umbral fijo de 2 puntos sobre esa
       ventana son 1.3 desviaciones: alarma por azar una de cada diez veces.
       Aqui el umbral son 3 desviaciones medidas sobre el propio historial.

    `metrica_esperada` se acepta solo para compatibilidad; no se usa como
    referencia por lo explicado en (1).
    """
    if historial.empty or "station_id" not in historial.columns:
        return None

    totales = historial[historial["station_id"].isna()].copy()
    if totales.empty:
        return None
    totales["computed_at"] = pd.to_datetime(totales["computed_at"], utc=True)
    totales = totales.sort_values("computed_at")

    # Hace falta la ventana actual mas suficientes anteriores para saber cual
    # es el comportamiento normal de ESTE modelo en ESTA competencia.
    minimo = CICLOS_POR_VENTANA + VENTANAS_DE_REFERENCIA
    if len(totales) < minimo:
        return None

    # Mediana y no promedio: un solo ciclo catastrofico -medido, bajan hasta
    # 57.79 sin que nada falle- arrastra el promedio de 24 mas de un punto,
    # suficiente para inventar una alarma. La mediana de 24 valores ni se
    # entera de uno suelto, pero se mueve entera si la caida es sostenida,
    # que es justo lo que se quiere detectar.
    ventanas = totales["accuracy"].rolling(CICLOS_POR_VENTANA).median().dropna()
    actual = float(ventanas.iloc[-1])
    referencia = ventanas.iloc[:-1]
    base = float(referencia.median())
    sigma = float(referencia.std())

    if not sigma or pd.isna(sigma):
        return None

    umbral = base - SIGMAS_PARA_ALARMA * sigma
    if actual >= umbral:
        return None

    return Senal(
        tipo="performance",
        descripcion=(
            f"El accuracy tipico de las ultimas {CICLOS_POR_VENTANA} horas cayo a {actual:.2f}, "
            f"{(base - actual) / sigma:.1f} desviaciones por debajo de su propio historial "
            f"(base {base:.2f}, desviacion {sigma:.2f})."
        ),
        valor=actual, umbral=umbral,
        accion="Investigar antes de reentrenar: revisar si la caida se concentra en pocas estaciones u horizontes.",
    )


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
