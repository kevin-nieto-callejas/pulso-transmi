"""Bateria de pruebas de esfuerzo del sistema completo.

Hasta aqui se probaron piezas sueltas: tests unitarios, un simulacro de un
ciclo y algunas corridas contra la API. Eso deja sin cubrir lo que mas
importa el dia de la competencia: como se comporta el sistema COMPLETO bajo
cientos de condiciones distintas, y que tan mal se pone cuando algo no sale
como se espera.

Cinco baterias:

  1. **Cobertura temporal**: cientos de ciclos repartidos por hora del dia y
     dia de la semana. Da la distribucion real del accuracy y descubre
     franjas donde el sistema falla de forma sistematica.
  2. **Recolector atrasado**: cuanto cuesta que el ancla vaya vieja. Es el
     fallo operacional mas probable y hasta ahora solo estaba mitigado "a
     ojo".
  3. **Detector de drift**: tasa de falsas alarmas sobre historiales sanos, y
     sensibilidad real ante caidas de distinto tamano. Se acaba de recalibrar
     y no se ha verificado a escala.
  4. **Robustez**: que pasa con estaciones ausentes, horizontes fuera de
     rango y valores extremos.
  5. **Tiempos**: cuanto tarda armar un batch, contra los 25 minutos de
     ventana que da la competencia.

Corre sin tocar la base de datos: usa el historico local y el artefacto del
champion. Asi se puede repetir cuantas veces haga falta sin ensuciar nada ni
gastar cuota.

    python src/stress_test.py
    python src/stress_test.py --ciclos 500
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import evaluate  # noqa: E402
from evaluate import detectar_cambio_en_datos, detectar_degradacion, detectar_falla_operacional  # noqa: E402


def evaluate_ciclos_por_ventana() -> int:
    return evaluate.CICLOS_POR_VENTANA


def _historial_desde_valores(valores):
    ahora = pd.Timestamp.now(tz="UTC")
    n = len(valores)
    return pd.DataFrame([
        {"cycle_id": f"c{i}", "station_id": None, "accuracy": v,
         "computed_at": (ahora - pd.Timedelta(hours=n - 1 - i)).isoformat()}
        for i, v in enumerate(valores)
    ])
from features import (  # noqa: E402
    EXTENDED_FEATURE_COLUMNS,
    HORIZONS_MINUTES,
    build_feature_frame,
    explode_horizons,
    station_dummy_columns,
    wape_accuracy,
)
from train import load_data  # noqa: E402

SEED = 20260921
hallazgos: list[str] = []


def anotar(texto: str) -> None:
    hallazgos.append(texto)
    print(f"    >> HALLAZGO: {texto}")


def entrenar_modelo(frame: pd.DataFrame, columnas: list[str], hasta: pd.Timestamp):
    """Entrena solo con datos anteriores a `hasta`: nada de mirar el futuro."""
    from lightgbm import LGBMRegressor

    train = frame[frame["observed_at"] < hasta].dropna(subset=columnas + ["target_demand"])
    modelo = LGBMRegressor(
        n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    modelo.fit(train[columnas], train["target_demand"])
    return modelo


def accuracy_de_ciclo(grupo: pd.DataFrame) -> float:
    return float(grupo.groupby("station_id").apply(
        lambda g: wape_accuracy(g["target_demand"].values, g["pred"].values), include_groups=False,
    ).mean())


# --- Bateria 1: cobertura temporal ------------------------------------------

def bateria_cobertura(evaluados: pd.DataFrame) -> pd.DataFrame:
    print("\n=== 1. COBERTURA TEMPORAL ===")
    ciclos = evaluados.groupby("observed_at").apply(accuracy_de_ciclo, include_groups=False).rename("acc").reset_index()
    ciclos["hora"] = ciclos["observed_at"].dt.hour
    ciclos["dia"] = ciclos["observed_at"].dt.dayofweek

    print(f"  Ciclos evaluados: {len(ciclos)}")
    print(f"  Accuracy: media {ciclos.acc.mean():.2f}  mediana {ciclos.acc.median():.2f}  "
          f"desviacion {ciclos.acc.std():.2f}")
    print(f"  Percentiles: p05 {ciclos.acc.quantile(.05):.2f}  p50 {ciclos.acc.median():.2f}  "
          f"p95 {ciclos.acc.quantile(.95):.2f}  min {ciclos.acc.min():.2f}")

    print("\n  Por hora del dia:")
    por_hora = ciclos.groupby("hora")["acc"].agg(["mean", "std", "count"])
    peor_h, mejor_h = por_hora["mean"].idxmin(), por_hora["mean"].idxmax()
    for h in range(0, 24, 3):
        if h in por_hora.index:
            r = por_hora.loc[h]
            print(f"    {h:>2}h  {r['mean']:.2f} +/- {r['std']:.2f}  (n={int(r['count'])})")
    brecha_h = por_hora["mean"].max() - por_hora["mean"].min()
    print(f"    Brecha entre la mejor ({mejor_h}h) y la peor ({peor_h}h): {brecha_h:.2f} puntos")
    if brecha_h > 5:
        anotar(f"El accuracy varia {brecha_h:.1f} puntos segun la hora: el umbral de drift "
               "no puede ser un numero fijo comparado contra ciclos sueltos.")

    print("\n  Por dia de la semana:")
    dias = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]
    por_dia = ciclos.groupby("dia")["acc"].agg(["mean", "count"])
    for d, r in por_dia.iterrows():
        print(f"    {dias[d]}  {r['mean']:.2f}  (n={int(r['count'])})")
    brecha_d = por_dia["mean"].max() - por_dia["mean"].min()
    print(f"    Brecha entre dias: {brecha_d:.2f} puntos")
    if brecha_d > 5:
        anotar(f"Hay {brecha_d:.1f} puntos de diferencia entre dias de la semana; "
               "conviene revisar si el modelo distingue bien fin de semana.")

    print("\n  Por horizonte:")
    for h in HORIZONS_MINUTES:
        sub = evaluados[evaluados["horizon_minutes"] == h]
        print(f"    +{h:>2} min  {accuracy_de_ciclo(sub):.2f}")

    print("\n  Estaciones mas dificiles:")
    por_est = evaluados.groupby("station_id").apply(
        lambda g: wape_accuracy(g["target_demand"].values, g["pred"].values), include_groups=False,
    ).sort_values()
    for est, acc in por_est.head(3).items():
        print(f"    {est}  {acc:.2f}")
    if por_est.max() - por_est.min() > 15:
        anotar(f"Entre la mejor y la peor estacion hay {por_est.max()-por_est.min():.1f} puntos "
               f"({por_est.idxmin()} es la mas dificil): vale la pena mirarla aparte.")
    return ciclos


# --- Bateria 2: recolector atrasado -----------------------------------------

def bateria_atraso(frame: pd.DataFrame, columnas: list[str], modelo, anclas_prueba: list[pd.Timestamp]) -> None:
    """Cuanto cuesta que el ancla vaya vieja.

    Es el fallo operacional mas probable: el recolector se retrasa y la
    ultima observacion disponible queda antes del corte del ciclo. El codigo
    ya mide el horizonte desde el ancla real, pero nunca se habia medido
    CUANTO empeora la prediccion.
    """
    print("\n=== 2. RECOLECTOR ATRASADO ===")
    print("  atraso   horizonte efectivo   accuracy   perdida")
    base = None
    for atraso in [0, 15, 30, 60, 120]:
        accs = []
        for ancla in anclas_prueba:
            ancla_real = ancla - pd.Timedelta(minutes=atraso)
            filas = frame[(frame["observed_at"] == ancla_real)].copy()
            if filas.empty:
                continue
            # El ciclo sigue pidiendo +15..+60 desde su propio corte, asi que
            # desde el ancla atrasada el salto real es mayor.
            filas = filas.assign(horizon_minutes=filas["horizon_minutes"] + atraso)
            filas = filas.dropna(subset=columnas + ["target_demand"])
            if filas.empty:
                continue
            filas["pred"] = modelo.predict(filas[columnas])
            accs.append(accuracy_de_ciclo(filas))
        if not accs:
            continue
        media = statistics.mean(accs)
        base = media if base is None else base
        print(f"  {atraso:>3} min   {15+atraso}-{60+atraso} min          {media:>6.2f}    {media-base:>+6.2f}")
        if atraso and media - base < -5:
            anotar(f"Con {atraso} min de atraso del recolector se pierden {base-media:.1f} puntos. "
                   "Conviene vigilar la frescura del ancla, no solo que el workflow corra.")


# --- Bateria 3: detector de drift -------------------------------------------

def _historial_sintetico(n, media, sigma, semilla, caida=0.0, desde=None):
    rng = random.Random(semilla)
    ahora = pd.Timestamp.now(tz="UTC")
    vals, arrastre = [], 0.0
    for i in range(n):
        arrastre = 0.75 * arrastre + rng.gauss(0, sigma * 0.55)
        v = media + arrastre + 1.2 * np.sin(2 * np.pi * i / 24)
        if desde is not None and i >= desde:
            v -= caida
        vals.append(v)
    return pd.DataFrame([
        {"cycle_id": f"c{i}", "station_id": None, "accuracy": v,
         "computed_at": (ahora - pd.Timedelta(hours=n - 1 - i)).isoformat()}
        for i, v in enumerate(vals)
    ])


def bateria_drift() -> None:
    print("\n=== 3. DETECTOR DE DRIFT ===")
    n, media, sigma = 80, 85.74, 2.60

    # Se cuentan EPISODIOS, no chequeos. Una misma excursion del accuracy
    # dispara muchos chequeos seguidos, asi que el porcentaje por chequeo
    # exagera: medido sobre una serie real de 43 dias, 101 alarmas resultaron
    # ser 3 episodios. Lo que importa operativamente es cada cuanto alguien
    # tiene que ir a mirar.
    serie = [v for v in _historial_sintetico(600, media, sigma, 1)["accuracy"]]
    minimo = 2 * evaluate_ciclos_por_ventana() + 24 * 4 + 10
    flags = []
    for i in range(minimo, len(serie)):
        h = _historial_desde_valores(serie[i - minimo:i])
        flags.append(detectar_degradacion(h) is not None)
    episodios = sum(1 for j, f in enumerate(flags) if f and (j == 0 or not flags[j - 1]))
    dias = len(flags) / 24
    por_semana = episodios / (dias / 7) if dias else 0
    print(f"  Historial sano de {dias:.0f} dias: {sum(flags)} chequeos en alarma, "
          f"{episodios} episodio(s) -> {por_semana:.1f} por semana")
    if por_semana > 3:
        anotar(f"El detector levantaria {por_semana:.1f} falsas alarmas por semana: "
               "suficiente para que alguien empiece a ignorarlas.")

    # El historial tiene que ser lo bastante largo para que exista referencia
    # limpia: la ventana actual, otra de separacion, y varios bloques
    # independientes antes. Con historiales cortos el detector se abstiene
    # -correctamente- y la medicion daria 0% en todos los tamanos, que fue lo
    # que paso la primera vez que se corrio esta bateria.
    largo = 2 * evaluate_ciclos_por_ventana() + 24 * 5
    print(f"\n  Sensibilidad (100 historiales de {largo} ciclos por tamano de caida):")
    for caida in [1, 2, 3, 4, 6, 10]:
        detectadas = sum(
            detectar_degradacion(
                _historial_sintetico(largo, media, sigma, s, caida=caida, desde=largo - 30)
            ) is not None
            for s in range(100)
        )
        marca = "  <-- la de perder el lag semanal" if caida == 2 else ""
        print(f"    caida de {caida:>2} puntos  detectada {detectadas:>3}%{marca}")
        if caida >= 6 and detectadas < 70:
            anotar(f"Una caida de {caida} puntos solo se detecta el {detectadas}% de las veces: "
                   "el detector es demasiado conservador para degradaciones grandes.")

    # Falla operacional
    ciclos = pd.DataFrame([{"cycle_id": f"cyc_{i}"} for i in range(10)])
    entregados = pd.DataFrame([{"cycle_id": f"cyc_{i}"} for i in range(7)])
    s = detectar_falla_operacional(ciclos, entregados)
    print(f"\n  Ciclos sin entregar detectados: {'si' if s else 'NO'}")
    if not s:
        anotar("No detecta ciclos sin entregar, que es la senal mas facil de las tres.")


# --- Bateria 4: robustez ----------------------------------------------------

def bateria_robustez(frame: pd.DataFrame, columnas: list[str], modelo) -> None:
    print("\n=== 4. ROBUSTEZ ===")
    from infer import build_batch_predictions

    muestra = frame.dropna(subset=columnas + ["target_demand"]).iloc[-500:]
    ancla_ts = muestra["observed_at"].max()
    anclas = muestra[muestra["observed_at"] == ancla_ts].drop_duplicates("station_id").set_index("station_id")

    casos = {
        "estacion inexistente": [{"station_id": "99999", "target_at": (ancla_ts + pd.Timedelta(minutes=15)).isoformat()}],
        "horizonte enorme (+300 min)": [{"station_id": anclas.index[0], "target_at": (ancla_ts + pd.Timedelta(minutes=300)).isoformat()}],
        "target en el pasado": [{"station_id": anclas.index[0], "target_at": (ancla_ts - pd.Timedelta(minutes=30)).isoformat()}],
        "lista de targets vacia": [],
    }
    for nombre, targets in casos.items():
        try:
            r = build_batch_predictions(modelo, columnas, anclas, targets, ancla_ts)
            valores = [p["value"] for p in r]
            ok = all(0 <= v <= 100000 and np.isfinite(v) for v in valores)
            print(f"  {nombre:30s} -> {len(r)} predicciones, valores validos: {ok}")
            if not ok:
                anotar(f"Con '{nombre}' se generaron valores fuera de rango.")
        except Exception as exc:
            esperado = nombre == "estacion inexistente"
            print(f"  {nombre:30s} -> {type(exc).__name__}: {str(exc)[:60]}")
            if not esperado:
                anotar(f"'{nombre}' lanza {type(exc).__name__} en vez de manejarse; "
                       "en competencia eso tumbaria el batch completo.")

    # Data drift con una estacion que se dispara
    ref = pd.DataFrame([{"station_id": "A", "demand": d} for d in [100, 105, 95, 102, 98, 101]])
    rec = pd.DataFrame([{"station_id": "A", "demand": 400}])
    print(f"  data drift extremo detectado: {'si' if detectar_cambio_en_datos(rec, ref) else 'NO'}")
    if not detectar_cambio_en_datos(rec, ref):
        anotar("Una estacion que cuadruplica su demanda no dispara data drift.")


# --- Bateria 5: tiempos -----------------------------------------------------

def bateria_tiempos(frame: pd.DataFrame, columnas: list[str], modelo) -> None:
    print("\n=== 5. TIEMPOS (ventana de entrega: 25 min) ===")
    from infer import build_batch_predictions

    muestra = frame.dropna(subset=columnas + ["target_demand"])
    ancla_ts = muestra["observed_at"].max()
    anclas = muestra[muestra["observed_at"] == ancla_ts].drop_duplicates("station_id").set_index("station_id")
    targets = [
        {"station_id": s, "target_at": (ancla_ts + pd.Timedelta(minutes=h)).isoformat()}
        for s in anclas.index for h in HORIZONS_MINUTES
    ]
    tiempos = []
    for _ in range(5):
        t0 = time.monotonic()
        build_batch_predictions(modelo, columnas, anclas, targets, ancla_ts)
        tiempos.append(time.monotonic() - t0)
    print(f"  Armar {len(targets)} predicciones: {statistics.mean(tiempos):.2f}s "
          f"(peor {max(tiempos):.2f}s)")
    if max(tiempos) > 60:
        anotar(f"Armar el batch tarda {max(tiempos):.0f}s; con la ventana de 25 min queda justo.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bateria de pruebas de esfuerzo")
    parser.add_argument("--ciclos", type=int, default=300, help="ciclos a evaluar en la bateria temporal")
    args = parser.parse_args()

    print("=" * 62)
    print(" BATERIA DE PRUEBAS DE ESFUERZO — Pulso TransMi")
    print("=" * 62)

    obs, ctx = load_data()
    frame = explode_horizons(build_feature_frame(obs, ctx))
    columnas = EXTENDED_FEATURE_COLUMNS + station_dummy_columns(obs)

    # Corte temporal: se entrena con el pasado y se evalua con el futuro.
    tiempos_unicos = np.sort(frame["observed_at"].unique())
    corte = pd.Timestamp(tiempos_unicos[int(len(tiempos_unicos) * 0.7)])
    print(f"\nEntrenando con datos anteriores a {corte} ...")
    modelo = entrenar_modelo(frame, columnas, corte)

    prueba = frame[frame["observed_at"] >= corte].dropna(subset=columnas + ["target_demand"]).copy()
    anclas_disponibles = np.sort(prueba["observed_at"].unique())
    elegidas = anclas_disponibles[:: max(1, len(anclas_disponibles) // args.ciclos)][: args.ciclos]
    evaluados = prueba[prueba["observed_at"].isin(elegidas)].copy()
    evaluados["pred"] = modelo.predict(evaluados[columnas])
    print(f"Evaluando {len(elegidas)} ciclos ({len(evaluados)} predicciones)")

    bateria_cobertura(evaluados)
    bateria_atraso(prueba, columnas, modelo, [pd.Timestamp(t) for t in elegidas[:40]])
    bateria_drift()
    bateria_robustez(frame, columnas, modelo)
    bateria_tiempos(frame, columnas, modelo)

    print("\n" + "=" * 62)
    if hallazgos:
        print(f" {len(hallazgos)} HALLAZGO(S):")
        for i, h in enumerate(hallazgos, 1):
            print(f"  {i}. {h}")
    else:
        print(" Sin hallazgos: el sistema se comporto como se esperaba.")
    print("=" * 62)


if __name__ == "__main__":
    main()
