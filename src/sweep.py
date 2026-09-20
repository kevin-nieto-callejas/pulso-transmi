"""Barrido de experimentos con seguimiento en MLflow.

La guia coloca MLflow como "bono de madurez": *"aporta una interfaz
especializada para registrar experimentos, metricas, artefactos, versiones y
linaje. Su uso suma valor si ayuda a responder mejor que se entreno, con
cuales datos y por que se promovio"*.

Este script explora muchas combinaciones de hiperparametros y conjuntos de
features, y registra CADA intento en MLflow con sus parametros, su metrica
global y su desglose por horizonte. A diferencia de `train.py` -que compara 8
candidatos elegidos a mano y decide la promocion- aqui el objetivo es
explorar en volumen para descubrir si existe una configuracion mejor.

Busqueda de dos etapas (coarse-to-fine), no fuerza bruta ciega:

  1. Exploracion amplia con validacion mas barata (3 cortes temporales) para
     recorrer muchas combinaciones en tiempo razonable.
  2. Los mejores se revalidan con el protocolo completo de `train.py`
     (5 cortes) antes de considerarlos promovibles, porque comparar contra el
     champion exige la MISMA validacion con la que se midio el champion.

Nada aqui promueve un modelo: promover sigue siendo decision de `train.py`,
que aplica la regla de la guia (una version nueva reemplaza al champion solo
si lo supera en la misma validacion).

Uso:
    python src/sweep.py                 # 60 experimentos (por defecto)
    python src/sweep.py --n-trials 300  # barrido grande
    mlflow ui                           # explorar los resultados
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import mlflow
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import (  # noqa: E402
    HORIZONS_MINUTES,
    MULTI_HORIZON_FEATURE_COLUMNS,
    MULTI_HORIZON_NO_WEEKLY_LAG_FEATURE_COLUMNS,
    build_feature_frame,
    explode_horizons,
    station_dummy_columns,
)
from train import evaluate_candidate, load_data  # noqa: E402

EXPERIMENT_NAME = "pulso-transmi-xgboost"
SEED = 20260918


def tracking_uri() -> str:
    """Donde guarda MLflow el historial de experimentos.

    MLflow 3.x retiro el backend de archivos y exige base de datos. Se usa
    SQLite, pero NO dentro del proyecto: este repo vive en una ruta de red
    (`\\\\wsl.localhost\\...`) y SQLite no puede tomar bloqueos ahi
    ("database is locked"). Por eso la base va a una carpeta local del
    usuario. Es estado local reproducible -se regenera corriendo el barrido-;
    la evidencia que se versiona es `eda/reports/sweep_results.csv`.

    Se puede sobrescribir con MLFLOW_TRACKING_URI (por ejemplo para apuntar a
    un Postgres compartido).
    """
    configurado = os.environ.get("MLFLOW_TRACKING_URI")
    if configurado:
        return configurado
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "pulso-transmi"
    base.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{base.as_posix()}/mlflow.db"


def espacio_de_busqueda(rng: random.Random) -> dict:
    """Una combinacion al azar del espacio de hiperparametros."""
    return {
        "n_estimators": rng.choice([200, 300, 400, 600, 800, 1200]),
        "max_depth": rng.choice([4, 5, 6, 7, 8, 10]),
        "learning_rate": rng.choice([0.01, 0.02, 0.05, 0.08, 0.1]),
        "subsample": rng.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
        "colsample_bytree": rng.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
        "min_child_weight": rng.choice([1, 3, 5, 10]),
        "reg_lambda": rng.choice([0.5, 1.0, 2.0, 5.0]),
        "reg_alpha": rng.choice([0.0, 0.1, 0.5, 1.0]),
    }


def conjuntos_de_features(observations: pd.DataFrame) -> dict[str, list[str]]:
    """Variantes de features a explorar, no solo hiperparametros.

    El hallazgo de la ronda anterior fue que agregar una senal que faltaba
    (la identidad de la estacion) movio mas la aguja que duplicar el numero
    de arboles. Por eso el barrido explora tambien el conjunto de features.
    """
    dummies = station_dummy_columns(observations)
    return {
        "con_estacion": MULTI_HORIZON_FEATURE_COLUMNS + dummies,
        "base": MULTI_HORIZON_FEATURE_COLUMNS,
        "sin_lag_semanal": MULTI_HORIZON_NO_WEEKLY_LAG_FEATURE_COLUMNS + dummies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Barrido de experimentos con MLflow")
    parser.add_argument("--n-trials", type=int, default=60, help="cuantas combinaciones probar")
    parser.add_argument("--n-splits", type=int, default=3, help="cortes temporales de la exploracion")
    parser.add_argument(
        "--desglose-horizontes", action="store_true",
        help="registra tambien el accuracy por horizonte (5x mas lento; util solo en barridos cortos)",
    )
    args = parser.parse_args()

    rng = random.Random(SEED)
    observations, context = load_data()
    feature_frame = explode_horizons(build_feature_frame(observations, context))
    variantes = conjuntos_de_features(observations)
    data_cutoff = observations["observed_at"].max().isoformat()

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"Backend de MLflow: {tracking_uri()}")
    print(f"Registrando en MLflow: experimento '{EXPERIMENT_NAME}' ({args.n_trials} intentos, {args.n_splits} cortes)")

    resultados = []
    fallidos = 0
    for i in range(1, args.n_trials + 1):
        params = espacio_de_busqueda(rng)
        nombre_features = rng.choice(list(variantes))
        columnas = variantes[nombre_features]
        inicio = time.monotonic()

        try:
            accuracy, duracion = correr_intento(i, params, nombre_features, columnas, feature_frame, data_cutoff, args)
        except Exception as exc:
            # Un barrido largo no puede morirse por un intento. Causas tipicas:
            # contencion de SQLite si algo mas consulta la base al tiempo, o una
            # combinacion de hiperparametros que el modelo rechaza. Se registra
            # y se sigue; al final se reporta cuantos fallaron.
            fallidos += 1
            print(f"  [{i:>4}/{args.n_trials}] FALLO ({type(exc).__name__}: {exc}). Se continua.")
            continue

        resultados.append({"trial": i, "feature_set": nombre_features, "accuracy": accuracy, **params})
        print(f"  [{i:>4}/{args.n_trials}] {nombre_features:16s} accuracy={accuracy:.2f}  ({duracion:.0f}s)")

        # Guardado incremental: si el proceso se corta, no se pierde el avance.
        if i % 10 == 0:
            pd.DataFrame(resultados).sort_values("accuracy", ascending=False).to_csv(
                ROOT / "eda" / "reports" / "sweep_results.csv", index=False,
            )

    tabla = pd.DataFrame(resultados).sort_values("accuracy", ascending=False)
    salida = ROOT / "eda" / "reports" / "sweep_results.csv"
    tabla.to_csv(salida, index=False)

    print(f"\nExperimentos completados: {len(resultados)} | fallidos: {fallidos}")
    print(f"Resultados guardados en {salida}")
    print("\n=== TOP 10 ===")
    print(tabla.head(10).to_string(index=False))
    print(
        "\nOjo: esta exploracion usa validacion mas barata que train.py. Para "
        "comparar contra el champion hay que revalidar con el protocolo "
        "completo (5 cortes), que es el que midio al champion."
    )


def correr_intento(i, params, nombre_features, columnas, feature_frame, data_cutoff, args) -> tuple[float, float]:
    """Ejecuta y registra un experimento. Devuelve (accuracy, segundos)."""
    inicio = time.monotonic()
    with mlflow.start_run(run_name=f"trial-{i:04d}"):
        mlflow.log_params(params)
        mlflow.log_param("feature_set", nombre_features)
        mlflow.log_param("n_features", len(columnas))
        mlflow.log_param("n_splits", args.n_splits)
        # Linaje de datos: hasta que instante vio datos este experimento.
        mlflow.log_param("data_cutoff", data_cutoff)
        mlflow.set_tag("etapa", "exploracion")
        mlflow.set_tag("modelo", "XGBRegressor")

        modelo_params = {**params, "random_state": SEED, "n_jobs": -1, "tree_method": "hist"}
        accuracy = evaluate_candidate(
            XGBRegressor, columnas, modelo_params, feature_frame, n_splits=args.n_splits,
        )
        mlflow.log_metric("accuracy", accuracy)

        # Desglose por horizonte: un promedio bueno puede esconder que el
        # modelo se cae justo en +60 min, que es el horizonte mas dificil.
        # Cuesta 4 evaluaciones extra (5x mas lento), asi que en un barrido
        # amplio se omite y se calcula despues solo para los mejores.
        if args.desglose_horizontes:
            for horizonte in HORIZONS_MINUTES:
                subset = feature_frame[feature_frame["horizon_minutes"] == horizonte]
                acc_h = evaluate_candidate(
                    XGBRegressor, columnas, modelo_params, subset, n_splits=args.n_splits,
                )
                mlflow.log_metric(f"accuracy_h{horizonte}", acc_h)

        duracion = time.monotonic() - inicio
        mlflow.log_metric("segundos", duracion)
        return accuracy, duracion


if __name__ == "__main__":
    main()
