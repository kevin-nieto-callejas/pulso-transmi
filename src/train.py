"""Entrena, compara y promueve el modelo champion de Pulso TransMi.

Corre varios candidatos con la misma validacion cruzada temporal
(TimeSeriesSplit, nunca aleatoria) para poder comparar de forma justa:

  1. rf_full          - Random Forest con todas las features.
  2. rf_no_weekly_lag - Random Forest SIN lag_672/roll_mean_96/roll_std_96,
                         para medir que tan fragil es el modelo si esa senal
                         de estacionalidad semanal deja de ser confiable
                         (el riesgo de drift que motiva todo el reto).
  3. gbr_full          - Gradient Boosting (sklearn) con todas las features.
  4. extra_trees_full  - Extra Trees (ensemble mas aleatorizado que RF).
  5. rf_tuned          - Random Forest mas grande (mas arboles, mas profundo).
  6. xgboost_full      - XGBoost, gradient boosting optimizado.
  7. xgboost_station   - XGBoost + identidad de estacion (one-hot). El EDA
                         mostro >3x de diferencia en demanda promedio entre
                         estaciones, pero ningun candidato anterior le decia
                         al modelo explicitamente "en que estacion estas".
  8. xgboost_tuned2    - XGBoost con mas arboles/menor learning rate, para
                         ver si converge a un optimo mejor que xgboost_full.

Regla de promocion (guia metodologica, seccion "El modelo promovido"):
una version nueva reemplaza al champion SOLO si lo supera en la misma
validacion. La novedad por si sola no es mejora. Si no lo supera, se
registra igual como "candidate" (evidencia del experimento) sin tocar
el champion vigente. La base de datos tiene un indice unico parcial
(one_champion_only) que impide tener dos champions a la vez.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from features import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    NO_WEEKLY_LAG_FEATURE_COLUMNS,
    build_feature_frame,
    station_dummy_columns,
    wape_accuracy,
)


def build_candidates(observations: pd.DataFrame) -> dict:
    with_station = ALL_FEATURE_COLUMNS + station_dummy_columns(observations)
    return {
        "rf_full": (
            RandomForestRegressor, ALL_FEATURE_COLUMNS,
            dict(n_estimators=200, max_depth=10, random_state=20260916, n_jobs=-1),
        ),
        "rf_no_weekly_lag": (
            RandomForestRegressor, NO_WEEKLY_LAG_FEATURE_COLUMNS,
            dict(n_estimators=200, max_depth=10, random_state=20260916, n_jobs=-1),
        ),
        "gbr_full": (
            GradientBoostingRegressor, ALL_FEATURE_COLUMNS,
            dict(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=20260916),
        ),
        "extra_trees_full": (
            ExtraTreesRegressor, ALL_FEATURE_COLUMNS,
            dict(n_estimators=300, max_depth=14, random_state=20260916, n_jobs=-1),
        ),
        "rf_tuned": (
            RandomForestRegressor, ALL_FEATURE_COLUMNS,
            dict(n_estimators=500, max_depth=16, min_samples_leaf=2, random_state=20260916, n_jobs=-1),
        ),
        "xgboost_full": (
            XGBRegressor, ALL_FEATURE_COLUMNS,
            dict(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=20260916,
                n_jobs=-1, tree_method="hist",
            ),
        ),
        "xgboost_station": (
            XGBRegressor, with_station,
            dict(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=20260916,
                n_jobs=-1, tree_method="hist",
            ),
        ),
        "xgboost_tuned2": (
            XGBRegressor, ALL_FEATURE_COLUMNS,
            dict(
                n_estimators=800, max_depth=5, learning_rate=0.02,
                subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
                random_state=20260916, n_jobs=-1, tree_method="hist",
            ),
        ),
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    observations = pd.read_csv(
        DATA / "observations.csv", dtype={"station_id": "string"}, parse_dates=["observed_at"]
    )
    context = pd.read_csv(DATA / "context.csv", parse_dates=["observed_at"])
    return observations, context


def evaluate_candidate(model_cls, feature_columns: list[str], params: dict, feature_frame: pd.DataFrame) -> float:
    model_frame = feature_frame.dropna(subset=feature_columns + ["demand"]).sort_values("observed_at")
    unique_times = model_frame["observed_at"].sort_values().unique()
    splitter = TimeSeriesSplit(n_splits=5)
    accuracies = []

    for train_idx, test_idx in splitter.split(unique_times):
        train_times = set(unique_times[train_idx])
        test_times = set(unique_times[test_idx])
        train = model_frame[model_frame["observed_at"].isin(train_times)]
        test = model_frame[model_frame["observed_at"].isin(test_times)]
        if train.empty or test.empty:
            continue

        model = model_cls(**params)
        model.fit(train[feature_columns], train["demand"])
        preds = model.predict(test[feature_columns])

        acc_per_station = (
            test.assign(prediction=preds)
            .groupby("station_id")[["demand", "prediction"]]
            .apply(lambda g: wape_accuracy(g["demand"].values, g["prediction"].values))
        )
        accuracies.append(acc_per_station.mean())

    return sum(accuracies) / len(accuracies)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=*", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def supabase_headers() -> dict:
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def get_current_champion() -> dict | None:
    url = os.environ["SUPABASE_URL"].rstrip("/")
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            f"{url}/rest/v1/model_versions",
            headers=supabase_headers(),
            params={"status": "eq.champion", "select": "version_id,validation_metric"},
        )
        response.raise_for_status()
        rows = response.json()
    return rows[0] if rows else None


def demote_to_historical(version_id: str) -> None:
    url = os.environ["SUPABASE_URL"].rstrip("/")
    with httpx.Client(timeout=30.0) as client:
        response = client.patch(
            f"{url}/rest/v1/model_versions",
            headers={**supabase_headers(), "Prefer": "return=minimal"},
            params={"version_id": f"eq.{version_id}"},
            json={"status": "historical"},
        )
        if response.status_code >= 300:
            raise RuntimeError(f"No se pudo degradar el champion anterior: {response.status_code} {response.text[:300]}")


def upload_to_storage(local_path: Path, remote_path: str) -> str:
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    bucket = "model-artifacts"

    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    with httpx.Client(timeout=60.0) as client:
        exists = client.get(f"{url}/storage/v1/bucket/{bucket}", headers=headers)
        if exists.status_code == 404:
            client.post(
                f"{url}/storage/v1/bucket",
                headers={**headers, "Content-Type": "application/json"},
                json={"id": bucket, "name": bucket, "public": False},
            )
        with open(local_path, "rb") as f:
            response = client.post(
                f"{url}/storage/v1/object/{bucket}/{remote_path}",
                headers=headers,
                params={"upsert": "true"},
                content=f.read(),
            )
        if response.status_code >= 300:
            raise RuntimeError(f"Storage upload fallo: {response.status_code} {response.text[:300]}")

    return f"supabase-storage://{bucket}/{remote_path}"


def register_model_version(version_id, data_cutoff, features, validation_metric, artifact_location, status) -> None:
    url = os.environ["SUPABASE_URL"].rstrip("/")
    payload = {
        "version_id": version_id,
        "data_cutoff": data_cutoff,
        "code_commit": get_git_commit(),
        "features": features,
        "validation_metric": validation_metric,
        "artifact_location": artifact_location,
        "status": status,
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{url}/rest/v1/model_versions",
            headers={**supabase_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"},
            params={"on_conflict": "version_id"}, json=payload,
        )
        if response.status_code >= 300:
            raise RuntimeError(f"No se pudo registrar el modelo: {response.status_code} {response.text[:300]}")


def main() -> None:
    observations, context = load_data()
    feature_frame = build_feature_frame(observations, context)
    candidates = build_candidates(observations)

    print("=== Comparacion de candidatos (validacion cruzada temporal, 5 folds) ===")
    results = {}
    for name, (model_cls, feature_columns, params) in candidates.items():
        accuracy = evaluate_candidate(model_cls, feature_columns, params, feature_frame)
        results[name] = accuracy
        print(f"{name:20s} accuracy_mean={accuracy:.2f}  (n_features={len(feature_columns)})")

    fragility_gap = results["rf_full"] - results["rf_no_weekly_lag"]
    print(f"\nBrecha de fragilidad (rf_full - rf_no_weekly_lag): {fragility_gap:.2f} puntos")

    winner_name = max(results, key=results.get)
    winner_accuracy = results[winner_name]
    print(f"\nMejor candidato de esta corrida: {winner_name} (accuracy_mean={winner_accuracy:.2f})")

    current_champion = get_current_champion()
    if current_champion is not None:
        print(f"Champion vigente: {current_champion['version_id']} (accuracy={current_champion['validation_metric']:.2f})")

    promote = current_champion is None or winner_accuracy > current_champion["validation_metric"]

    model_cls, feature_columns, params = candidates[winner_name]
    model_frame = feature_frame.dropna(subset=feature_columns + ["demand"])
    final_model = model_cls(**params)
    final_model.fit(model_frame[feature_columns], model_frame["demand"])

    data_cutoff = observations["observed_at"].max().isoformat()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_id = f"{winner_name}-{timestamp}"
    artifact_name = f"{version_id}.joblib"
    local_path = ARTIFACTS / artifact_name
    joblib.dump({"model": final_model, "feature_columns": feature_columns}, local_path)
    print(f"\nArtefacto guardado local: {local_path}")

    artifact_location = upload_to_storage(local_path, artifact_name)
    print(f"Artefacto subido a Supabase Storage: {artifact_location}")

    if promote:
        if current_champion is not None:
            demote_to_historical(current_champion["version_id"])
            print(f"Champion anterior degradado a 'historical': {current_champion['version_id']}")
        status = "champion"
        print(f"PROMOVIDO a champion: {version_id} (superó al anterior)" if current_champion else f"PROMOVIDO a champion (primer modelo): {version_id}")
    else:
        status = "candidate"
        print(
            f"NO promovido: {winner_accuracy:.2f} no supera al champion vigente "
            f"({current_champion['validation_metric']:.2f}). Se registra como 'candidate'."
        )

    register_model_version(
        version_id=version_id,
        data_cutoff=data_cutoff,
        features=feature_columns,
        validation_metric=winner_accuracy,
        artifact_location=artifact_location,
        status=status,
    )
    print(f"Registrado en model_versions como {status.upper()}: {version_id}")

    summary = {
        "candidates": results,
        "fragility_gap": fragility_gap,
        "winner_this_run": winner_name,
        "version_id": version_id,
        "promoted": bool(promote),
        "previous_champion": current_champion,
        "data_cutoff": data_cutoff,
        "artifact_location": artifact_location,
    }
    (ROOT / "eda" / "reports" / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
