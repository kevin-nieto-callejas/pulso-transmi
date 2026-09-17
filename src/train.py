"""Entrena, compara y promueve el primer modelo champion de Pulso TransMi.

Corre 3 candidatos con la misma validacion cruzada temporal (TimeSeriesSplit,
nunca aleatoria) para poder comparar de forma justa:

  1. rf_full          - Random Forest con todas las features (incluye lag_672,
                         la senal de estacionalidad semanal que domina el EDA).
  2. rf_no_weekly_lag - Random Forest SIN lag_672/roll_mean_96/roll_std_96,
                         para medir que tan fragil es el modelo si esa senal
                         deja de ser confiable (esto es exactamente el riesgo
                         de drift que discutimos: un modelo que solo copia
                         "la semana pasada" se rompe cuando el patron cambia).
  3. gbr_full          - Gradient Boosting con todas las features, para
                         comparar contra una familia de algoritmo distinta.

El ganador se promueve como candidato "champion" en Supabase (tabla
model_versions) solo si es el primero (no hay champion previo que superar
todavia). El artefacto se guarda local en artifacts/ y se sube a Supabase
Storage para que sea reproducible desde cualquier maquina, no solo la local.
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from features import ALL_FEATURE_COLUMNS, NO_WEEKLY_LAG_FEATURE_COLUMNS, build_feature_frame, wape_accuracy  # noqa: E402

CANDIDATES = {
    "rf_full": (RandomForestRegressor, ALL_FEATURE_COLUMNS, dict(n_estimators=200, max_depth=10, random_state=20260916, n_jobs=-1)),
    "rf_no_weekly_lag": (RandomForestRegressor, NO_WEEKLY_LAG_FEATURE_COLUMNS, dict(n_estimators=200, max_depth=10, random_state=20260916, n_jobs=-1)),
    "gbr_full": (GradientBoostingRegressor, ALL_FEATURE_COLUMNS, dict(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=20260916)),
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


def upload_to_storage(local_path: Path, remote_path: str) -> str:
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    bucket = "model-artifacts"

    with httpx.Client(timeout=60.0) as client:
        client.post(
            f"{url}/storage/v1/bucket",
            headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"id": bucket, "name": bucket, "public": False},
        )
        with open(local_path, "rb") as f:
            response = client.post(
                f"{url}/storage/v1/object/{bucket}/{remote_path}",
                headers={"apikey": key, "Authorization": f"Bearer {key}"},
                params={"upsert": "true"},
                content=f.read(),
            )
        if response.status_code >= 300:
            raise RuntimeError(f"Storage upload fallo: {response.status_code} {response.text[:300]}")

    return f"supabase-storage://{bucket}/{remote_path}"


def register_model_version(version_id, data_cutoff, features, validation_metric, artifact_location, status) -> None:
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    headers = {
        "apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
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
            headers=headers, params={"on_conflict": "version_id"}, json=payload,
        )
        if response.status_code >= 300:
            raise RuntimeError(f"No se pudo registrar el modelo: {response.status_code} {response.text[:300]}")


def main() -> None:
    observations, context = load_data()
    feature_frame = build_feature_frame(observations, context)

    print("=== Comparacion de candidatos (validacion cruzada temporal, 5 folds) ===")
    results = {}
    for name, (model_cls, feature_columns, params) in CANDIDATES.items():
        accuracy = evaluate_candidate(model_cls, feature_columns, params, feature_frame)
        results[name] = accuracy
        print(f"{name:20s} accuracy_mean={accuracy:.2f}  (n_features={len(feature_columns)})")

    fragility_gap = results["rf_full"] - results["rf_no_weekly_lag"]
    print(f"\nBrecha de fragilidad (rf_full - rf_no_weekly_lag): {fragility_gap:.2f} puntos")
    print("Esa brecha es cuanto se perderia si lag_672 dejara de ser confiable por drift.\n")

    winner_name = max(results, key=results.get)
    print(f"Ganador: {winner_name} (accuracy_mean={results[winner_name]:.2f})")

    model_cls, feature_columns, params = CANDIDATES[winner_name]
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

    register_model_version(
        version_id=version_id,
        data_cutoff=data_cutoff,
        features=feature_columns,
        validation_metric=results[winner_name],
        artifact_location=artifact_location,
        status="champion",
    )
    print(f"Registrado en model_versions como CHAMPION: {version_id}")

    summary = {
        "candidates": results,
        "fragility_gap": fragility_gap,
        "winner": winner_name,
        "version_id": version_id,
        "data_cutoff": data_cutoff,
        "artifact_location": artifact_location,
    }
    (ROOT / "eda" / "reports" / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
