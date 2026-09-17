"""Carga el modelo champion registrado en Supabase y genera una prediccion
de humo (smoke test) para confirmar que el artefacto es reproducible.

No esta conectado a un ciclo real de la API (el reloj de competencia sigue
en `waiting` y no hay targets que la API haya solicitado todavia). Esto
solo demuestra: "el modelo guardado se puede volver a cargar y predecir".
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

sys.path.insert(0, str(ROOT / "src"))
from features import build_feature_frame  # noqa: E402


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


def load_model_from_storage(artifact_location: str) -> dict:
    bucket, path = artifact_location.replace("supabase-storage://", "").split("/", 1)
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"]
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    with httpx.Client(timeout=60.0) as client:
        response = client.get(f"{url}/storage/v1/object/{bucket}/{path}", headers=headers)
        response.raise_for_status()
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
    feature_frame = build_feature_frame(observations, context).dropna(subset=feature_columns)

    last_rows = feature_frame.sort_values("observed_at").groupby("station_id").tail(1)
    predictions = model.predict(last_rows[feature_columns])
    result = last_rows[["station_id", "observed_at", "demand"]].assign(prediction=predictions)
    print("\nSmoke test (ultimo periodo conocido por estacion, real vs prediccion):")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
