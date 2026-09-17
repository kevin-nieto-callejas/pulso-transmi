"""Migra el historico local (../data/*.csv) a Supabase via PostgREST.

Usa upsert (Prefer: resolution=merge-duplicates) sobre las mismas llaves
unicas del esquema, por lo que volver a correr este script es idempotente:
no duplica filas si ya existen.

RLS esta activo con politicas de solo lectura publica: este script necesita
SUPABASE_SERVICE_ROLE_KEY (Supabase Dashboard > Settings > API) para poder
escribir. La anon/publishable key (SUPABASE_KEY) ya no alcanza para insertar
o actualizar filas. Nunca uses la service_role key en un entorno que llegue
al navegador (Vercel, frontend); solo en scripts locales o GitHub Actions
Secrets.
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"]

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates,return=minimal",
}


def upsert(table: str, rows: list[dict], on_conflict: str, batch_size: int = 2000) -> None:
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    params = {"on_conflict": on_conflict}
    with httpx.Client(timeout=60.0) as client:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            response = client.post(url, headers=HEADERS, params=params, json=batch)
            if response.status_code >= 300:
                raise RuntimeError(
                    f"{table}: fallo batch {start}-{start + len(batch)}: "
                    f"{response.status_code} {response.text[:500]}"
                )
            print(f"{table}: filas {start}-{start + len(batch)} ok")


def main() -> None:
    stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": "string"})
    observations = pd.read_csv(
        DATA / "observations.csv", dtype={"station_id": "string"}, parse_dates=["observed_at"]
    )
    context = pd.read_csv(DATA / "context.csv", parse_dates=["observed_at"])

    station_rows = stations.to_dict(orient="records")
    upsert("stations", station_rows, on_conflict="station_id")

    obs = observations.copy()
    obs["observed_at"] = obs["observed_at"].apply(lambda ts: ts.isoformat())
    obs_rows = obs[["station_id", "observed_at", "demand"]].to_dict(orient="records")
    upsert("observations", obs_rows, on_conflict="station_id,observed_at")

    ctx = context.copy()
    ctx["observed_at"] = ctx["observed_at"].apply(lambda ts: ts.isoformat())
    ctx_rows = ctx.to_dict(orient="records")
    upsert("context_readings", ctx_rows, on_conflict="observed_at")

    print(f"Migracion completa: {len(station_rows)} estaciones, "
          f"{len(obs_rows)} observaciones, {len(ctx_rows)} lecturas de contexto")


if __name__ == "__main__":
    main()
