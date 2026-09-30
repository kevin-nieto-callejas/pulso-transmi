"""Exporta observaciones y contexto frescos de Supabase a data/*.csv.

`train.py` lee sus datos de `data/observations.csv` y `data/context.csv` en
disco (no consulta Supabase directamente) - un diseno pensado para correr
localmente contra una foto ya descargada. El problema: esa foto se creo una
sola vez a mano el 16-sep y nunca se volvio a actualizar, y `data/` esta en
.gitignore (nunca llega al runner de GitHub Actions). El primer intento de
reentreno automatico (30-sep) fallo con `FileNotFoundError` por esto mismo.

Este script cierra ese hueco: reconstruye los dos CSV con el MISMO esquema
que ya usaba `train.py`, pero leyendo todo el historico actual de Supabase
(`observations` y `context_readings`, las mismas tablas que llena
`src/ingest.py`). Se corre una vez antes de `train.py` en el workflow de
reentreno, asi el modelo siempre se entrena con los datos mas recientes
disponibles - incluidas las horas de drift severo que motivaron el reentreno.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def supabase_headers() -> dict:
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return {"apikey": key, "Authorization": f"Bearer {key}"}


def traer(client: httpx.Client, url: str, tabla: str, columnas: str) -> pd.DataFrame:
    """Lee una tabla completa paginando (PostgREST corta en 1000 filas)."""
    filas, inicio = [], 0
    while True:
        r = client.get(
            f"{url}/rest/v1/{tabla}",
            headers={**supabase_headers(), "Range-Unit": "items", "Range": f"{inicio}-{inicio + 999}"},
            params={"select": columnas, "order": "observed_at.asc"},
        )
        r.raise_for_status()
        pagina = r.json()
        filas.extend(pagina)
        if len(pagina) < 1000:
            return pd.DataFrame(filas)
        inicio += 1000


def main() -> None:
    url = os.environ["SUPABASE_URL"].rstrip("/")
    DATA.mkdir(exist_ok=True)

    with httpx.Client(timeout=60.0) as client:
        observations = traer(client, url, "observations", "station_id,observed_at,demand")
        context = traer(client, url, "context_readings",
                         "observed_at,rain_mm,rain_forecast,temperature_c,temperature_forecast,event_intensity")

    if observations.empty:
        print("Sin observaciones en Supabase: no se sobrescriben los CSV existentes.", file=sys.stderr)
        sys.exit(1)

    # Mismo orden de columnas que el CSV original, para que train.py (que
    # parsea observed_at con parse_dates) no note ninguna diferencia.
    observations = observations[["observed_at", "station_id", "demand"]]
    observations.to_csv(DATA / "observations.csv", index=False)
    print(f"observations.csv: {len(observations)} filas "
          f"({observations['observed_at'].min()} a {observations['observed_at'].max()})")

    if not context.empty:
        context = context[["observed_at", "rain_mm", "rain_forecast", "temperature_c",
                            "temperature_forecast", "event_intensity"]]
        context.to_csv(DATA / "context.csv", index=False)
        print(f"context.csv: {len(context)} filas")
    else:
        print("Sin lecturas de contexto en Supabase: se deja context.csv vacio con encabezado.", file=sys.stderr)
        pd.DataFrame(columns=["observed_at", "rain_mm", "rain_forecast", "temperature_c",
                               "temperature_forecast", "event_intensity"]).to_csv(DATA / "context.csv", index=False)


if __name__ == "__main__":
    main()
