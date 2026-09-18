"""Collector incremental idempotente para Pulso TransMi.

Se puede correr manualmente o via cron (GitHub Actions, cuando se conecte).
Aunque el stream este vacio (reloj en `waiting`), debe poder correr sin
fallar y dejar evidencia en `collector_runs` (asi lo pide la guia
metodologica: "debe dejar evidencia incluso cuando no encuentre novedades").

Idempotencia:
  - El cursor de observaciones se retoma del ULTIMO `collector_runs` con
    status='success' (nunca se fabrica localmente a partir de la hora).
  - El "cursor" de contexto es el maximo `observed_at` ya guardado en
    `context_readings` (esa tabla no tiene cursor propio en la API).
  - Los upsert usan las mismas llaves unicas del esquema
    (station_id+observed_at / observed_at), asi que correr esto dos veces
    con el mismo cursor NUNCA duplica filas.
  - El cursor solo "avanza" (se registra como cursor_after) despues de que
    el upsert a Supabase ya se confirmo, no antes.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

API_URL = os.environ.get("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")


def supabase_headers() -> dict:
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def get_last_confirmed_cursor(client: httpx.Client, supabase_url: str) -> str | None:
    response = client.get(
        f"{supabase_url}/rest/v1/collector_runs",
        headers=supabase_headers(),
        params={
            "status": "eq.success",
            "select": "cursor_after",
            "order": "finished_at.desc",
            "limit": 1,
        },
    )
    response.raise_for_status()
    rows = response.json()
    return rows[0]["cursor_after"] if rows and rows[0]["cursor_after"] else None


def get_last_context_timestamp(client: httpx.Client, supabase_url: str) -> str | None:
    response = client.get(
        f"{supabase_url}/rest/v1/context_readings",
        headers=supabase_headers(),
        params={"select": "observed_at", "order": "observed_at.desc", "limit": 1},
    )
    response.raise_for_status()
    rows = response.json()
    return rows[0]["observed_at"] if rows else None


def fetch_new_observations(client: httpx.Client, cursor: str | None) -> tuple[list[dict], str | None]:
    """Pagina /v1/stream/observations desde `cursor` hasta agotar next_cursor.

    Regla de resumen (no documentada explicitamente por la API, asumida por
    seguridad): si la ULTIMA pagina ya no trae next_cursor, se confirma el
    cursor con el que se PIDIO esa pagina (no `None`), para nunca terminar
    reiniciando el stream desde cero por accidente. El peor caso posible es
    re-pedir esa misma pagina la proxima corrida, lo cual es inofensivo
    porque el upsert es idempotente.
    """
    all_rows: list[dict] = []
    request_cursor = cursor
    confirmed_cursor = cursor
    while True:
        params = {"limit": 5000}
        if request_cursor:
            params["cursor"] = request_cursor
        response = client.get(f"{API_URL}/v1/stream/observations", params=params)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data", [])
        all_rows.extend(rows)
        next_cursor = payload.get("next_cursor")
        if next_cursor is None:
            confirmed_cursor = request_cursor
            break
        request_cursor = next_cursor
        confirmed_cursor = next_cursor
    return all_rows, confirmed_cursor


def fetch_new_context(client: httpx.Client, since: str | None) -> list[dict]:
    all_rows: list[dict] = []
    cursor = None
    while True:
        params = {"limit": 5000}
        if since:
            params["start"] = since
        if cursor:
            params["cursor"] = cursor
        response = client.get(f"{API_URL}/v1/context", params=params)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data", [])
        # "start" es inclusivo, y Supabase (+00:00) vs la API (-05:00) devuelven
        # el mismo instante con offsets de texto distintos, asi que hay que
        # comparar como fechas reales, no como strings, para no recontar la
        # ultima fila ya conocida como si fuera nueva.
        if since:
            since_dt = pd.Timestamp(since)
            rows = [r for r in rows if pd.Timestamp(r["observed_at"]) != since_dt]
        all_rows.extend(rows)
        cursor = payload.get("next_cursor")
        if not rows or cursor is None:
            break
    return all_rows


def upsert(client: httpx.Client, supabase_url: str, table: str, rows: list[dict], on_conflict: str, batch_size: int = 2000) -> None:
    if not rows:
        return
    headers = {**supabase_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"}
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        response = client.post(
            f"{supabase_url}/rest/v1/{table}",
            headers=headers, params={"on_conflict": on_conflict}, json=batch,
        )
        if response.status_code >= 300:
            raise RuntimeError(f"upsert {table} fallo: {response.status_code} {response.text[:300]}")


def log_run(client: httpx.Client, supabase_url: str, *, started_at, cursor_before, cursor_after, rows_ingested, status, error_message=None) -> None:
    payload = {
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cursor_before": cursor_before,
        "cursor_after": cursor_after,
        "rows_ingested": rows_ingested,
        "status": status,
        "error_message": error_message,
    }
    response = client.post(
        f"{supabase_url}/rest/v1/collector_runs",
        headers={**supabase_headers(), "Prefer": "return=minimal"},
        json=payload,
    )
    if response.status_code >= 300:
        raise RuntimeError(f"No se pudo registrar collector_runs: {response.status_code} {response.text[:300]}")


def main() -> None:
    supabase_url = os.environ["SUPABASE_URL"].rstrip("/")
    started_at = datetime.now(timezone.utc).isoformat()

    with httpx.Client(timeout=60.0) as client:
        cursor_before = get_last_confirmed_cursor(client, supabase_url)
        print(f"Cursor de partida (ultimo confirmado): {cursor_before!r}")

        try:
            new_observations, cursor_after = fetch_new_observations(client, cursor_before)
            last_context_ts = get_last_context_timestamp(client, supabase_url)
            new_context = fetch_new_context(client, last_context_ts)
        except httpx.HTTPError as exc:
            log_run(
                client, supabase_url, started_at=started_at, cursor_before=cursor_before,
                cursor_after=cursor_before, rows_ingested=0, status="failed", error_message=str(exc)[:500],
            )
            print(f"Collector FALLO: {exc}")
            raise

        rows_ingested = len(new_observations) + len(new_context)

        if rows_ingested == 0:
            log_run(
                client, supabase_url, started_at=started_at, cursor_before=cursor_before,
                cursor_after=cursor_before, rows_ingested=0, status="no_new_data",
            )
            print("Sin novedades (stream vacio o al dia). Bitacora registrada igual.")
            return

        obs_rows = [
            {"station_id": str(r["station_id"]), "observed_at": r["observed_at"], "demand": r["demand"]}
            for r in new_observations
        ]
        upsert(client, supabase_url, "observations", obs_rows, on_conflict="station_id,observed_at")
        upsert(client, supabase_url, "context_readings", new_context, on_conflict="observed_at")

        # El cursor solo "avanza" (se guarda como confirmado) despues de que
        # el upsert de arriba ya se ejecuto sin lanzar excepcion.
        log_run(
            client, supabase_url, started_at=started_at, cursor_before=cursor_before,
            cursor_after=cursor_after, rows_ingested=rows_ingested, status="success",
        )
        print(f"Collector OK: {len(obs_rows)} observaciones nuevas, {len(new_context)} lecturas de contexto nuevas.")
        print(f"Cursor confirmado: {cursor_after!r}")


if __name__ == "__main__":
    main()
