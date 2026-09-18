import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import ingest  # noqa: E402


def test_fetch_new_context_excludes_boundary_row_across_tz_formats(monkeypatch) -> None:
    """Regresion: el collector confundio el offset -05:00 (API) con +00:00
    (Supabase) para el MISMO instante y lo conto como fila nueva."""
    monkeypatch.setattr(ingest, "API_URL", "https://example.test")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/context"
        return httpx.Response(200, json={
            "data": [
                {"observed_at": "2026-09-09T04:45:00+00:00", "rain_mm": 0.0},
                {"observed_at": "2026-09-09T05:00:00+00:00", "rain_mm": 1.2},
            ],
            "count": 2,
            "next_cursor": None,
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    rows = ingest.fetch_new_context(client, since="2026-09-08T23:45:00-05:00")

    assert [r["observed_at"] for r in rows] == ["2026-09-09T05:00:00+00:00"]


def test_fetch_new_observations_confirms_cursor_used_for_last_page(monkeypatch) -> None:
    """La ultima pagina no trae next_cursor: se confirma el cursor con el que
    se PIDIO esa pagina ('page-2'), no None - para no reiniciar el stream
    desde cero en la proxima corrida."""
    monkeypatch.setattr(ingest, "API_URL", "https://example.test")

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        if cursor is None:
            return httpx.Response(200, json={
                "data": [{"station_id": "03000", "observed_at": "t1", "demand": 5}],
                "count": 1, "next_cursor": "page-2",
            })
        return httpx.Response(200, json={
            "data": [{"station_id": "03000", "observed_at": "t2", "demand": 7}],
            "count": 1, "next_cursor": None,
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    rows, confirmed_cursor = ingest.fetch_new_observations(client, cursor=None)

    assert [r["demand"] for r in rows] == [5, 7]
    assert confirmed_cursor == "page-2"


def test_fetch_new_observations_preserves_cursor_when_stream_is_empty(monkeypatch) -> None:
    """Caso real verificado en produccion: reloj en 'waiting', el stream
    devuelve data=[] y next_cursor=None. El cursor de entrada debe
    conservarse tal cual, no perderse."""
    monkeypatch.setattr(ingest, "API_URL", "https://example.test")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [], "count": 0, "next_cursor": None})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    rows, confirmed_cursor = ingest.fetch_new_observations(client, cursor="cursor-abc")

    assert rows == []
    assert confirmed_cursor == "cursor-abc"
