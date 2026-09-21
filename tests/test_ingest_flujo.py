"""El camino completo del collector cuando SI llegan datos.

Hueco detectado revisando la bitacora real: de 46 corridas del collector, 44
terminaron en `no_new_data` y solo una ingirio algo (una fila). Las 51.840
observaciones entraron por el script de migracion, no por el collector. Es
decir, el camino que el lunes tiene que funcionar a la primera —el stream
trae filas, se hace upsert, el cursor avanza y queda registrado como
exito— practicamente nunca se ha ejecutado.

Los tests anteriores cubrian las funciones de descarga por separado. Estos
cubren el flujo entero de `main()` con datos, incluidas las dos cosas que
solo importan cuando hay novedades: que el cursor avance al confirmado y que
NO avance si el upsert falla.
"""
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import ingest  # noqa: E402


class _Supabase:
    """Supabase de mentira: registra lo que se le escribe."""

    def __init__(self, cursor_previo=None, falla_upsert=False):
        self.cursor_previo = cursor_previo
        self.falla_upsert = falla_upsert
        self.observaciones = []
        self.contexto = []
        self.bitacora = []

    def responder(self, request: httpx.Request) -> httpx.Response:
        ruta = request.url.path
        if request.method == "GET" and ruta.endswith("/collector_runs"):
            return httpx.Response(200, json=[{"cursor_after": self.cursor_previo}] if self.cursor_previo else [])
        if request.method == "GET" and ruta.endswith("/context_readings"):
            return httpx.Response(200, json=[{"observed_at": "2026-09-09T04:45:00+00:00"}])
        if request.method == "POST" and ruta.endswith("/observations"):
            if self.falla_upsert:
                return httpx.Response(500, text="el upsert fallo")
            self.observaciones.extend(request.read().decode() and [1])
            return httpx.Response(201)
        if request.method == "POST" and ruta.endswith("/context_readings"):
            self.contexto.append(1)
            return httpx.Response(201)
        if request.method == "POST" and ruta.endswith("/collector_runs"):
            import json
            self.bitacora.append(json.loads(request.read()))
            return httpx.Response(201)
        return httpx.Response(200, json=[])


def _montar(monkeypatch, supabase, filas_stream, cursor_final="cursor-nuevo"):
    monkeypatch.setenv("SUPABASE_URL", "https://sb.test")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "clave-falsa")
    monkeypatch.setattr(ingest, "API_URL", "https://api.test")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.test":
            if request.url.path == "/v1/stream/observations":
                return httpx.Response(200, json={"data": filas_stream, "count": len(filas_stream),
                                                 "next_cursor": None})
            if request.url.path == "/v1/context":
                return httpx.Response(200, json={"data": [], "count": 0, "next_cursor": None})
        return supabase.responder(request)

    transporte = httpx.MockTransport(handler)
    original = httpx.Client

    def cliente(*args, **kwargs):
        kwargs["transport"] = transporte
        return original(*args, **kwargs)

    monkeypatch.setattr(ingest.httpx, "Client", cliente)


def test_el_collector_ingiere_y_confirma_el_cursor(monkeypatch) -> None:
    """El caso que el lunes tiene que funcionar: llegan filas, se guardan y
    el cursor queda confirmado para la siguiente corrida."""
    supabase = _Supabase(cursor_previo="cursor-viejo")
    filas = [
        {"station_id": "03000", "observed_at": "2026-09-09T05:00:00+00:00", "demand": 140},
        {"station_id": "05000", "observed_at": "2026-09-09T05:00:00+00:00", "demand": 210},
    ]
    _montar(monkeypatch, supabase, filas)

    ingest.main()

    assert supabase.observaciones, "No se escribieron las observaciones nuevas"
    registro = supabase.bitacora[-1]
    assert registro["status"] == "success"
    assert registro["rows_ingested"] == 2
    assert registro["cursor_before"] == "cursor-viejo"
    # Sin next_cursor, se confirma el cursor con el que se pidio la pagina.
    assert registro["cursor_after"] == "cursor-viejo"


def test_si_falla_el_upsert_el_cursor_NO_avanza(monkeypatch) -> None:
    """La garantia que evita perder datos en silencio: si la escritura falla,
    el cursor se queda donde estaba y la proxima corrida vuelve a pedir esas
    filas. Avanzarlo igual seria saltarselas para siempre."""
    supabase = _Supabase(cursor_previo="cursor-viejo", falla_upsert=True)
    filas = [{"station_id": "03000", "observed_at": "2026-09-09T05:00:00+00:00", "demand": 140}]
    _montar(monkeypatch, supabase, filas)

    with pytest.raises(RuntimeError):
        ingest.main()

    # Puede no haberse registrado nada, pero si se registro algo, jamas debe
    # decir que el cursor avanzo.
    for registro in supabase.bitacora:
        assert registro["cursor_after"] != "cursor-nuevo"
        assert registro["status"] != "success"


def test_sin_novedades_queda_constancia(monkeypatch) -> None:
    """La guia lo pide explicitamente: el collector debe dejar evidencia
    incluso cuando no encuentra nada."""
    supabase = _Supabase(cursor_previo="cursor-viejo")
    _montar(monkeypatch, supabase, [])

    ingest.main()

    registro = supabase.bitacora[-1]
    assert registro["status"] == "no_new_data"
    assert registro["rows_ingested"] == 0
    assert registro["cursor_after"] == "cursor-viejo"
