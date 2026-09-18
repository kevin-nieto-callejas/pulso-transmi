import sys
from pathlib import Path

import httpx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import infer  # noqa: E402


class _FakeModel:
    """Devuelve valores predefinidos, uno por cada llamada a predict (cada
    llamada de build_batch_predictions predice una sola fila)."""

    def __init__(self, values):
        self._values = list(values)

    def predict(self, frame):
        return [self._values.pop(0)] * len(frame)


def test_get_current_cycle_returns_none_when_no_open_cycle(monkeypatch) -> None:
    monkeypatch.setattr(infer, "API_URL", "https://example.test")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/forecast-cycles/current"
        return httpx.Response(404, json={"detail": {"code": "no_open_cycle", "message": "no cycle"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert infer.get_current_cycle(client) is None


def test_get_current_cycle_returns_payload_when_open(monkeypatch) -> None:
    monkeypatch.setattr(infer, "API_URL", "https://example.test")
    cycle_payload = {"cycle_id": "cyc_abc123", "data_cutoff": "2026-09-20T10:00:00+00:00", "closes_at": "2026-09-20T10:25:00+00:00", "targets": []}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=cycle_payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert infer.get_current_cycle(client) == cycle_payload


def test_get_current_cycle_raises_on_unexpected_error(monkeypatch) -> None:
    monkeypatch.setattr(infer, "API_URL", "https://example.test")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        infer.get_current_cycle(client)


def test_parse_targets_finds_known_key() -> None:
    cycle = {"cycle_id": "cyc_1", "targets": [{"station_id": "03000", "target_at": "t"}]}
    assert infer.parse_targets(cycle) == cycle["targets"]


def test_parse_targets_raises_with_payload_when_missing() -> None:
    cycle = {"cycle_id": "cyc_1", "something_else": []}
    with pytest.raises(KeyError, match="cyc_1"):
        infer.parse_targets(cycle)


def test_build_batch_predictions_computes_horizon_and_clips_values() -> None:
    data_cutoff = pd.Timestamp("2026-09-20T10:00:00+00:00")
    anchor_by_station = pd.DataFrame(
        {"some_feat": [1.0, 2.0]}, index=pd.Index(["03000", "05000"], name="station_id"),
    )
    targets = [
        {"station_id": "03000", "target_at": "2026-09-20T10:15:00+00:00"},
        {"station_id": "03000", "target_at": "2026-09-20T10:30:00+00:00"},
        {"station_id": "05000", "target_at": "2026-09-20T10:15:00+00:00"},
    ]
    # -50 debe recortarse a 0.0 (no negativa); 999999 debe recortarse a 100000.0
    model = _FakeModel([-50, 42.345, 999999])

    predictions = infer.build_batch_predictions(
        model, ["some_feat", "horizon_minutes"], anchor_by_station, targets, data_cutoff,
    )

    assert predictions[0] == {"station_id": "03000", "target_at": "2026-09-20T10:15:00+00:00", "value": 0.0}
    assert predictions[1]["value"] == 42.35 or predictions[1]["value"] == 42.34  # redondeo a 2 decimales
    assert predictions[2] == {"station_id": "05000", "target_at": "2026-09-20T10:15:00+00:00", "value": 100000.0}


def test_build_batch_predictions_raises_for_unknown_station() -> None:
    data_cutoff = pd.Timestamp("2026-09-20T10:00:00+00:00")
    anchor_by_station = pd.DataFrame({"some_feat": [1.0]}, index=pd.Index(["03000"], name="station_id"))
    targets = [{"station_id": "99999", "target_at": "2026-09-20T10:15:00+00:00"}]

    with pytest.raises(ValueError, match="99999"):
        infer.build_batch_predictions(_FakeModel([1.0]), ["some_feat"], anchor_by_station, targets, data_cutoff)


def test_make_idempotency_key_is_stable_per_cycle_and_model() -> None:
    key1 = infer.make_idempotency_key("cyc_abc123", "xgboost_station-20260920T100000Z")
    key2 = infer.make_idempotency_key("cyc_abc123", "xgboost_station-20260920T100000Z")
    key3 = infer.make_idempotency_key("cyc_other", "xgboost_station-20260920T100000Z")

    assert key1 == key2
    assert key1 != key3
    assert 8 <= len(key1) <= 128


def test_fetch_all_rows_pagina_hasta_el_final(monkeypatch) -> None:
    """Regresion real (18/09): Supabase corta TODA respuesta en 1000 filas
    aunque se pida limit=5000. La paginacion anterior asumia paginas de 5000,
    asi que paraba en la primera y solo cargaba 1 de las 12 estaciones -
    rompio la primera corrida contra el ciclo de practica."""
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "fake-key")
    rangos_pedidos = []

    def handler(request: httpx.Request) -> httpx.Response:
        rangos_pedidos.append(request.headers["Range"])
        start = int(request.headers["Range"].split("-")[0])
        total = 2300  # dos paginas llenas + una parcial
        restantes = max(0, total - start)
        n = min(1000, restantes)
        return httpx.Response(200, json=[{"i": start + k} for k in range(n)])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    filas = infer.fetch_all_rows(client, "https://sb.test/rest/v1/observations", {}, {})

    assert len(filas) == 2300
    assert rangos_pedidos == ["0-999", "1000-1999", "2000-2999"]


def test_already_submitted_detects_existing_accepted_submission(monkeypatch) -> None:
    """Corridas solapadas (el cron dispara cada 10 min a proposito, porque
    GitHub descarta corridas) no deben rearmar ni reenviar un ciclo que la
    API ya acepto."""
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "fake-key")
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json=[{"submission_id": "sub_123"}])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert infer.already_submitted(client, "https://sb.test", "cyc_abc", "model-v1") is True
    # El filtro debe ser por ciclo Y modelo Y aceptada: si se filtrara solo
    # por ciclo, un reentrenamiento a mitad de ventana nunca podria entregar.
    assert captured["params"]["cycle_id"] == "eq.cyc_abc"
    assert captured["params"]["model_version_id"] == "eq.model-v1"
    assert captured["params"]["accepted"] == "is.true"


def test_already_submitted_false_when_no_rows(monkeypatch) -> None:
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "fake-key")

    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[])))
    assert infer.already_submitted(client, "https://sb.test", "cyc_abc", "model-v1") is False


def test_make_submission_payload_shape() -> None:
    cycle = {"cycle_id": "cyc_abc123", "data_cutoff": "2026-09-20T10:00:00+00:00"}
    champion = {"version_id": "xgboost_station-20260920T100000Z", "created_at": "2026-09-17T04:29:28+00:00",
                "data_cutoff": "2026-09-08T23:45:00-05:00", "code_commit": "9005e8c"}
    predictions = [{"station_id": "03000", "target_at": "2026-09-20T10:15:00+00:00", "value": 42.0}]

    payload = infer.make_submission_payload(cycle, champion, predictions, client_run_id="run-1")

    assert payload["schema_version"] == "1.0"
    assert payload["cycle_id"] == "cyc_abc123"
    assert payload["client_run_id"] == "run-1"
    assert payload["predictions"] == predictions
    assert payload["model"]["version"] == champion["version_id"]
    assert payload["model"]["git_commit"] == "9005e8c"
