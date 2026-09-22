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
    # Caso normal: el ancla coincide con el corte de datos (collector al dia).
    anchor_by_station = pd.DataFrame(
        {"some_feat": [1.0, 2.0], "observed_at": [data_cutoff, data_cutoff]},
        index=pd.Index(["03000", "05000"], name="station_id"),
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


def test_horizonte_se_mide_desde_el_ancla_no_desde_el_data_cutoff() -> None:
    """Riesgo detectado antes de la competencia real: si el collector va
    atrasado, el ancla queda ANTES del data_cutoff. Medir el horizonte desde
    el data_cutoff le diria al modelo "salta 15 min" cuando en realidad debe
    saltar 45 -> predicciones malas y ningun error visible."""
    data_cutoff = pd.Timestamp("2026-09-20T10:00:00+00:00")
    # El ancla va 30 minutos atrasada respecto al corte de datos.
    anchor_by_station = pd.DataFrame(
        {"some_feat": [1.0], "observed_at": [pd.Timestamp("2026-09-20T09:30:00+00:00")]},
        index=pd.Index(["03000"], name="station_id"),
    )
    targets = [{"station_id": "03000", "target_at": "2026-09-20T10:15:00+00:00"}]

    capturado = {}

    class _ModeloQueEspia:
        def predict(self, frame):
            capturado["horizon"] = frame["horizon_minutes"].iloc[0]
            return [42.0]

    infer.build_batch_predictions(
        _ModeloQueEspia(), ["some_feat", "horizon_minutes"], anchor_by_station, targets, data_cutoff,
    )

    # 10:15 - 09:30 = 45 min reales, no los 15 que sugiere el data_cutoff.
    assert capturado["horizon"] == 45


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


def test_las_features_del_target_se_calculan_en_inferencia() -> None:
    """El bug mas caro del proyecto: 18.5 puntos de accuracy, en silencio.

    `explode_horizons` calcula la hora y el dia del instante OBJETIVO, pero
    `build_feature_frame` -lo unico que corre en inferencia- no. Esas cinco
    columnas caian en el relleno de one-hot y se mandaban en 0. Un seno y un
    coseno valiendo 0 a la vez es un punto que no existe en el circulo
    unitario y que el modelo jamas vio entrenando.

    Nada fallaba: el modelo predecia, la API aceptaba, los tests pasaban.
    """
    import numpy as np
    from features import TARGET_TIME_COLUMNS

    data_cutoff = pd.Timestamp("2026-09-20T10:00:00+00:00")
    anchor_by_station = pd.DataFrame(
        {"some_feat": [1.0], "observed_at": [data_cutoff]},
        index=pd.Index(["03000"], name="station_id"),
    )
    visto = {}

    class _Espia:
        def predict(self, frame):
            visto.update(frame.iloc[0].to_dict())
            return [10.0]

    # target a las 10:45 de un domingo -> ninguna de las cinco es 0
    infer.build_batch_predictions(
        _Espia(), ["some_feat", "horizon_minutes", *TARGET_TIME_COLUMNS], anchor_by_station,
        [{"station_id": "03000", "target_at": "2026-09-20T10:45:00+00:00"}], data_cutoff,
    )

    for columna in TARGET_TIME_COLUMNS:
        assert columna in visto, f"la inferencia no mando {columna}"

    # 10:45 UTC son las 05:45 en Bogota, y la hora LOCAL es la que aprendio
    # el modelo: la gente toma el bus segun su reloj. Usar la hora UTC aqui
    # deja el modelo corrido cinco horas sobre su senal mas fuerte.
    periodo = 5 * 4 + 45 / 15
    assert visto["target_hour_sin"] == pytest.approx(np.sin(2 * np.pi * periodo / 96))
    assert visto["target_hour_cos"] == pytest.approx(np.cos(2 * np.pi * periodo / 96))
    # sin^2 + cos^2 = 1: imposible si ambas llegaran en 0
    assert visto["target_hour_sin"] ** 2 + visto["target_hour_cos"] ** 2 == pytest.approx(1.0)
    assert visto["target_is_weekend"] == 1  # 2026-09-20 es domingo


def test_una_feature_desconocida_no_se_rellena_con_cero() -> None:
    """Causa raiz del bug anterior: rellenar con 0 cualquier columna ausente.

    Para los one-hot de otras estaciones 0 es correcto por definicion. Para
    cualquier otra cosa es una diferencia entre entrenar y predecir, y debe
    reventar en vez de devolver una prediccion mala en silencio.
    """
    data_cutoff = pd.Timestamp("2026-09-20T10:00:00+00:00")
    anchor_by_station = pd.DataFrame(
        {"some_feat": [1.0], "observed_at": [data_cutoff]},
        index=pd.Index(["03000"], name="station_id"),
    )
    targets = [{"station_id": "03000", "target_at": "2026-09-20T10:15:00+00:00"}]

    # station_99999 es un one-hot: se rellena con 0 sin quejarse.
    infer.build_batch_predictions(
        _FakeModel([5.0]), ["some_feat", "horizon_minutes", "station_99999"],
        anchor_by_station, targets, data_cutoff,
    )

    # roll_mean_8 no lo es: la inferencia no sabe calcularla, debe fallar.
    with pytest.raises(ValueError, match="roll_mean_8"):
        infer.build_batch_predictions(
            _FakeModel([5.0]), ["some_feat", "horizon_minutes", "roll_mean_8"],
            anchor_by_station, targets, data_cutoff,
        )
