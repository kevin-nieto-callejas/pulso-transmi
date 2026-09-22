import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import build_feature_frame  # noqa: E402


def _datos(zona: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dos dias de una estacion, el mismo instante expresado en `zona`."""
    instantes = pd.date_range("2026-09-01 00:00", periods=192, freq="15min", tz="UTC").tz_convert(zona)
    obs = pd.DataFrame({
        "station_id": pd.Series(["03000"] * len(instantes), dtype="string"),
        "observed_at": instantes,
        "demand": np.arange(len(instantes), dtype=float) % 96 + 10,
    })
    ctx = pd.DataFrame({
        "observed_at": instantes, "rain_mm": 0.0, "rain_forecast": 0.0,
        "temperature_c": 14.0, "temperature_forecast": 14.0, "event_intensity": 0.0,
    })
    return obs, ctx


def test_la_hora_no_depende_de_la_zona_en_que_llegan_los_datos() -> None:
    """El CSV del historico se lee como UTC-05:00 y Supabase devuelve UTC.

    Son los mismos instantes, asi que lags y horizontes salen bien por los
    dos caminos. Pero `.dt.hour` no: el entrenamiento veia la hora de Bogota
    y la inferencia la de UTC, cinco horas corrida. Llevo el accuracy de un
    ciclo real hasta 30 sin lanzar un solo error.
    """
    columnas = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]

    desde_csv = build_feature_frame(*_datos("-05:00"))
    desde_supabase = build_feature_frame(*_datos("UTC"))

    pd.testing.assert_frame_equal(
        desde_csv[columnas].reset_index(drop=True),
        desde_supabase[columnas].reset_index(drop=True),
    )


def test_la_hora_que_ve_el_modelo_es_la_de_bogota() -> None:
    """No basta con que los dos caminos coincidan: tienen que coincidir en la
    hora LOCAL. Las 12:00 UTC son las 7:00 en Bogota, plena hora pico."""
    frame = build_feature_frame(*_datos("UTC"))
    fila = frame[frame["observed_at"] == pd.Timestamp("2026-09-01 12:00", tz="UTC")].iloc[0]

    periodo_bogota = 7 * 4
    assert fila["hour_sin"] == np.sin(2 * np.pi * periodo_bogota / 96)
    assert fila["hour_cos"] == np.cos(2 * np.pi * periodo_bogota / 96)
