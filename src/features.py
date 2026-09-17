"""Feature engineering compartido entre el EDA y el entrenamiento.

Mantenido en un solo lugar para que el modelo entrenado en produccion use
exactamente las mismas features que se validaron en el EDA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ALL_FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "lag_1", "lag_4", "lag_96", "lag_672", "roll_mean_4", "roll_mean_96",
    "roll_std_96", "rain_mm", "temperature_c", "event_intensity",
]

# Subconjunto sin lag_672 (demanda de hace una semana): sirve para medir que
# tan fragil es el modelo si esa senal deja de ser confiable por drift.
NO_WEEKLY_LAG_FEATURE_COLUMNS = [
    c for c in ALL_FEATURE_COLUMNS if c not in ("lag_672", "roll_mean_96", "roll_std_96")
]


def build_feature_frame(observations: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    frame = observations.sort_values(["station_id", "observed_at"]).copy()

    frame["hour"] = frame["observed_at"].dt.hour
    frame["minute"] = frame["observed_at"].dt.minute
    frame["day_of_week"] = frame["observed_at"].dt.dayofweek
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)
    frame["hour_sin"] = np.sin(2 * np.pi * (frame["hour"] * 4 + frame["minute"] / 15) / 96)
    frame["hour_cos"] = np.cos(2 * np.pi * (frame["hour"] * 4 + frame["minute"] / 15) / 96)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)

    grouped = frame.groupby("station_id")["demand"]
    frame["lag_1"] = grouped.shift(1)
    frame["lag_4"] = grouped.shift(4)
    frame["lag_96"] = grouped.shift(96)
    frame["lag_672"] = grouped.shift(672)
    frame["roll_mean_4"] = grouped.shift(1).rolling(4).mean()
    frame["roll_mean_96"] = grouped.shift(1).rolling(96).mean()
    frame["roll_std_96"] = grouped.shift(1).rolling(96).std()

    frame = frame.merge(context, on="observed_at", how="left")
    return frame


def wape_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    wape = np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()
    return 100 * max(0.0, 1 - wape)
