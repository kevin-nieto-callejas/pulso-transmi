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

# La API pide 4 horizontes por ciclo (+15/+30/+45/+60 min). Un modelo
# entrenado para predecir la propia fila (lag_1 = paso anterior) solo
# generaliza a +15: para +30 necesitaria lag_1 = demanda en target-15, que
# todavia no existe en data_cutoff. La solucion es "horizonte directo": una
# sola familia de modelos que recibe `horizon_minutes` como feature y cuyo
# target es la demanda desplazada hacia adelante ese horizonte, siempre
# usando unicamente lags anclados en el momento de prediccion (nunca en el
# futuro). Ver train.py y README ("Modelo") para el detalle.
HORIZONS_MINUTES = (15, 30, 45, 60)
MULTI_HORIZON_FEATURE_COLUMNS = ALL_FEATURE_COLUMNS + ["horizon_minutes"]
MULTI_HORIZON_NO_WEEKLY_LAG_FEATURE_COLUMNS = NO_WEEKLY_LAG_FEATURE_COLUMNS + ["horizon_minutes"]


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

    # Identidad de la estacion como one-hot: el EDA mostro >3x de diferencia
    # en demanda promedio entre estaciones (mapa geografico), pero ninguna
    # feature anterior le decia al modelo "en que estacion estas parado" -
    # solo lo inferia indirectamente via los lags. Se agrega sin reemplazar
    # station_id (se sigue necesitando para agrupar por estacion).
    dummies = pd.get_dummies(frame["station_id"], prefix="station", dtype=int)
    frame = pd.concat([frame, dummies], axis=1)

    return frame


def station_dummy_columns(observations: pd.DataFrame) -> list[str]:
    return [f"station_{sid}" for sid in sorted(observations["station_id"].unique())]


def explode_horizons(anchor_frame: pd.DataFrame, horizons_minutes: tuple[int, ...] = HORIZONS_MINUTES) -> pd.DataFrame:
    """Convierte cada fila ancla en 4 filas de entrenamiento, una por horizonte.

    Todas las features siguen ancladas en `observed_at` (lo que se sabe al
    predecir); solo cambia `horizon_minutes` y el target `target_demand`
    (la demanda real en `observed_at + horizonte`, tomada directamente del
    futuro conocido en el historico, nunca reconstruida con lags de ese
    futuro).
    """
    grouped = anchor_frame.sort_values(["station_id", "observed_at"]).groupby("station_id")["demand"]
    variants = []
    for horizon in horizons_minutes:
        steps = horizon // 15
        variant = anchor_frame.copy()
        variant["horizon_minutes"] = horizon
        variant["target_at"] = variant["observed_at"] + pd.Timedelta(minutes=horizon)
        variant["target_demand"] = grouped.shift(-steps).reindex(variant.index)
        variants.append(variant)
    return pd.concat(variants, ignore_index=True)


def wape_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    wape = np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()
    return 100 * max(0.0, 1 - wape)
