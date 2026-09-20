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

# --- Conjunto ampliado (ronda de mejora) -----------------------------------
# Tres huecos detectados al revisar por que el modelo se estanca:
#
#  1. `rain_forecast` y `temperature_forecast` existen en la API y NUNCA se
#     usaron. Para predecir el futuro, el PRONOSTICO del clima es la variable
#     correcta; la lluvia observada ahora describe el pasado.
#  2. El modelo conocia la hora del ancla y el horizonte, pero no la hora del
#     TARGET. Deducirla exige una aritmetica que un arbol no hace bien, y el
#     perfil de demanda depende sobre todo de la hora objetivo.
#  3. Faltaban escalas intermedias (2 h, 2 dias, 2 semanas) y la tendencia
#     reciente (la demanda esta subiendo o bajando respecto a hace una hora).
EXTRA_LAG_COLUMNS = [
    "lag_2", "lag_3", "lag_8", "lag_192", "lag_1344",
    "roll_mean_8", "roll_mean_48", "roll_std_4", "roll_max_96", "roll_min_96",
    "diff_1h", "diff_1d", "ratio_1h",
]
FORECAST_COLUMNS = ["rain_forecast", "temperature_forecast"]
SLOT_COLUMNS = ["slot_mean"]
TARGET_TIME_COLUMNS = ["target_hour_sin", "target_hour_cos", "target_dow_sin", "target_dow_cos", "target_is_weekend"]

EXTENDED_FEATURE_COLUMNS = (
    MULTI_HORIZON_FEATURE_COLUMNS + EXTRA_LAG_COLUMNS + FORECAST_COLUMNS + SLOT_COLUMNS + TARGET_TIME_COLUMNS
)


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

    # --- Escalas intermedias y tendencia (ronda de mejora) ---
    # Todo parte de shift(>=1): nunca mira el presente ni el futuro.
    frame["lag_2"] = grouped.shift(2)
    frame["lag_3"] = grouped.shift(3)
    frame["lag_8"] = grouped.shift(8)        # 2 horas
    frame["lag_192"] = grouped.shift(192)    # 2 dias
    frame["lag_1344"] = grouped.shift(1344)  # 2 semanas
    frame["roll_mean_8"] = grouped.shift(1).rolling(8).mean()
    frame["roll_mean_48"] = grouped.shift(1).rolling(48).mean()
    frame["roll_std_4"] = grouped.shift(1).rolling(4).std()
    frame["roll_max_96"] = grouped.shift(1).rolling(96).max()
    frame["roll_min_96"] = grouped.shift(1).rolling(96).min()

    # Tendencia: ¿viene subiendo o bajando respecto a hace una hora / un dia?
    frame["diff_1h"] = frame["lag_1"] - frame["lag_4"]
    frame["diff_1d"] = frame["lag_1"] - frame["lag_96"]
    frame["ratio_1h"] = frame["lag_1"] / frame["lag_4"].replace(0, np.nan)

    # Perfil historico de cada estacion en cada franja de la semana, calculado
    # de forma causal: para cada fila, el promedio de TODAS las ocurrencias
    # ANTERIORES de esa misma franja (shift(1) antes de expanding). Le da al
    # modelo "cuanta gente suele haber aqui un martes a las 7:15" sin filtrar
    # el valor que se quiere predecir.
    slot = frame["day_of_week"] * 96 + frame["hour"] * 4 + frame["minute"] // 15
    frame["_slot"] = slot
    frame["slot_mean"] = (
        frame.groupby(["station_id", "_slot"])["demand"].transform(lambda s: s.shift(1).expanding().mean())
    )
    frame = frame.drop(columns="_slot")

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

        # Hora y dia del INSTANTE OBJETIVO. No es fuga: el reloj del futuro se
        # conoce de antemano. Sin esto el modelo tenia la hora del ancla y el
        # horizonte por separado, y deducir "entonces el target cae a las 7:15"
        # exige una aritmetica que los arboles no hacen bien - justo cuando el
        # perfil de demanda depende sobre todo de la hora objetivo.
        t = variant["target_at"]
        periodo = t.dt.hour * 4 + t.dt.minute / 15
        variant["target_hour_sin"] = np.sin(2 * np.pi * periodo / 96)
        variant["target_hour_cos"] = np.cos(2 * np.pi * periodo / 96)
        variant["target_dow_sin"] = np.sin(2 * np.pi * t.dt.dayofweek / 7)
        variant["target_dow_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 7)
        variant["target_is_weekend"] = t.dt.dayofweek.isin([5, 6]).astype(int)

        variants.append(variant)
    return pd.concat(variants, ignore_index=True)


def wape_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    wape = np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()
    return 100 * max(0.0, 1 - wape)
