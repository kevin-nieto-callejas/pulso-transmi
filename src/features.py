"""Feature engineering compartido entre el EDA y el entrenamiento.

Mantenido en un solo lugar para que el modelo entrenado en produccion use
exactamente las mismas features que se validaron en el EDA.
"""
from __future__ import annotations

from datetime import timedelta, timezone

import numpy as np
import pandas as pd

# Bogota, UTC-5 todo el ano (Colombia no cambia la hora). Se usa un offset
# fijo a proposito, en vez de la zona "America/Bogota", para no depender de
# que la base de zonas horarias este instalada en el runner de GitHub.
ZONA_BOGOTA = timezone(timedelta(hours=-5))


def a_hora_local(valores):
    """Lleva cualquier marca de tiempo a hora de Bogota.

    La razon de que esto exista: el historico en CSV se lee como UTC-05:00 y
    Supabase devuelve UTC. Son el MISMO instante, asi que los lags, las
    diferencias y los horizontes salen bien por ambos caminos y nada falla.
    Pero `.dt.hour` da 11 por un lado y 16 por el otro, y de ahi salen
    `hour_sin`/`hour_cos` y el perfil por franja.

    O sea: el modelo se entrenaba con la hora de Bogota y se le preguntaba en
    UTC, cinco horas corrido, sobre la senal que mas manda en demanda de
    transporte. Ningun error, ninguna excepcion, ningun test en rojo.

    La hora local es la correcta: la gente toma el bus segun su reloj, no
    segun UTC. Todo se normaliza aqui, en un solo lugar.
    """
    if isinstance(valores, pd.Timestamp):
        if valores.tzinfo is None:
            return valores.tz_localize(ZONA_BOGOTA)
        return valores.tz_convert(ZONA_BOGOTA)
    serie = pd.to_datetime(valores)
    if serie.dt.tz is None:
        return serie.dt.tz_localize(ZONA_BOGOTA)
    return serie.dt.tz_convert(ZONA_BOGOTA)

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
    observations = observations.copy()
    context = context.copy()
    # Unico punto donde se fija la zona horaria. Da igual si los datos vienen
    # del CSV (UTC-05:00) o de Supabase (UTC): a partir de aqui, una sola
    # convencion para entrenar y para predecir.
    observations["observed_at"] = a_hora_local(observations["observed_at"])
    if not context.empty and "observed_at" in context.columns:
        context["observed_at"] = a_hora_local(context["observed_at"])

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


def aplicar_features_de_target(target_at: "pd.Series | pd.Timestamp") -> dict:
    """Hora y dia del INSTANTE OBJETIVO, la unica definicion que existe.

    No es fuga de futuro: el reloj del futuro se conoce de antemano. Sin
    esto el modelo tiene la hora del ancla y el horizonte por separado, y
    deducir "entonces el target cae a las 7:15" exige una aritmetica que los
    arboles no hacen bien, justo cuando el perfil de demanda depende sobre
    todo de la hora objetivo.

    Vive aqui, y no dentro de `explode_horizons`, porque el entrenamiento y
    la inferencia TIENEN que calcularla igual. Cuando estuvo solo en el
    camino de entrenamiento, la inferencia no la calculaba y estas cinco
    columnas se rellenaban con ceros: un seno y un coseno valiendo 0 a la vez
    es un punto que no existe en el circulo y que el modelo nunca vio
    entrenando. Costaba 18.5 puntos de accuracy y no lanzaba ningun error.

    Acepta una Serie (entrenamiento, muchas filas) o un Timestamp suelto
    (inferencia, una prediccion).
    """
    target_at = a_hora_local(target_at)
    if isinstance(target_at, pd.Timestamp):
        periodo = target_at.hour * 4 + target_at.minute / 15
        dow = target_at.dayofweek
        return {
            "target_hour_sin": float(np.sin(2 * np.pi * periodo / 96)),
            "target_hour_cos": float(np.cos(2 * np.pi * periodo / 96)),
            "target_dow_sin": float(np.sin(2 * np.pi * dow / 7)),
            "target_dow_cos": float(np.cos(2 * np.pi * dow / 7)),
            "target_is_weekend": int(dow in (5, 6)),
        }

    periodo = target_at.dt.hour * 4 + target_at.dt.minute / 15
    dow = target_at.dt.dayofweek
    return {
        "target_hour_sin": np.sin(2 * np.pi * periodo / 96),
        "target_hour_cos": np.cos(2 * np.pi * periodo / 96),
        "target_dow_sin": np.sin(2 * np.pi * dow / 7),
        "target_dow_cos": np.cos(2 * np.pi * dow / 7),
        "target_is_weekend": dow.isin([5, 6]).astype(int),
    }


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
        for columna, valores in aplicar_features_de_target(variant["target_at"]).items():
            variant[columna] = valores

        variants.append(variant)
    return pd.concat(variants, ignore_index=True)


def wape_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    wape = np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()
    return 100 * max(0.0, 1 - wape)
