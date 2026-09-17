"""Analisis exploratorio, correlacion, feature engineering, seleccion de
features y validacion cruzada temporal para el reto Pulso TransMi.

Corre de forma aislada: lee de ../data (generado por examples/01_download.py)
y escribe resultados solo dentro de esta carpeta (eda/figures, eda/reports).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"
FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": "string"})
    observations = pd.read_csv(
        DATA / "observations.csv",
        dtype={"station_id": "string"},
        parse_dates=["observed_at"],
    )
    context = pd.read_csv(DATA / "context.csv", parse_dates=["observed_at"])
    return stations, observations, context


# ---------------------------------------------------------------------------
# Fase 1a: calidad de datos (continuidad, duplicados, tipos, cobertura)
# ---------------------------------------------------------------------------

def data_quality_report(observations: pd.DataFrame, stations: pd.DataFrame) -> dict:
    report: dict = {}

    report["duplicated_rows"] = int(
        observations.duplicated(subset=["station_id", "observed_at"]).sum()
    )
    report["null_counts"] = observations.isna().sum().to_dict()
    report["dtypes"] = {k: str(v) for k, v in observations.dtypes.items()}

    expected_periods = observations["observed_at"].nunique()
    coverage = observations.groupby("station_id").size()
    report["coverage_per_station"] = coverage.to_dict()
    report["expected_periods"] = int(expected_periods)
    report["stations_with_gaps"] = coverage[coverage != expected_periods].to_dict()

    full_index = pd.date_range(
        observations["observed_at"].min(),
        observations["observed_at"].max(),
        freq="15min",
    )
    gap_summary = {}
    for station_id, group in observations.groupby("station_id"):
        missing = full_index.difference(group["observed_at"])
        if len(missing) > 0:
            gap_summary[station_id] = len(missing)
    report["missing_timestamps_per_station"] = gap_summary

    report["demand_negative_count"] = int((observations["demand"] < 0).sum())
    report["demand_stats"] = observations["demand"].describe().to_dict()

    known_stations = set(stations["station_id"])
    obs_stations = set(observations["station_id"])
    report["stations_without_metadata"] = list(obs_stations - known_stations)

    return report


# ---------------------------------------------------------------------------
# Fase 1b: analisis exploratorio temporal y geografico
# ---------------------------------------------------------------------------

def plot_temporal_overview(observations: pd.DataFrame, stations: pd.DataFrame) -> None:
    merged = observations.merge(stations, on="station_id", how="left")

    total_by_time = observations.groupby("observed_at")["demand"].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    total_by_time.plot(ax=ax, linewidth=0.8, color="#1f6f4a")
    ax.set_title("Demanda total del sistema (todas las estaciones)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Pasajeros por periodo de 15 min")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_demanda_total_tiempo.png", dpi=140)
    plt.close(fig)

    hourly = merged.copy()
    hourly["hour"] = hourly["observed_at"].dt.hour
    hourly["dow"] = hourly["observed_at"].dt.day_name()
    pivot = hourly.pivot_table(
        index="hour", columns="dow", values="demand", aggfunc="mean"
    )
    day_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    ]
    pivot = pivot.reindex(columns=[d for d in day_order if d in pivot.columns])
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGn")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels(range(0, 24, 2))
    ax.set_xlabel("Dia de la semana")
    ax.set_ylabel("Hora del dia")
    ax.set_title("Estacionalidad: demanda promedio por hora x dia")
    fig.colorbar(im, ax=ax, label="Demanda promedio")
    fig.tight_layout()
    fig.savefig(FIGURES / "02_heatmap_hora_dia.png", dpi=140)
    plt.close(fig)

    by_station = merged.groupby(["station_id", "station_name"])["demand"].mean().reset_index()
    by_station = by_station.sort_values("demand", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(by_station["station_name"], by_station["demand"], color="#2f7a56")
    ax.set_xlabel("Demanda promedio por periodo (15 min)")
    ax.set_title("Demanda promedio por estacion")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_demanda_promedio_por_estacion.png", dpi=140)
    plt.close(fig)

    demand_by_id = merged.groupby("station_id")["demand"].mean()
    sizes = stations["station_id"].map(demand_by_id)
    fig, ax = plt.subplots(figsize=(6, 7))
    sc = ax.scatter(
        stations["longitude"], stations["latitude"],
        s=sizes / sizes.max() * 800 + 40,
        c=sizes, cmap="YlGn", edgecolors="black", linewidths=0.5,
    )
    for _, row in stations.iterrows():
        ax.annotate(row["station_name"], (row["longitude"], row["latitude"]), fontsize=7)
    ax.set_title("Mapa geografico: tamano/color = demanda promedio")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    fig.colorbar(sc, ax=ax, label="Demanda promedio")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_mapa_geografico_demanda.png", dpi=140)
    plt.close(fig)

    daily_series = observations.groupby(observations["observed_at"].dt.date)["demand"].sum()
    fig, ax = plt.subplots(figsize=(12, 4))
    daily_series.plot(ax=ax, color="#1f6f4a", marker="o", markersize=3)
    ax.set_title("Demanda total diaria (45 dias de historia)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Demanda total del dia")
    fig.tight_layout()
    fig.savefig(FIGURES / "05_demanda_diaria.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

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


def correlation_analysis(feature_frame: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "demand", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
        "lag_1", "lag_4", "lag_96", "lag_672", "roll_mean_4", "roll_mean_96",
        "roll_std_96", "rain_mm", "rain_forecast", "temperature_c",
        "temperature_forecast", "event_intensity",
    ]
    corr = feature_frame[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols, fontsize=7)
    ax.set_title("Matriz de correlacion (demanda + features + contexto)")
    fig.colorbar(im, ax=ax, label="Correlacion de Pearson")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_matriz_correlacion.png", dpi=140)
    plt.close(fig)

    return corr


# ---------------------------------------------------------------------------
# Seleccion de features + validacion cruzada temporal
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "lag_1", "lag_4", "lag_96", "lag_672", "roll_mean_4", "roll_mean_96",
    "roll_std_96", "rain_mm", "temperature_c", "event_intensity",
]


def feature_selection(feature_frame: pd.DataFrame) -> pd.Series:
    model_frame = feature_frame.dropna(subset=FEATURE_COLUMNS + ["demand"]).copy()
    X = model_frame[FEATURE_COLUMNS]
    y = model_frame["demand"]

    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=20260916, n_jobs=-1
    )
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.sort_values().plot(kind="barh", ax=ax, color="#2f7a56")
    ax.set_title("Importancia de features (Random Forest)")
    ax.set_xlabel("Importancia")
    fig.tight_layout()
    fig.savefig(FIGURES / "07_feature_importance.png", dpi=140)
    plt.close(fig)

    return importances


def wape_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    wape = np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()
    return 100 * max(0.0, 1 - wape)


def temporal_cross_validation(feature_frame: pd.DataFrame, top_features: list[str]) -> pd.DataFrame:
    model_frame = feature_frame.dropna(subset=top_features + ["demand"]).copy()
    model_frame = model_frame.sort_values("observed_at")

    unique_times = model_frame["observed_at"].sort_values().unique()
    splitter = TimeSeriesSplit(n_splits=5)
    results = []

    time_index = pd.Series(range(len(unique_times)), index=unique_times)
    model_frame["_time_rank"] = model_frame["observed_at"].map(time_index)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(unique_times), start=1):
        train_times = set(unique_times[train_idx])
        test_times = set(unique_times[test_idx])

        train = model_frame[model_frame["observed_at"].isin(train_times)]
        test = model_frame[model_frame["observed_at"].isin(test_times)]
        if train.empty or test.empty:
            continue

        model = RandomForestRegressor(
            n_estimators=150, max_depth=10, random_state=20260916, n_jobs=-1
        )
        model.fit(train[top_features], train["demand"])
        preds = model.predict(test[top_features])

        mae = mean_absolute_error(test["demand"], preds)
        acc_per_station = (
            test.assign(prediction=preds)
            .groupby("station_id")
            .apply(lambda g: wape_accuracy(g["demand"].values, g["prediction"].values))
        )
        results.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "test_rows": len(test),
                "mae": mae,
                "accuracy_mean": acc_per_station.mean(),
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------

def main() -> None:
    stations, observations, context = load_data()

    quality = data_quality_report(observations, stations)
    (REPORTS / "data_quality.json").write_text(
        json.dumps(quality, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )

    plot_temporal_overview(observations, stations)

    feature_frame = build_feature_frame(observations, context)
    corr = correlation_analysis(feature_frame)
    corr.to_csv(REPORTS / "correlation_matrix.csv")

    importances = feature_selection(feature_frame)
    importances.to_csv(REPORTS / "feature_importance.csv", header=["importance"])

    top_features = importances.head(8).index.tolist()
    cv_results = temporal_cross_validation(feature_frame, top_features)
    cv_results.to_csv(REPORTS / "cross_validation_results.csv", index=False)

    print("=== Calidad de datos ===")
    print(f"Duplicados: {quality['duplicated_rows']}")
    print(f"Demanda negativa: {quality['demand_negative_count']}")
    print(f"Estaciones con huecos: {quality['stations_with_gaps']}")
    print()
    print("=== Top features por importancia ===")
    print(importances.head(8).round(4))
    print()
    print("=== Validacion cruzada temporal (5 folds, TimeSeriesSplit) ===")
    print(cv_results.round(2))
    print()
    print(f"Accuracy promedio CV: {cv_results['accuracy_mean'].mean():.2f}")
    print()
    print(f"Graficos guardados en: {FIGURES}")
    print(f"Reportes guardados en: {REPORTS}")


if __name__ == "__main__":
    main()
