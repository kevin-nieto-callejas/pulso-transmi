"""Correccion aprendida de la capa adaptativa en la revision 4 (hallazgo #42).

La onda larga + persistencia+tendencia (perfil.py) comete errores con patron:
llega tarde a ciertos giros y exagera otros. Un LightGBM aprende ese residuo
con lo que ya paso en la revision 4: por estacion, corte y horizonte, las
ultimas 8 lecturas, la pendiente y las tres predicciones (mezcla, onda sola,
persistencia+tendencia), todo dividido por el nivel reciente. Se reentrena en
cada ciclo solo con cortes cuyo target ya ocurrio (causal).

Backtest walk-forward, 41 cortes 00:00-10:00Z del 21-sep, reentrenando cada
hora: produccion 87.59 -> corregida 89.34 (primera mitad 87.01 -> 88.82,
segunda 88.14 -> 89.84, p10 85.45 -> 87.15).

Este modulo replica en numpy, sobre la rejilla de 15 min, las mismas reglas de
perfil.py (onda larga con 1-2 armonicos y peso 0.5-0.7, persistencia+1/2
tendencia), para que entrenamiento y prediccion vean exactamente lo mismo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from perfil import (
    INICIO_REVISION_4,
    ONDA_LARGA_MIN_PUNTOS,
    ONDA_LARGA_PERIODOS_H,
    ONDA_LARGA_PUNTOS_2_ARMONICOS,
    ONDA_LARGA_PUNTOS_PESO_ALTO,
    PESO_ONDA_LARGA,
    PESO_ONDA_LARGA_ALTO,
)

HORIZONTES = np.arange(1, 5)          # +15..+60 min en pasos de 15
MIN_CORTES_ENTRENAMIENTO = 24         # 6 h de cortes con la onda ya ajustable
COLUMNAS = [f"l{k}" for k in range(8)] + ["pend", "fprod", "fonda", "fpt", "h", "hrs"]


def _pt(y: np.ndarray, k: float = 0.5, w: int = 4) -> np.ndarray:
    j = np.flatnonzero(~np.isnan(y))[-w:]
    pend = np.polyfit(j, y[j], 1)[0] if len(j) >= 2 else 0.0
    return np.maximum(y[j[-1]] + k * pend * (len(y) - 1 + HORIZONTES - j[-1]), 0)


def _onda(y: np.ndarray) -> np.ndarray | None:
    n = len(y)
    j = np.flatnonzero(~np.isnan(y))
    if len(j) < ONDA_LARGA_MIN_PUNTOS:
        return None
    armonicos = 2 if len(j) >= ONDA_LARGA_PUNTOS_2_ARMONICOS else 1
    x = (j - (n - 1)).astype(float)
    v = y[j]

    def columnas(z: np.ndarray, w: float) -> np.ndarray:
        cols = [np.ones_like(z)]
        for k in range(1, armonicos + 1):
            cols += [np.sin(k * w * z), np.cos(k * w * z)]
        return np.column_stack(cols)

    mejor = None
    for horas in ONDA_LARGA_PERIODOS_H:
        w = 2 * np.pi / (horas * 4)
        matriz = columnas(x, w)
        coef, *_ = np.linalg.lstsq(matriz, v, rcond=None)
        r = float(((matriz @ coef - v) ** 2).sum())
        if mejor is None or r < mejor[0]:
            mejor = (r, coef, w)
    _, coef, w = mejor
    curva = lambda z: columnas(np.atleast_1d(z).astype(float), w) @ coef  # noqa: E731
    return np.maximum(v[-1] + curva(HORIZONTES.astype(float)) - curva(x[-1])[0], 0)


def _filas(serie: np.ndarray, c: int, i_rev4: int) -> list[dict] | None:
    """Features de una estacion en el corte c (indice de la rejilla)."""
    y = serie[:c + 1]
    validos = np.flatnonzero(~np.isnan(y))
    if validos.size < 2:
        return None
    v = y[validos[-8:]]
    esc = max(float(np.mean(v[-4:])), 5.0)
    base = _pt(y)
    onda = _onda(y[i_rev4:]) if c >= i_rev4 else None
    if onda is None:
        prod, onda = base, base
    else:
        n = int((~np.isnan(y[i_rev4:])).sum())
        w = PESO_ONDA_LARGA_ALTO if n >= ONDA_LARGA_PUNTOS_PESO_ALTO else PESO_ONDA_LARGA
        prod = w * onda + (1 - w) * base
    ult4 = v[-4:]
    pend = np.polyfit(np.arange(len(ult4)), ult4, 1)[0] if len(ult4) >= 2 else 0.0
    lags = {f"l{k}": (v[-1 - k] / esc if k < len(v) else 1.0) for k in range(8)}
    return [dict(h=h + 1, esc=esc, prod=prod[h], fprod=prod[h] / esc, fonda=onda[h] / esc, fpt=base[h] / esc,
                 pend=pend / esc, hrs=(c - i_rev4) / 4, **lags) for h in range(4)]


def corregir(observaciones: pd.DataFrame, data_cutoff: pd.Timestamp) -> dict[tuple[str, int], float] | None:
    """{(estacion, pasos de 15 min desde el corte): prediccion corregida}, o
    `None` si todavia no hay suficientes cortes de la revision 4 para entrenar."""
    import lightgbm as lgb

    obs = observaciones.copy()
    obs["observed_at"] = pd.to_datetime(obs["observed_at"], utc=True)
    rev4 = INICIO_REVISION_4.tz_convert("UTC")
    corte = pd.Timestamp(data_cutoff).tz_convert("UTC")
    desde = rev4 - pd.Timedelta(hours=4)
    obs = obs[(obs["observed_at"] >= desde) & (obs["observed_at"] <= corte)]
    rejilla = pd.date_range(desde, corte.floor("15min"), freq="15min")
    P = obs.pivot_table(index="observed_at", columns="station_id", values="demand").reindex(rejilla)
    A = P.to_numpy(float)
    i_rev4 = rejilla.get_loc(rev4)
    c_final = len(rejilla) - 1
    primer = i_rev4 + ONDA_LARGA_MIN_PUNTOS

    filas = []
    for c in range(primer, c_final + 1):
        for s, estacion in enumerate(P.columns):
            fs = _filas(A[:, s], c, i_rev4)
            if fs is None:
                continue
            for f in fs:
                t_obj = c + f["h"]
                real = A[t_obj, s] if t_obj <= c_final else np.nan
                filas.append({**f, "c": c, "estacion": str(estacion), "t_obj": t_obj, "real": real})
    D = pd.DataFrame(filas)
    if D.empty:
        return None
    entrenamiento = D[(D["t_obj"] <= c_final) & D["real"].notna()]
    if entrenamiento["c"].nunique() < MIN_CORTES_ENTRENAMIENTO:
        return None
    modelo = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, objective="l1", verbose=-1, random_state=0,
    ).fit(entrenamiento[COLUMNAS], entrenamiento["real"] / entrenamiento["esc"] - entrenamiento["fprod"])

    actual = D[D["c"] == c_final]
    correccion = modelo.predict(actual[COLUMNAS])
    valores = np.maximum(actual["fprod"].to_numpy() + correccion, 0) * actual["esc"].to_numpy()
    return {(e, int(h)): float(v) for e, h, v in zip(actual["estacion"], actual["h"], valores)}
