import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import correccion  # noqa: E402
from perfil import INICIO_REVISION_4  # noqa: E402


def _obs(horas_rev4: float, estaciones=("01000", "02000")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    inicio = INICIO_REVISION_4 - pd.Timedelta(hours=6)
    pasos = int((6 + horas_rev4) * 4) + 1
    filas = []
    for k, est in enumerate(estaciones):
        for i in range(pasos):
            t = inicio + pd.Timedelta(minutes=15 * i)
            x = (t - INICIO_REVISION_4) / pd.Timedelta(hours=1)
            valor = 300.0 if t < INICIO_REVISION_4 else 500 + 300 * np.sin(2 * np.pi * x / 5.5 + k)
            filas.append({"station_id": est, "observed_at": t, "demand": valor + rng.normal(0, 10)})
    return pd.DataFrame(filas)


def test_sin_suficientes_cortes_no_corrige():
    obs = _obs(horas_rev4=6.0)
    assert correccion.corregir(obs, obs["observed_at"].max()) is None


def test_con_historia_devuelve_las_cuatro_predicciones_por_estacion():
    obs = _obs(horas_rev4=14.0)
    corte = obs["observed_at"].max()
    resultado = correccion.corregir(obs, corte)
    assert resultado is not None
    assert set(resultado) == {(e, h) for e in ("01000", "02000") for h in (1, 2, 3, 4)}
    for (est, h), valor in resultado.items():
        x = (corte + pd.Timedelta(minutes=15 * h) - INICIO_REVISION_4) / pd.Timedelta(hours=1)
        k = 0 if est == "01000" else 1
        esperado = 500 + 300 * np.sin(2 * np.pi * x / 5.5 + k)
        assert abs(valor - esperado) < 120, (est, h, valor, esperado)
