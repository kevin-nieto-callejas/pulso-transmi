"""El perfil adaptativo entra al camino de entrega, asi que se prueba lo que
puede romper un ciclo: que no invente datos, que se calle cuando no sabe y que
reaccione a un cambio de nivel sin dispararse.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from perfil import LIMITES_FACTOR, PerfilAdaptativo, mezclar  # noqa: E402

ZONA = "America/Bogota"


def _observaciones(dias: int = 21, base: float = 100.0, factor_ultimo_dia: float = 1.0) -> pd.DataFrame:
    """Demanda sintetica con un pico diario claro, opcionalmente escalada el
    ultimo dia para simular un level_shift."""
    inicio = pd.Timestamp("2026-08-01 00:00", tz=ZONA)
    filas = []
    for dia in range(dias):
        for paso in range(96):
            hora = paso / 4
            valor = base * (1 + np.sin(2 * np.pi * (hora - 6) / 24))
            if dia == dias - 1:
                valor *= factor_ultimo_dia
            filas.append({
                "station_id": "01000",
                "observed_at": inicio + pd.Timedelta(days=dia, minutes=15 * paso),
                "demand": max(valor, 1.0),
            })
    return pd.DataFrame(filas)


def test_perfil_reproduce_el_patron_diario():
    """Con dias identicos, el perfil de una franja debe ser el valor de esa franja."""
    obs = _observaciones()
    perfil = PerfilAdaptativo(obs)
    ancla = pd.Timestamp("2026-08-21 08:00", tz=ZONA)
    prediccion = perfil.predecir("01000", ancla + pd.Timedelta(minutes=60), ancla)

    esperado = obs[obs["observed_at"] == ancla + pd.Timedelta(minutes=60)]["demand"].iloc[0]
    assert prediccion == pytest.approx(esperado, rel=0.02)


def test_el_factor_de_nivel_absorbe_un_salto_de_nivel():
    """Esto es lo que justifica el modulo: si la estacion lleva una hora 30%
    arriba de su perfil, el pronostico debe subir, no quedarse en el patron
    viejo como hace un modelo entrenado."""
    obs = _observaciones(factor_ultimo_dia=1.3)
    perfil = PerfilAdaptativo(obs)
    ancla = pd.Timestamp("2026-08-21 08:00", tz=ZONA)

    factor = perfil.factor_de_nivel("01000", ancla)
    assert factor == pytest.approx(1.3, rel=0.05)

    sin_drift = PerfilAdaptativo(_observaciones()).predecir("01000", ancla + pd.Timedelta(minutes=60), ancla)
    con_drift = perfil.predecir("01000", ancla + pd.Timedelta(minutes=60), ancla)
    assert con_drift > sin_drift


def test_el_factor_no_se_dispara_con_datos_absurdos():
    """Un pico de 100x es un dato roto, no un drift: se recorta."""
    obs = _observaciones(factor_ultimo_dia=100.0)
    factor = PerfilAdaptativo(obs).factor_de_nivel("01000", pd.Timestamp("2026-08-21 08:00", tz=ZONA))
    assert factor == pytest.approx(LIMITES_FACTOR[1])


def test_devuelve_none_para_una_estacion_desconocida():
    """None significa 'no tengo opinion' y deja pasar el champion solo."""
    perfil = PerfilAdaptativo(_observaciones())
    ancla = pd.Timestamp("2026-08-21 08:00", tz=ZONA)
    assert perfil.predecir("99999", ancla + pd.Timedelta(minutes=15), ancla) is None
    assert perfil.factor_de_nivel("99999", ancla) == 1.0


def test_no_usa_el_futuro():
    """Si el perfil mirara el mismo dia que predice, tendria fuga: el valor
    para un dia con un salto solo puede salir de los dias anteriores."""
    obs = _observaciones(factor_ultimo_dia=2.0)
    perfil = PerfilAdaptativo(obs)
    # Se consulta el perfil crudo (sin factor de nivel) del ultimo dia.
    ancla = pd.Timestamp("2026-08-21 03:00", tz=ZONA)
    crudo = perfil._perfil["01000"][perfil._paso(ancla)]
    normal = _observaciones()
    esperado = normal[normal["observed_at"] == pd.Timestamp("2026-08-21 03:00", tz=ZONA)]["demand"].iloc[0]
    assert crudo == pytest.approx(esperado, rel=0.02)


def test_exige_historia_minima():
    with pytest.raises(ValueError):
        PerfilAdaptativo(_observaciones(dias=2))
    with pytest.raises(ValueError):
        PerfilAdaptativo(pd.DataFrame(columns=["station_id", "observed_at", "demand"]))


def test_mezclar_respeta_al_champion_cuando_el_perfil_calla():
    assert mezclar(100.0, None) == 100.0
    assert mezclar(100.0, float("nan")) == 100.0
    assert mezclar(100.0, 200.0, peso_perfil=0.4) == pytest.approx(140.0)


def test_el_piso_permite_seguir_un_cierre():
    """El generador programa un `closure` que deja una estacion en el 40% de
    su demanda. Si el piso del factor fuera 0.5, el perfil no podria seguir
    esa caida aunque la viera. Este test fija esa capacidad."""
    assert LIMITES_FACTOR[0] < 0.40, "el piso debe dejar seguir un cierre a x0.40"

    obs = _observaciones(factor_ultimo_dia=0.40)
    factor = PerfilAdaptativo(obs).factor_de_nivel("01000", pd.Timestamp("2026-08-21 08:00", tz=ZONA))
    assert factor == pytest.approx(0.40, rel=0.05)
