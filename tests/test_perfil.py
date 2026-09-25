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

from perfil import (  # noqa: E402
    LIMITES_FACTOR,
    MEZCLA_PERFIL,
    PESO_PERFIL_CIERRE,
    PerfilAdaptativo,
    mezclar,
)

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


def _observaciones_con_cierre(dias: int = 21, factor_cierre: float = 0.40, horas_de_cierre: int = 6) -> pd.DataFrame:
    """Como `_observaciones`, pero la estacion cae a `factor_cierre` durante las
    ultimas `horas_de_cierre` horas de la serie (un closure ya en marcha)."""
    obs = _observaciones(dias=dias)
    corte = obs["observed_at"].max() - pd.Timedelta(hours=horas_de_cierre)
    obs.loc[obs["observed_at"] > corte, "demand"] *= factor_cierre
    return obs


def test_un_cierre_le_quita_la_voz_al_champion():
    """Con el nivel desplomado en las dos ventanas, manda el perfil: el
    champion sigue creyendo en el regimen normal y arrastraria la mezcla."""
    obs = _observaciones_con_cierre()
    perfil = PerfilAdaptativo(obs)
    ancla = obs["observed_at"].max()
    assert perfil.peso_de_mezcla("01000", ancla) == PESO_PERFIL_CIERRE


def test_en_regimen_normal_no_se_dispara_el_detector():
    """Un falso positivo apagaria al champion sin motivo, asi que el detector
    tiene que callar cuando la estacion va en su perfil."""
    obs = _observaciones()
    perfil = PerfilAdaptativo(obs)
    for ancla in obs["observed_at"].iloc[-96::8]:
        assert perfil.peso_de_mezcla("01000", ancla) == MEZCLA_PERFIL


def test_un_valle_de_una_sola_hora_no_cuenta_como_cierre():
    """Exigir las dos ventanas evita disparar con una caida breve: en una
    hora el factor corto puede bajar de 0.7 pero el largo no llega a 0.8."""
    obs = _observaciones()
    corte = obs["observed_at"].max() - pd.Timedelta(hours=1)
    obs.loc[obs["observed_at"] > corte, "demand"] *= 0.30
    perfil = PerfilAdaptativo(obs)
    # solo la ultima hora cae; el tramo previo (2 h) va en su perfil
    assert perfil.factor_de_nivel("01000", obs["observed_at"].max()) < 0.70
    assert perfil.peso_de_mezcla("01000", obs["observed_at"].max()) == MEZCLA_PERFIL


def test_una_estacion_desconocida_no_dispara_el_detector():
    perfil = PerfilAdaptativo(_observaciones())
    assert perfil.peso_de_mezcla("99999", pd.Timestamp("2026-08-21 08:00", tz=ZONA)) == MEZCLA_PERFIL


def _observaciones_con_pico_corrido(dias: int = 21, pasos: int = 3) -> pd.DataFrame:
    """Como `_observaciones`, pero el ULTIMO dia el pico llega `pasos` pasos de
    15 min mas tarde (un peak_shift ya en marcha)."""
    obs = _observaciones(dias=dias).reset_index(drop=True)
    ultimo = obs["observed_at"].dt.normalize() == obs["observed_at"].dt.normalize().max()
    demanda = obs["demand"].to_numpy().copy()
    idx = np.flatnonzero(ultimo.to_numpy())
    demanda[idx] = obs["demand"].to_numpy()[idx - pasos]
    obs["demand"] = demanda
    return obs


def test_detecta_un_pico_corrido():
    """El factor de NIVEL no arregla un desfase temporal: hay que detectarlo."""
    obs = _observaciones_con_pico_corrido(pasos=3)
    perfil = PerfilAdaptativo(obs)
    assert perfil.desfase("01000", obs["observed_at"].max()) == 3


def test_en_regimen_normal_el_desfase_es_cero():
    """Un falso positivo movería el perfil sin motivo: tiene que ser 0 cuando la
    curva ya coincide."""
    obs = _observaciones()
    perfil = PerfilAdaptativo(obs)
    for ancla in obs["observed_at"].iloc[-96::12]:
        assert perfil.desfase("01000", ancla) == 0


def test_corregir_la_fase_acerca_la_prediccion():
    """Con el pico corrido, predecir con el desfase estimado debe quedar mas
    cerca de lo real que predecir con el perfil sin mover."""
    obs = _observaciones_con_pico_corrido(pasos=3)
    perfil = PerfilAdaptativo(obs)
    ancla = obs["observed_at"].max() - pd.Timedelta(hours=8)
    objetivo = ancla + pd.Timedelta(minutes=60)
    real = obs[obs["observed_at"] == objetivo]["demand"].iloc[0]

    con_fase = perfil.predecir("01000", objetivo, ancla)
    sin_fase = perfil._perfil["01000"][perfil._paso(objetivo)] * perfil.factor_de_nivel("01000", ancla)
    assert abs(con_fase - real) < abs(sin_fase - real)


def test_sin_historia_suficiente_el_desfase_es_cero():
    """Con pocos dias la ventana de 24 h no cabe: no se corrige nada."""
    obs = _observaciones(dias=3)
    perfil = PerfilAdaptativo(obs)
    assert perfil.desfase("01000", obs["observed_at"].max()) == 0
    assert perfil.desfase("99999", obs["observed_at"].max()) == 0
