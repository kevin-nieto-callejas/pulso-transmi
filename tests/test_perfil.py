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
    PESO_PERFIL_ALZA,
    PESO_PERFIL_CIERRE,
    PESO_PERFIL_CON_DESFASE,
    UMBRAL_EXTRAPOLACION_ALZA,
    UMBRAL_EXTRAPOLACION_CIERRE,
    INICIO_REVISION_4,
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


def _observaciones_con_alza(dias: int = 21, factor_alza: float = 2.0, horas_de_alza: int = 4) -> pd.DataFrame:
    """Como `_observaciones_con_cierre`, pero al reves: la estacion sube a
    `factor_alza` durante las ultimas `horas_de_alza` horas (una alza ya en
    marcha, como la que trae un cambio de forma en la demanda)."""
    obs = _observaciones(dias=dias)
    corte = obs["observed_at"].max() - pd.Timedelta(hours=horas_de_alza)
    obs.loc[obs["observed_at"] > corte, "demand"] *= factor_alza
    return obs


def test_una_alza_sostenida_le_quita_la_voz_al_champion():
    """Simetrico al cierre: con el nivel disparado en las dos ventanas, el
    champion sigue creyendo en el regimen viejo y manda el perfil escalado."""
    obs = _observaciones_con_alza()
    perfil = PerfilAdaptativo(obs)
    ancla = obs["observed_at"].max()
    assert perfil.peso_de_mezcla("01000", ancla) == PESO_PERFIL_ALZA


def test_una_alza_extrema_activa_la_extrapolacion():
    """Con el factor MUY por encima del umbral de alza (no solo del umbral
    normal), manda la extrapolacion lineal sobre la serie cruda, no el perfil
    historico escalado (hallazgo #31/#34: con un pico que se revierte de
    golpe, hasta el perfil con peso reducido se queda extrapolando el nivel
    alto)."""
    obs = _observaciones_con_alza(factor_alza=4.0)
    perfil = PerfilAdaptativo(obs)
    ancla = obs["observed_at"].max()
    objetivo = ancla + pd.Timedelta(minutes=15)
    assert perfil.factor_de_nivel("01000", ancla) > UMBRAL_EXTRAPOLACION_ALZA
    assert perfil.extrapolacion_extrema("01000", ancla, objetivo) is not None


def test_una_alza_moderada_no_activa_la_extrapolacion():
    """Justo encima del umbral normal de alza pero lejos del umbral extremo:
    se queda con la mezcla champion+perfil de siempre, no la extrapolacion."""
    obs = _observaciones_con_alza(factor_alza=1.8)
    perfil = PerfilAdaptativo(obs)
    ancla = obs["observed_at"].max()
    objetivo = ancla + pd.Timedelta(minutes=15)
    assert perfil.factor_de_nivel("01000", ancla) < UMBRAL_EXTRAPOLACION_ALZA
    assert perfil.extrapolacion_extrema("01000", ancla, objetivo) is None


def test_un_cierre_extremo_activa_la_extrapolacion():
    """Simetrico: un cierre muy por debajo del umbral extremo tambien manda
    la extrapolacion en vez del perfil escalado."""
    obs = _observaciones_con_cierre(factor_cierre=0.10)
    perfil = PerfilAdaptativo(obs)
    ancla = obs["observed_at"].max()
    objetivo = ancla + pd.Timedelta(minutes=15)
    assert perfil.factor_de_nivel("01000", ancla) < UMBRAL_EXTRAPOLACION_CIERRE
    assert perfil.extrapolacion_extrema("01000", ancla, objetivo) is not None


def _observaciones_con_onda(dias: int = 21, horas_onda: int = 24, periodo_pasos: int = 16) -> pd.DataFrame:
    """Como `_observaciones`, pero las ultimas `horas_onda` horas son una onda
    cuadrada de `periodo_pasos` (alta la mitad, baja la otra mitad): la forma
    que trae la revision 3 de drift (hallazgo #35)."""
    obs = _observaciones(dias=dias).reset_index(drop=True)
    corte = obs["observed_at"].max() - pd.Timedelta(hours=horas_onda)
    tramo = obs.index[obs["observed_at"] > corte]
    fase = np.arange(len(tramo)) % periodo_pasos
    obs.loc[tramo, "demand"] = np.where(fase < periodo_pasos // 2, 1000.0, 200.0)
    return obs


def test_una_onda_de_4_horas_se_detecta_y_se_copia():
    obs = _observaciones_con_onda()
    perfil = PerfilAdaptativo(obs)
    ancla = obs["observed_at"].max()
    p, acierto = perfil.periodo_corto("01000", ancla)
    assert p == 16 and acierto > 99
    for minutos in (15, 30, 45, 60):
        objetivo = ancla + pd.Timedelta(minutes=minutos)
        esperado = obs.loc[obs["observed_at"] == objetivo - pd.Timedelta(hours=4), "demand"].iloc[0]
        assert perfil.prediccion_periodica("01000", ancla, objetivo) == pytest.approx(esperado)


def test_la_curva_diaria_normal_no_activa_la_onda_corta():
    obs = _observaciones()
    perfil = PerfilAdaptativo(obs)
    ancla = obs["observed_at"].max()
    assert perfil.prediccion_periodica("01000", ancla, ancla + pd.Timedelta(minutes=15)) is None


def test_la_extrapolacion_extrema_sigue_la_tendencia_reciente_no_el_pico():
    """La extrapolacion debe seguir la caida reciente, no el nivel alto de
    hace unas horas: con una subida que ya esta revirtiendo en las ultimas
    lecturas, la prediccion debe quedar por DEBAJO del ultimo valor alto, no
    repetirlo."""
    obs = _observaciones_con_alza(factor_alza=4.0, horas_de_alza=6)
    # las ultimas 4 lecturas (1h) caen en picada, simulando el colapso
    # posterior al pico que describe el hallazgo #31
    idx_ultimas = obs.index[-4:]
    obs.loc[idx_ultimas, "demand"] = [1000.0, 700.0, 400.0, 150.0]
    perfil = PerfilAdaptativo(obs)
    ancla = obs["observed_at"].max()
    objetivo = ancla + pd.Timedelta(minutes=15)
    valor = perfil.extrapolacion_extrema("01000", ancla, objetivo)
    assert valor is not None
    assert valor < 150.0  # sigue bajando, no rebota al nivel del pico


def _observaciones_con_pico_corrido(dias: int = 21, pasos: int = 3, dias_corridos: int = 2) -> pd.DataFrame:
    """Como `_observaciones`, pero en los ULTIMOS `dias_corridos` dias el pico
    llega `pasos` pasos de 15 min mas tarde (un peak_shift ya establecido: en
    la realidad la transicion dura 24 h y el desfase lleva un dia o mas)."""
    obs = _observaciones(dias=dias).reset_index(drop=True)
    dia = obs["observed_at"].dt.normalize()
    corrido = (dia > dia.max() - pd.Timedelta(days=dias_corridos)).to_numpy()
    demanda = obs["demand"].to_numpy().copy()
    idx = np.flatnonzero(corrido)
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


def test_una_rampa_retrasada_no_se_confunde_con_un_cierre():
    """Regresion de un falso positivo real: con el pico corrido, la demanda sube
    tarde y el perfil sin mover ya esta arriba, asi que el cociente cae. Con el
    desfase corregido el detector NO debe tomarlo por un cierre: el perfil pesa
    lo que corresponde a un desfase, no el peso total de un cierre."""
    obs = _observaciones_con_pico_corrido(pasos=3)
    perfil = PerfilAdaptativo(obs)
    # justo cuando la curva sube: el perfil sin mover adelanta a la demanda real
    for hora in range(4, 12):
        ancla = obs["observed_at"].max().normalize() + pd.Timedelta(hours=hora)
        assert perfil.peso_de_mezcla("01000", ancla) != PESO_PERFIL_CIERRE, f"falsa alarma a las {hora}:00"


def test_con_un_desfase_confirmado_el_perfil_pesa_mas():
    """El champion sigue anclado a la hora vieja del pico; donde el desfase esta
    confirmado se le da mas voz al perfil."""
    obs = _observaciones_con_pico_corrido(pasos=3)
    perfil = PerfilAdaptativo(obs)
    ancla = obs["observed_at"].max()
    assert perfil.desfase("01000", ancla) == 3
    assert perfil.peso_de_mezcla("01000", ancla) == PESO_PERFIL_CON_DESFASE
    assert PESO_PERFIL_CON_DESFASE > MEZCLA_PERFIL


def test_rellenar_contexto_extiende_con_la_mediana_de_cada_franja():
    from features import CONTEXT_COLUMNS, rellenar_contexto
    t = pd.date_range("2026-09-01 00:00", "2026-09-08 23:45", freq="15min", tz=ZONA)
    ctx = pd.DataFrame({"observed_at": t})
    for c in CONTEXT_COLUMNS:
        ctx[c] = t.hour.astype(float)  # cada franja vale su hora
    hasta = pd.Timestamp("2026-09-10 12:00", tz=ZONA)
    out = rellenar_contexto(ctx, hasta)
    assert out["observed_at"].max() == hasta.tz_convert("UTC")
    assert out[CONTEXT_COLUMNS].isna().sum().sum() == 0
    nuevo = out[out["observed_at"] > t.max()].set_index("observed_at")
    horas = nuevo.index.tz_convert(ZONA).hour
    assert (nuevo["temperature_c"].to_numpy() == horas.to_numpy()).all()
    # y no toca lo que ya existia, ni rellena si no hace falta
    assert len(rellenar_contexto(ctx, t.max())) == len(ctx)


def _observaciones_revision_4(horas_rev4: float = 6.0, periodo_h: float = 5.5) -> pd.DataFrame:
    """Historia normal hasta el inicio de la revision 4 y despues una onda
    sinusoidal lenta: la forma del hallazgo #40."""
    inicio = INICIO_REVISION_4 - pd.Timedelta(days=5)
    pasos = int((5 * 24 + horas_rev4) * 4) + 1
    instantes = [inicio + pd.Timedelta(minutes=15 * i) for i in range(pasos)]
    valores = []
    for t in instantes:
        if t < INICIO_REVISION_4:
            valores.append(300.0)
        else:
            x = (t - INICIO_REVISION_4) / pd.Timedelta(hours=1)
            valores.append(500.0 + 400.0 * np.sin(2 * np.pi * x / periodo_h))
    return pd.DataFrame({"station_id": "01000", "observed_at": instantes, "demand": valores})


def test_la_onda_larga_sigue_la_sinusoide_de_la_revision_4():
    obs = _observaciones_revision_4()
    perfil = PerfilAdaptativo(obs)
    ancla = obs["observed_at"].max()
    for minutos in (15, 30, 45, 60):
        objetivo = ancla + pd.Timedelta(minutes=minutos)
        x = (objetivo - INICIO_REVISION_4) / pd.Timedelta(hours=1)
        esperado = 500.0 + 400.0 * np.sin(2 * np.pi * x / 5.5)
        assert perfil.onda_larga("01000", ancla, objetivo) == pytest.approx(esperado, abs=1.0)


def test_la_onda_larga_espera_cuatro_horas_de_la_revision_4():
    obs = _observaciones_revision_4(horas_rev4=3.0)
    perfil = PerfilAdaptativo(obs)
    ancla = obs["observed_at"].max()
    assert perfil.onda_larga("01000", ancla, ancla + pd.Timedelta(minutes=15)) is None


def test_la_onda_larga_no_existe_antes_de_la_revision_4():
    obs = _observaciones()
    perfil = PerfilAdaptativo(obs)
    ancla = obs["observed_at"].max()
    assert perfil.onda_larga("01000", ancla, ancla + pd.Timedelta(minutes=15)) is None


def test_con_ocho_horas_la_onda_larga_usa_dos_armonicos_y_sigue_la_curva():
    obs = _observaciones_revision_4(horas_rev4=10.0)
    perfil = PerfilAdaptativo(obs)
    ancla = obs["observed_at"].max()
    for minutos in (15, 60):
        objetivo = ancla + pd.Timedelta(minutes=minutos)
        x = (objetivo - INICIO_REVISION_4) / pd.Timedelta(hours=1)
        esperado = 500.0 + 400.0 * np.sin(2 * np.pi * x / 5.5)
        assert perfil.onda_larga("01000", ancla, objetivo) == pytest.approx(esperado, abs=2.0)
