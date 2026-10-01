"""Perfil adaptativo de demanda: la mitad del pronostico que se adapta sola.

El champion (CatBoost) aprende el regimen del entrenamiento y se queda ahi. El
generador del profesor, en cambio, mueve parametros causales durante la
competencia (`peak_shift` corre el pico y sube el nivel; `closure` tumba una
estacion y reparte su demanda a las vecinas). Frente a eso un modelo congelado
sobrepredice durante horas hasta que alguien lo reentrena.

Este modulo calcula, sin entrenar nada, el perfil tipico de cada estacion y lo
reescala con lo que acaba de pasar:

    prediccion = perfil(estacion, franja del target, tipo de dia)
                 x  factor_de_nivel(ultima hora)

  * El PERFIL es la media de la misma franja de 15 min en dias previos del
    mismo tipo (habil / sabado / domingo), ponderada con vida media de 14
    dias: lo reciente pesa mas, pero una semana rara no borra el patron.
  * El FACTOR DE NIVEL compara la ultima hora observada contra lo que el
    perfil esperaba para esa misma hora. Si la estacion viene corriendo 20%
    por encima de su perfil, el pronostico sube 20%. Ahi es donde se absorbe
    un `level_shift` sin reentrenar, y en minutos en vez de en dias.

Validado con origen rodante sobre 5 ventanas de un dia (ver docs/HALLAZGOS.md):
mezclado al 40% con el champion sube el peor dia de 83.41 a 85.29 y la media
de 87.27 a 87.73. Gana justo donde mas dolia - los ciclos con drift - y cede
~1 punto en los dias tranquilos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import a_hora_local

PASOS_POR_DIA = 96  # 24 h en intervalos de 15 min
PASOS_POR_HORA = 4

# Valores elegidos por barrido sobre 5 ventanas de validacion. No son magicos,
# pero moverlos a ciegas cuesta puntos: el barrido esta en
# scratchpad/test_perfil_v2.py y la tabla completa en docs/HALLAZGOS.md.
VIDA_MEDIA_DIAS = 14.0
VENTANA_NIVEL = 4          # 4 pasos = la ultima hora
# Un factor fuera de estos limites es un dato roto, no un drift. El piso no es
# 0.5 por una razon concreta: el contrato del generador programa un `closure`
# que deja una estacion en el 40% de su demanda. Con piso 0.5 el perfil no
# podria seguir esa caida NI AUNQUE la viera - se quedaria corto por diseno
# justo en el evento que mas castiga. Bajarlo a 0.30 no cuesta nada en regimen
# normal (mismo resultado en las 5 ventanas de validacion, porque el factor
# nunca llega ahi) y da margen para el cierre.
#
# Ampliado el 29-sep-2026 tras la continuacion del escenario (docs/drift-control.md,
# niveles 0-3): con ground truth ya resuelto se confirmo el mismo patron pero mas
# extremo en ambas puntas. 02300 y 05000 corrieron 4 ciclos seguidos con
# pred/real = 0.44-0.58 (factor de nivel pegado al tope 1.6 en las dos ventanas,
# la demanda real real pedia ~2.0x) y 05100 al reves: cierre detectado, factor
# pegado al piso 0.30, pero pred/real subio 1.11 -> 1.57 ciclo a ciclo porque la
# demanda real seguia cayendo por debajo del 30% que el piso permitia. Mismo
# razonamiento que la vez anterior: en regimen normal el factor nunca se acerca
# a 2.5 ni a 0.15 (no cambia nada en los ciclos sanos), y da margen para seguir
# un salto de nivel o un cierre mas severo que los ya vistos.
#
# Ampliado de nuevo el 30-sep-2026: el profesor activo la revision 3 de drift
# (docs/HALLAZGOS.md #28), mas exigente que la 2. El tope de 2.5 duro menos de
# un dia: el accuracy por ciclo cayo de ~90 a 34.6 (ciclo T050000Z), con las 12
# estaciones cayendo A LA VEZ (no 2-4 como antes) y el factor pegado exacto en
# 2.5 en casi todas. Medido sin tope en el momento: 07111 en 6.42x, 02300 en
# 5.95x, varias entre 3x y 6x, y el factor CORTO por encima del LARGO en casi
# todas - la subida seguia acelerando, no se habia estabilizado. Backtest causal
# contra el ciclo T050000Z ya resuelto: tope 2.5 -> 41.3 de accuracy (perfil
# solo, 12 estaciones); tope 5.0 -> 50.6; tope 8.0 y 12.0 dan lo mismo que 5.0
# (no hace falta mas para ESE ciclo, pero la demanda seguia subiendo despues).
# Se sube a 10.0 con margen sobre el 6.42x ya visto, para no repetir este mismo
# ajuste en un par de ciclos si la subida no toco techo todavia. Mismo
# razonamiento de siempre: en regimen sano el factor nunca se acerca ni a 10 ni
# a 0.15, asi que no cambia nada fuera de estos episodios.
LIMITES_FACTOR = (0.15, 10.0)
DIAS_DE_HISTORIA = 28      # con vida media 14 d, mas atras pesa <0.25
MEZCLA_PERFIL = 0.4        # peso del perfil frente al champion

# Deteccion de cierre. El generador programa un `closure`: una estacion cae a
# ~40% de su demanda. El champion sigue prediciendo el regimen normal (lags de
# 1 dia y 1 semana, slot_mean) y con el 60% de la mezcla arrastra a la
# estacion cerrada hasta ~45 puntos de accuracy. Cuando el nivel de una
# estacion se desploma se le quita la voz al champion y manda el perfil solo.
# Sobre 20 escenarios simulados (4 estaciones, 2 horas de inicio, transicion
# corta y larga) sube las 12 estaciones ~3.0 puntos y la estacion cerrada de
# ~45 a ~82; en regimen normal el resultado es identico (87.89 / 86.05, se
# dispara 12 veces en 5520 predicciones). Exigir DOS ventanas evita disparar
# con un valle de una sola hora.
UMBRAL_CIERRE_CORTO = 0.70  # factor de nivel de la ultima hora (4 pasos)
UMBRAL_CIERRE_LARGO = 0.80  # factor de las ultimas 2 horas (8 pasos)
VENTANA_CIERRE_LARGA = 8
PESO_PERFIL_CIERRE = 1.0

# Deteccion de alza sostenida, simetrica al cierre. El 30-sep el profesor activo
# una revision de drift que cambia la FORMA temporal de la demanda (no solo el
# nivel) - su propia bitacora dice que "la antigua referencia adaptativa que
# solo ajustaba la escala del perfil deja de ser apropiada". Con el tope recien
# ampliado a 2.5 se vio en vivo: 02300 y 05000 llevan horas con el factor pegado
# al tope en las dos ventanas, y como no tienen desfase se quedan en el 40% de
# mezcla de siempre - el champion (60%) no sabe nada de la subida y arrastra la
# prediccion. 06000 paso de factor4=1.28 a 2.50 en una sola hora y su accuracy
# se desplomo de 82 a 37 en el mismo salto. Con el perfil solo como referencia
# optimista, las estaciones en alza sostenida (02300, 05000, 07111) miden
# 85-90 de accuracy en las 7 horas de este episodio - muy por encima de lo que
# puede dar un champion ciego al evento. Chequeado contra 15 anclas de regimen
# sano (5-9 sep, 180 combinaciones estacion x hora): CERO disparos falsos,
# porque el factor nunca se acerca a estos umbrales sin una subida real.
#
# Actualizacion 1-oct (hallazgo #31): a diferencia de un cierre (que una vez
# empieza tiende a persistir), una subida de esta revision puede revertirse de
# golpe. Caso real: 05000/07105/09122 subieron hasta 10x y luego el profesor
# los hizo caer de 1627/868/1244 a 167/75/114 en apenas 2 horas (ciclo
# T080000Z). Con peso=1.0 el perfil siguio extrapolando el pico (876-1252)
# justo cuando la demanda real ya se habia desplomado -> 0.0 de accuracy en
# las tres. El problema no es de deteccion tardia: al momento de predecir
# (ancla 08:00) la caida todavia no habia ocurrido, era informacion del
# futuro. Bajar el peso deja una fraccion de champion como colchon para
# cuando la subida se revierte sin aviso. Backtest causal contra los 4
# ciclos del colapso (T050000Z-T080000Z, 12 estaciones, ground truth real):
# peso 1.0 -> 32.3 de accuracy; 0.7 -> 35.1; 0.5 -> 35.9 (mejor punto medido);
# 0.3 -> 35.0; 0.0 (solo champion) -> 29.8, peor que cualquier mezcla. Se deja
# en 0.5: conserva la mayor parte de la ventaja de #28 en una subida genuina
# y limita el dano cuando se revierte sin aviso.
UMBRAL_ALZA_CORTO = 1.30
UMBRAL_ALZA_LARGO = 1.20
PESO_PERFIL_ALZA = 0.5

# Peso del perfil cuando hay un desfase confirmado. El champion sigue anclado a
# la hora vieja del pico: con el peak_shift la rampa de la manana llega ~45 min
# tarde y el champion la predice antes, sobrepredice las horas previas y
# subpredice las siguientes (05000 marco 23 de accuracy en un ciclo). Donde el
# detector de fase ya confirma un corrimiento, el perfil sabe algo que el
# champion no. Medido con 38 h de drift activo: +0.5 en la ventana del 12-sep y
# +0.7 en las ultimas horas, y -0.03 en los dias sin drift (0.55: +0.4, 0.70:
# +0.6, 0.85: +0.7, 1.00: +0.5; se toma 0.70 por prudencia, casi toda la ganancia).
PESO_PERFIL_CON_DESFASE = 0.70

# Correccion de fase. El `peak_shift` del generador corre el centro del pico
# diario (+45 min en el ejemplo del profesor) ademas de subir el nivel. El
# perfil sigue anclado a la hora vieja y predice el pico donde ya no esta, y
# ningun factor de NIVEL arregla un desfase temporal. Por estacion se estima
# con las ultimas 24 h cuantos pasos de 15 min conviene desplazar el perfil.
# Con la evidencia en la mano: en las 5 ventanas de validacion es inocuo (el
# resultado es identico, porque casi siempre el desfase estimado es 0) y en
# las horas con el drift activo sube las 12 estaciones ~1.2 puntos.
DESFASES = range(-4, 5)      # de -1 h a +1 h, en pasos de 15 min
VENTANA_FASE = 96            # las ultimas 24 h
MEJORA_MINIMA_FASE = 0.10    # exige 10% menos de error para moverse de 0


def _tipo_de_dia(fechas: np.ndarray) -> np.ndarray:
    """0 = habil, 1 = sabado, 2 = domingo. Sabado y domingo tienen perfiles
    muy distintos entre si, asi que promediarlos juntos emborrona los dos."""
    dow = np.array([f.dayofweek for f in fechas])
    return np.where(dow < 5, 0, np.where(dow == 5, 1, 2))


class PerfilAdaptativo:
    """Perfil por estacion, listo para consultar en cualquier instante.

    Se construye una vez por ciclo con el historico reciente y despues se
    consulta 48 veces (12 estaciones x 4 horizontes) sin recalcular nada.
    """

    def __init__(
        self,
        observaciones: pd.DataFrame,
        vida_media_dias: float = VIDA_MEDIA_DIAS,
        ventana_nivel: int = VENTANA_NIVEL,
    ) -> None:
        if observaciones.empty:
            raise ValueError("El perfil adaptativo necesita observaciones y llego un frame vacio")

        obs = observaciones.copy()
        obs["observed_at"] = a_hora_local(obs["observed_at"])
        obs = obs.sort_values(["station_id", "observed_at"])

        # Rejilla plana: una casilla por cada 15 min desde el primer dia, con
        # NaN donde el collector no alcanzo a traer el dato. Trabajar sobre
        # indices enteros hace que "la misma franja de ayer" sea una resta.
        self._origen = obs["observed_at"].min().normalize()
        pasos = ((obs["observed_at"] - self._origen) / pd.Timedelta(minutes=15)).round().astype(int)
        dias_con_datos = int(pasos.max()) // PASOS_POR_DIA + 1
        if dias_con_datos < 3:
            raise ValueError(f"El perfil necesita al menos 3 dias de historia y hay {dias_con_datos}")
        # Un dia de mas, sin observaciones: el perfil tambien se calcula para
        # "manana". Sin el, un target que cruza la medianoche caia fuera del
        # arreglo y el perfil se quedaba sin opinion justo en esa franja, y la
        # correccion de fase no podia mirar mas alla del ultimo paso del dia.
        self._n_dias = dias_con_datos + 1

        self._ventana_nivel = ventana_nivel
        fechas = np.array([self._origen + pd.Timedelta(days=d) for d in range(self._n_dias)])
        tipos = _tipo_de_dia(fechas)

        self._serie: dict[str, np.ndarray] = {}
        self._perfil: dict[str, np.ndarray] = {}
        for estacion, grupo in obs.groupby("station_id"):
            serie = np.full(self._n_dias * PASOS_POR_DIA, np.nan)
            indices = pasos.loc[grupo.index].to_numpy()
            dentro = indices < serie.size
            serie[indices[dentro]] = grupo["demand"].to_numpy()[dentro]
            self._serie[str(estacion)] = serie
            self._perfil[str(estacion)] = self._calcular_perfil(serie, tipos, vida_media_dias)

    def _calcular_perfil(self, serie: np.ndarray, tipos: np.ndarray, vida_media: float) -> np.ndarray:
        """Para cada dia, el perfil de sus 96 franjas segun los dias ANTERIORES
        del mismo tipo. Solo mira hacia atras: nunca usa el dia que predice."""
        por_dia = serie.reshape(self._n_dias, PASOS_POR_DIA)
        perfil = np.full_like(por_dia, np.nan)
        for dia in range(self._n_dias):
            previos = np.flatnonzero(tipos[:dia] == tipos[dia])
            if previos.size == 0:
                continue
            pesos = 0.5 ** ((dia - previos) / vida_media)
            valores = por_dia[previos]
            presentes = ~np.isnan(valores)
            suma_pesos = (pesos[:, None] * presentes).sum(axis=0)
            suma = (np.where(presentes, valores, 0.0) * pesos[:, None]).sum(axis=0)
            perfil[dia] = np.where(suma_pesos > 0, suma / np.maximum(suma_pesos, 1e-9), np.nan)
        return perfil.reshape(-1)

    def _paso(self, instante: pd.Timestamp) -> int:
        instante = a_hora_local(pd.Series([instante])).iloc[0]
        return int(round((instante - self._origen) / pd.Timedelta(minutes=15)))

    def factor_de_nivel(
        self, estacion: str, ancla_at: pd.Timestamp, ventana: int | None = None, desfase: int = 0,
    ) -> float:
        """Cuanto se desvia la ultima hora real respecto a lo que el perfil
        esperaba. 1.0 = la estacion va justo en su perfil. `ventana` (en pasos
        de 15 min) permite mirar mas atras; por defecto la ventana del perfil.
        `desfase` compara contra el perfil corrido esa cantidad de pasos."""
        ventana = self._ventana_nivel if ventana is None else ventana
        serie, perfil = self._serie.get(str(estacion)), self._perfil.get(str(estacion))
        if serie is None:
            return 1.0
        fin = self._paso(ancla_at) + 1
        inicio = max(0, fin - ventana)
        if inicio - desfase < 0 or fin - desfase > perfil.size:
            return 1.0
        reales, esperados = serie[inicio:fin], perfil[inicio - desfase:fin - desfase]
        validos = ~np.isnan(reales) & ~np.isnan(esperados)
        # Con menos de media ventana, o con un perfil que suma cero, el factor
        # seria ruido amplificado: mejor no corregir nada.
        if validos.sum() < max(2, ventana // 2) or esperados[validos].sum() <= 0:
            return 1.0
        return float(np.clip(reales[validos].sum() / esperados[validos].sum(), *LIMITES_FACTOR))

    def desfase(self, estacion: str, ancla_at: pd.Timestamp) -> int:
        """Pasos de 15 min que conviene correr el perfil para alinearlo con lo
        que acaba de pasar (positivo = el pico real llega mas tarde que en el
        perfil). 0 salvo que correrlo baje el error al menos MEJORA_MINIMA_FASE.
        Solo mira datos <= `ancla_at`."""
        serie, perfil = self._serie.get(str(estacion)), self._perfil.get(str(estacion))
        if serie is None:
            return 0
        fin = self._paso(ancla_at) + 1
        inicio = fin - VENTANA_FASE
        margen = max(abs(d) for d in DESFASES)
        if inicio - margen < 0 or fin + margen > perfil.size:
            return 0
        t = np.arange(inicio, fin)
        reales = serie[t]
        errores: dict[int, float] = {}
        for d in DESFASES:
            esperados = perfil[t - d]
            validos = ~np.isnan(reales) & ~np.isnan(esperados)
            if validos.sum() < VENTANA_FASE // 2 or esperados[validos].sum() <= 0:
                continue
            # Cada desfase se compara con su mejor nivel: lo que importa aqui
            # es la FORMA de la curva, el nivel ya lo corrige el factor.
            k = reales[validos].sum() / esperados[validos].sum()
            errores[d] = float(np.abs(reales[validos] - k * esperados[validos]).sum() / max(reales[validos].sum(), 1e-9))
        if 0 not in errores:
            return 0
        mejor = min(errores, key=errores.get)
        return mejor if errores[0] - errores[mejor] > MEJORA_MINIMA_FASE * errores[0] else 0

    def peso_de_mezcla(self, estacion: str, ancla_at: pd.Timestamp) -> float:
        """Peso del perfil frente al champion para esta estacion y este ancla.

        Normalmente MEZCLA_PERFIL. Si el nivel se desploma en las dos ventanas
        (un cierre) o sube sostenido en las dos ventanas (una alza), el
        champion deja de opinar y manda el perfil escalado: en ambos casos el
        champion esta anclado al regimen viejo y el perfil ya vio lo que paso.

        El nivel se mide contra el perfil YA desplazado por el desfase estimado.
        Sin eso, una estacion con el pico corrido (peak_shift) parece hundirse
        cada vez que la demanda sube: la curva real llega tarde, el perfil sin
        mover ya esta arriba y el cociente cae. Asi confundio la rampa de la
        manana de 05000 con un cierre y le quito la voz al champion cuando el
        champion iba bien.
        """
        try:
            desfase = self.desfase(estacion, ancla_at)
        except Exception:
            desfase = 0
        corto = self.factor_de_nivel(estacion, ancla_at, desfase=desfase)
        largo = self.factor_de_nivel(estacion, ancla_at, VENTANA_CIERRE_LARGA, desfase)
        if corto < UMBRAL_CIERRE_CORTO and largo < UMBRAL_CIERRE_LARGO:
            return PESO_PERFIL_CIERRE
        if corto > UMBRAL_ALZA_CORTO and largo > UMBRAL_ALZA_LARGO:
            return PESO_PERFIL_ALZA
        return PESO_PERFIL_CON_DESFASE if desfase != 0 else MEZCLA_PERFIL

    def predecir(self, estacion: str, target_at: pd.Timestamp, ancla_at: pd.Timestamp) -> float | None:
        """Demanda esperada en `target_at`, o None si no hay perfil para esa
        franja (estacion nueva, hueco del collector). None significa "no tengo
        opinion": quien llama se queda con el champion solo."""
        perfil = self._perfil.get(str(estacion))
        if perfil is None:
            return None
        # La correccion de fase es una mejora, no un requisito: si falla por
        # lo que sea se predice con el perfil sin desplazar.
        try:
            desfase = self.desfase(estacion, ancla_at)
        except Exception:
            desfase = 0
        paso = self._paso(target_at) - desfase
        if not 0 <= paso < perfil.size or np.isnan(perfil[paso]):
            return None
        return float(perfil[paso] * self.factor_de_nivel(estacion, ancla_at, desfase=desfase))


def mezclar(valor_champion: float, valor_perfil: float | None, peso_perfil: float = MEZCLA_PERFIL) -> float:
    """Combina las dos opiniones. Si el perfil no tiene una, manda el champion.

    Los dos modelos fallan distinto: el champion se equivoca cuando el regimen
    cambia, el perfil cuando el dia es atipico respecto a los anteriores.
    Promediarlos sale mejor que cualquiera de los dos por separado.
    """
    if valor_perfil is None or not np.isfinite(valor_perfil):
        return valor_champion
    return (1.0 - peso_perfil) * valor_champion + peso_perfil * valor_perfil
