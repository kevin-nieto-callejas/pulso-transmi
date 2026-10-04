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

# Extrapolacion lineal directa sobre la serie cruda (sin pasar por el perfil
# historico) para el caso EXTREMO: un pico que se revierte de golpe (hallazgo
# #31) deja al perfil extrapolando el nivel alto justo cuando la demanda real
# ya se desplomo, y bajar su peso a 0.5 no alcanza (T100000Z: 05000/07105/
# 09122/02300/06000/07111 en 0.0 de accuracy exacto). La extrapolacion lineal
# de las ultimas 4 lecturas (1 hora) no sabe nada de patrones historicos, solo
# seguir la tendencia MAS RECIENTE - exactamente lo que hace falta cuando el
# evento ya esta tan lejos de cualquier dia anterior que el perfil historico
# estorba mas de lo que ayuda. Confirmado con la literatura: el aprendizaje
# online/incremental recupera de un drift en <1h vs 24h-7 dias de un reentreno
# por lotes (ver docs/HALLAZGOS.md #34).
#
# Pero NO es universal: en alza/cierre MODERADO (el caso normal que ya cubren
# los umbrales de arriba) el perfil historico SI aporta forma que una recta no
# tiene - probado en 8 ciclos de regimen sano, la extrapolacion sin filtro
# pierde 1.9 puntos de promedio frente a la mezcla actual. Por eso el umbral
# para activarla es mas exigente que el de alza/cierre normal: solo cuando el
# factor esta MUY lejos de 1.0, no en el borde.
#
# Backtest causal real (T080000Z y T100000Z, accuracy oficial por estacion):
# umbral 2.0/2.5 -> 64.35 y 52.12 (vs produccion real 41.23 y 32.70, +20 a +23
# puntos); umbral 3.0 -> 49.32 y 46.12 (empieza a perder el beneficio, el
# corte queda muy alto). Con 2.5 el costo en regimen sano baja a -0.82 puntos
# (82.70 vs ~83.52) sin perder nada del beneficio en el colapso. Se deja en
# 2.5/0.4.
UMBRAL_EXTRAPOLACION_ALZA = 2.5
UMBRAL_EXTRAPOLACION_CIERRE = 0.4
PASOS_EXTRAPOLACION = 4  # 1 hora de historia (4 x 15 min)

# Periodicidad corta (hallazgo #35). La revision 3 de drift no es un cambio de
# nivel ni de fase: convierte la curva diaria en una ONDA de ~4 horas (2 h alta,
# 2 h baja) en 10 de 12 estaciones (autocorrelacion 0.65-0.88 en el lag de 16
# pasos; 02300 y 07107 a 8 h). Los "picos que se revierten de golpe" de #31-#34
# eran esta onda. Ni el champion (lags 15 min/1 h/1 d/1 sem) ni el perfil diario
# la pueden ver. El naive estacional "lo que paso hace un periodo" acierta 91%
# el 19-sep, contra ~40% del pipeline.
#
# Por estacion y en cada ciclo, con datos <= ancla: se busca el periodo P entre
# 2 h y 6 h que mejor habria acertado en las ultimas 12 h (naive retrospectivo).
# Si ese acierto supera el umbral, se predice el promedio de la demanda en
# target-P y target-2P; si no, sigue el pipeline de siempre. P <= 6 h cabe al
# menos dos veces en la ventana de 12 h: con periodos mas largos el detector
# encontraba coincidencias espurias en regimen normal.
#
# Backtest sobre los 231 ciclos oficiales resueltos (metrica oficial):
# todo 80.14 -> 83.29, regimen normal 85.25 -> 85.25 (identico: nunca se
# activa ahi), drift 59.04 -> 75.20, ultimos 12 ciclos 41.01 -> 91.78. Umbral
# 70 da casi lo mismo en drift pero cuesta -0.8 en normal; 80 es el corte.
PERIODO_MIN = 8          # 2 h
PERIODO_MAX = 24         # 6 h
VENTANA_PERIODO = 48     # 12 h de evidencia retrospectiva
UMBRAL_PERIODO = 80.0    # accuracy retrospectiva minima para confiar en la onda
CICLOS_PERIODO = 2       # promedia target-P y target-2P

# Peso del perfil cuando hay un desfase confirmado. El champion sigue anclado a
# la hora vieja del pico: con el peak_shift la rampa de la manana llega ~45 min
# tarde y el champion la predice antes, sobrepredice las horas previas y
# subpredice las siguientes (05000 marco 23 de accuracy en un ciclo). Donde el
# detector de fase ya confirma un corrimiento, el perfil sabe algo que el
# champion no. Medido con 38 h de drift activo: +0.5 en la ventana del 12-sep y
# +0.7 en las ultimas horas, y -0.03 en los dias sin drift (0.55: +0.4, 0.70:
# +0.6, 0.85: +0.7, 1.00: +0.5; se toma 0.70 por prudencia, casi toda la ganancia).
PESO_PERFIL_CON_DESFASE = 0.70

# Revision 4 del drift (hallazgo #40): la demanda sigue una onda lenta de ~5-6 h
# con periodo y fase propios por estacion. La onda corta no la ve (busca P <= 6 h
# con 12 h de evidencia, y esas 12 h siguen mezclando la revision 3) y la
# persistencia+tendencia llega tarde a cada giro. Por estacion se ajusta
# a + b*sin(wx) + c*cos(wx) por minimos cuadrados SOLO con datos desde el inicio
# de la revision 4, con el periodo de la rejilla que mejor ajusta, y se suma al
# ultimo valor real el cambio que la curva predice hasta el target. Va mezclada
# 50/50 con la persistencia+tendencia. Backtest causal, 9 cortes (16:00-18:00Z
# virtual): persistencia+tendencia 79.96 -> mezcla 83.28, peor corte 76.6 -> 80.7.
INICIO_REVISION_4 = pd.Timestamp("2026-09-20T12:00:00Z")  # frontera publicada por el profesor
ONDA_LARGA_PERIODOS_H = np.arange(3.5, 8.01, 0.25)
ONDA_LARGA_MIN_PUNTOS = 16   # 4 h de la revision 4 antes de confiar en el ajuste
PESO_ONDA_LARGA = 0.5

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

    def periodo_corto(self, estacion: str, ancla_at: pd.Timestamp) -> tuple[int | None, float]:
        """(P, acierto) del periodo entre PERIODO_MIN y PERIODO_MAX pasos que
        mejor habria predicho las ultimas VENTANA_PERIODO lecturas copiando la
        de P pasos antes. Solo mira datos <= `ancla_at`."""
        serie = self._serie.get(str(estacion))
        if serie is None:
            return None, -1.0
        fin = self._paso(ancla_at) + 1
        ini = fin - VENTANA_PERIODO
        if ini - PERIODO_MAX < 0 or fin > serie.size:
            return None, -1.0
        reales = serie[ini:fin]
        mejor, acierto = None, -1.0
        for p in range(PERIODO_MIN, PERIODO_MAX + 1):
            previos = serie[ini - p:fin - p]
            ok = ~np.isnan(reales) & ~np.isnan(previos)
            if ok.sum() < VENTANA_PERIODO // 2 or reales[ok].sum() <= 0:
                continue
            a = 100 * (1 - np.abs(previos[ok] - reales[ok]).sum() / reales[ok].sum())
            if a > acierto:
                mejor, acierto = p, float(a)
        return mejor, acierto

    def prediccion_periodica(
        self, estacion: str, ancla_at: pd.Timestamp, target_at: pd.Timestamp,
    ) -> float | None:
        """Naive estacional con el periodo corto detectado, o `None` si no hay
        una onda confiable (quien llama sigue con el pipeline normal). Ver el
        comentario de PERIODO_* arriba."""
        p, acierto = self.periodo_corto(estacion, ancla_at)
        if p is None or acierto < UMBRAL_PERIODO:
            return None
        serie = self._serie[str(estacion)]
        tope = self._paso(ancla_at)
        # La ventana de 12 h tarda horas en notar que la onda se acabo (revision
        # 4: la onda siguio activa con 85% retrospectivo y acerto 0%). Se exige
        # que tambien haya acertado en las ultimas 2 h.
        recientes = np.arange(tope - 7, tope + 1)
        reales, previos = serie[recientes], serie[recientes - p]
        ok = ~np.isnan(reales) & ~np.isnan(previos)
        if ok.sum() >= 4 and reales[ok].sum() > 0:
            if 100 * (1 - np.abs(previos[ok] - reales[ok]).sum() / reales[ok].sum()) < UMBRAL_PERIODO:
                return None
        paso = self._paso(target_at)
        valores = [serie[paso - q * p] for q in range(1, CICLOS_PERIODO + 1)
                   if 0 <= paso - q * p <= tope]
        valores = [v for v in valores if not np.isnan(v)]
        return float(np.mean(valores)) if valores else None

    def persistencia_tendencia(
        self, estacion: str, ancla_at: pd.Timestamp, target_at: pd.Timestamp,
    ) -> float | None:
        """Ultimo valor real + la mitad de la pendiente de la ultima hora. En la
        revision 4 (tendencias lentas sin onda) midio 72.6 contra 64.0 de la
        mezcla champion+perfil y ~0 de la onda corta vieja."""
        serie = self._serie.get(str(estacion))
        if serie is None:
            return None
        tope = self._paso(ancla_at)
        validos = np.flatnonzero(~np.isnan(serie[max(0, tope - 7):tope + 1])) + max(0, tope - 7)
        if validos.size == 0:
            return None
        ultimo = validos[-1]
        ventana = validos[validos > ultimo - 4]
        pendiente = np.polyfit(ventana, serie[ventana], 1)[0] if ventana.size >= 2 else 0.0
        return float(max(0.0, serie[ultimo] + 0.5 * pendiente * (self._paso(target_at) - ultimo)))

    def onda_larga(
        self, estacion: str, ancla_at: pd.Timestamp, target_at: pd.Timestamp,
    ) -> float | None:
        """Ultimo valor real + el cambio que predice la sinusoide ajustada con
        los datos de la revision 4, o `None` si todavia no hay suficientes
        (ver INICIO_REVISION_4)."""
        serie = self._serie.get(str(estacion))
        if serie is None:
            return None
        tope = self._paso(ancla_at)
        inicio = max(0, self._paso(INICIO_REVISION_4))
        if tope < inicio:
            return None
        indices = np.arange(inicio, tope + 1)
        indices = indices[~np.isnan(serie[indices])]
        if indices.size < ONDA_LARGA_MIN_PUNTOS:
            return None
        valores = serie[indices]
        x = (indices - tope).astype(float)
        mejor = None
        for horas in ONDA_LARGA_PERIODOS_H:
            w = 2 * np.pi / (horas * PASOS_POR_HORA)
            matriz = np.column_stack([np.ones_like(x), np.sin(w * x), np.cos(w * x)])
            coef, *_ = np.linalg.lstsq(matriz, valores, rcond=None)
            residuo = float(((matriz @ coef - valores) ** 2).sum())
            if mejor is None or residuo < mejor[0]:
                mejor = (residuo, coef, w)
        _, coef, w = mejor

        def curva(z: float) -> float:
            return coef[0] + coef[1] * np.sin(w * z) + coef[2] * np.cos(w * z)

        z_target = float(self._paso(target_at) - tope)
        return float(max(0.0, valores[-1] + curva(z_target) - curva(x[-1])))

    def extrapolacion_extrema(
        self, estacion: str, ancla_at: pd.Timestamp, target_at: pd.Timestamp,
    ) -> float | None:
        """Extrapolacion lineal directa sobre la serie cruda (sin perfil
        historico), solo para el caso EXTREMO de alza/cierre. `None` si el
        factor no esta en zona extrema (quien llama se queda con la mezcla
        normal champion+perfil). Ver el comentario de UMBRAL_EXTRAPOLACION_*
        arriba para la evidencia.
        """
        serie = self._serie.get(str(estacion))
        if serie is None:
            return None
        try:
            desfase = self.desfase(estacion, ancla_at)
        except Exception:
            desfase = 0
        corto = self.factor_de_nivel(estacion, ancla_at, desfase=desfase)
        extremo_alza = corto > UMBRAL_EXTRAPOLACION_ALZA
        extremo_cierre = corto < UMBRAL_EXTRAPOLACION_CIERRE
        if not (extremo_alza or extremo_cierre):
            return None

        fin = self._paso(ancla_at) + 1
        inicio = max(0, fin - PASOS_EXTRAPOLACION)
        ventana = serie[inicio:fin]
        validos = ~np.isnan(ventana)
        if validos.sum() < 2:
            return None
        y = ventana[validos]
        x = np.arange(len(ventana))[validos]
        pendiente, intercepto = np.polyfit(x, y, 1)

        pasos_objetivo = self._paso(target_at) - (fin - 1)
        valor = intercepto + pendiente * (len(ventana) - 1 + pasos_objetivo)
        return float(max(0.0, valor))


def mezclar(valor_champion: float, valor_perfil: float | None, peso_perfil: float = MEZCLA_PERFIL) -> float:
    """Combina las dos opiniones. Si el perfil no tiene una, manda el champion.

    Los dos modelos fallan distinto: el champion se equivoca cuando el regimen
    cambia, el perfil cuando el dia es atipico respecto a los anteriores.
    Promediarlos sale mejor que cualquiera de los dos por separado.
    """
    if valor_perfil is None or not np.isfinite(valor_perfil):
        return valor_champion
    return (1.0 - peso_perfil) * valor_champion + peso_perfil * valor_perfil
