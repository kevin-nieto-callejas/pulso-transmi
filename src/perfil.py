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
LIMITES_FACTOR = (0.5, 1.6)  # un factor fuera de esto es un dato roto, no un drift
DIAS_DE_HISTORIA = 28      # con vida media 14 d, mas atras pesa <0.25
MEZCLA_PERFIL = 0.4        # peso del perfil frente al champion


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
        self._n_dias = int(pasos.max()) // PASOS_POR_DIA + 1
        if self._n_dias < 3:
            raise ValueError(f"El perfil necesita al menos 3 dias de historia y hay {self._n_dias}")

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

    def factor_de_nivel(self, estacion: str, ancla_at: pd.Timestamp) -> float:
        """Cuanto se desvia la ultima hora real respecto a lo que el perfil
        esperaba. 1.0 = la estacion va justo en su perfil."""
        serie, perfil = self._serie.get(str(estacion)), self._perfil.get(str(estacion))
        if serie is None:
            return 1.0
        fin = self._paso(ancla_at) + 1
        inicio = max(0, fin - self._ventana_nivel)
        reales, esperados = serie[inicio:fin], perfil[inicio:fin]
        validos = ~np.isnan(reales) & ~np.isnan(esperados)
        # Con menos de media ventana, o con un perfil que suma cero, el factor
        # seria ruido amplificado: mejor no corregir nada.
        if validos.sum() < max(2, self._ventana_nivel // 2) or esperados[validos].sum() <= 0:
            return 1.0
        return float(np.clip(reales[validos].sum() / esperados[validos].sum(), *LIMITES_FACTOR))

    def predecir(self, estacion: str, target_at: pd.Timestamp, ancla_at: pd.Timestamp) -> float | None:
        """Demanda esperada en `target_at`, o None si no hay perfil para esa
        franja (estacion nueva, hueco del collector). None significa "no tengo
        opinion": quien llama se queda con el champion solo."""
        perfil = self._perfil.get(str(estacion))
        if perfil is None:
            return None
        paso = self._paso(target_at)
        if not 0 <= paso < perfil.size or np.isnan(perfil[paso]):
            return None
        return float(perfil[paso] * self.factor_de_nivel(estacion, ancla_at))


def mezclar(valor_champion: float, valor_perfil: float | None, peso_perfil: float = MEZCLA_PERFIL) -> float:
    """Combina las dos opiniones. Si el perfil no tiene una, manda el champion.

    Los dos modelos fallan distinto: el champion se equivoca cuando el regimen
    cambia, el perfil cuando el dia es atipico respecto a los anteriores.
    Promediarlos sale mejor que cualquiera de los dos por separado.
    """
    if valor_perfil is None or not np.isfinite(valor_perfil):
        return valor_champion
    return (1.0 - peso_perfil) * valor_champion + peso_perfil * valor_perfil
