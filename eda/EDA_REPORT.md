# EDA — Pulso TransMi

Generado por `eda.py` a partir del histórico descargado en `../data/`
(45 días, 12 estaciones, 15 min de frecuencia, 51.840 observaciones).

## Variable objetivo

`demand`: número de pasajeros observados en una estación de TransMilenio
durante un periodo de 15 minutos. Es un conteo no negativo, entero, con
fuerte componente temporal (hora del día, día de la semana) y espacial
(la estación).

## Diccionario de datos

| Tabla | Columna | Tipo | Descripción |
|---|---|---|---|
| stations.csv | station_id | string (con ceros a la izquierda) | Identificador oficial de la estación |
| stations.csv | station_name | string | Nombre de la estación |
| stations.csv | corridor | string | Troncal/corredor al que pertenece |
| stations.csv | latitude, longitude | float | Coordenadas geográficas |
| observations.csv | observed_at | datetime (America/Bogota, offset -05:00) | Timestamp del periodo de 15 min |
| observations.csv | station_id | string | FK a stations |
| observations.csv | demand | int | Pasajeros del periodo (variable objetivo) |
| context.csv | observed_at | datetime | Timestamp del periodo de 15 min (compartido por todas las estaciones) |
| context.csv | rain_mm, rain_forecast | float | Lluvia observada / pronosticada |
| context.csv | temperature_c, temperature_forecast | float | Temperatura observada / pronosticada |
| context.csv | event_intensity | float | Intensidad de eventos (0 = sin evento) |

`context.csv` no está desagregado por estación: aplica igual a las 12 al
unir por `observed_at`.

## Calidad de datos (`reports/data_quality.json`)

- **0 duplicados** en `(station_id, observed_at)`.
- **0 nulos** en ninguna columna clave.
- **0 valores negativos** de `demand`.
- **Cobertura perfecta**: las 12 estaciones tienen exactamente 4.320 filas
  (45 días × 96 periodos/día), sin timestamps faltantes.
- Este dataset histórico está limpio por construcción (es sintético); el
  collector incremental de la fase de operación sí deberá seguir
  validando esto en cada corte, porque el stream en vivo puede llegar con
  huecos o duplicados de red.

## Hipótesis exploradas

1. **Estacionalidad diaria fuerte**: el heatmap hora×día
   (`figures/02_heatmap_hora_dia.png`) muestra picos claros en horas pico
   (mañana y tarde) y valles en la madrugada, consistente en todos los
   días hábiles.
2. **Estacionalidad semanal**: la serie diaria total
   (`figures/05_demanda_diaria.png`) y la importancia de `lag_672`
   (demanda de hace exactamente 7 días, ya que 672 × 15 min = 7 días) —
   con **91.8% de importancia** en el Random Forest — confirman que el
   patrón semanal es la señal más fuerte del dataset.
3. **Heterogeneidad entre estaciones**: la demanda promedio por estación
   (`figures/03_demanda_promedio_por_estacion.png`) varía en más de 3x
   entre la estación más y menos concurrida; el mapa geográfico
   (`figures/04_mapa_geografico_demanda.png`) sugiere que los corredores
   troncales con más transferencias concentran más demanda.
4. **Contexto (clima/eventos) con efecto marginal**: en la matriz de
   correlación (`figures/06_matriz_correlacion.png`) y en la importancia
   de features, `rain_mm` y `temperature_c` aportan pero muy por debajo
   de los lags autorregresivos — el clima ajusta la demanda, no la
   determina.

## Feature engineering

Construido en `build_feature_frame()`:

- Cíclicas de tiempo: `hour_sin/cos` (ciclo de 96 periodos = 1 día),
  `dow_sin/cos` (ciclo semanal), `is_weekend`.
- Autorregresivas por estación: `lag_1`, `lag_4` (1h), `lag_96` (1 día),
  `lag_672` (1 semana), `roll_mean_4`, `roll_mean_96`, `roll_std_96`.
- Contexto unido por `observed_at`: lluvia, temperatura, eventos.

## Selección de features

Random Forest sobre todas las features candidatas
(`figures/07_feature_importance.png`,
`reports/feature_importance.csv`). Top 8 usadas para el modelo de
validación cruzada:

| Feature | Importancia |
|---|---:|
| lag_672 | 0.918 |
| lag_1 | 0.048 |
| lag_96 | 0.011 |
| roll_mean_96 | 0.005 |
| roll_mean_4 | 0.005 |
| hour_cos | 0.003 |
| temperature_c | 0.002 |
| rain_mm | 0.002 |

## Validación cruzada temporal

`TimeSeriesSplit` de 5 folds (nunca aleatorio, respetando el orden
temporal como exige la guía metodológica). Resultados en
`reports/cross_validation_results.csv`:

| Fold | Train rows | Test rows | MAE | Accuracy |
|---|---:|---:|---:|---:|
| 1 | 7,296 | 7,296 | 49.44 | 86.02 |
| 2 | 14,592 | 7,296 | 48.31 | 86.21 |
| 3 | 21,888 | 7,296 | 48.02 | 86.13 |
| 4 | 29,184 | 7,296 | 47.23 | 86.24 |
| 5 | 36,480 | 7,296 | 49.06 | 86.43 |

**Accuracy promedio: 86.21%**, comparado contra el baseline ingenuo de
`examples/02_naive_baseline.py` (lag de 24h, accuracy 77.89%). El modelo
con features autorregresivas + estacionalidad + contexto mejora
~8.3 puntos sobre el baseline naive.

## Próximo paso sugerido

Con `lag_672` dominando tan claramente, vale la pena probar un segundo
baseline explícito "repetir el valor de hace una semana" como
comparación directa, y evaluar si un modelo más simple (ej. regresión
lineal solo con lags) generaliza igual de bien con menos riesgo de
overfitting que el Random Forest.
