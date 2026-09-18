# Pulso TransMi — Proyecto del equipo

Implementación del reto MLOps **Pulso TransMi** (Universidad Externado,
Departamento de Matemáticas). Pronostica demanda de pasajeros para 12
estaciones de TransMilenio cada 15 minutos, con un pipeline pensado para
operar de forma continua, no solo entrenar un modelo una vez.

Este repo parte del [starter kit oficial](https://github.com/uexternadojz/pulso-transmi-sdk)
del profesor (remoto `upstream`), que provee el SDK cliente
(`src/pulso_transmi/`) para consumir la API pública. Todo lo demás —
análisis, esquema de datos, migración— es trabajo propio del equipo.

## Estado actual

Fases 1, 2 y 3 de la guía metodológica ("Comprender", "Construir la
memoria" y "Experimentar"):

- ✅ SDK instalado, histórico descargado (12 estaciones, 45 días, 51.840
  observaciones).
- ✅ Análisis exploratorio completo: calidad de datos, estacionalidad,
  correlaciones, feature engineering, selección de features y
  validación cruzada temporal — ver [`eda/EDA_REPORT.md`](eda/EDA_REPORT.md).
- ✅ Base de datos en Supabase con el histórico migrado — ver
  [`docs/entity-relation.md`](docs/entity-relation.md) para el esquema,
  el diagrama entidad-relación y las políticas de seguridad (RLS).
- ✅ Modelo champion entrenado, comparado contra 7 alternativas y
  registrado en `model_versions` (artefacto en Supabase Storage) — ver
  [`src/train.py`](src/train.py) y la sección "Modelo" abajo.
- ✅ Collector incremental idempotente (`src/ingest.py`) — probado en
  vivo contra el stream real (vacío, reloj en `waiting`); deja evidencia
  en `collector_runs` en cada corrida, incluso sin novedades.
- ⬜ Inferencia periódica vía GitHub Actions y submissions (bloqueado:
  requiere que el profesor active el reloj y habilite la API key).
- ⬜ Monitoreo de drift y reentrenamiento.
- ⬜ Dashboard (bono).

## Collector incremental

`src/ingest.py` sigue el patrón que exige la guía metodológica
("Automatizaciones esperadas — Collector"):

- Retoma el cursor del **último `collector_runs` con `status='success'`**
  guardado en Supabase — nunca lo fabrica a partir de la hora local.
- Pagina `/v1/stream/observations` hasta agotar `next_cursor`.
- Para `context_readings` (que no tiene cursor propio en la API), usa el
  máximo `observed_at` ya guardado como filtro `start`.
- El upsert usa las mismas llaves únicas del esquema → correrlo N veces
  seguidas nunca duplica filas (verificado: 4 corridas seguidas, mismo
  conteo exacto de filas en `observations`/`context_readings`).
- Deja evidencia en `collector_runs` **siempre**, incluso cuando no hay
  novedades (`status='no_new_data'`) — así se ve en la bitácora real de
  hoy: una corrida con un bug de comparación de fechas (detectado y
  corregido, con test de regresión en `tests/test_ingest.py`), y las
  siguientes ya limpias.
- Regla de seguridad ante ambigüedad de la API: si la API deja de
  paginar (`next_cursor: null`) sin dejar claro cómo continuar después,
  el collector conserva el cursor con el que pidió esa última página en
  vez de arriesgarse a perderlo — el peor caso es repetir una página
  (inofensivo, gracias al upsert idempotente), nunca perder el progreso.

```bash
python src/ingest.py
```

## Modelo

`src/train.py` entrena y compara 8 candidatos con la misma validación
cruzada temporal (5 folds, `TimeSeriesSplit`, nunca aleatoria):

| Candidato | Features | Accuracy (CV) |
|---|---:|---:|
| **`xgboost_station`** (champion vigente) | 27 (15 + one-hot de estación) | **86.77** |
| `xgboost_full` (champion anterior → `historical`) | 15 | 86.61 |
| `xgboost_tuned2` (más árboles, learning rate más bajo) | 15 | 86.59 |
| `extra_trees_full` | 15 | 86.56 |
| `rf_tuned` (RF, 500 árboles, más profundo) | 15 | 86.48 |
| `rf_full` (primer champion → `historical`) | 15 | 86.34 |
| `gbr_full` | 15 | 86.16 |
| `rf_no_weekly_lag` (prueba de fragilidad) | 12 (sin `lag_672`/`roll_mean_96`/`roll_std_96`) | 85.48 |

**Hallazgo de esta ronda de refinamiento:** afinar hiperparámetros de
XGBoost (`xgboost_tuned2`: más árboles, learning rate más bajo) casi no
movió la aguja (86.59 vs 86.61, incluso un poco peor). Lo que sí ayudó
fue agregar una **señal nueva**: la identidad de la estación (one-hot).
El EDA ya había mostrado que la demanda promedio varía más de 3x entre
estaciones, pero ningún candidato anterior le decía eso al modelo
explícitamente — solo lo inferían indirectamente vía los lags. Lección:
más cómputo/arboles no sustituye una feature que de verdad falta.

`rf_no_weekly_lag` existe a propósito para medir la fragilidad del
modelo: ¿qué tan mal quedaríamos si `lag_672` (demanda de hace una
semana) dejara de ser confiable por drift? La brecha real es de solo
**0.87 puntos** — mucho menor de lo que la altísima importancia de esa
feature en el EDA (91.8%) hacía temer. Esto corrige una preocupación
que teníamos: el modelo no depende tan ciegamente de una sola señal
como parecía; las demás features (lags cortos, hora, contexto) cubren
razonablemente si esa señal falla. Aun así, no hay garantía de que esto
se sostenga ante un drift real de la competencia — solo mide robustez
ante la *ausencia* de esa feature, no ante un cambio en su significado.

**Regla de promoción real, no solo "el número más alto":** `train.py`
consulta el champion vigente en Supabase antes de decidir. Un candidato
nuevo solo se promueve si supera esa métrica; si no, se registra igual
como `candidate` (evidencia del experimento) sin tocar el champion. Un
índice único parcial en Postgres (`one_champion_only`) impide a nivel
de base de datos que existan dos champions a la vez. Ya pasó dos veces
en la práctica: `xgboost_full` reemplazó a `rf_full` (86.61 > 86.34), y
luego `xgboost_station` reemplazó a `xgboost_full` (86.77 > 86.61).
Ambos anteriores quedaron marcados `historical`, no borrados — se
conservan como evidencia de cada experimento.

El modelo ganador se guarda en `artifacts/` (ignorado por git) y se
sube a Supabase Storage (`model-artifacts` bucket) para que sea
reproducible desde cualquier máquina, no solo la que lo entrenó.
`src/predict.py` demuestra que el artefacto se puede volver a cargar
desde Storage y producir predicciones (smoke test, no conectado a un
ciclo real todavía).

## Arquitectura

```
API Pulso TransMi  →  collector  →  Supabase (Postgres)  →  experimentos / modelo
                                          ↑                        ↓
                                   GitHub Actions  ←───────  predicciones + submission
```

- **API Pulso TransMi**: fuente de verdad compartida (estaciones,
  observaciones, contexto, ciclos).
- **Supabase**: memoria del equipo. 11 tablas (`supabase/schema.sql`):
  catálogo de estaciones/observaciones/contexto, bitácora del collector,
  ciclos, versiones de modelo, predicciones, submissions, evaluaciones,
  métricas y señales de drift. RLS activo: lectura pública, escritura
  solo con `service_role` key.
- **GitHub Actions**: pendiente — correrá el collector, el entrenamiento
  y la inferencia periódica.

## Decisiones tomadas y por qué

- **Split siempre temporal, nunca aleatorio**: la validación cruzada del
  EDA usa `TimeSeriesSplit` (5 folds expandiendo la ventana de
  entrenamiento). Mezclar aleatoriamente futuro y pasado da métricas
  optimistas que no representan la competencia.
- **`lag_672` (demanda de hace exactamente 1 semana) es la feature más
  fuerte** (91.8% de importancia en el Random Forest de selección de
  features) — hay estacionalidad semanal muy marcada en el histórico.
  Esto encendió una alarma inicial sobre fragilidad ante drift, pero al
  entrenar sin esa feature (ver sección "Modelo") la caída real es de
  solo 0.87 puntos — menor de lo temido, aunque sigue sin ser una
  garantía de que el modelo aguante un drift real en la competencia. No
  se debe confundir el accuracy en el histórico estático con un
  problema resuelto.
- **RLS con lectura pública**: el esquema no contiene PII (es demanda
  sintética de transporte público), así que se optó por lectura abierta
  para simplificar el futuro dashboard, y escritura restringida a la
  `service_role` key para que el collector y las migraciones no dependan
  de una key pública.
- **Migración de datos vía PostgREST (`upsert`)** en vez de SQL con
  arrays gigantes: más simple de mantener y ya es idempotente por
  diseño (mismas llaves únicas que usará el futuro collector
  incremental).

## Cómo reproducir

Requiere Python 3.11+.

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e '.[ml]'
cp .env.example .env  # completar SUPABASE_URL / SUPABASE_KEY

python examples/01_download.py     # descarga el historico a data/
python examples/02_naive_baseline.py

python eda/eda.py                  # calidad de datos, graficos, feature
                                    # engineering, seleccion de features,
                                    # validacion cruzada temporal
```

Correr el collector incremental (requiere `SUPABASE_SERVICE_ROLE_KEY`):

```bash
python src/ingest.py
```

Migrar (o re-migrar; es idempotente) el histórico a Supabase requiere la
`service_role` key en `SUPABASE_SERVICE_ROLE_KEY` (ver `.env.example`):

```bash
python supabase/migrate_via_rest.py
```

Entrenar, comparar candidatos, y registrar el champion (requiere
`SUPABASE_SERVICE_ROLE_KEY`, sube el artefacto a Supabase Storage):

```bash
python src/train.py
```

Probar que el champion registrado se puede volver a cargar y predecir:

```bash
python src/predict.py
```

## Estructura del repo

```
src/pulso_transmi/     SDK cliente (heredado del starter kit)
src/features.py         Feature engineering compartido (EDA + entrenamiento)
src/ingest.py           Collector incremental idempotente
src/train.py            Entrena, compara candidatos y registra el champion
src/predict.py          Recarga el champion desde Storage y predice (smoke test)
examples/               Scripts de ejemplo del starter kit
eda/                    Analisis exploratorio, graficos, reportes
supabase/               Esquema de la base de datos y script de migracion
docs/                   Documentacion: API, guia del proyecto, entidad-relacion
artifacts/              Modelos entrenados localmente (ignorado por git)
tests/                  Tests del SDK (heredados del starter kit) + tests del collector
```

## Fuentes

- Starter kit y API: https://github.com/uexternadojz/pulso-transmi-sdk
- Documentación interactiva de la API: https://pulso-transmi.72-60-245-2.sslip.io/docs
- Guía metodológica del curso (MLOps · Ciencia de Datos, Universidad
  Externado de Colombia, Depto. de Matemáticas, docente Julián Zuluaga)
