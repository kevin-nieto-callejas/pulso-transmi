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

Las cinco fases de la guía están implementadas y operando. Lo único que
falta es evidencia con ciclos reales, que depende de que el docente active
la competencia: el reloj sigue en `waiting`.

**Estado en una línea:** champion `ensamble_extendido` con 86.61 de
accuracy, una submission aceptada en la ronda de práctica, collector e
inferencia entregando solos, 36 tests en verde y los cinco bonos cubiertos.

- ✅ SDK instalado, histórico descargado (12 estaciones, 45 días, 51.840
  observaciones).
- 🚨 **Runbook del día de competencia** — qué mirar, qué es normal, y qué
  hacer cuando algo falla: [`docs/RUNBOOK.md`](docs/RUNBOOK.md).
- 📄 **Informe final** — qué cambió, qué funcionó, qué no, y qué haríamos
  después: [`docs/INFORME_FINAL.md`](docs/INFORME_FINAL.md).
- 🔎 **Hallazgos** — lo que aprendimos midiendo, agrupado por tema:
  [`docs/HALLAZGOS.md`](docs/HALLAZGOS.md).
- 📓 **Bitácora del proyecto** — qué se construyó, qué se rompió, cómo se
  detectó y qué se decidió: [`docs/BITACORA.md`](docs/BITACORA.md).
- 📤 **Cómo enviar predicciones** — el contrato exacto, las reglas que no se
  pueden romper, los códigos de error y los problemas reales que ya nos
  pasaron: [`.claude/skills/enviar-predicciones/`](.claude/skills/enviar-predicciones/SKILL.md).
- ✅ Análisis exploratorio completo: calidad de datos, estacionalidad,
  correlaciones, feature engineering, selección de features y
  validación cruzada temporal — ver [`eda/EDA_REPORT.md`](eda/EDA_REPORT.md).
  El EDA es deliberadamente de un solo horizonte (+15 min, "¿qué tan
  predecible es esto en principio?"); el modelo que realmente opera
  (`src/train.py`) usa el esquema multi-horizonte correcto — ver
  "Modelo" más abajo.
- ✅ Base de datos en Supabase con el histórico migrado — ver
  [`docs/entity-relation.md`](docs/entity-relation.md) para el esquema,
  el diagrama entidad-relación y las políticas de seguridad (RLS).
- ✅ Modelo champion multi-horizonte (+15/+30/+45/+60 min), comparado
  contra 11 alternativas y registrado en `model_versions` (artefacto en
  Supabase Storage) — ver [`src/train.py`](src/train.py) y la sección
  "Modelo" abajo.
- ✅ Collector incremental idempotente (`src/ingest.py`), **automatizado
  vía GitHub Actions** (`.github/workflows/collector.yml`) — probado en vivo contra el stream real (vacío, reloj
  en `waiting`); deja evidencia en `collector_runs` en cada corrida,
  incluso sin novedades.
- ✅ Inferencia y submission (`src/infer.py` +
  `.github/workflows/inference.yml`): consulta el ciclo vigente, arma el
  batch y lo firma con una llave de idempotencia estable. **Entrega real
  aceptada** en la ronda de práctica del 18/09
  (`sub_b64352e18de5486ead9478e66d57c523`, 12/12, `is_official: true`).
  Termina sin error mientras no haya ciclo abierto.
- ✅ Monitoreo, drift y decisión de reentrenamiento (`src/evaluate.py`):
  une predicción con realidad, calcula accuracy por estación y ciclo,
  vigila las tres señales de la guía y decide mantener/investigar/
  reentrenar con umbrales justificados. Corre tras cada recolección.
- ✅ Estrategia de rollback (`src/rollback.py`, bono): vuelve a un champion
  anterior verificando primero que cargue, y deja la decisión registrada.
- ✅ Dashboard (bono) desplegado en Vercel:
  **https://pulso-transmi-one.vercel.app** — estado en vivo leído con la
  llave pública de solo lectura ([`dashboard/`](dashboard/README.md)).
- ✅ MLflow (bono): `src/sweep.py` explora al azar las tres familias de
  boosting y registra cada intento; `src/revalidar.py` vuelve a medir los
  mejores con el protocolo oficial antes de considerarlos promovibles.
- ✅ Pruebas automatizadas (bono): 36 tests en CI, más una batería de
  esfuerzo (`src/stress_test.py`) y un simulacro de ciclo completo
  (`src/simulate_cycle.py`).
- ⬜ Evaluación de accuracy con ciclos reales: el código está completo y
  ejercitado de punta a punta, pero `prediction_evaluations` y
  `cycle_metrics` siguen vacías porque la ronda de práctica pidió un
  instante cuyo valor real nunca se publicó. Depende del docente.

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
- Automatizado vía `.github/workflows/collector.yml`, cron en los minutos
  `8,23,38,53` (UTC). La guía pide cada 30 minutos; se dispara cada 15 y
  fuera de `:00`/`:30` porque GitHub descarta corridas programadas bajo
  carga (ver "Hallazgo operacional" más abajo), así la cadencia *efectiva*
  se acerca a los 30 minutos pedidos. Perder una corrida no pierde datos
  —el cursor es incremental y el upsert idempotente, la siguiente corrida
  recupera el atraso— pero sí deja huecos en la bitácora. También se puede
  disparar a mano desde la pestaña Actions (`workflow_dispatch`).

```bash
python src/ingest.py
```

## Inferencia y submission (Fase 4)

`src/infer.py` sigue "Cómo se entrega una predicción" y "Automatizaciones
esperadas → Inferencia y submission" de la guía:

1. Consulta `/v1/clock` y `/v1/forecast-cycles/current`. Si no hay ciclo
   abierto, **termina sin error** — ese es el comportamiento esperado
   ahora mismo (reloj en `waiting`) y también el comportamiento normal
   entre ciclos una vez activa la competencia.
2. Si hay ciclo abierto, usa el `data_cutoff`, los targets
   (`station_id` + `target_at`) y el `closes_at` que devuelve la API —
   nunca los calcula ni los asume por horario local.
3. Carga el champion vigente desde Supabase Storage y reconstruye, por
   estación, las features tal como se veían en `data_cutoff` (histórico
   de hasta 8 días atrás, para cubrir `lag_672`).
4. Arma el batch completo (hasta 100 predicciones, normalmente 48 =
   12 estaciones × 4 horizontes) y lo firma con una llave de idempotencia
   estable — el mismo ciclo y el mismo modelo generan siempre la misma
   llave, así que un reintento nunca crea una segunda entrega.
5. Envía a `POST /v1/submissions` con la API key en el header
   `Authorization`. Si esa key no está configurada, el script arma y valida
   el batch igual, lo imprime, y se detiene explícitamente en vez de fallar.

Probado con `httpx.MockTransport` (`tests/test_infer.py`, 13 tests): el
camino "sin ciclo abierto", el armado del batch (horizonte correcto por
target, valores recortados a `[0, 100000]` como exige el contrato,
estación desconocida levanta error), la llave de idempotencia estable, la
detección de "este ciclo ya fue entregado" y la forma exacta del payload
de `SubmissionInput`. En total el repo corre 36 tests en CI.

Automatizado vía `.github/workflows/inference.yml`. Los minutos son una
recomendación operacional: la decisión real siempre la toma `src/infer.py`
consultando el estado de la API, nunca el cron por sí solo.

### Hallazgo operacional: GitHub descarta corridas programadas

La guía advierte que "los workflows programados pueden presentar
retrasos". Medido en este repo el 18/09, el problema resultó ser peor que
un retraso: de ~28 corridas programadas esperadas en 7 horas, **solo se
ejecutaron 2**. GitHub no encola las corridas atrasadas — las descarta
bajo carga, y los minutos `:00` y `:30` (donde estaba el cron original del
collector) son los más congestionados de la plataforma. No es un problema
de cuota: el repo es público, con minutos ilimitados.

Durante la competencia eso sería grave: perder la corrida de un ciclo
significa perder su ventana de 25 minutos, o sea 48 predicciones en cero.
Es exactamente la tercera señal que describe la guía ("falla operacional:
el pipeline no produjo o envió resultados — corregir la operación antes de
culpar al modelo"). Mitigación en dos capas:

1. **Más intentos, en minutos menos congestionados.** Inferencia cada 10
   min (`3,13,23,33,43,53`), collector cada 15 (`8,23,38,53`) — nunca en
   `:00` ni `:30`.
2. **Cada corrida vigila 8 minutos más** (`PULSO_POLL_MINUTES`), revisando
   la API cada 2 minutos, para que una sola corrida que sí arranque cubra
   buena parte de la ventana aunque las demás se descarten. Un fallo
   pasajero de red dentro de la ventana se reintenta en vez de abortar la
   corrida completa.

Reenviar no duplica ni gasta intentos: la llave de idempotencia es estable
por (ciclo, modelo) y, antes de rearmar el batch, `src/infer.py` consulta
en Supabase si ese ciclo ya fue entregado con ese mismo modelo.

```bash
python src/infer.py
```

## Modelo

`src/train.py` entrena y compara 12 candidatos con la misma validación
cruzada temporal (5 folds, `TimeSeriesSplit`, nunca aleatoria). Evidencia
completa en [`eda/reports/candidate_comparison.csv`](eda/reports/candidate_comparison.csv):

| Candidato | Features | Accuracy (CV) |
|---|---:|---:|
| **`ensamble_extendido`** (champion vigente) | 49 + one-hot de estación | **86.61** |
| `xgboost_extendido` | 49 | 86.47 |
| `lgbm_extendido` | 49 | 86.44 |
| `catboost_extendido` | 49 | 86.41 |
| `xgboost_station` (champion anterior → `historical`) | 28 | 85.22 |
| `rf_tuned` (RF, 500 árboles, más profundo) | 16 | 84.55 |
| `extra_trees_full` | 16 | 84.37 |
| `xgboost_full` | 16 | 84.28 |
| `xgboost_tuned2` (más árboles, learning rate más bajo) | 16 | 83.60 |
| `rf_full` | 16 | 82.92 |
| `gbr_full` | 16 | 80.82 |
| `rf_no_weekly_lag` (prueba de fragilidad) | 13 (sin `lag_672`/`roll_mean_96`/`roll_std_96`) | 80.81 |
| *baseline ingenuo (repetir el valor de hace 24 h)* | *0* | *77.89* |

El champion es un ensamble de LightGBM + XGBoost + CatBoost: tres familias
de boosting se equivocan en sitios distintos, así que promediarlas le gana a
cualquiera por separado (86.61 contra 86.47 del mejor individual).

Desglose del champion por horizonte (misma validación, filtrando cada
subconjunto):

| Horizonte | Accuracy |
|---|---:|
| +15 min | 86.95 |
| +30 min | 86.52 |
| +45 min | 86.23 |
| +60 min | 85.99 |

Además de esos 12 candidatos elegidos a mano, `src/sweep.py` explora cientos
de combinaciones al azar con validación más barata y las registra en MLflow
([`eda/reports/sweep_results.csv`](eda/reports/sweep_results.csv)). Esos
números **no son comparables** con los de arriba —se miden con 3 cortes en
vez de 5—, por eso `src/revalidar.py` vuelve a medir los mejores con el
protocolo oficial antes de considerarlos promovibles.

**Corrección importante de esta ronda — horizonte directo, no
"nowcasting":** la primera versión del modelo (86.77 de accuracy, la que
aparecía antes en este README) tenía un problema real: predecía la
demanda de **su propia fila** usando `lag_1` = el período anterior. Eso
solo generaliza a +15 min. Para +30/+45/+60 min, ese mismo truco exigiría
conocer demanda que todavía no existe en `data_cutoff` (fuga de futuro al
revés). La API pide los 4 horizontes en cada ciclo, así que ese modelo no
podía usarse para submissions reales sin hacer trampa.

La corrección: cada fila ancla (lo que se sabe en un momento dado) se
apila 4 veces, una por horizonte, agregando `horizon_minutes` como
feature y usando como target la demanda real desplazada ese horizonte
hacia adelante — siempre con lags anclados en el momento de predicción,
nunca en el futuro. Es la estrategia de "horizonte directo" (un solo
modelo, el horizonte como feature) en vez de recursiva (encadenar 4
predicciones de un paso, que acumula error). El accuracy bajó de 86.77 a
85.22 — **eso no es una regresión**: predecir 4 horizontes de verdad es
un problema más difícil que predecir solo el siguiente paso, y el número
anterior nunca hubiera sido alcanzable en la competencia real. El
desglose por horizonte (arriba) muestra exactamente el patrón esperado:
más lejos en el futuro, más difícil — 86.23 en +15 min bajando a 84.59 en
+60 min.

**Hallazgo que se mantiene:** agregar la identidad de la estación
(one-hot) sigue siendo la mejora más grande sobre el resto de candidatos
(85.22 vs 84.28 de `xgboost_full`), igual que en la ronda anterior. El
EDA ya había mostrado que la demanda promedio varía más de 3x entre
estaciones; afinar hiperparámetros de XGBoost sin esa señal
(`xgboost_tuned2`) sigue moviendo la aguja mucho menos.

`rf_no_weekly_lag` existe a propósito para medir la fragilidad del
modelo: ¿qué tan mal quedaríamos si `lag_672` (demanda de hace una
semana) dejara de ser confiable por drift? La brecha real es de **2.11
puntos** (antes 0.87, con el esquema de un solo horizonte) — mayor que
antes, algo esperable: en horizontes largos el modelo depende un poco
más de la estacionalidad semanal porque los lags cortos (`lag_1`,
`lag_4`) ya no son tan informativos sobre el futuro lejano. Sigue sin ser
una brecha catastrófica, pero es una señal real a vigilar si el patrón
semanal cambia durante la competencia.

**Regla de promoción real, no solo "el número más alto":** `train.py`
consulta el champion vigente en Supabase antes de decidir. Un candidato
nuevo solo se promueve si supera esa métrica; si no, se registra igual
como `candidate` (evidencia del experimento) sin tocar el champion. Un
índice único parcial en Postgres (`one_champion_only`) impide a nivel
de base de datos que existan dos champions a la vez.

Esta corrección de esquema es también un caso real de esa regla puesta a
prueba: como el champion anterior (86.77) y el nuevo (85.22) usan
definiciones de métrica distintas (un horizonte vs cuatro apilados),
`train.py` detecta la incompatibilidad revisando si `horizon_minutes`
está en la lista de features del champion vigente, y si no lo está,
reemplaza sin comparar los dos números directamente — la comparación
numérica ciega hubiera sido incorrecta (habría "reprobado" un modelo
mejor solo porque resuelve un problema más difícil). Antes de esta
corrección, sí hubo dos promociones válidas y comparables entre sí:
`xgboost_full` reemplazó a `rf_full` (86.61 > 86.34), y `xgboost_station`
reemplazó a `xgboost_full` (86.77 > 86.61). Todos los champions
anteriores quedaron marcados `historical`, no borrados — se conservan
como evidencia de cada experimento.

El modelo ganador se guarda en `artifacts/` (ignorado por git) y se
sube a Supabase Storage (`model-artifacts` bucket) para que sea
reproducible desde cualquier máquina, no solo la que lo entrenó.
`src/predict.py` hace una prueba honesta de pronóstico (no solo "el
archivo carga"): toma un ancla 60 minutos antes del último dato conocido
por estación, predice los 4 horizontes y compara contra el valor real ya
conocido en el histórico. Última corrida: 88.03% de accuracy agregada
sobre esa muestra puntual (referencial — la cifra oficial del modelo es
la de la validación cruzada en `train.py`, no esta).

## Arquitectura

```
API Pulso TransMi  →  collector  →  Supabase (Postgres)  →  experimentos / modelo
      ↑                  (cron 30min)                            ↓
      └──────────────  GitHub Actions  ←───────────────  predicciones + submission
                        (cron min 7,37)
```

- **API Pulso TransMi**: fuente de verdad compartida (estaciones,
  observaciones, contexto, ciclos).
- **Supabase**: memoria del equipo. 11 tablas (`supabase/schema.sql`):
  catálogo de estaciones/observaciones/contexto, bitácora del collector,
  ciclos, versiones de modelo, predicciones, submissions, evaluaciones,
  métricas y señales de drift. RLS activo: lectura pública, escritura
  solo con `service_role` key.
- **GitHub Actions**: dos workflows programados, ambos corriendo ya
  (`.github/workflows/collector.yml` y `.github/workflows/inference.yml`):
  el collector deja evidencia cada 30 minutos aunque no haya novedades; el
  de inferencia consulta el ciclo vigente y termina sin error mientras no
  haya uno abierto. Falta conectar el entrenamiento (hoy es manual, por
  decisión: la guía dice explícitamente "no es obligatorio reentrenar
  cada 30 minutos").

## Decisiones tomadas y por qué

- **Split siempre temporal, nunca aleatorio**: la validación cruzada del
  EDA usa `TimeSeriesSplit` (5 folds expandiendo la ventana de
  entrenamiento). Mezclar aleatoriamente futuro y pasado da métricas
  optimistas que no representan la competencia.
- **`lag_672` (demanda de hace exactamente 1 semana) es la feature más
  fuerte** (91.8% de importancia en el Random Forest de selección de
  features) — hay estacionalidad semanal muy marcada en el histórico.
  Esto encendió una alarma inicial sobre fragilidad ante drift; entrenar
  sin esa feature (ver sección "Modelo") cuesta 2.11 puntos de accuracy —
  no catastrófico, pero tampoco despreciable, y mayor que en el esquema
  de un solo horizonte (0.87) porque en horizontes largos los lags cortos
  pesan menos. No se debe confundir el accuracy en el histórico estático
  con un problema resuelto.
- **Horizonte directo (una familia de modelos con `horizon_minutes` como
  feature) en vez de recursivo (encadenar 4 predicciones de un paso)**:
  la API pide +15/+30/+45/+60 min en cada ciclo. Un modelo entrenado solo
  para predecir su propia fila (esquema inicial, ver "Modelo") únicamente
  generaliza a +15 min — usarlo en cascada para los demás horizontes
  acumula error a cada paso y, peor, la primera versión ni siquiera podía
  hacer eso sin usar datos del futuro que no existirían en `data_cutoff`.
  Apilar 4 copias de cada fila ancla (una por horizonte) con el target
  correctamente desplazado hacia adelante es más simple de mantener que 4
  modelos separados y evita la fuga de futuro por diseño.
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

Probar que el champion registrado se puede volver a cargar y pronosticar
los 4 horizontes:

```bash
python src/predict.py
```

Correr la inferencia/submission (requiere `SUPABASE_SERVICE_ROLE_KEY`;
`PULSO_API_KEY` todavia no existe, asi que arma y valida el batch sin
enviarlo si no hay ciclo abierto o falta la key):

```bash
python src/infer.py
```

## Estructura del repo

**Lo que opera** (corre solo, vía GitHub Actions)

```
src/ingest.py           Collector incremental idempotente (Fase 2)
src/infer.py            Inferencia + submission por ciclo (Fase 4)
src/evaluate.py         Evaluacion, drift y decision de reentrenamiento (Fase 5)
src/check_contract.py   Avisa si el docente cambia el contrato de la API
src/check_model.py      Comprueba que el champion cargue y prediga
.github/workflows/      ci.yml · collector.yml · inference.yml · contract-watch.yml
```

**Lo que se corre a mano**

```
src/train.py            Compara candidatos y promueve el champion (Fase 3)
src/rollback.py         Vuelve a un champion anterior, verificandolo antes
src/sweep.py            Barrido de experimentos con MLflow (bono)
src/revalidar.py        Revalida los mejores del barrido con el protocolo oficial
src/predict.py          Recarga el champion y pronostica (smoke test)
src/simulate_cycle.py   Simulacro de un ciclo completo, y limpia lo que crea
src/stress_test.py      Bateria de esfuerzo del sistema completo
```

**Codigo compartido, datos y documentacion**

```
src/features.py         Feature engineering (EDA + entrenamiento + inferencia)
src/pulso_transmi/      SDK cliente (heredado del starter kit)
supabase/               Esquema de la base de datos y script de migracion
contract/expected.json  Estado del contrato del docente que ya revisamos
dashboard/              Pagina estatica desplegada en Vercel (bono)
eda/                    Analisis exploratorio, graficas, reportes de experimentos
docs/                   Bitacora, informe final, runbook, entidad-relacion, traspaso
tests/                  36 tests: SDK, collector, inferencia y monitoreo
artifacts/              Modelos entrenados y cache local (ignorado por git)
```

**Heredado del starter kit, conservado como referencia**

```
docs/student-project.md   Resumen de requisitos del docente
examples/                 Descarga del historico y baseline ingenuo
templates/pipeline.yml    Plantilla de workflow del starter kit. NO se usa:
                          referencia src/pipeline.py y requirements.txt, que
                          no existen aqui. Nuestros workflows reales estan en
                          .github/workflows/.
```

## Fuentes

- Starter kit y API: https://github.com/uexternadojz/pulso-transmi-sdk
- Documentación interactiva de la API: https://pulso-transmi.72-60-245-2.sslip.io/docs
- Guía metodológica del curso (MLOps · Ciencia de Datos, Universidad
  Externado de Colombia, Depto. de Matemáticas, docente Julián Zuluaga)
