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

Fase 1 de la guía metodológica ("Comprender" + "Construir la memoria"):

- ✅ SDK instalado, histórico descargado (12 estaciones, 45 días, 51.840
  observaciones).
- ✅ Análisis exploratorio completo: calidad de datos, estacionalidad,
  correlaciones, feature engineering, selección de features y
  validación cruzada temporal — ver [`eda/EDA_REPORT.md`](eda/EDA_REPORT.md).
- ✅ Base de datos en Supabase con el histórico migrado — ver
  [`docs/entity-relation.md`](docs/entity-relation.md) para el esquema,
  el diagrama entidad-relación y las políticas de seguridad (RLS).
- ⬜ Collector incremental automatizado (próxima fase, cuando se active
  el stream en vivo).
- ⬜ Modelo champion registrado y promovido.
- ⬜ Inferencia periódica vía GitHub Actions y submissions.
- ⬜ Monitoreo de drift y reentrenamiento.
- ⬜ Dashboard (bono).

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
  Esto es una señal útil hoy, pero también una alerta: un modelo que
  solo copia el patrón de la semana pasada es frágil ante el drift que
  el reto va a introducir cuando el stream esté activo. No se debe
  confundir el 86.21% de accuracy en el histórico estático con un
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

Migrar (o re-migrar; es idempotente) el histórico a Supabase requiere la
`service_role` key en `SUPABASE_SERVICE_ROLE_KEY` (ver `.env.example`):

```bash
python supabase/migrate_via_rest.py
```

## Estructura del repo

```
src/pulso_transmi/     SDK cliente (heredado del starter kit)
examples/               Scripts de ejemplo del starter kit
eda/                    Analisis exploratorio, graficos, reportes
supabase/               Esquema de la base de datos y script de migracion
docs/                   Documentacion: API, guia del proyecto, entidad-relacion
tests/                  Tests del SDK (heredados del starter kit)
```

## Fuentes

- Starter kit y API: https://github.com/uexternadojz/pulso-transmi-sdk
- Documentación interactiva de la API: https://pulso-transmi.72-60-245-2.sslip.io/docs
- Guía metodológica del curso (MLOps · Ciencia de Datos, Universidad
  Externado de Colombia, Depto. de Matemáticas, docente Julián Zuluaga)
