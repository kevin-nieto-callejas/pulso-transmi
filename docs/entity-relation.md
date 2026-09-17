# Modelo entidad-relación — Supabase

Proyecto Supabase: `pulso-transmi` (region `us-east-1`, Postgres 17).
Esquema completo en [`../supabase/schema.sql`](../supabase/schema.sql),
aplicado como la migración `initial_schema`.

## Diagrama

```mermaid
erDiagram
    STATIONS ||--o{ OBSERVATIONS : "registra demanda de"
    STATIONS ||--o{ PREDICTIONS : "se predice para"
    STATIONS ||--o{ CYCLE_METRICS : "se mide por"
    CYCLES ||--o{ PREDICTIONS : "agrupa"
    CYCLES ||--o{ SUBMISSIONS : "se entrega en"
    CYCLES ||--o{ CYCLE_METRICS : "se evalua en"
    MODEL_VERSIONS ||--o{ PREDICTIONS : "genera"
    MODEL_VERSIONS ||--o{ SUBMISSIONS : "respalda"
    PREDICTIONS ||--o| PREDICTION_EVALUATIONS : "se evalua como"

    STATIONS {
        text station_id PK
        text station_name
        text corridor
        float latitude
        float longitude
    }
    OBSERVATIONS {
        bigint id PK
        text station_id FK
        timestamptz observed_at
        int demand
        timestamptz ingested_at
    }
    CONTEXT_READINGS {
        timestamptz observed_at PK
        float rain_mm
        float rain_forecast
        float temperature_c
        float temperature_forecast
        float event_intensity
    }
    COLLECTOR_RUNS {
        bigint id PK
        timestamptz started_at
        timestamptz finished_at
        text cursor_before
        text cursor_after
        int rows_ingested
        text status
    }
    CYCLES {
        text cycle_id PK
        timestamptz opens_at
        timestamptz data_cutoff
        timestamptz closes_at
        text status
    }
    MODEL_VERSIONS {
        text version_id PK
        timestamptz data_cutoff
        text code_commit
        jsonb features
        float validation_metric
        text artifact_location
        text status
    }
    PREDICTIONS {
        bigint id PK
        text cycle_id FK
        text station_id FK
        timestamptz target_at
        int horizon_minutes
        float predicted_value
        text model_version_id FK
    }
    SUBMISSIONS {
        text submission_id PK
        text cycle_id FK
        text model_version_id FK
        text idempotency_key
        boolean accepted
        jsonb receipt
    }
    PREDICTION_EVALUATIONS {
        bigint prediction_id PK_FK
        float actual_value
        float absolute_error
    }
    CYCLE_METRICS {
        bigint id PK
        text cycle_id FK
        text station_id FK
        float wape
        float accuracy
    }
    DRIFT_SIGNALS {
        bigint id PK
        text signal_type
        text description
        float metric_value
        float threshold_value
    }
```

`context_readings` y `drift_signals` no tienen FK entrantes de otras
entidades en el diagrama (el contexto se une por `observed_at` en tiempo de
feature engineering, no por llave foránea, porque es compartido por las 12
estaciones; las señales de drift se calculan sobre `cycle_metrics` pero se
guardan desacopladas para no atar cada señal a un ciclo específico).

## Justificación de cada tabla

Mapea directamente a lo que la guía metodológica pide que Supabase
conserve (sección "Supabase — la memoria operacional"):

| Tabla | Por qué existe | Requisito de la guía |
|---|---|---|
| `stations` | Catálogo geográfico, no cambia con el tiempo | "las estaciones... descargadas" |
| `observations` | Demanda real por estación y periodo de 15 min | "las estaciones y observaciones descargadas" |
| `context_readings` | Clima/eventos disponibles en cada corte | "el contexto disponible al momento de predecir" |
| `collector_runs` | Bitácora de cada corrida del collector (cursor antes/después, filas, éxito/error) | "el historial de ejecuciones del collector" |
| `cycles` | Identidad de cada ciclo de predicción (cutoff, cierre, estado) — evita que una predicción quede mal atribuida | "Identidad del ciclo" (guía, sección "Cómo se entrega una predicción") |
| `model_versions` | Ficha de cada versión: commit, features, métrica, ubicación del artefacto, estado (`candidate`/`champion`/`historical`) | "las versiones promovidas de modelos" |
| `predictions` | Pareja exacta `(station_id, target_at)` por ciclo y modelo, con `unique(cycle_id, station_id, target_at)` para no duplicar | "las predicciones emitidas" |
| `submissions` | Evidencia de entrega a la API, con `idempotency_key` única para reintentos seguros | "Reintentos sin duplicados" |
| `prediction_evaluations` | Valor real y error una vez se conoce el target | "el resultado real cuando se conozca" |
| `cycle_metrics` | WAPE/accuracy agregados por ciclo y estación | "las métricas de accuracy y drift" |
| `drift_signals` | Señales de performance/data/operational drift con la acción tomada | monitoreo y decisión de reentrenamiento (Fase 5) |

## Reglas de integridad relevantes

- `observations` tiene `unique(station_id, observed_at)` y
  `check (demand >= 0)`: el collector puede reintentar sin duplicar y
  nunca puede insertar demanda negativa.
- `predictions` tiene `unique(cycle_id, station_id, target_at)` y
  `check (horizon_minutes in (15,30,45,60))`: refleja el contrato de 48
  predicciones por ciclo (12 estaciones × 4 horizontes) que exige la API.
- `submissions.idempotency_key` es `unique`: si el workflow reintenta tras
  un fallo de red, un `upsert` sobre esa llave nunca crea una segunda
  entrega.
- `model_versions.status` restringido a `candidate | champion | historical`
  modela la regla de promoción: solo una fila puede/debería estar en
  `champion` a la vez (a reforzar más adelante con un índice parcial
  único si se automatiza la promoción).

## Nota de seguridad pendiente

Row Level Security (RLS) está **deshabilitado** en las 11 tablas del
esquema (advertencia crítica del linter de Supabase). Mientras el
proyecto es de uso interno del equipo (collector y notebooks locales
usando la `anon key`), esto no expone datos a terceros porque el
proyecto no es público. Pero **antes de conectar el dashboard de Vercel
(bono) o de compartir la anon key**, hay que:

1. Habilitar RLS en cada tabla.
2. Agregar políticas explícitas: lectura pública solo para las tablas que
   alimentarán el dashboard (`cycle_metrics`, `model_versions` con
   columnas no sensibles), y ninguna escritura desde el rol `anon`.
3. Mover la ingesta (collector, submissions) a la `service_role` key
   dentro de GitHub Actions Secrets, nunca en el navegador — tal como
   exige la guía metodológica.

SQL de referencia para habilitar RLS (no aplicado todavía; requiere
definir las políticas antes de correrlo o el acceso actual del equipo se
bloquea):

```sql
alter table stations enable row level security;
alter table observations enable row level security;
alter table context_readings enable row level security;
alter table collector_runs enable row level security;
alter table cycles enable row level security;
alter table model_versions enable row level security;
alter table predictions enable row level security;
alter table submissions enable row level security;
alter table prediction_evaluations enable row level security;
alter table cycle_metrics enable row level security;
alter table drift_signals enable row level security;
```
