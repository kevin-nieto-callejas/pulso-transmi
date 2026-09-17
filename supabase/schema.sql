-- Esquema de Supabase para Pulso TransMi.
-- Aplicado via `apply_migration` (migraciones "initial_schema" y
-- "enable_rls_public_read"). Se conserva aqui como documentacion
-- versionada del esquema (ver docs/entity-relation.md para el diagrama
-- y la justificacion de cada tabla y politica).

create table stations (
    station_id text primary key,
    station_name text not null,
    corridor text not null,
    latitude double precision not null,
    longitude double precision not null
);

create table observations (
    id bigint generated always as identity primary key,
    station_id text not null references stations (station_id),
    observed_at timestamptz not null,
    demand integer not null check (demand >= 0),
    ingested_at timestamptz not null default now(),
    unique (station_id, observed_at)
);
create index observations_station_time_idx on observations (station_id, observed_at);

create table context_readings (
    observed_at timestamptz primary key,
    rain_mm double precision,
    rain_forecast double precision,
    temperature_c double precision,
    temperature_forecast double precision,
    event_intensity double precision,
    ingested_at timestamptz not null default now()
);

create table collector_runs (
    id bigint generated always as identity primary key,
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    cursor_before text,
    cursor_after text,
    rows_ingested integer not null default 0,
    status text not null check (status in ('success', 'failed', 'no_new_data')),
    error_message text
);

create table cycles (
    cycle_id text primary key,
    opens_at timestamptz not null,
    data_cutoff timestamptz not null,
    closes_at timestamptz not null,
    status text not null default 'open' check (status in ('open', 'closed', 'evaluated'))
);

create table model_versions (
    version_id text primary key,
    created_at timestamptz not null default now(),
    data_cutoff timestamptz not null,
    code_commit text not null,
    features jsonb not null,
    validation_metric double precision not null,
    artifact_location text not null,
    status text not null default 'candidate' check (status in ('candidate', 'champion', 'historical'))
);

create table predictions (
    id bigint generated always as identity primary key,
    cycle_id text not null references cycles (cycle_id),
    station_id text not null references stations (station_id),
    target_at timestamptz not null,
    horizon_minutes integer not null check (horizon_minutes in (15, 30, 45, 60)),
    predicted_value double precision not null check (predicted_value >= 0),
    model_version_id text not null references model_versions (version_id),
    emitted_at timestamptz not null default now(),
    unique (cycle_id, station_id, target_at)
);

create table submissions (
    submission_id text primary key,
    cycle_id text not null references cycles (cycle_id),
    model_version_id text not null references model_versions (version_id),
    idempotency_key text not null unique,
    submitted_at timestamptz not null default now(),
    accepted boolean not null,
    receipt jsonb
);

create table prediction_evaluations (
    prediction_id bigint primary key references predictions (id),
    actual_value double precision not null,
    absolute_error double precision not null,
    evaluated_at timestamptz not null default now()
);

create table cycle_metrics (
    id bigint generated always as identity primary key,
    cycle_id text not null references cycles (cycle_id),
    station_id text references stations (station_id),
    wape double precision,
    accuracy double precision,
    computed_at timestamptz not null default now()
);

create table drift_signals (
    id bigint generated always as identity primary key,
    detected_at timestamptz not null default now(),
    signal_type text not null check (signal_type in ('performance', 'data', 'operational')),
    description text not null,
    metric_value double precision,
    threshold_value double precision,
    action_taken text
);

-- Indices para las FK que el linter de performance de Supabase marco sin
-- cobertura (predictions y submissions creceran rapido en fase operativa:
-- 48 filas por ciclo, un ciclo por hora).
create index cycle_metrics_cycle_id_idx on cycle_metrics (cycle_id);
create index cycle_metrics_station_id_idx on cycle_metrics (station_id);
create index predictions_model_version_id_idx on predictions (model_version_id);
create index predictions_station_id_idx on predictions (station_id);
create index submissions_cycle_id_idx on submissions (cycle_id);
create index submissions_model_version_id_idx on submissions (model_version_id);

-- Row Level Security: lectura publica (sin PII en este esquema), escritura
-- restringida a la service_role key (ver docs/entity-relation.md).
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

create policy "public_read" on stations for select using (true);
create policy "public_read" on observations for select using (true);
create policy "public_read" on context_readings for select using (true);
create policy "public_read" on collector_runs for select using (true);
create policy "public_read" on cycles for select using (true);
create policy "public_read" on model_versions for select using (true);
create policy "public_read" on predictions for select using (true);
create policy "public_read" on submissions for select using (true);
create policy "public_read" on prediction_evaluations for select using (true);
create policy "public_read" on cycle_metrics for select using (true);
create policy "public_read" on drift_signals for select using (true);
