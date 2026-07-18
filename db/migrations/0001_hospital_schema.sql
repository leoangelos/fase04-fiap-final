-- =============================================================
-- FIAP Fase 04 - Sistema Hospitalar Multimodal
-- 0001: schema 'hospital' completo (tabelas, índices, RLS, policies)
-- Executar em um projeto Supabase NOVO, na ordem 0001 → 0005.
-- =============================================================

create extension if not exists "uuid-ossp";
create extension if not exists vector;

create schema if not exists hospital;

-- Profissionais (espelha auth.users)
create table hospital.professionals (
  id uuid primary key references auth.users(id),
  full_name text not null,
  crm text,
  role text not null check (role in ('medico','enfermagem','fisio','admin')),
  created_at timestamptz default now()
);

-- Pacientes
create table hospital.patients (
  id uuid primary key default uuid_generate_v4(),
  mrn text unique not null,
  full_name text not null,
  birth_date date not null,
  sex text check (sex in ('M','F','O')),
  cpf text unique check (cpf is null or cpf ~ '^[0-9]{11}$'),
  admission_status text default 'outpatient'
    check (admission_status in ('outpatient','admitted','icu','discharged')),
  created_by uuid references hospital.professionals(id),
  created_at timestamptz default now()
);

-- Encontros clínicos
create table hospital.encounters (
  id uuid primary key default uuid_generate_v4(),
  patient_id uuid not null references hospital.patients(id),
  professional_id uuid references hospital.professionals(id),
  encounter_type text not null
    check (encounter_type in ('consulta','cirurgia','fisioterapia','internacao','telemedicina')),
  started_at timestamptz not null default now(),
  ended_at timestamptz,
  notes text
);
create index idx_hosp_encounters_patient on hospital.encounters (patient_id, started_at desc);

-- Mídia multimodal
create table hospital.media_assets (
  id uuid primary key default uuid_generate_v4(),
  patient_id uuid not null references hospital.patients(id),
  encounter_id uuid references hospital.encounters(id),
  uploaded_by uuid not null references hospital.professionals(id),
  modality text not null check (modality in ('video','audio','text','image')),
  category text,
  storage_path text not null,
  mime_type text,
  size_bytes bigint,
  duration_seconds numeric,
  checksum_sha256 text,
  processing_status text default 'pending'
    check (processing_status in ('pending','processing','done','failed')),
  error_message text,
  created_at timestamptz default now()
);
create index idx_hosp_media_patient on hospital.media_assets (patient_id, created_at desc);
create index idx_hosp_media_status on hospital.media_assets (processing_status)
  where processing_status in ('pending','processing');

-- Resultados de análise
create table hospital.analysis_results (
  id uuid primary key default uuid_generate_v4(),
  media_asset_id uuid references hospital.media_assets(id),
  patient_id uuid not null references hospital.patients(id),
  analysis_type text not null,
  engine text not null,
  result jsonb not null,
  risk_score numeric check (risk_score between 0 and 1),
  embedding vector(1536),
  created_at timestamptz default now()
);
create index idx_hosp_analysis_patient on hospital.analysis_results (patient_id, created_at desc);
create index idx_hosp_analysis_asset on hospital.analysis_results (media_asset_id);

-- Sinais vitais (série temporal)
create table hospital.vital_signs (
  id bigint generated always as identity primary key,
  patient_id uuid not null references hospital.patients(id),
  measured_at timestamptz not null,
  heart_rate int,
  spo2 numeric,
  temperature numeric,
  systolic_bp int,
  diastolic_bp int,
  respiratory_rate int,
  source text default 'manual' check (source in ('manual','monitor','wearable','import')),
  created_at timestamptz default now()
);
create index idx_hosp_vitals_patient_time on hospital.vital_signs (patient_id, measured_at desc);

-- Prescrições
create table hospital.prescriptions (
  id uuid primary key default uuid_generate_v4(),
  patient_id uuid not null references hospital.patients(id),
  prescribed_by uuid not null references hospital.professionals(id),
  medication text not null,
  dosage text,
  frequency text,
  route text,
  starts_at timestamptz default now(),
  ends_at timestamptz,
  active boolean default true
);
create index idx_hosp_rx_patient on hospital.prescriptions (patient_id) where active;

-- Alertas
create table hospital.alerts (
  id uuid primary key default uuid_generate_v4(),
  patient_id uuid not null references hospital.patients(id),
  source_type text not null
    check (source_type in ('vital_signs','audio','video','prescription','nlp','fusion')),
  source_id text,
  severity text not null check (severity in ('info','warning','critical')),
  title text not null,
  details jsonb,
  acknowledged_by uuid references hospital.professionals(id),
  acknowledged_at timestamptz,
  created_at timestamptz default now()
);
create index idx_hosp_alerts_open on hospital.alerts (patient_id, created_at desc)
  where acknowledged_at is null;

-- Auditoria LGPD
create table hospital.audit_logs (
  id bigint generated always as identity primary key,
  professional_id uuid references hospital.professionals(id),
  action text not null,
  entity text,
  entity_id text,
  created_at timestamptz default now()
);

-- Fila de jobs para workers (padrão FOR UPDATE SKIP LOCKED)
create table hospital.jobs (
  id bigint generated always as identity primary key,
  job_type text not null,
  payload jsonb not null,
  status text default 'queued' check (status in ('queued','running','done','failed')),
  attempts int default 0,
  max_attempts int default 3,
  last_error text,
  run_after timestamptz default now(),
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);
create index idx_hosp_jobs_pick on hospital.jobs (run_after) where status = 'queued';

-- =============================================================
-- RLS habilitado em todas as tabelas
-- =============================================================
alter table hospital.professionals enable row level security;
alter table hospital.patients enable row level security;
alter table hospital.encounters enable row level security;
alter table hospital.media_assets enable row level security;
alter table hospital.analysis_results enable row level security;
alter table hospital.vital_signs enable row level security;
alter table hospital.prescriptions enable row level security;
alter table hospital.alerts enable row level security;
alter table hospital.audit_logs enable row level security;
alter table hospital.jobs enable row level security;

-- helper: usuário logado é profissional cadastrado
create or replace function hospital.is_professional()
returns boolean language sql stable security definer set search_path = hospital as $$
  select exists (select 1 from hospital.professionals p where p.id = auth.uid());
$$;

-- Policies (profissionais autenticados; fila jobs fica SEM policy = só service_role)
create policy prof_self on hospital.professionals
  for select to authenticated using (true);
create policy patients_rw on hospital.patients
  for all to authenticated using (hospital.is_professional()) with check (hospital.is_professional());
create policy encounters_rw on hospital.encounters
  for all to authenticated using (hospital.is_professional()) with check (hospital.is_professional());
create policy media_rw on hospital.media_assets
  for all to authenticated using (hospital.is_professional()) with check (hospital.is_professional());
create policy analysis_ro on hospital.analysis_results
  for select to authenticated using (hospital.is_professional());
create policy vitals_rw on hospital.vital_signs
  for all to authenticated using (hospital.is_professional()) with check (hospital.is_professional());
create policy rx_rw on hospital.prescriptions
  for all to authenticated using (hospital.is_professional()) with check (hospital.is_professional());
create policy alerts_rw on hospital.alerts
  for all to authenticated using (hospital.is_professional()) with check (hospital.is_professional());
create policy audit_insert on hospital.audit_logs
  for insert to authenticated with check (hospital.is_professional());
create policy audit_ro on hospital.audit_logs
  for select to authenticated using (hospital.is_professional());
