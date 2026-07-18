-- 0004: exposição do schema no PostgREST + privilégios mínimos por papel
--
-- Expõe 'hospital' na API REST sem passo manual no dashboard: o PostgREST do
-- Supabase lê a configuração do papel 'authenticator'.
-- Rollback da exposição: ALTER ROLE authenticator RESET pgrst.db_schemas;
--                        NOTIFY pgrst, 'reload config';

ALTER ROLE authenticator SET pgrst.db_schemas = 'public, graphql_public, hospital';
NOTIFY pgrst, 'reload config';

-- Privilégios (as policies RLS do 0001 só valem se houver GRANT de tabela):
grant usage on schema hospital to anon, authenticated, service_role;

-- service_role: backend/workers (bypassa RLS, mas precisa dos privilégios)
grant all privileges on all tables in schema hospital to service_role;
grant usage, select on all sequences in schema hospital to service_role;
grant execute on all functions in schema hospital to service_role;

-- authenticated: CRUD limitado pelas policies RLS
grant select, insert, update, delete on all tables in schema hospital to authenticated;
grant usage, select on all sequences in schema hospital to authenticated;

-- fila de processamento é EXCLUSIVA do backend
revoke all on table hospital.jobs from authenticated;
revoke execute on function hospital.pick_job(text[]) from public, anon, authenticated;
revoke execute on function hospital.complete_job(bigint, boolean, text) from public, anon, authenticated;

-- anon: apenas USAGE no schema (sem privilégio de tabela — nada de dados de
-- pacientes sem login)

-- objetos futuros herdam os mesmos privilégios
alter default privileges for role postgres in schema hospital
  grant select, insert, update, delete on tables to authenticated;
alter default privileges for role postgres in schema hospital
  grant all on tables to service_role;
alter default privileges for role postgres in schema hospital
  grant usage, select on sequences to authenticated, service_role;
alter default privileges for role postgres in schema hospital
  grant execute on functions to service_role;
