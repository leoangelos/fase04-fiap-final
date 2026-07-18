-- 0005: desativação lógica (CRUD sem apagar dados clínicos — LGPD/auditoria)
--        + alertas em tempo real via Supabase Realtime

alter table hospital.professionals add column if not exists active boolean not null default true;
alter table hospital.patients add column if not exists active boolean not null default true;

-- Publica hospital.alerts no Realtime (frontend recebe alertas ao vivo)
do $$
begin
  if exists (select 1 from pg_publication where pubname = 'supabase_realtime')
     and not exists (
       select 1 from pg_publication_tables
        where pubname = 'supabase_realtime'
          and schemaname = 'hospital' and tablename = 'alerts'
     ) then
    alter publication supabase_realtime add table hospital.alerts;
  end if;
end $$;
