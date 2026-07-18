-- 0002: funções da fila de processamento (chamadas via RPC com service_role)

create or replace function hospital.pick_job(p_job_types text[] default null)
returns setof hospital.jobs
language plpgsql security definer set search_path = hospital as $$
begin
  return query
  update hospital.jobs j
     set status = 'running',
         attempts = j.attempts + 1,
         updated_at = now()
   where j.id = (
     select id from hospital.jobs
      where status = 'queued'
        and run_after <= now()
        and (p_job_types is null or job_type = any(p_job_types))
      order by id
      for update skip locked
      limit 1
   )
  returning j.*;
end;
$$;

create or replace function hospital.complete_job(p_job_id bigint, p_success boolean, p_error text default null)
returns void
language plpgsql security definer set search_path = hospital as $$
declare v_attempts int; v_max int;
begin
  select attempts, max_attempts into v_attempts, v_max from hospital.jobs where id = p_job_id;
  if p_success then
    update hospital.jobs set status = 'done', updated_at = now() where id = p_job_id;
  elsif v_attempts >= v_max then
    update hospital.jobs set status = 'failed', last_error = p_error, updated_at = now() where id = p_job_id;
  else
    -- retry com backoff progressivo (2^attempts minutos)
    update hospital.jobs
       set status = 'queued',
           last_error = p_error,
           run_after = now() + (power(2, v_attempts) || ' minutes')::interval,
           updated_at = now()
     where id = p_job_id;
  end if;
end;
$$;

revoke execute on function hospital.pick_job(text[]) from public, anon, authenticated;
revoke execute on function hospital.complete_job(bigint, boolean, text) from public, anon, authenticated;
grant execute on function hospital.pick_job(text[]) to service_role;
grant execute on function hospital.complete_job(bigint, boolean, text) to service_role;
