# Banco de dados — bootstrap do zero (Supabase)

Migrations SQL que recriam **toda** a camada hospitalar em um projeto Supabase
novo: schema `hospital` (10 tabelas, RLS, policies), funções da fila de workers,
buckets privados de Storage, exposição no PostgREST, grants por papel e Realtime.

> Estes arquivos espelham exatamente o que está aplicado no projeto de
> referência (histórico `supabase_migrations.schema_migrations`), consolidando
> também o que foi feito fora de migration (buckets e publicação Realtime).

## Subindo em um ambiente novo

1. **Crie um projeto** em [supabase.com](https://supabase.com) (tier gratuito serve).
2. **Execute as migrations na ordem**, por um dos caminhos:
   - *Dashboard* → SQL Editor → cole e rode cada arquivo `0001` → `0005`; **ou**
   - *psql/CLI* (connection string em *Settings → Database*):

     ```bash
     for f in db/migrations/0*.sql; do psql "$DATABASE_URL" -f "$f"; done
     ```

3. **Chaves** → *Settings → API*: copie `Project URL`, `anon` e `service_role`
   para o `.env` na raiz (`SUPABASE_URL`, `SUPABASE_ANON_KEY`,
   `SUPABASE_SERVICE_KEY`). `REDIS_URL` (Upstash) é opcional — sem ela a janela
   z-score roda em memória.
4. **Seed de demonstração** (login demo, 3 pacientes, prescrições com interação
   e o cenário de deterioração NEWS2 2→14):

   ```bash
   uv run python scripts/seed_hospital.py
   ```

5. Pronto: dashboard (`uv run streamlit run app/Home.py`), API
   (`uv run uvicorn hospital_ai.main:app --port 8000`) e worker
   (`uv run python -m hospital_ai.workers.runner`).

**Nenhum passo manual no dashboard do Supabase é necessário** — inclusive a
exposição do schema no PostgREST (0004) e o Realtime (0005) são feitos por SQL.

## Conteúdo de cada migration

| Arquivo | O que cria |
|---|---|
| `0001_hospital_schema.sql` | Schema `hospital`, 10 tabelas, índices, RLS + policies (fila `jobs` sem policy = só backend) |
| `0002_job_queue_functions.sql` | `pick_job`/`complete_job` (`FOR UPDATE SKIP LOCKED`, retry com backoff 2^n min) |
| `0003_storage_buckets.sql` | Buckets privados `medical-videos` (1 GB), `medical-audio` (200 MB), `medical-docs` (50 MB) com MIME types tolerantes a navegadores |
| `0004_postgrest_and_grants.sql` | Expõe `hospital` no PostgREST via `ALTER ROLE authenticator` + grants mínimos (anon sem acesso a dados; `jobs`/RPCs só service key) |
| `0005_active_flags_and_realtime.sql` | Colunas `active` (exclusão lógica) + `hospital.alerts` na publication do Realtime |

## Observações

- **Ordem importa**: 0004 referencia funções criadas no 0002.
- Projetadas para projeto **novo**; o `0003` e o `0005` são idempotentes, e os
  demais assumem primeira execução.
- Em conta compartilhada, o `0004` **preserva** os schemas expostos padrão
  (`public, graphql_public`) — ajuste a lista se o seu projeto expõe outros.
- Storage não usa policies em `storage.objects`: todo acesso a arquivos passa
  pelo backend com service key e **signed URLs** de curta duração.
