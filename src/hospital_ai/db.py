"""Clientes Supabase da camada hospitalar.

- ``service_client``: SERVICE_KEY, ignora RLS — uso exclusivo do backend/workers.
- ``anon_client``: ANON_KEY, apenas para validar o JWT dos usuários.
Todas as tabelas vivem no schema ``hospital`` (exposto no PostgREST).
"""

from __future__ import annotations

from functools import lru_cache

from .config import SCHEMA, settings


@lru_cache
def service_client():
    if not settings.supabase_configured:
        raise RuntimeError(
            "Supabase não configurado: defina SUPABASE_URL e SUPABASE_SERVICE_KEY no .env"
        )
    from supabase import create_client
    from supabase.lib.client_options import SyncClientOptions

    return create_client(
        settings.supabase_url,
        settings.supabase_service_key,
        options=SyncClientOptions(schema=SCHEMA),
    )


@lru_cache
def anon_client():
    if not (settings.supabase_url and settings.supabase_anon_key):
        raise RuntimeError(
            "Supabase não configurado: defina SUPABASE_URL e SUPABASE_ANON_KEY no .env"
        )
    from supabase import create_client

    return create_client(settings.supabase_url, settings.supabase_anon_key)


def table(name: str):
    return service_client().table(name)
