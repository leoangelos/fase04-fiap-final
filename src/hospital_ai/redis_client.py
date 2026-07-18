"""Cliente Redis (Upstash) — opcional.

Sem ``REDIS_URL`` configurada, devolve ``None`` e a janela deslizante de sinais
vitais passa a viver em memória do processo (suficiente para demo e testes;
em produção o Redis garante janela compartilhada entre réplicas da API).
"""

from __future__ import annotations

from functools import lru_cache

from .config import settings


@lru_cache
def get_redis():
    if not settings.redis_configured:
        return None
    import redis

    return redis.from_url(settings.redis_url, decode_responses=True)
