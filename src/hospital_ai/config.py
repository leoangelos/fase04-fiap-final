"""Configuração da camada hospitalar (Supabase, Redis e thresholds).

Segue o mesmo padrão de ``multimodal_monitor.config``: variáveis do ``.env`` na
raiz do projeto, dataclass congelada e *flags* de disponibilidade — sem chaves,
a API/worker falham com mensagem clara, mas os módulos puros (NEWS2, z-score,
fusão) continuam importáveis e testáveis offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

SCHEMA = "hospital"


@dataclass(frozen=True)
class HospitalSettings:
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_anon_key: str = os.getenv("SUPABASE_ANON_KEY", "")
    supabase_service_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")
    redis_url: str = os.getenv("REDIS_URL", "")

    # Detecção de anomalias em sinais vitais
    vitals_window_size: int = 50        # amostras na janela deslizante
    vitals_min_window: int = 10         # mínimo de histórico p/ z-score confiável
    vitals_zscore_threshold: float = 3.0

    # Fusão multimodal
    fusion_alert_threshold: float = 0.7

    signed_url_ttl_seconds: int = 300

    @property
    def supabase_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_key)

    @property
    def redis_configured(self) -> bool:
        return bool(self.redis_url)


settings = HospitalSettings()


def get_settings() -> HospitalSettings:
    return settings
