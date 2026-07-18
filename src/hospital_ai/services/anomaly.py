"""Detecção estatística de anomalias em sinais vitais (camada 2).

Janela deslizante por paciente/métrica com z-score contra a linha de base do
PRÓPRIO paciente — complementa o NEWS2 (camada 1, populacional): um paciente
cronicamente taquicárdico não dispara z-score por estar "alto", e sim quando
muda em relação a si mesmo.

Backend da janela: Redis (chave ``vitals:{patient_id}:{metric}``, TTL 7 dias)
quando ``REDIS_URL`` está configurada; caso contrário, memória do processo.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

from ..config import settings
from ..redis_client import get_redis

METRICS = ["heart_rate", "spo2", "temperature", "systolic_bp",
           "diastolic_bp", "respiratory_rate"]

# Fallback em memória: {chave: deque de floats} — usado sem Redis (demo/testes).
_memory_windows: dict[str, deque] = defaultdict(
    lambda: deque(maxlen=settings.vitals_window_size)
)


def _key(patient_id: str, metric: str) -> str:
    return f"vitals:{patient_id}:{metric}"


def _read_window(key: str) -> list[float]:
    r = get_redis()
    if r is None:
        return list(_memory_windows[key])
    return [float(v) for v in r.lrange(key, 0, -1)]


def _append(key: str, value: float) -> None:
    r = get_redis()
    if r is None:
        _memory_windows[key].append(float(value))
        return
    pipe = r.pipeline()
    pipe.rpush(key, value)
    pipe.ltrim(key, -settings.vitals_window_size, -1)
    pipe.expire(key, 60 * 60 * 24 * 7)  # 7 dias
    pipe.execute()


def reset_memory_windows() -> None:
    """Limpa o fallback em memória (usado nos testes)."""
    _memory_windows.clear()


def push_and_score(patient_id: str, readings: dict) -> list[dict]:
    """Compara cada leitura com a janela do paciente e depois a incorpora.

    Retorna a lista de anomalias (|z| ≥ threshold com janela mínima).
    """
    anomalies = []
    for metric in METRICS:
        value = readings.get(metric)
        if value is None:
            continue
        key = _key(patient_id, metric)
        window = _read_window(key)
        if len(window) >= settings.vitals_min_window:
            mean = sum(window) / len(window)
            var = sum((x - mean) ** 2 for x in window) / len(window)
            std = math.sqrt(var)
            if std > 1e-6:
                z = (float(value) - mean) / std
                if abs(z) >= settings.vitals_zscore_threshold:
                    anomalies.append({
                        "metric": metric, "value": value, "zscore": round(z, 2),
                        "baseline_mean": round(mean, 2),
                        "baseline_std": round(std, 2),
                        "window_size": len(window),
                    })
        _append(key, float(value))
    return anomalies
