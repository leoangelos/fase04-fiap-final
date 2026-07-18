"""NEWS2 (National Early Warning Score 2) - Royal College of Physicians.

Baseline clinico deterministico para deteccao de deterioracao.
Implementa os parametros disponiveis no modelo de dados
(sem nivel de consciencia ACVPU e sem O2 suplementar, documentado no relatorio).
"""
from typing import Optional


def _band(value, bands):
    if value is None:
        return 0
    for low, high, score in bands:
        if (low is None or value >= low) and (high is None or value <= high):
            return score
    return 0


def news2_score(
    respiratory_rate: Optional[int] = None,
    spo2: Optional[float] = None,
    systolic_bp: Optional[int] = None,
    heart_rate: Optional[int] = None,
    temperature: Optional[float] = None,
) -> dict:
    parts = {
        "respiratory_rate": _band(respiratory_rate, [
            (None, 8, 3), (9, 11, 1), (12, 20, 0), (21, 24, 2), (25, None, 3)]),
        "spo2": _band(spo2, [
            (None, 91, 3), (92, 93, 2), (94, 95, 1), (96, None, 0)]),
        "systolic_bp": _band(systolic_bp, [
            (None, 90, 3), (91, 100, 2), (101, 110, 1), (111, 219, 0), (220, None, 3)]),
        "heart_rate": _band(heart_rate, [
            (None, 40, 3), (41, 50, 1), (51, 90, 0), (91, 110, 1), (111, 130, 2), (131, None, 3)]),
        "temperature": _band(temperature, [
            (None, 35.0, 3), (35.1, 36.0, 1), (36.1, 38.0, 0), (38.1, 39.0, 1), (39.1, None, 2)]),
    }
    total = sum(parts.values())
    any_red = any(v == 3 for v in parts.values())

    if total >= 7:
        severity = "critical"
    elif total >= 5 or any_red:
        severity = "warning"
    elif total >= 1:
        severity = "info"
    else:
        severity = None  # sem alerta

    return {"total": total, "parts": parts, "severity": severity,
            "risk_score": min(total / 10.0, 1.0)}
