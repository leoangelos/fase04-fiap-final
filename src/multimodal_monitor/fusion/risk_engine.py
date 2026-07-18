"""Fusão multimodal → índice de risco do paciente.

Combina os alertas das quatro modalidades (vídeo, áudio, sinais vitais e
prescrições) em um único índice de risco 0–100, ponderando por severidade e por
modalidade. É a etapa que responde ao objetivo do edital: "análise e fusão de
diferentes tipos de dados médicos".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..alerts.alert_manager import Alert, AlertLevel, MODALITY_PT

# Peso por severidade de cada alerta.
_LEVEL_WEIGHT = {AlertLevel.INFO: 1.0, AlertLevel.WARNING: 4.0, AlertLevel.CRITICAL: 10.0}

# Peso por modalidade (importância clínica relativa na fusão).
_MODALITY_WEIGHT = {
    "vitals": 1.0,
    "audio": 0.8,
    "video": 0.8,
    "prescription": 0.9,
    "fusion": 1.0,
}


@dataclass
class RiskAssessment:
    patient_id: str
    score: float                       # 0–100
    level: AlertLevel
    per_modality: dict[str, float] = field(default_factory=dict)
    n_alerts: int = 0
    contributing: list[str] = field(default_factory=list)

    @property
    def level_label(self) -> str:
        return {AlertLevel.INFO: "BAIXO", AlertLevel.WARNING: "MODERADO",
                AlertLevel.CRITICAL: "ALTO"}[self.level]

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "score": round(self.score, 1),
            "level": self.level_label,
            "per_modality": {k: round(v, 1) for k, v in self.per_modality.items()},
            "n_alerts": self.n_alerts,
            "contributing": self.contributing,
        }


def compute_risk(alerts: list[Alert], patient_id: str = "P001") -> RiskAssessment:
    """Agrega alertas em um índice de risco 0–100.

    A pontuação bruta soma severidade × peso da modalidade; é então comprimida por
    saturação (1 - e^-x) para 0–100, de modo que muitos alertas leves não estouram
    a escala, mas alertas críticos elevam o índice rapidamente.
    """
    import math

    per_modality_raw: dict[str, float] = {}
    for a in alerts:
        w = _LEVEL_WEIGHT[a.level] * _MODALITY_WEIGHT.get(a.modality, 0.8)
        per_modality_raw[a.modality] = per_modality_raw.get(a.modality, 0.0) + w

    raw_total = sum(per_modality_raw.values())
    # saturação: 0 → 0, cresce rápido e satura perto de 100
    score = 100 * (1 - math.exp(-raw_total / 20.0))

    has_critical = any(a.level == AlertLevel.CRITICAL for a in alerts)
    if has_critical or score >= 60:
        level = AlertLevel.CRITICAL
    elif score >= 25:
        level = AlertLevel.WARNING
    else:
        level = AlertLevel.INFO

    # modalidades ordenadas pela contribuição, para explicar o índice
    per_modality = {
        k: 100 * (1 - math.exp(-v / 20.0)) for k, v in per_modality_raw.items()
    }
    contributing = [
        f"{MODALITY_PT.get(m, m)} ({sum(1 for a in alerts if a.modality == m)} alertas)"
        for m in sorted(per_modality_raw, key=per_modality_raw.get, reverse=True)
    ]

    return RiskAssessment(
        patient_id=patient_id,
        score=score,
        level=level,
        per_modality=per_modality,
        n_alerts=len(alerts),
        contributing=contributing,
    )
