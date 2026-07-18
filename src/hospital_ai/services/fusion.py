"""Fusao multimodal tardia (late fusion).

Combina os risk_scores mais recentes de cada fonte com pesos e decaimento
temporal, gerando um patient_risk_score unico. Score acima do limiar gera
alerta de fusao.
"""
from datetime import datetime, timezone
from ..config import get_settings
from ..db import table

WEIGHTS = {
    "vital_signs": 0.35,
    "audio": 0.20,
    "video": 0.20,
    "nlp": 0.15,
    "prescription": 0.10,
}
HALF_LIFE_HOURS = 24.0


def _decay(created_at: str) -> float:
    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    return 0.5 ** (age_h / HALF_LIFE_HOURS)


def compute_patient_risk(patient_id: str) -> dict:
    res = (table("analysis_results").select("analysis_type,engine,risk_score,created_at")
           .eq("patient_id", patient_id).not_.is_("risk_score", "null")
           .order("created_at", desc=True).limit(50).execute())

    latest: dict[str, dict] = {}
    for row in res.data:
        src = _source_of(row["analysis_type"])
        if src and src not in latest:
            latest[src] = row

    num, den = 0.0, 0.0
    contributions = {}
    for src, row in latest.items():
        w = WEIGHTS.get(src, 0.1) * _decay(row["created_at"])
        num += w * float(row["risk_score"])
        den += w
        contributions[src] = {"risk_score": row["risk_score"], "weight": round(w, 3)}

    fused = num / den if den > 0 else 0.0
    return {"patient_risk_score": round(fused, 3), "contributions": contributions}


def _source_of(analysis_type: str) -> str | None:
    mapping = {
        "news2": "vital_signs", "vitals_zscore": "vital_signs",
        "transcricao": "audio", "disartria": "audio", "prosodia": "audio",
        "pose_anomaly": "video", "movement_metrics": "video",
        "ner_clinico": "nlp", "laudo_nlp": "nlp",
        "interacao_medicamentosa": "prescription",
    }
    return mapping.get(analysis_type)


def maybe_alert_fusion(patient_id: str) -> dict | None:
    s = get_settings()
    fused = compute_patient_risk(patient_id)
    if fused["patient_risk_score"] < s.fusion_alert_threshold:
        return None

    # dedupe: enquanto houver alerta de fusão NÃO reconhecido, não repetir a cada
    # nova leitura — a equipe já foi notificada do risco elevado.
    existing = (table("alerts").select("id").eq("patient_id", patient_id)
                .eq("source_type", "fusion").is_("acknowledged_at", "null")
                .limit(1).execute())
    if existing.data:
        return None

    alert = {
        "patient_id": patient_id, "source_type": "fusion",
        "severity": "critical" if fused["patient_risk_score"] >= 0.85 else "warning",
        "title": f"Risco multimodal elevado: {fused['patient_risk_score']:.0%}",
        "details": fused,
    }
    table("alerts").insert(alert).execute()
    return alert
