"""Ingestão de sinais vitais: persiste a leitura e roda as camadas de detecção.

Compartilhado pela rota FastAPI e pela página de pacientes do dashboard:
  1. NEWS2 (baseline clínico determinístico, populacional);
  2. z-score em janela deslizante (baseline estatístico do próprio paciente);
  3. fusão multimodal (recalculada após cada ingestão).
"""

from __future__ import annotations

from ..db import table
from .anomaly import push_and_score
from .fusion import maybe_alert_fusion
from .labels import VITAL_PT
from .news2 import news2_score

VITAL_FIELDS = ("heart_rate", "spo2", "temperature", "systolic_bp",
                "diastolic_bp", "respiratory_rate")


def ingest_vitals(patient_id: str, readings: dict, measured_at: str,
                  source: str = "manual") -> dict:
    """Insere a leitura, roda NEWS2 + z-score e cria alertas relevantes."""
    row = {k: readings.get(k) for k in VITAL_FIELDS}
    table("vital_signs").insert(
        row | {"patient_id": patient_id, "measured_at": measured_at,
               "source": source}).execute()

    news2 = news2_score(**{k: row.get(k) for k in
                           ("respiratory_rate", "spo2", "systolic_bp",
                            "heart_rate", "temperature")})
    anomalies = push_and_score(patient_id, row)

    table("analysis_results").insert({
        "patient_id": patient_id, "analysis_type": "news2",
        "engine": "local_rules",
        "result": {"news2": news2, "zscore_anomalies": anomalies},
        "risk_score": news2["risk_score"],
    }).execute()

    alerts_created = 0
    # nível "info" fica só no analysis_result — alertas apenas p/ warning/critical
    if news2["severity"] in ("warning", "critical"):
        table("alerts").insert({
            "patient_id": patient_id, "source_type": "vital_signs",
            "severity": news2["severity"], "title": f"NEWS2 = {news2['total']}",
            "details": news2,
        }).execute()
        alerts_created += 1
    for anom in anomalies:
        metrica = VITAL_PT.get(anom["metric"], anom["metric"])
        table("alerts").insert({
            "patient_id": patient_id, "source_type": "vital_signs",
            "severity": "warning",
            "title": f"Desvio da linha de base em {metrica} (z={anom['zscore']})",
            "details": anom,
        }).execute()
        alerts_created += 1

    fusion_alert = maybe_alert_fusion(patient_id)
    return {"news2": news2, "anomalies": anomalies,
            "alerts_created": alerts_created, "fusion_alert": fusion_alert}
