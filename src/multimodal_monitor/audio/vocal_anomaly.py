"""Score de alteração vocal a partir das features acústicas.

Compara cada feature com faixas de referência (``AudioThresholds``) e acumula um
score interpretável, mapeando para alertas. Não é diagnóstico — é uma triagem que
sinaliza voz compatível com fadiga/esforço para revisão da equipe.
"""

from __future__ import annotations

import math

from ..alerts.alert_manager import Alert, AlertLevel
from ..config import AudioThresholds, settings
from .acoustic_features import AcousticFeatures


def score_vocal_fatigue(
    feats: AcousticFeatures, thresholds: AudioThresholds | None = None
) -> dict:
    """Retorna dict com score 0–1, nível e a lista de indicadores acionados."""
    th = thresholds or settings.audio
    indicators: list[str] = []
    score = 0.0

    def ok(x: float) -> bool:
        return x is not None and not math.isnan(x)

    if ok(feats.jitter_local_pct) and feats.jitter_local_pct > th.jitter_local_pct:
        indicators.append(f"jitter elevado ({feats.jitter_local_pct:.2f}% > {th.jitter_local_pct}%)")
        score += 0.25
    if ok(feats.shimmer_local_pct) and feats.shimmer_local_pct > th.shimmer_local_pct:
        indicators.append(f"shimmer elevado ({feats.shimmer_local_pct:.2f}% > {th.shimmer_local_pct}%)")
        score += 0.25
    if ok(feats.hnr_db) and feats.hnr_db < th.hnr_db_min:
        indicators.append(f"HNR baixo ({feats.hnr_db:.1f} dB < {th.hnr_db_min} dB) — voz soprosa/ruidosa")
        score += 0.2
    if ok(feats.pause_ratio) and feats.pause_ratio > th.pause_ratio_max:
        indicators.append(f"muitas pausas ({feats.pause_ratio*100:.0f}% do tempo em silêncio)")
        score += 0.15
    if ok(feats.speech_rate_syl_s) and 0 < feats.speech_rate_syl_s < th.speech_rate_min:
        indicators.append(f"fala lentificada ({feats.speech_rate_syl_s:.1f} síl/s < {th.speech_rate_min})")
        score += 0.15

    score = min(score, 1.0)
    # Triagem por ACÚMULO: um único indicador fraco (score ~0.2) fica em INFO e não
    # gera alerta — reduz falso-positivo em vozes com variação natural. São precisos
    # ≥2 indicadores (score ≳0.35) para WARNING e sinais fortes/múltiplos para CRITICAL.
    if score >= 0.5:
        level = AlertLevel.CRITICAL
    elif score >= 0.35:
        level = AlertLevel.WARNING
    else:
        level = AlertLevel.INFO
    return {"score": round(score, 2), "level": level, "indicators": indicators}


def vocal_anomaly_alert(
    feats: AcousticFeatures, patient_id: str = "P001", thresholds: AudioThresholds | None = None
) -> Alert | None:
    """Gera um ``Alert`` quando a voz apresenta alteração relevante (nível ≥ WARNING).

    Indícios isolados/fracos (nível INFO) não geram alerta — só entram no relatório
    detalhado das features.
    """
    result = score_vocal_fatigue(feats, thresholds)
    if result["level"] == AlertLevel.INFO or not result["indicators"]:
        return None
    return Alert(
        level=result["level"],
        modality="audio",
        message=(
            f"Alteração vocal (índice {result['score']:.2f}): "
            + "; ".join(result["indicators"])
        ),
        patient_id=patient_id,
        metric="vocal_fatigue_score",
        value=result["score"],
        details={"indicators": result["indicators"], "rule": "vocal_features"},
    )
