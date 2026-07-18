"""Funções auxiliares compartilhadas pelas páginas do dashboard Streamlit.

Centraliza o processamento (com cache do Streamlit) de cada modalidade, para que
as páginas apenas consumam os resultados. Evita reprocessar vídeo/áudio a cada
interação.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# garante que o pacote src/ seja importável ao rodar `streamlit run app/Home.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from multimodal_monitor.alerts.alert_manager import Alert, AlertLevel, MODALITY_PT  # noqa: E402
from multimodal_monitor.config import OUTPUTS_DIR, SAMPLES_DIR, settings  # noqa: E402

# cores por chave interna (level em inglês) e rótulos de exibição em português
LEVEL_COLOR = {"INFO": "#16a34a", "WARNING": "#eab308", "CRITICAL": "#dc2626"}
LEVEL_PT = {"INFO": "INFORMATIVO", "WARNING": "ATENÇÃO", "CRITICAL": "CRÍTICO"}
RISK_COLOR = {"BAIXO": "#16a34a", "MODERADO": "#eab308", "ALTO": "#dc2626"}

# rótulos em português das regras de detecção (chave interna → texto exibido)
RULE_PT = {
    "fall": "Queda", "immobility": "Imobilidade", "posture": "Postura/amplitude",
    "safe_zone": "Zona segura", "movement_pattern": "Padrão de movimento",
    "zone_intrusion": "Área crítica", "unexpected_object": "Objeto na cena",
    "threshold": "Limite clínico", "zscore": "Variação (z-score)",
    "isolation_forest": "Isolation Forest", "dose_change": "Variação de dose",
    "discontinuation": "Descontinuação", "interaction": "Interação",
    "vocal_features": "Alteração vocal", "critical_terms": "Termos críticos",
    "sentiment": "Sentimento",
}

# rótulos em português dos sinais/métricas
METRIC_PT = {
    "hr": "FC", "spo2": "SpO₂", "resp": "Resp.", "sbp": "PA sistólica",
    "movement_index": "Índice de movimento",
}

# rótulos em português dos parâmetros acústicos (voz)
FEATURE_PT = {
    "duration_s": "Duração (s)",
    "f0_mean_hz": "F0 média (Hz)",
    "f0_std_hz": "F0 desvio (Hz)",
    "jitter_local_pct": "Jitter (%)",
    "shimmer_local_pct": "Shimmer (%)",
    "hnr_db": "HNR (dB)",
    "pause_ratio": "Razão de pausas",
    "speech_rate_syl_s": "Taxa de fala (síl/s)",
    "voiced_ratio": "Fração sonorizada",
}


def rule_pt(rule: str | None) -> str:
    return RULE_PT.get(rule, rule or "—")


def metric_pt(metric: str | None) -> str:
    return METRIC_PT.get(metric, metric or "—")

DEFAULT_VIDEO = SAMPLES_DIR / "video" / "patient_immobility_demo.mp4"
DEFAULT_AUDIO_CRITICAL = SAMPLES_DIR / "audio" / "consulta_critica.wav"
DEFAULT_AUDIO_NEUTRAL = SAMPLES_DIR / "audio" / "consulta_neutra.wav"


# ----------------------------- Sinais vitais ---------------------------------
@st.cache_data(show_spinner=False)
def get_vitals(seed: int = 42) -> pd.DataFrame:
    from multimodal_monitor.vitals.loaders import generate_synthetic_vitals

    return generate_synthetic_vitals(seed=seed, inject_anomalies=True)


@st.cache_data(show_spinner=False)
def get_vitals_alerts(seed: int = 42, patient_id: str = "P001") -> list[dict]:
    from multimodal_monitor.vitals.detectors import detect_all

    return [a.to_dict() for a in detect_all(get_vitals(seed), patient_id)]


@st.cache_data(show_spinner=False)
def get_prescriptions() -> pd.DataFrame:
    from multimodal_monitor.vitals.prescriptions import sample_prescriptions

    return sample_prescriptions()


@st.cache_data(show_spinner=False)
def get_prescription_alerts(patient_id: str = "P001") -> list[dict]:
    from multimodal_monitor.vitals.prescriptions import (
        detect_prescription_anomalies, sample_prescriptions,
    )

    return [a.to_dict() for a in detect_prescription_anomalies(sample_prescriptions(), patient_id)]


# --------------------------------- Vídeo -------------------------------------
@st.cache_data(show_spinner=True)
def process_video(video_path: str, patient_id: str = "P001",
                  file_sig: str | None = None) -> dict:
    """Processa o vídeo (pose + objetos/área crítica), gera o VÍDEO ANOTADO e
    devolve features, alertas, resumo e o caminho do anotado.

    ``file_sig`` (mtime+tamanho) entra na chave do cache: substituir o arquivo
    mantendo o nome invalida o resultado automaticamente.
    """
    from multimodal_monitor.video.anomaly import detect_video_anomalies
    from multimodal_monitor.video.movement_features import compute_movement_features
    from multimodal_monitor.video.object_detection import (
        detect_scene_anomalies, extract_object_detections,
    )
    from multimodal_monitor.video.pose_extractor import extract_pose_sequence
    from multimodal_monitor.video.report import annotate_video, build_video_report

    seq = extract_pose_sequence(video_path, stride=2)
    feats = compute_movement_features(seq)
    alerts = detect_video_anomalies(feats, patient_id)

    det_seq = extract_object_detections(video_path, stride=2)
    alerts = sorted(alerts + detect_scene_anomalies(det_seq, patient_id),
                    key=lambda a: a.timestamp)

    report = build_video_report(seq, feats, alerts)

    th = settings.scene
    annotated = OUTPUTS_DIR / f"annotated_{Path(video_path).stem}.mp4"
    annotate_video(video_path, seq, alerts, annotated, stride=2,
                   zone=(th.zone_x1, th.zone_y1, th.zone_x2, th.zone_y2),
                   zone_label=th.zone_name.upper())

    return {
        "features": feats.to_dict(orient="list"),
        "alerts": [a.to_dict() for a in alerts],
        "report": report,
        "objects": det_seq.class_counts(),
        "annotated": str(annotated),
    }


# --------------------------------- Áudio -------------------------------------
@st.cache_data(show_spinner=True)
def process_audio(audio_path: str, patient_id: str = "P001") -> dict:
    """Extrai features vocais, alerta vocal, transcrição (Azure) e análise de texto."""
    from multimodal_monitor.audio.acoustic_features import extract_acoustic_features
    from multimodal_monitor.audio.azure_speech import transcribe
    from multimodal_monitor.audio.azure_text import analyze_text, text_alerts
    from multimodal_monitor.audio.vocal_anomaly import score_vocal_fatigue, vocal_anomaly_alert

    feats = extract_acoustic_features(audio_path)
    score = score_vocal_fatigue(feats)
    vocal_alert = vocal_anomaly_alert(feats, patient_id)

    stt = transcribe(audio_path)
    if stt.configured and stt.text:
        text, transcript_source = stt.text, "Azure Speech to Text"
    else:
        # fallback offline p/ demonstrar a análise de texto sem chaves Azure
        text = ("Doutor estou com dor no peito e falta de ar tive tontura e quase desmaiei"
                if "critic" in Path(audio_path).name
                else "Bom dia doutor estou me sentindo bem hoje sem queixas")
        transcript_source = "roteiro local (Azure Speech não configurado)"

    analysis = analyze_text(text)
    talerts = text_alerts(analysis, patient_id)

    alerts = ([vocal_alert.to_dict()] if vocal_alert else []) + [a.to_dict() for a in talerts]
    return {
        "features": feats.to_dict(),
        "score": score["score"],
        "score_level": score["level"].label,
        "indicators": score["indicators"],
        "text": text,
        "transcript_source": transcript_source,
        "sentiment": analysis.sentiment,
        "sentiment_scores": analysis.sentiment_scores,
        "key_phrases": analysis.key_phrases,
        "critical_terms": analysis.critical_terms,
        "critical_terms_azure": analysis.critical_terms_azure,
        "azure_configured": analysis.configured,
        "alerts": alerts,
    }


# --------------------------------- Fusão -------------------------------------
def compute_patient_risk(all_alerts: list[dict], patient_id: str = "P001") -> dict:
    """Reconstrói Alert objects a partir dos dicts e calcula o risco."""
    from multimodal_monitor.fusion.risk_engine import compute_risk

    alerts = [_dict_to_alert(d) for d in all_alerts]
    return compute_risk(alerts, patient_id).to_dict()


def _dict_to_alert(d: dict) -> Alert:
    return Alert(
        level=AlertLevel[d["level"]],
        modality=d["modality"],
        message=d["message"],
        patient_id=d.get("patient_id", "P001"),
        timestamp=d.get("timestamp", 0.0),
        metric=d.get("metric"),
        value=d.get("value"),
        details=d.get("details", {}),
    )


def azure_status() -> tuple[bool, bool]:
    return settings.azure.speech_configured, settings.azure.language_configured
