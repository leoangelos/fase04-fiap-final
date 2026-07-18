"""Testes do módulo de áudio.

Cobrem a lógica que NÃO depende do Azure: detecção de termos críticos, score de
fadiga vocal a partir de features e a degradação graciosa quando as chaves faltam.
"""

from multimodal_monitor.alerts.alert_manager import AlertLevel
from multimodal_monitor.audio.acoustic_features import AcousticFeatures
from multimodal_monitor.audio.vocal_anomaly import score_vocal_fatigue, vocal_anomaly_alert
from multimodal_monitor.audio.azure_text import (
    analyze_text,
    crosscheck_terms_with_phrases,
    find_critical_terms,
    text_alerts,
)


def _healthy_features() -> AcousticFeatures:
    return AcousticFeatures(
        duration_s=15, f0_mean_hz=180, f0_std_hz=20, jitter_local_pct=0.8,
        shimmer_local_pct=3.0, hnr_db=20, pause_ratio=0.1, speech_rate_syl_s=4.0, voiced_ratio=0.8,
    )


def _fatigued_features() -> AcousticFeatures:
    return AcousticFeatures(
        duration_s=18, f0_mean_hz=180, f0_std_hz=30, jitter_local_pct=1.6,
        shimmer_local_pct=8.0, hnr_db=10, pause_ratio=0.55, speech_rate_syl_s=1.5, voiced_ratio=0.6,
    )


def test_healthy_voice_no_alert():
    result = score_vocal_fatigue(_healthy_features())
    assert result["level"] == AlertLevel.INFO
    assert vocal_anomaly_alert(_healthy_features()) is None


def test_fatigued_voice_is_critical():
    result = score_vocal_fatigue(_fatigued_features())
    assert result["level"] == AlertLevel.CRITICAL
    alert = vocal_anomaly_alert(_fatigued_features())
    assert alert is not None and alert.level == AlertLevel.CRITICAL
    assert len(result["indicators"]) >= 3


def test_single_weak_indicator_stays_info():
    # apenas shimmer levemente alto → 1 indicador → INFO, sem alerta
    feats = _healthy_features()
    feats.shimmer_local_pct = 4.0
    assert score_vocal_fatigue(feats)["level"] == AlertLevel.INFO
    assert vocal_anomaly_alert(feats) is None


def test_find_critical_terms_accent_insensitive():
    assert find_critical_terms("Estou com dor no peito") == ["dor no peito"]
    # sem acento também casa
    assert "tontura" in find_critical_terms("sinto tontura e falta de ar")
    assert find_critical_terms("me sinto bem, sem queixas") == []


def _sem_azure(monkeypatch):
    """Força o caminho offline mesmo com chaves reais no .env da máquina."""
    from types import SimpleNamespace

    from multimodal_monitor.audio import azure_text

    monkeypatch.setattr(
        azure_text, "settings",
        SimpleNamespace(azure=SimpleNamespace(language_configured=False)),
    )


def test_analyze_text_degrades_without_azure(monkeypatch):
    # sem chaves Azure, ainda detecta termos críticos localmente
    _sem_azure(monkeypatch)
    result = analyze_text("dor no peito e falta de ar")
    assert result.configured is False
    assert set(["dor no peito", "falta de ar"]).issubset(set(result.critical_terms))


def test_text_alerts_flags_critical_terms(monkeypatch):
    _sem_azure(monkeypatch)
    result = analyze_text("tenho dor torácica e sangramento")
    alerts = text_alerts(result)
    assert any(a.level == AlertLevel.CRITICAL for a in alerts)
    assert any(a.details.get("rule") == "critical_terms" for a in alerts)


def test_crosscheck_terms_with_azure_key_phrases():
    # termo contido em frase-chave do Azure (e vice-versa), sem acento
    confirmed = crosscheck_terms_with_phrases(
        ["dor no peito", "queda", "falta de ar"],
        ["forte dor no péito", "ar"],
    )
    assert confirmed == ["dor no peito"]
    assert crosscheck_terms_with_phrases(["tontura"], []) == []
