"""Testes do gerenciador de alertas."""

from multimodal_monitor.alerts.alert_manager import Alert, AlertLevel, AlertManager


def test_levels_order_and_labels():
    assert AlertLevel.CRITICAL > AlertLevel.WARNING > AlertLevel.INFO
    assert AlertLevel.CRITICAL.label == "CRITICAL"
    assert AlertLevel.WARNING.emoji == "🟡"


def test_manager_sorts_by_severity_then_time():
    m = AlertManager()
    m.emit(AlertLevel.WARNING, "vitals", "w", timestamp=10)
    m.emit(AlertLevel.CRITICAL, "audio", "c", timestamp=50)
    m.emit(AlertLevel.INFO, "video", "i", timestamp=1)
    ordered = m.all()
    assert ordered[0].level == AlertLevel.CRITICAL
    assert ordered[-1].level == AlertLevel.INFO


def test_manager_filters_and_summary():
    m = AlertManager()
    m.emit(AlertLevel.CRITICAL, "vitals", "a", patient_id="P1")
    m.emit(AlertLevel.WARNING, "audio", "b", patient_id="P2")
    assert len(m.by_modality("vitals")) == 1
    assert len(m.by_patient("P2")) == 1
    assert m.max_level() == AlertLevel.CRITICAL
    assert m.summary()["CRITICAL"] == 1


def test_alert_to_dict_is_serializable():
    a = Alert(AlertLevel.INFO, "video", "ok", value=1.0)
    d = a.to_dict()
    assert d["level"] == "INFO"
    assert d["modality"] == "video"
