"""Testes dos detectores de sinais vitais e prescrições.

Princípio: injetamos anomalias conhecidas e verificamos que viram alertas com o
nível e a métrica esperados, dentro da janela de tempo em que foram plantadas.
"""

from multimodal_monitor.alerts.alert_manager import AlertLevel
from multimodal_monitor.vitals.loaders import generate_synthetic_vitals
from multimodal_monitor.vitals.detectors import (
    detect_all,
    detect_multivariate_anomalies,
    detect_threshold_violations,
    detect_zscore_anomalies,
)
from multimodal_monitor.vitals.prescriptions import (
    detect_prescription_anomalies,
    sample_prescriptions,
)


def _in_any_window(ts, windows, tol=15.0):
    return any(w["start_s"] - tol <= ts <= w["end_s"] + tol for w in windows)


def test_synthetic_generator_marks_three_events():
    df = generate_synthetic_vitals(seed=1)
    assert len(df) == 600
    assert set(["t", "hr", "spo2", "resp", "sbp"]).issubset(df.columns)
    assert len(df.attrs["anomaly_windows"]) == 3


def test_threshold_detector_flags_injected_events():
    df = generate_synthetic_vitals(seed=1)
    alerts = detect_threshold_violations(df)
    assert alerts, "esperado ao menos um alerta de limite clínico"
    # a taquicardia injetada (~+70 bpm sobre 75) deve estourar hr_high=120
    hr_alerts = [a for a in alerts if a.metric == "hr"]
    assert hr_alerts
    assert any(a.level == AlertLevel.CRITICAL for a in hr_alerts)


def test_zscore_detector_catches_abrupt_change():
    df = generate_synthetic_vitals(seed=1)
    alerts = detect_zscore_anomalies(df)
    windows = df.attrs["anomaly_windows"]
    assert alerts
    # ao menos um alerta z-score deve cair dentro de uma janela injetada
    assert any(_in_any_window(a.timestamp, windows) for a in alerts)


def test_isolation_forest_returns_alerts():
    df = generate_synthetic_vitals(seed=1)
    alerts = detect_multivariate_anomalies(df)
    assert alerts, "Isolation Forest deveria sinalizar padrões raros"


def test_no_anomalies_when_disabled_is_quiet_on_thresholds():
    df = generate_synthetic_vitals(seed=2, inject_anomalies=False)
    # sem eventos plantados, limites clínicos não devem disparar
    assert detect_threshold_violations(df) == []


def test_detect_all_covers_injected_metrics():
    df = generate_synthetic_vitals(seed=1)
    metrics = {a.metric for a in detect_all(df)}
    # os três eventos envolvem hr, spo2 e sbp
    assert {"hr", "spo2", "sbp"} & metrics


def test_prescription_dose_change_and_interaction():
    alerts = detect_prescription_anomalies(sample_prescriptions())
    rules = {a.details.get("rule") for a in alerts}
    assert "dose_change" in rules       # warfarina 5→15
    assert "discontinuation" in rules   # warfarina some no dia 5
    assert "interaction" in rules       # warfarina+aspirina / enalapril+espironolactona
    # interação é crítica
    assert any(a.level == AlertLevel.CRITICAL for a in alerts if a.details.get("rule") == "interaction")
