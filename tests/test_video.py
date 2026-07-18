"""Testes do módulo de vídeo usando poses SINTÉTICAS.

Não dependem do modelo YOLO (que exige download/inferência): construímos
``PoseSequence`` artificiais e verificamos as features e os detectores.
"""

import numpy as np

from multimodal_monitor.video.pose_extractor import KP, FramePose, PoseSequence
from multimodal_monitor.video.movement_features import compute_movement_features
from multimodal_monitor.video.anomaly import detect_falls, detect_immobility, detect_video_anomalies
from multimodal_monitor.config import settings


def _standing_keypoints(hip_y: float = 0.5, jitter: float = 0.0, height: int = 720) -> np.ndarray:
    """Keypoints de uma pessoa em pé, centrada, com quadril em hip_y (fração da altura)."""
    kp = np.full((17, 2), np.nan)
    cx = 360.0
    hy = hip_y * height
    r = np.random.default_rng(0)
    j = (lambda: r.normal(0, jitter) if jitter else 0.0)
    kp[KP["left_shoulder"]] = [cx - 40, hy - 200 + j()]
    kp[KP["right_shoulder"]] = [cx + 40, hy - 200 + j()]
    kp[KP["left_hip"]] = [cx - 30, hy + j()]
    kp[KP["right_hip"]] = [cx + 30, hy + j()]
    kp[KP["left_knee"]] = [cx - 30, hy + 120]
    kp[KP["right_knee"]] = [cx + 30, hy + 120]
    kp[KP["left_ankle"]] = [cx - 30, hy + 240]
    kp[KP["right_ankle"]] = [cx + 30, hy + 240]
    kp[KP["left_elbow"]] = [cx - 60, hy - 120]
    kp[KP["right_elbow"]] = [cx + 60, hy - 120]
    kp[KP["left_wrist"]] = [cx - 70, hy - 40]
    kp[KP["right_wrist"]] = [cx + 70, hy - 40]
    return kp


def _make_sequence(hip_ys, fps=10.0, height=720, jitter=0.0) -> PoseSequence:
    poses = []
    for i, hy in enumerate(hip_ys):
        kp = _standing_keypoints(hy, jitter=jitter, height=height)
        poses.append(FramePose(i, i / fps, kp, np.ones(17), (300, 100, 420, 700), 1))
    return PoseSequence(poses, fps, 720, height, "synthetic")


def test_features_have_expected_columns_and_validity():
    seq = _make_sequence([0.5] * 20)
    feats = compute_movement_features(seq)
    for c in ("t", "valid", "knee_left", "hip_y_norm", "motion"):
        assert c in feats.columns
    assert feats["valid"].all()          # pessoa sempre bem detectada


def test_immobility_detected_when_still():
    # 12s completamente parado a 10fps → deve exceder immobility_seconds (5s)
    seq = _make_sequence([0.5] * 120)
    feats = compute_movement_features(seq)
    # injeta um pouco de movimento no começo para haver mediana de referência > 0
    feats.loc[:5, "motion"] = 0.2
    alerts = detect_immobility(feats, settings.video, "P001")
    assert any(a.details["rule"] == "immobility" for a in alerts)


def test_fall_detected_on_hip_drop():
    # quadril estável e depois cai bruscamente (y aumenta) → queda
    hip_ys = [0.4] * 15 + [0.8] * 15
    seq = _make_sequence(hip_ys)
    feats = compute_movement_features(seq)
    alerts = detect_falls(feats, settings.video, "P001")
    assert alerts, "esperado alerta de queda"
    assert alerts[0].level.label == "CRITICAL"


def test_no_alerts_on_stable_moving_person():
    # pessoa presente, quadril estável, movimento moderado constante → sem queda/imobilidade
    rng = np.random.default_rng(1)
    hip_ys = 0.5 + rng.normal(0, 0.002, 40)
    seq = _make_sequence(hip_ys, jitter=8.0)
    feats = compute_movement_features(seq)
    alerts = [a for a in detect_video_anomalies(feats, "P001")
              if a.details.get("rule") in ("fall", "immobility")]
    assert alerts == []
