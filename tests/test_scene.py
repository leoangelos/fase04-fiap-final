"""Testes da detecção de objetos e áreas críticas usando detecções SINTÉTICAS.

Não dependem do modelo YOLO: construímos ``DetectionSequence`` artificiais e
verificamos as regras de zona crítica e objeto inesperado.
"""

from multimodal_monitor.config import SceneThresholds
from multimodal_monitor.video.object_detection import (
    Detection,
    DetectionSequence,
    FrameDetections,
    detect_scene_anomalies,
    detect_unexpected_objects,
    detect_zone_intrusions,
)

# Zona crítica de teste: terço esquerdo do quadro 1000x500.
TH = SceneThresholds(zone_x1=0.0, zone_y1=0.0, zone_x2=0.33, zone_y2=1.0,
                     expected_classes=("person",), min_object_frames=3, min_conf=0.35,
                     object_alert_conf=0.60)


def _person(x_center: float, conf: float = 0.9) -> Detection:
    """Pessoa com bbox de 60px de largura centrada em x_center, base em y=400."""
    return Detection("person", conf, (x_center - 30, 100.0, x_center + 30, 400.0))


def _seq(frames: list[list[Detection]], fps: float = 10.0) -> DetectionSequence:
    return DetectionSequence(
        frames=[FrameDetections(i, i / fps, dets) for i, dets in enumerate(frames)],
        fps=fps, width=1000, height=500, source="synthetic",
    )


def test_zone_intrusion_fires_when_person_enters_zone():
    # pessoa caminha da direita (fora) para dentro do terço esquerdo
    xs = [900, 800, 700, 600, 500, 400, 300, 200, 100, 100]
    seq = _seq([[_person(x)] for x in xs])
    alerts = detect_zone_intrusions(seq, "P001", TH)
    assert alerts, "esperado alerta de área crítica"
    assert all(a.details["rule"] == "zone_intrusion" for a in alerts)
    # dedupe: permanência contínua de <3s não gera vários alertas
    assert len(alerts) == 1


def test_no_zone_alert_when_person_stays_outside():
    seq = _seq([[_person(800)] for _ in range(10)])
    assert detect_zone_intrusions(seq, "P001", TH) == []


def test_unexpected_object_needs_persistence():
    chair = Detection("chair", 0.8, (500, 300, 600, 400))
    # cadeira em apenas 2 frames (< min_object_frames=3) → sem alerta
    seq = _seq([[_person(800), chair], [_person(800), chair], [_person(800)]])
    assert detect_unexpected_objects(seq, "P001", TH) == []
    # em 3 frames → 1 alerta INFO, com a classe nos detalhes
    seq = _seq([[_person(800), chair]] * 3)
    alerts = detect_unexpected_objects(seq, "P001", TH)
    assert len(alerts) == 1
    assert alerts[0].details["rule"] == "unexpected_object"
    assert alerts[0].details["class"] == "chair"
    assert alerts[0].details["median_conf"] == 0.8


def test_expected_classes_do_not_alert():
    seq = _seq([[_person(800)]] * 5)
    assert detect_unexpected_objects(seq, "P001", TH) == []


def test_low_confidence_class_is_filtered():
    # classe persistente porém com confiança mediana baixa (erro de classificação
    # do modelo, ex.: guarda-chuva lido como skate) → sem alerta
    fantasma = Detection("skateboard", 0.5, (500, 300, 600, 400))
    seq = _seq([[_person(800), fantasma]] * 6)
    assert detect_unexpected_objects(seq, "P001", TH) == []


def test_scene_anomalies_combines_rules_sorted():
    chair = Detection("chair", 0.8, (500, 300, 600, 400))
    frames = [[_person(900), chair], [_person(700), chair], [_person(200), chair]]
    alerts = detect_scene_anomalies(_seq(frames), "P001", TH)
    rules = {a.details["rule"] for a in alerts}
    assert rules == {"zone_intrusion", "unexpected_object"}
    assert [a.timestamp for a in alerts] == sorted(a.timestamp for a in alerts)
