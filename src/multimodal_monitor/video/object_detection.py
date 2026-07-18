"""Detecção de objetos e áreas críticas com YOLOv8 (detecção padrão, COCO).

Complementa a análise de pose e atende à parte do edital "YOLOv8 para detecção
de objetos e áreas críticas":

  - inventário de objetos da cena (classes COCO) por frame;
  - **zona crítica**: retângulo configurável (ex.: área restrita de acesso a
    medicamentos) — pessoa cuja base entra na zona gera alerta;
  - **objeto inesperado**: classe fora do esperado que persiste na cena.

Usa o peso ``yolov8n.pt`` (baixado automaticamente pelo ultralytics, como o de
pose). As regras trabalham sobre dataclasses simples, permitindo testes com
detecções sintéticas sem modelo/rede.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..alerts.alert_manager import Alert, AlertLevel
from ..config import SceneThresholds, settings

# Tradução PT-BR das classes COCO mais prováveis nas cenas (fallback: nome cru).
COCO_PT = {
    "person": "pessoa",
    "chair": "cadeira",
    "bench": "banco",
    "couch": "sofá",
    "bed": "cama",
    "bottle": "garrafa",
    "cup": "copo",
    "cell phone": "celular",
    "laptop": "notebook",
    "tv": "televisão",
    "backpack": "mochila",
    "handbag": "bolsa",
    "suitcase": "mala",
    "scissors": "tesoura",
    "knife": "faca",
    "clock": "relógio",
    "book": "livro",
    "car": "carro",
    "truck": "caminhão",
    "bicycle": "bicicleta",
    "motorcycle": "moto",
    "umbrella": "guarda-chuva",
    "potted plant": "vaso de planta",
    "skateboard": "skate",
    "kite": "pipa",
    "bird": "pássaro",
    "dog": "cachorro",
    "cat": "gato",
    "traffic light": "semáforo",
    "frisbee": "frisbee",
}


def class_pt(name: str) -> str:
    return COCO_PT.get(name, name)


@dataclass
class Detection:
    """Um objeto detectado em um frame."""

    class_name: str
    conf: float
    bbox: tuple[float, float, float, float]   # x1,y1,x2,y2 em pixels

    @property
    def base_point(self) -> tuple[float, float]:
        """Ponto de contato com o chão (centro da base da bbox)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


@dataclass
class FrameDetections:
    frame_idx: int
    time_s: float
    detections: list[Detection] = field(default_factory=list)


@dataclass
class DetectionSequence:
    """Detecções de objetos de um vídeo, com metadados."""

    frames: list[FrameDetections]
    fps: float
    width: int
    height: int
    source: str

    def __len__(self) -> int:
        return len(self.frames)

    def class_counts(self) -> dict[str, int]:
        """Nº de frames em que cada classe aparece (inventário para o relatório)."""
        counts: dict[str, int] = {}
        for f in self.frames:
            for name in {d.class_name for d in f.detections}:
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def extract_object_detections(
    video_path: str | Path,
    model_name: str | None = None,
    conf: float | None = None,
    max_frames: int | None = None,
    stride: int = 1,
    device: str | None = None,
) -> DetectionSequence:
    """Roda o YOLOv8 de detecção de objetos sobre o vídeo.

    Mesmo padrão de leitura do ``pose_extractor`` (stride/max_frames); devolve
    todas as detecções por frame para as regras de zona/objeto. O modelo padrão
    vem de ``settings.scene.model_name`` (yolov8s: classificação mais confiável).
    """
    import cv2
    from ultralytics import YOLO

    model_name = model_name or settings.scene.model_name
    conf = settings.scene.min_conf if conf is None else conf
    video_path = str(video_path)
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: list[FrameDetections] = []
    frame_idx = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            result = model.predict(frame, conf=conf, verbose=False, device=device)[0]
            frames.append(_parse_result(result, frame_idx, frame_idx / fps))
            processed += 1
            if max_frames and processed >= max_frames:
                break
        frame_idx += 1
    cap.release()

    return DetectionSequence(frames=frames, fps=fps, width=width, height=height, source=video_path)


def _parse_result(result, frame_idx: int, time_s: float) -> FrameDetections:
    boxes = getattr(result, "boxes", None)
    dets: list[Detection] = []
    if boxes is not None and len(boxes) > 0:
        names = result.names
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)
        for bb, cf, cl in zip(xyxy, confs, clses):
            dets.append(Detection(names[int(cl)], float(cf), tuple(map(float, bb))))
    return FrameDetections(frame_idx, time_s, dets)


def zone_rect_px(th: SceneThresholds, width: int, height: int) -> tuple[float, float, float, float]:
    """Retângulo da zona crítica em pixels a partir das frações da config."""
    return (th.zone_x1 * width, th.zone_y1 * height, th.zone_x2 * width, th.zone_y2 * height)


def detect_zone_intrusions(
    seq: DetectionSequence,
    patient_id: str = "P001",
    thresholds: SceneThresholds | None = None,
) -> list[Alert]:
    """Pessoa com a base dentro da zona crítica → alerta (dedupe de 3 s)."""
    th = thresholds or settings.scene
    zx1, zy1, zx2, zy2 = zone_rect_px(th, seq.width, seq.height)
    alerts: list[Alert] = []
    fired_until = -1.0
    for f in seq.frames:
        if f.time_s < fired_until:
            continue
        persons = [d for d in f.detections if d.class_name == "person" and d.conf >= th.min_conf]
        inside = [d for d in persons
                  if zx1 <= d.base_point[0] <= zx2 and zy1 <= d.base_point[1] <= zy2]
        if inside:
            fired_until = f.time_s + 3.0
            alerts.append(
                Alert(
                    level=AlertLevel.WARNING,
                    modality="video",
                    message=f"Pessoa dentro da área crítica ({th.zone_name})",
                    patient_id=patient_id,
                    timestamp=float(f.time_s),
                    metric="zone_intrusion",
                    value=float(len(inside)),
                    details={"rule": "zone_intrusion", "zone": th.zone_name},
                )
            )
    return alerts


def detect_unexpected_objects(
    seq: DetectionSequence,
    patient_id: str = "P001",
    thresholds: SceneThresholds | None = None,
) -> list[Alert]:
    """Classe fora de ``expected_classes`` persistente na cena → alerta INFO.

    Dois filtros de falso positivo (calibrados nos vídeos de demonstração):
      1. presença em ≥ ``min_object_frames`` frames (descarta glitches isolados);
      2. confiança MEDIANA da classe ≥ ``object_alert_conf`` — erros de
         classificação do modelo (ex.: guarda-chuva lido como skate) ficam com
         mediana baixa, enquanto objetos reais sustentam confiança alta.
    Emite um único alerta por classe (no primeiro instante em que apareceu).
    """
    th = thresholds or settings.scene
    expected = {c.lower() for c in th.expected_classes}
    first_seen: dict[str, float] = {}
    confs: dict[str, list[float]] = {}
    for f in seq.frames:
        vistos: dict[str, float] = {}
        for d in f.detections:
            if d.conf >= th.min_conf and d.class_name.lower() not in expected:
                vistos[d.class_name] = max(vistos.get(d.class_name, 0.0), d.conf)
        for name, conf in vistos.items():
            confs.setdefault(name, []).append(conf)
            first_seen.setdefault(name, f.time_s)

    alerts: list[Alert] = []
    for name, valores in confs.items():
        count = len(valores)
        mediana = sorted(valores)[count // 2]
        if count < th.min_object_frames or mediana < th.object_alert_conf:
            continue
        alerts.append(
            Alert(
                level=AlertLevel.INFO,
                modality="video",
                message=(f"Objeto inesperado na cena: {class_pt(name)} "
                         f"(presente em {count} quadros, confiança {mediana:.0%})"),
                patient_id=patient_id,
                timestamp=float(first_seen[name]),
                metric="unexpected_object",
                value=float(count),
                details={"rule": "unexpected_object", "class": name,
                         "median_conf": round(mediana, 2)},
            )
        )
    return sorted(alerts, key=lambda a: a.timestamp)


def detect_scene_anomalies(
    seq: DetectionSequence,
    patient_id: str = "P001",
    thresholds: SceneThresholds | None = None,
) -> list[Alert]:
    """Executa as regras de cena (zona crítica + objetos inesperados)."""
    th = thresholds or settings.scene
    alerts = detect_zone_intrusions(seq, patient_id, th) + detect_unexpected_objects(
        seq, patient_id, th
    )
    return sorted(alerts, key=lambda a: a.timestamp)
