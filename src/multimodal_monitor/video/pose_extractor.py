"""Extração de pose corporal quadro a quadro com YOLOv8-pose.

YOLOv8-pose estima 17 keypoints (formato COCO) por pessoa detectada. Roda em CPU
ou Apple Silicon (MPS). Baixa o peso ``yolov8n-pose.pt`` automaticamente na
primeira execução (ultralytics cuida do download).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Índices dos 17 keypoints COCO usados pelo YOLOv8-pose.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


@dataclass
class FramePose:
    """Pose da pessoa principal em um frame."""

    frame_idx: int
    time_s: float
    keypoints: np.ndarray            # (17, 2) em pixels; NaN quando ausente
    confidences: np.ndarray          # (17,)
    bbox: tuple[float, float, float, float] | None = None  # x1,y1,x2,y2
    n_persons: int = 0


@dataclass
class PoseSequence:
    """Sequência de poses de um vídeo, com metadados."""

    poses: list[FramePose]
    fps: float
    width: int
    height: int
    source: str
    extra: dict = field(default_factory=dict)   # ex.: contagem de pessoas por frame

    def __len__(self) -> int:
        return len(self.poses)

    def keypoint_series(self, name: str) -> np.ndarray:
        """Array (T, 2) da trajetória de um keypoint ao longo do tempo."""
        i = KP[name]
        return np.array([p.keypoints[i] for p in self.poses])

    def times(self) -> np.ndarray:
        return np.array([p.time_s for p in self.poses])


def extract_pose_sequence(
    video_path: str | Path,
    model_name: str = "yolov8n-pose.pt",
    conf: float = 0.25,
    max_frames: int | None = None,
    stride: int = 1,
    device: str | None = None,
) -> PoseSequence:
    """Roda YOLOv8-pose sobre o vídeo e devolve a ``PoseSequence``.

    Args:
        stride: processa 1 a cada ``stride`` frames (acelera vídeos longos).
        max_frames: teto de frames processados (útil para demo).
        device: "cpu", "mps" ou None (autodetect do ultralytics).

    Mantém, por frame, a pessoa de maior bounding box (paciente em foco) e conta
    quantas pessoas apareceram (usado depois para "pessoa fora da zona segura").
    """
    import cv2
    from ultralytics import YOLO

    video_path = str(video_path)
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir o vídeo: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    poses: list[FramePose] = []
    frame_idx = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            result = model.predict(frame, conf=conf, verbose=False, device=device)[0]
            poses.append(_parse_result(result, frame_idx, frame_idx / fps))
            processed += 1
            if max_frames and processed >= max_frames:
                break
        frame_idx += 1
    cap.release()

    return PoseSequence(poses=poses, fps=fps, width=width, height=height, source=video_path)


def _parse_result(result, frame_idx: int, time_s: float) -> FramePose:
    """Extrai a pessoa de maior bbox de um resultado YOLO."""
    empty_kp = np.full((17, 2), np.nan)
    empty_conf = np.zeros(17)

    kps = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)
    if kps is None or kps.xy is None or len(kps.xy) == 0:
        return FramePose(frame_idx, time_s, empty_kp, empty_conf, None, 0)

    xy = kps.xy.cpu().numpy()          # (n_persons, 17, 2)
    conf = kps.conf.cpu().numpy() if kps.conf is not None else np.ones((len(xy), 17))
    n_persons = len(xy)

    # escolhe a maior bounding box
    areas = []
    xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else None
    if xyxy is not None:
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        main = int(np.argmax(areas))
        bbox = tuple(map(float, xyxy[main]))
    else:
        main, bbox = 0, None

    kp = xy[main].astype(float)
    c = conf[main].astype(float)
    kp[c < 0.01] = np.nan            # keypoints sem confiança viram NaN
    return FramePose(frame_idx, time_s, kp, c, bbox, n_persons)
