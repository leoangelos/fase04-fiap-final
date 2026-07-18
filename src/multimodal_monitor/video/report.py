"""Geração de saídas do módulo de vídeo.

  - ``annotate_video``: reescreve o vídeo com skeleton da pose + banner de alertas,
    atendendo "relatórios automáticos indicando desvios".
  - ``build_video_report``: dict resumo (eventos, contagens) para o relatório e o
    dashboard.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..alerts.alert_manager import Alert
from .pose_extractor import KEYPOINT_NAMES, KP, PoseSequence

# pares de conexão do esqueleto (COCO)
SKELETON = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


def build_video_report(seq: PoseSequence, features, alerts: list[Alert]) -> dict:
    """Resumo estruturado do processamento de vídeo."""
    duration = seq.times()[-1] if len(seq) else 0.0
    detected = sum(1 for p in seq.poses if p.n_persons > 0)
    by_rule: dict[str, int] = {}
    for a in alerts:
        by_rule[a.details.get("rule", "?")] = by_rule.get(a.details.get("rule", "?"), 0) + 1
    return {
        "source": seq.source,
        "fps": round(seq.fps, 2),
        "resolution": f"{seq.width}x{seq.height}",
        "frames_processed": len(seq),
        "duration_s": round(float(duration), 2),
        "frames_with_person_pct": round(100 * detected / max(len(seq), 1), 1),
        "n_alerts": len(alerts),
        "alerts_by_rule": by_rule,
        "events": [a.to_dict() for a in alerts],
    }


def annotate_video(
    video_path: str | Path,
    seq: PoseSequence,
    alerts: list[Alert],
    out_path: str | Path,
    stride: int = 1,
    zone: tuple[float, float, float, float] | None = None,
    zone_label: str = "AREA CRITICA",
) -> Path:
    """Regrava o vídeo com esqueleto sobreposto e banner do alerta ativo.

    ``zone`` (x1,y1,x2,y2 em frações 0–1 da imagem) desenha a área crítica
    monitorada pela detecção de objetos.
    """
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or seq.fps or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (w, h))

    pose_by_frame = {p.frame_idx: p for p in seq.poses}
    # alertas indexados por segundo para exibir banner no instante certo
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            if zone is not None:
                _draw_zone(frame, zone, zone_label)
            pose = pose_by_frame.get(frame_idx)
            if pose is not None:
                _draw_skeleton(frame, pose.keypoints)
            _draw_active_alerts(frame, frame_idx / fps, alerts)
            writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return _to_h264(out_path)


def _to_h264(path: Path) -> Path:
    """Converte o mp4v do OpenCV para H.264 (navegadores não reproduzem mp4v).

    Usa o ffmpeg (pré-requisito do projeto); se indisponível, mantém o arquivo
    original — reproduzível em players de desktop, mas não no dashboard.
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return path
    tmp = path.with_suffix(".h264.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
    return path


def _draw_zone(frame, zone: tuple[float, float, float, float], label: str) -> None:
    """Retângulo tracejado translúcido da zona crítica configurada."""
    import cv2

    h, w = frame.shape[:2]
    x1, y1 = int(zone[0] * w), int(zone[1] * h)
    x2, y2 = int(zone[2] * w), int(zone[3] * h)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, label, (x1 + 6, min(y1 + 22, h - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)


def _draw_skeleton(frame, kp: np.ndarray) -> None:
    import cv2

    for a, b in SKELETON:
        pa, pb = kp[KP[a]], kp[KP[b]]
        if np.any(np.isnan([*pa, *pb])):
            continue
        cv2.line(frame, tuple(pa.astype(int)), tuple(pb.astype(int)), (0, 255, 0), 2)
    for i in range(len(KEYPOINT_NAMES)):
        if not np.any(np.isnan(kp[i])):
            cv2.circle(frame, tuple(kp[i].astype(int)), 3, (0, 200, 255), -1)


def _draw_active_alerts(frame, t: float, alerts: list[Alert], window: float = 2.0) -> None:
    import cv2

    active = [a for a in alerts if a.timestamp <= t <= a.timestamp + window]
    if not active:
        return
    top = max(active, key=lambda a: a.level)
    color = {0: (0, 180, 0), 1: (0, 200, 255), 2: (0, 0, 255)}[int(top.level)]
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 40), color, -1)
    cv2.putText(
        frame, f"{top.level.pt}: {top.message[:60]}", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
