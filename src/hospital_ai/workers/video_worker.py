"""Worker de vídeo: reutiliza o pipeline de visão da ``multimodal_monitor``.

Por asset (sessão de fisioterapia / cirurgia / monitoramento):
  1. YOLOv8-pose → features de movimento → detectores (queda, imobilidade,
     desvio postural, padrão de movimentação).
  2. YOLOv8 objetos → área crítica e objetos inesperados na cena.
  3. Métricas-resumo de movimento comparadas ao BASELINE HISTÓRICO do próprio
     paciente (sessões anteriores em ``analysis_results``) — desvios > 60%
     entram no resultado como ``baseline_deviations``.

Grava ``analysis_type='pose_anomaly'`` (eventos + risk) e
``'movement_metrics'`` (métricas p/ baseline futuro); eventos WARNING/CRITICAL
viram ``alerts``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..db import table
from .common import (
    create_alert, download_asset, save_result, set_asset_status, upload_sibling,
)

_LEVEL_TO_SEVERITY = {"INFO": "info", "WARNING": "warning", "CRITICAL": "critical"}
_LEVEL_RISK = {"INFO": 0.2, "WARNING": 0.6, "CRITICAL": 0.9}


def _movement_summary(features) -> dict:
    """Métricas-resumo da sessão (base de comparação entre sessões)."""
    motion = features["motion"].dropna()
    knee = features[["knee_left", "knee_right"]].stack().dropna()
    return {
        "frames": int(len(features)),
        "valid_pct": round(100 * float(features["valid"].mean()), 1),
        "mean_motion": round(float(motion.mean()), 5) if len(motion) else None,
        "p95_motion": round(float(motion.quantile(0.95)), 5) if len(motion) else None,
        "knee_rom_deg": round(float(knee.max() - knee.min()), 1) if len(knee) else None,
    }


def _baseline(patient_id: str, exclude_asset: str) -> dict | None:
    """Média das métricas das últimas sessões do MESMO paciente."""
    res = (table("analysis_results").select("result,media_asset_id")
           .eq("patient_id", patient_id).eq("analysis_type", "movement_metrics")
           .order("created_at", desc=True).limit(5).execute())
    rows = [r["result"] for r in res.data
            if r.get("media_asset_id") != exclude_asset
            and r["result"].get("mean_motion") is not None]
    if not rows:
        return None
    keys = ("mean_motion", "p95_motion", "knee_rom_deg")
    return {k: float(np.mean([r[k] for r in rows if r.get(k) is not None]))
            for k in keys if any(r.get(k) is not None for r in rows)}


def handle(payload: dict) -> None:
    from multimodal_monitor.video.anomaly import detect_video_anomalies
    from multimodal_monitor.video.movement_features import compute_movement_features
    from multimodal_monitor.video.object_detection import (
        detect_scene_anomalies, extract_object_detections,
    )
    from multimodal_monitor.video.pose_extractor import extract_pose_sequence

    asset_id, patient_id = payload["asset_id"], payload["patient_id"]
    set_asset_status(asset_id, "processing")
    path = download_asset(payload["storage_path"])
    try:
        # 1) pose + regras de movimento
        seq = extract_pose_sequence(path, stride=2)
        features = compute_movement_features(seq)
        alerts = detect_video_anomalies(features, patient_id)

        # 2) objetos + área crítica
        det_seq = extract_object_detections(path, stride=2)
        alerts += detect_scene_anomalies(det_seq, patient_id)

        events = [a.to_dict() for a in sorted(alerts, key=lambda x: x.timestamp)]
        risk = max((_LEVEL_RISK[a.level.label] for a in alerts), default=0.05)

        # vídeo anotado (esqueleto + área crítica + banner, H.264) armazenado ao
        # lado do original no Storage — o cartão da mídia exibe via signed URL
        annotated_path = None
        try:
            import os
            import tempfile

            from multimodal_monitor.config import settings as mm_settings
            from multimodal_monitor.video.report import annotate_video

            th = mm_settings.scene
            # mkstemp devolve um fd ABERTO — fechar já, senão o Windows bloqueia
            # o os.replace do transcode H.264 (WinError 32)
            fd, tmp_name = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            tmp_annotated = Path(tmp_name)
            annotate_video(path, seq, alerts, tmp_annotated, stride=2,
                           zone=(th.zone_x1, th.zone_y1, th.zone_x2, th.zone_y2),
                           zone_label=th.zone_name.upper())
            annotated_path = upload_sibling(payload["storage_path"], tmp_annotated,
                                            "_annotated", "video/mp4")
            tmp_annotated.unlink(missing_ok=True)
        except Exception as exc:  # anotação é acessória: não derruba a análise
            annotated_path = None
            print(f"[video_worker] anotação falhou (análise segue válida): {exc}")

        save_result(patient_id, asset_id, "pose_anomaly", "yolov8",
                    {"events": events, "objects": det_seq.class_counts(),
                     "annotated_storage_path": annotated_path}, risk)

        # 3) métricas da sessão vs. baseline histórico do paciente
        summary = _movement_summary(features)
        base = _baseline(patient_id, exclude_asset=asset_id)
        deviations = {}
        if base and summary.get("mean_motion") is not None:
            for k, ref in base.items():
                cur = summary.get(k)
                if cur is not None and ref > 1e-9:
                    dev = abs(cur - ref) / ref
                    if dev > 0.6:
                        deviations[k] = round(dev, 2)
        summary["baseline_deviations"] = deviations
        save_result(patient_id, asset_id, "movement_metrics", "yolov8_pose",
                    summary, None)

        for a in alerts:
            severity = _LEVEL_TO_SEVERITY[a.level.label]
            if severity in ("warning", "critical"):
                create_alert(patient_id, "video", severity, a.message,
                             {"t_s": a.timestamp, **a.details}, asset_id)
        if deviations:
            create_alert(patient_id, "video", "warning",
                         "Desvio do padrão motor vs. baseline do paciente",
                         {"deviations": deviations, "baseline": base,
                          "current": summary}, asset_id)

        set_asset_status(asset_id, "done")
    except Exception as exc:
        set_asset_status(asset_id, "failed", str(exc)[:500])
        raise
    finally:
        path.unlink(missing_ok=True)
