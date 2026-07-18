"""Pipeline multimodal ponta a ponta (CLI).

Processa as quatro modalidades para um paciente, funde os resultados em um índice
de risco e gera um relatório consolidado em Markdown. Módulos indisponíveis
(ex.: vídeo sem arquivo, Azure sem chaves) são pulados com aviso — o pipeline
nunca quebra por falta de um insumo.

Uso:
    uv run python scripts/run_pipeline.py
    uv run python scripts/run_pipeline.py --patient P002 --no-video
"""

from __future__ import annotations

import argparse
from pathlib import Path

from multimodal_monitor.alerts.alert_manager import AlertManager
from multimodal_monitor.config import OUTPUTS_DIR, SAMPLES_DIR, settings
from multimodal_monitor.fusion.risk_engine import compute_risk
from multimodal_monitor.reporting.generator import build_markdown_report, save_report


def run_vitals(mgr: AlertManager, patient_id: str) -> None:
    from multimodal_monitor.vitals.detectors import detect_all
    from multimodal_monitor.vitals.loaders import load_vitals
    from multimodal_monitor.vitals.prescriptions import (
        detect_prescription_anomalies, sample_prescriptions,
    )

    print("• Sinais vitais: gerando série e detectando anomalias...")
    df = load_vitals(prefer_real=False, inject_anomalies=True)
    mgr.extend(detect_all(df, patient_id))

    print("• Prescrições: avaliando evolução...")
    mgr.extend(detect_prescription_anomalies(sample_prescriptions(), patient_id))


def run_video(mgr: AlertManager, patient_id: str, video_path: Path, annotate: bool = True) -> None:
    if not video_path.exists():
        print(f"• Vídeo: '{video_path.name}' não encontrado — pulando "
              "(rode scripts/download_data.py).")
        return
    from multimodal_monitor.video.anomaly import detect_video_anomalies
    from multimodal_monitor.video.movement_features import compute_movement_features
    from multimodal_monitor.video.object_detection import (
        detect_scene_anomalies, extract_object_detections,
    )
    from multimodal_monitor.video.pose_extractor import extract_pose_sequence
    from multimodal_monitor.video.report import annotate_video

    print(f"• Vídeo: processando pose em '{video_path.name}' (YOLOv8-pose)...")
    seq = extract_pose_sequence(video_path, stride=2)
    feats = compute_movement_features(seq)
    video_alerts = detect_video_anomalies(feats, patient_id)

    print("• Vídeo: detectando objetos e área crítica (YOLOv8)...")
    det_seq = extract_object_detections(video_path, stride=2)
    objects = det_seq.class_counts()
    if objects:
        print("  objetos na cena:", ", ".join(f"{k}×{v} frames" for k, v in objects.items()))
    video_alerts += detect_scene_anomalies(det_seq, patient_id)
    mgr.extend(video_alerts)

    if annotate:
        th = settings.scene
        out = OUTPUTS_DIR / f"annotated_{video_path.stem}.mp4"
        annotate_video(video_path, seq, video_alerts, out, stride=2,
                       zone=(th.zone_x1, th.zone_y1, th.zone_x2, th.zone_y2),
                       zone_label=th.zone_name.upper())
        print(f"  vídeo anotado salvo em: {out}")


def run_audio(mgr: AlertManager, patient_id: str, audio_path: Path) -> None:
    if not audio_path.exists():
        print(f"• Áudio: '{audio_path.name}' não encontrado — pulando "
              "(rode scripts/generate_audio_samples.py).")
        return
    from multimodal_monitor.audio.acoustic_features import extract_acoustic_features
    from multimodal_monitor.audio.azure_speech import transcribe
    from multimodal_monitor.audio.azure_text import analyze_text, text_alerts
    from multimodal_monitor.audio.vocal_anomaly import vocal_anomaly_alert

    print(f"• Áudio: extraindo features vocais de '{audio_path.name}'...")
    feats = extract_acoustic_features(audio_path)
    if (alert := vocal_anomaly_alert(feats, patient_id)) is not None:
        mgr.add(alert)

    result = transcribe(audio_path)
    if result.configured and result.text:
        print(f"  transcrição (Azure): “{result.text[:80]}...”")
        text = result.text
    else:
        print("  Azure Speech indisponível — usando termos críticos locais sobre roteiro.")
        text = ("Doutor estou com dor no peito e falta de ar tive tontura e quase desmaiei"
                if "critic" in audio_path.name else "Bom dia estou bem sem queixas")
    mgr.extend(text_alerts(analyze_text(text), patient_id))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline multimodal de monitoramento")
    parser.add_argument("--patient", default="P001")
    parser.add_argument("--video", default=str(SAMPLES_DIR / "video" / "patient_immobility_demo.mp4"))
    parser.add_argument("--audio", default=str(SAMPLES_DIR / "audio" / "consulta_critica.wav"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--no-annotate", action="store_true",
                        help="não gerar o vídeo anotado (mais rápido)")
    parser.add_argument("--out", default=str(OUTPUTS_DIR / "relatorio_consolidado.md"))
    args = parser.parse_args()

    print(f"\n=== Monitoramento multimodal — paciente {args.patient} ===")
    print(f"Azure Speech: {'OK' if settings.azure.speech_configured else 'não configurado'} | "
          f"Azure Language: {'OK' if settings.azure.language_configured else 'não configurado'}\n")

    mgr = AlertManager()
    run_vitals(mgr, args.patient)
    if not args.no_video:
        run_video(mgr, args.patient, Path(args.video), annotate=not args.no_annotate)
    if not args.no_audio:
        run_audio(mgr, args.patient, Path(args.audio))

    alerts = mgr.all()
    risk = compute_risk(alerts, args.patient)

    print(f"\n→ Índice de risco: {risk.score:.0f}/100 ({risk.level_label}) | "
          f"{risk.n_alerts} alertas ({len(mgr.critical())} críticos)")

    sources = []
    if not args.no_video and Path(args.video).exists():
        sources.append(Path(args.video).name)
    if not args.no_audio and Path(args.audio).exists():
        sources.append(Path(args.audio).name)
    md = build_markdown_report(alerts, risk, args.patient, context={"sources": ", ".join(sources)})
    out = save_report(md, args.out)
    print(f"→ Relatório consolidado salvo em: {out}")


if __name__ == "__main__":
    main()
