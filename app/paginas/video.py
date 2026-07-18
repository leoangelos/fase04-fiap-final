"""Página: Vídeo — pose, features de movimento e anomalias."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

import monitor_ui as ui

st.title("🎥 Análise de Vídeo")
st.caption("Análise avulsa com vídeos de exemplo · YOLOv8-pose (queda, imobilidade, postura) "
           "+ YOLOv8 (objetos e área crítica) · Para analisar o vídeo de um paciente "
           "cadastrado, use 🏥 Hospital → Pacientes → Enviar mídia")

video_dir = ui.SAMPLES_DIR / "video"
videos = sorted(video_dir.glob("*.mp4")) if video_dir.exists() else []

if not videos:
    st.warning("Nenhum vídeo em `data/samples/video/`. Rode `python scripts/download_data.py`.")
    st.stop()

names = [v.name for v in videos]
default_idx = names.index("patient_immobility_demo.mp4") if "patient_immobility_demo.mp4" in names else 0
choice = st.selectbox("Vídeo clínico", names, index=default_idx)
video_path = video_dir / choice

col_a, col_b = st.columns([1, 1])
with col_a:
    st.video(str(video_path))

if st.button("▶️ Processar vídeo", type="primary"):
    # limpa o cache para reprocessar de verdade (o arquivo ou os detectores
    # podem ter mudado desde a última execução com o mesmo caminho)
    ui.process_video.clear()
    stat = video_path.stat()
    with st.spinner("Extraindo pose, detectando anomalias e gerando o vídeo anotado..."):
        result = ui.process_video(str(video_path),
                                  file_sig=f"{stat.st_mtime_ns}-{stat.st_size}")
    st.session_state["video_result"] = result

result = st.session_state.get("video_result")
if result and result["report"]["source"].endswith(choice):
    report = result["report"]
    with col_b:
        st.subheader("Resumo")
        st.json({
            "resolução": report["resolution"],
            "fps": report["fps"],
            "frames": report["frames_processed"],
            "% com pessoa": report["frames_with_person_pct"],
            "alertas": report["n_alerts"],
            "por regra": {ui.rule_pt(k): v for k, v in report["alerts_by_rule"].items()},
        })
        objects = result.get("objects", {})
        if objects:
            from multimodal_monitor.video.object_detection import class_pt

            st.markdown("**Objetos detectados (YOLOv8):** "
                        + ", ".join(f"{class_pt(k)} ({v} quadros)" for k, v in objects.items()))
            zona = ui.settings.scene
            st.caption(f"Área crítica monitorada: “{zona.zone_name}” — retângulo "
                       f"x∈[{zona.zone_x1:.0%}, {zona.zone_x2:.0%}], y∈[{zona.zone_y1:.0%}, {zona.zone_y2:.0%}] do quadro.")

    st.subheader("Índice de movimento ao longo do tempo")
    feats = result["features"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=feats["t"], y=feats["motion"], line=dict(color="#3b82f6"),
                             name="movimento"))
    for a in result["alerts"]:
        color = ui.LEVEL_COLOR[a["level"]]
        fig.add_vline(x=a["timestamp"], line=dict(color=color, dash="dot"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="t (s)", yaxis_title="movimento (norm.)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Eventos detectados")
    if result["alerts"]:
        st.dataframe(
            [{"Nível": a["level_pt"], "t (s)": round(a["timestamp"], 1),
              "Regra": ui.rule_pt(a.get("details", {}).get("rule")), "Evento": a["message"]}
             for a in result["alerts"]],
            width="stretch", hide_index=True,
        )
    else:
        st.success("Nenhum evento anômalo detectado.")

    annotated = result.get("annotated")
    if annotated and Path(annotated).exists():
        st.subheader("Vídeo anotado (esqueleto + área crítica + banner de alerta)")
        st.video(annotated)
else:
    with col_b:
        st.info("Clique em **Processar vídeo** para extrair a pose e detectar anomalias.")
