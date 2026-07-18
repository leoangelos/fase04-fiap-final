"""Página inicial — visão geral do paciente e índice de risco fundido."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import monitor_ui as ui

st.title("🧪 Demonstração do Pipeline Multimodal")
st.caption("FIAP 8IADT — Fase 4 · Fusão de vídeo, áudio e sinais vitais com detecção de anomalias")
st.info("Esta seção processa **dados de exemplo** (`data/samples/`, paciente fictício) para "
        "demonstrar cada motor de análise isoladamente — **não está vinculada aos pacientes "
        "cadastrados**. Para o fluxo hospitalar real (upload por paciente, histórico e "
        "alertas persistidos), use a seção **🏥 Hospital** no menu.")

if "alerts" not in st.session_state:
    st.session_state.alerts = {}


def set_alerts(modality: str, alerts: list[dict]) -> None:
    st.session_state.alerts[modality] = alerts


# ----- barra lateral: status Azure e controles -----
speech_ok, lang_ok = ui.azure_status()
with st.sidebar:
    st.header("Configuração")
    st.markdown(
        f"**Azure Fala:** {'🟢 conectado' if speech_ok else '🔴 não configurado'}  \n"
        f"**Azure Linguagem:** {'🟢 conectado' if lang_ok else '🔴 não configurado'}"
    )
    if not (speech_ok and lang_ok):
        st.info("Sem chaves Azure, a análise de texto usa correspondência local de "
                "termos críticos (veja o README para configurar o `.env`).")
    st.divider()
    processar_midia = st.button("▶️ Processar vídeo + áudio de exemplo", type="primary")
    st.caption("Sinais vitais e prescrições são processados automaticamente.")

# ----- sinais vitais + prescrições (rápidos, sempre) -----
set_alerts("vitals", ui.get_vitals_alerts())
set_alerts("prescription", ui.get_prescription_alerts())

# ----- vídeo + áudio (sob demanda) -----
if processar_midia:
    if ui.DEFAULT_VIDEO.exists():
        with st.spinner("Processando vídeo (YOLOv8-pose)..."):
            set_alerts("video", ui.process_video(str(ui.DEFAULT_VIDEO))["alerts"])
    else:
        st.warning("Vídeo de exemplo ausente — rode `python scripts/download_data.py`.")
    if ui.DEFAULT_AUDIO_CRITICAL.exists():
        with st.spinner("Processando áudio (features vocais + análise de texto)..."):
            set_alerts("audio", ui.process_audio(str(ui.DEFAULT_AUDIO_CRITICAL))["alerts"])
    else:
        st.warning("Áudio de exemplo ausente — rode `python scripts/generate_audio_samples.py`.")

# ----- fusão -----
all_alerts = [a for alist in st.session_state.alerts.values() for a in alist]
risk = ui.compute_patient_risk(all_alerts)

# ----- layout principal -----
col1, col2 = st.columns([1, 1.4])

with col1:
    st.subheader("Índice de risco do paciente")
    color = ui.RISK_COLOR[risk["level"]]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk["score"],
        number={"suffix": "/100", "font": {"size": 44}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 25], "color": "#dcfce7"},
                {"range": [25, 60], "color": "#fef9c3"},
                {"range": [60, 100], "color": "#fee2e2"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        f"<h3 style='text-align:center;color:{color}'>Nível: {risk['level']}</h3>",
        unsafe_allow_html=True,
    )

with col2:
    st.subheader("Contribuição por modalidade")
    per_mod = risk["per_modality"]
    if per_mod:
        bar = go.Figure(go.Bar(
            x=list(per_mod.values()),
            y=[ui.MODALITY_PT.get(k, k) for k in per_mod],
            orientation="h",
            marker_color="#3b82f6",
        ))
        bar.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Risco parcial (0–100)")
        st.plotly_chart(bar, width="stretch")
    st.metric("Total de alertas", risk["n_alerts"],
              f"{sum(1 for a in all_alerts if a['level']=='CRITICAL')} críticos",
              delta_color="inverse")

st.divider()

# ----- alertas críticos -----
st.subheader("🔴 Alertas críticos")
crit = [a for a in all_alerts if a["level"] == "CRITICAL"]
if crit:
    for a in sorted(crit, key=lambda x: x["modality"]):
        st.error(f"**[{a['modality_pt']}]** {a['message']}  ·  t={a['timestamp']:.0f}s")
else:
    st.success("Nenhum alerta crítico no momento.")

st.divider()
st.caption("Use o menu lateral para explorar cada modalidade em detalhe. "
           "Projeto acadêmico — dados públicos/simulados; não constitui diagnóstico médico.")
