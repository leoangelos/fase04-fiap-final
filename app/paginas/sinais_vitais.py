"""Página: Sinais Vitais — séries temporais, anomalias e simulação em tempo real."""

from __future__ import annotations

import time

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import monitor_ui as ui

st.title("📈 Sinais Vitais")
st.caption("Análise avulsa com série sintética/PhysioNet · limites clínicos + z-score móvel + "
           "Isolation Forest · Para registrar vitais de um paciente cadastrado (NEWS2 + "
           "baseline individual), use 🏥 Hospital → Pacientes")

df = ui.get_vitals()
alerts = ui.get_vitals_alerts()

SIGNALS = [("hr", "FC (bpm)", "#ef4444"), ("spo2", "SpO₂ (%)", "#3b82f6"),
           ("resp", "Resp (irpm)", "#10b981"), ("sbp", "PA sist. (mmHg)", "#a855f7")]

tab_static, tab_live = st.tabs(["Visão completa", "▶️ Simulação em tempo real"])

with tab_static:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=[s[1] for s in SIGNALS], vertical_spacing=0.06)
    for i, (col, label, color) in enumerate(SIGNALS, start=1):
        fig.add_trace(go.Scatter(x=df["t"], y=df[col], line=dict(color=color, width=1.5),
                                 name=label, showlegend=False), row=i, col=1)
    for a in alerts:
        fig.add_vline(x=a["timestamp"], line=dict(color="rgba(220,38,38,0.35)", dash="dot"))
    fig.update_layout(height=680, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Anomalias detectadas")
    if alerts:
        st.dataframe(
            [{"Nível": a["level_pt"], "Sinal": ui.metric_pt(a.get("metric")),
              "t (s)": round(a["timestamp"], 0), "Evento": a["message"]} for a in alerts],
            width="stretch", hide_index=True,
        )
    else:
        st.success("Nenhuma anomalia detectada.")

with tab_live:
    st.write("Reproduz a chegada dos sinais amostra a amostra; alertas aparecem no instante da detecção.")
    speed = st.select_slider("Velocidade", options=[10, 20, 40, 80], value=40)
    placeholder_chart = st.empty()
    placeholder_alert = st.empty()
    start = st.button("▶️ Iniciar simulação", type="primary")

    if start:
        from multimodal_monitor.vitals.simulator import VitalsStreamSimulator

        sim = VitalsStreamSimulator(df, window=60)
        xs, hr, spo2 = [], [], []
        fired: list[str] = []
        step = max(len(df) // 150, 1)
        for i, update in enumerate(sim.stream(realtime=False)):
            row = update["row"]
            xs.append(row["t"]); hr.append(row["hr"]); spo2.append(row["spo2"])
            for a in update["new_alerts"]:
                fired.append(f"🔴 t={a.timestamp:.0f}s — {a.message}")
            if i % step == 0 or update["new_alerts"]:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    subplot_titles=["FC (bpm)", "SpO₂ (%)"])
                fig.add_trace(go.Scatter(x=xs, y=hr, line=dict(color="#ef4444")), row=1, col=1)
                fig.add_trace(go.Scatter(x=xs, y=spo2, line=dict(color="#3b82f6")), row=2, col=1)
                fig.update_layout(height=440, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
                placeholder_chart.plotly_chart(fig, width="stretch")
                if fired:
                    placeholder_alert.error("  \n".join(fired[-5:]))
                time.sleep(1.0 / speed)
        st.success(f"Simulação concluída — {len(fired)} alertas disparados.")
