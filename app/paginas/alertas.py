"""Página: Alertas — central consolidada com linha do tempo e filtros."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

import monitor_ui as ui

st.title("🔔 Central de Alertas (demonstração)")
st.caption("Consolidação dos alertas das análises de demonstração desta seção · Os alertas dos "
           "pacientes reais ficam em 🏥 Hospital → Visão Geral / Pacientes")

# reúne alertas do estado da sessão (preenchido pela página inicial) + vitais/prescrições
alerts = dict(st.session_state.get("alerts", {}))
alerts.setdefault("vitals", ui.get_vitals_alerts())
alerts.setdefault("prescription", ui.get_prescription_alerts())
all_alerts = [a for alist in alerts.values() for a in alist]

if not all_alerts:
    st.info("Nenhum alerta ainda. Volte à página **Início** e clique em "
            "**Processar vídeo + áudio**.")
    st.stop()

# filtros (rótulos em português; valor interno em inglês)
c1, c2 = st.columns(2)
with c1:
    modalidades = sorted({a["modality"] for a in all_alerts})
    mods = st.multiselect("Modalidades", modalidades, default=modalidades,
                          format_func=lambda m: ui.MODALITY_PT.get(m, m))
with c2:
    niveis = ["CRITICAL", "WARNING", "INFO"]
    levels = st.multiselect("Níveis", niveis, default=niveis,
                            format_func=lambda l: ui.LEVEL_PT.get(l, l))
filtered = [a for a in all_alerts if a["modality"] in mods and a["level"] in levels]

# resumo
counts = {lvl: sum(1 for a in filtered if a["level"] == lvl) for lvl in ["CRITICAL", "WARNING", "INFO"]}
m1, m2, m3 = st.columns(3)
m1.metric("🔴 Críticos", counts["CRITICAL"])
m2.metric("🟡 Atenção", counts["WARNING"])
m3.metric("🟢 Informativos", counts["INFO"])

# linha do tempo
st.subheader("Linha do tempo")
if filtered:
    fig = px.scatter(
        x=[a["timestamp"] for a in filtered],
        y=[a["modality_pt"] for a in filtered],
        color=[a["level_pt"] for a in filtered],
        size=[10] * len(filtered),
        color_discrete_map={"CRÍTICO": "#dc2626", "ATENÇÃO": "#eab308", "INFORMATIVO": "#16a34a"},
        labels={"x": "t (s)", "y": "modalidade", "color": "nível"},
        hover_name=[a["message"] for a in filtered],
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

# lista ordenada por severidade
st.subheader("Alertas")
order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
for a in sorted(filtered, key=lambda x: (order[x["level"]], x["timestamp"])):
    render = {"CRITICAL": st.error, "WARNING": st.warning, "INFO": st.info}[a["level"]]
    render(f"**[{a['modality_pt']}]** {a['message']}  ·  t={a['timestamp']:.0f}s")
