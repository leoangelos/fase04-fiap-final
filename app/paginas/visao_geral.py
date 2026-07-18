"""Página: Visão Geral — painel do hospital com TODOS os pacientes.

Responde "como está o hospital agora": censo de pacientes, risco multimodal de
cada um, alertas em aberto e últimas análises. Para o detalhe de um paciente
(upload, vitais, prescrições, histórico), use a página Pacientes.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import monitor_ui as ui  # noqa: F401 — garante src/ no sys.path

from hospital_ai.config import settings as hsettings

st.title("📊 Visão Geral do Hospital")
st.caption("Situação de TODOS os pacientes cadastrados — censo, risco multimodal e "
           "alertas em aberto. Detalhe individual na página 🗂️ Pacientes.")

if not hsettings.supabase_configured:
    st.warning(
        "**Supabase não configurado.** Preencha `SUPABASE_URL`, `SUPABASE_ANON_KEY` e "
        "`SUPABASE_SERVICE_KEY` no `.env` (README → *Camada hospitalar*). Enquanto isso, "
        "explore as **🔬 Análises de demonstração** no menu lateral.")
    st.stop()

from hospital_ai.db import table  # noqa: E402
from hospital_ai.services.fusion import compute_patient_risk  # noqa: E402
from hospital_ai.services.labels import (  # noqa: E402
    ADMISSION_PT, SOURCE_PT, title_pt,
)

SEV_ICON = {"info": "🟢", "warning": "🟡", "critical": "🔴"}


@st.cache_data(ttl=20, show_spinner=False)
def _carrega_painel() -> dict:
    pacientes = (table("patients").select("*").eq("active", True)
                 .order("created_at", desc=True).limit(50).execute().data)
    alertas = (table("alerts").select("id,patient_id,severity,title,source_type,created_at")
               .is_("acknowledged_at", "null").order("created_at", desc=True)
               .limit(100).execute().data)
    riscos = {p["id"]: compute_patient_risk(p["id"]) for p in pacientes}
    return {"pacientes": pacientes, "alertas": alertas, "riscos": riscos}


col_at, _ = st.columns([1, 5])
if col_at.button("🔄 Atualizar painel"):
    _carrega_painel.clear()
    st.rerun()

painel = _carrega_painel()
pacientes, alertas, riscos = painel["pacientes"], painel["alertas"], painel["riscos"]

if not pacientes:
    st.info("Nenhum paciente cadastrado. Cadastre na página **🗂️ Pacientes** ou rode "
            "`uv run python scripts/seed_hospital.py`.")
    st.stop()

# ── censo e indicadores ──────────────────────────────────────────────────────
abertos_por_pac: dict[str, list[dict]] = {}
for a in alertas:
    abertos_por_pac.setdefault(a["patient_id"], []).append(a)
criticos = [a for a in alertas if a["severity"] == "critical"]
internados = sum(1 for p in pacientes if p["admission_status"] in ("admitted", "icu"))
em_risco = sum(1 for p in pacientes if riscos[p["id"]]["patient_risk_score"] >= 0.7)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Pacientes ativos", len(pacientes))
k2.metric("Internados/UTI", internados)
k3.metric("🔴 Alertas críticos em aberto", len(criticos))
k4.metric("Pacientes em risco alto", em_risco)

st.divider()

# ── tabela de pacientes com risco ────────────────────────────────────────────
st.subheader("Pacientes")
linhas = []
for p in pacientes:
    r = riscos[p["id"]]
    n_ab = abertos_por_pac.get(p["id"], [])
    linhas.append({
        "Paciente": p["full_name"],
        "Prontuário": p["mrn"],
        "Situação": ADMISSION_PT.get(p["admission_status"], p["admission_status"]),
        "Risco multimodal": r["patient_risk_score"],
        "Alertas abertos": len(n_ab),
        "Fontes": ", ".join(SOURCE_PT.get(s, s) for s in r["contributions"]) or "—",
    })
linhas.sort(key=lambda x: -x["Risco multimodal"])
st.dataframe(
    linhas, width="stretch", hide_index=True,
    column_config={
        "Risco multimodal": st.column_config.ProgressColumn(
            "Risco multimodal", min_value=0.0, max_value=1.0, format="percent"),
    },
)

# ── gráfico de risco por paciente ────────────────────────────────────────────
if len(pacientes) > 1:
    fig = go.Figure(go.Bar(
        x=[l["Risco multimodal"] * 100 for l in linhas],
        y=[l["Paciente"] for l in linhas],
        orientation="h",
        marker_color=["#dc2626" if l["Risco multimodal"] >= 0.7
                      else "#eab308" if l["Risco multimodal"] >= 0.4
                      else "#16a34a" for l in linhas],
    ))
    fig.update_layout(height=90 + 46 * len(linhas), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Risco multimodal (0–100)", xaxis_range=[0, 100])
    st.plotly_chart(fig, width="stretch")

st.divider()

# ── alertas em aberto (todos os pacientes) ───────────────────────────────────
st.subheader(f"🔔 Alertas em aberto ({len(alertas)})")
nome_por_id = {p["id"]: p["full_name"] for p in pacientes}
if alertas:
    for a in alertas[:15]:
        st.write(f"{SEV_ICON.get(a['severity'], '·')} "
                 f"**{nome_por_id.get(a['patient_id'], 'Paciente')}** — "
                 f"{title_pt(a['title'])} · {a['created_at'][:16].replace('T', ' ')}")
    if len(alertas) > 15:
        st.caption(f"... e mais {len(alertas) - 15}. Dê ciência nos alertas na página "
                   "🗂️ Pacientes (seção *Alertas em aberto* de cada paciente).")
else:
    st.success("Nenhum alerta pendente de ciência. 🎉")
