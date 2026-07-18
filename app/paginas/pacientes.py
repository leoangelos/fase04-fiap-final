"""Página: Pacientes — gestão hospitalar (Supabase).

Cliente de demonstração da camada hospitalar: cadastro de pacientes, upload de
mídias multimodais por paciente (fila de análise), sinais vitais com NEWS2 +
z-score, prescrições com checagem de interação e histórico unificado.
Em produção, o caminho equivalente é a API FastAPI (``hospital_ai.main``).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import plotly.graph_objects as go
import streamlit as st

import monitor_ui as ui  # noqa: F401 — garante src/ no sys.path

from hospital_ai.config import settings as hsettings

st.title("🗂️ Pacientes")
st.caption("Gestão hospitalar: cadastro, upload multimodal por paciente, "
           "sinais vitais (NEWS2 + z-score) e histórico unificado — Supabase")

if not hsettings.supabase_configured:
    st.warning(
        "**Supabase não configurado.** Preencha `SUPABASE_URL`, `SUPABASE_ANON_KEY` "
        "e `SUPABASE_SERVICE_KEY` no `.env` (ver README, seção *Camada hospitalar*). "
        "O restante do dashboard segue funcionando sem esta página.")
    st.stop()

from hospital_ai.db import service_client, table  # noqa: E402
from hospital_ai.services.fusion import compute_patient_risk  # noqa: E402
from hospital_ai.services.interactions import check_drug_interactions  # noqa: E402
from hospital_ai.services.vitals_ingest import ingest_vitals  # noqa: E402
from hospital_ai.services import jobs  # noqa: E402
from hospital_ai.services.labels import (  # noqa: E402
    ADMISSION_PT, ANALYSIS_PT, ENGINE_PT, MODALITY_ICON, MODALITY_PT,
    NEWS2_PART_PT, SENTIMENT_PT, SOURCE_PT, STATUS_PT, VITAL_PT, title_pt,
)
from hospital_ai.api.media import BUCKETS, JOB_BY_MODALITY  # noqa: E402

SEV_ICON = {"info": "🟢", "warning": "🟡", "critical": "🔴"}

# MIME canônico por extensão — navegadores reportam variantes inconsistentes
# (ex.: Windows envia .m4a como audio/x-m4a), então normalizamos no cliente.
MIME_BY_EXT = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
    ".ogg": "audio/ogg", ".mp4": "video/mp4", ".mov": "video/quicktime",
    ".webm": "video/webm", ".txt": "text/plain", ".pdf": "application/pdf",
}


def _quando(iso: str) -> str:
    return iso[:16].replace("T", " ")


def render_analysis(a: dict) -> None:
    """Apresenta um ``analysis_result`` em linguagem clínica, por tipo."""
    tipo, r = a["analysis_type"], a.get("result") or {}
    rotulo = ANALYSIS_PT.get(tipo, tipo)
    motor = ENGINE_PT.get(a.get("engine"), a.get("engine", ""))
    st.markdown(f"**{rotulo}**  ·  {motor}  ·  {_quando(a['created_at'])}")

    if tipo == "transcricao":
        st.info("🗣️ “" + r.get("text", "(vazia)") + "”")

    elif tipo == "disartria":
        score = float(a.get("risk_score") or 0)
        cor = "#dc2626" if score >= 0.5 else "#eab308" if score >= 0.35 else "#16a34a"
        rotulo_score = ("alterações relevantes" if score >= 0.5
                        else "indícios moderados" if score >= 0.35 else "sem alteração relevante")
        st.markdown(f"Índice de alteração vocal: "
                    f"<span style='color:{cor};font-weight:700'>{score:.2f}</span> — {rotulo_score}",
                    unsafe_allow_html=True)
        for ind in r.get("indicators", []):
            st.write("• ", ind)
        feats = r.get("features", {})
        if feats:
            with st.popover("Parâmetros acústicos medidos"):
                st.dataframe(
                    [{"Parâmetro": ui.FEATURE_PT.get(k, k),
                      "Valor": round(v, 3) if isinstance(v, float) else v}
                     for k, v in feats.items()],
                    width="stretch", hide_index=True)

    elif tipo in ("ner_clinico", "laudo_nlp"):
        sent = SENTIMENT_PT.get(r.get("sentiment"), r.get("sentiment", "—"))
        neg = (r.get("sentiment_scores") or {}).get("negative")
        st.write(f"**Sentimento da fala:** {sent}"
                 + (f" (negativo: {neg:.0%})" if isinstance(neg, (int, float)) else ""))
        if r.get("key_phrases"):
            st.write("**Assuntos mencionados:** " + ", ".join(r["key_phrases"]))
        termos = r.get("critical_terms") or []
        if termos:
            azure_ok = set(r.get("critical_terms_azure") or [])
            st.error("⚠️ **Termos clínicos críticos:** "
                     + ", ".join(t + (" ☁️" if t in azure_ok else "") for t in termos))
        else:
            st.success("Nenhum termo clínico crítico identificado.")

    elif tipo == "pose_anomaly":
        eventos = r.get("events", [])
        if eventos:
            st.dataframe(
                [{"Nível": e.get("level_pt", e.get("level", "")),
                  "Instante (s)": round(e.get("timestamp", 0), 1),
                  "Evento": e.get("message", "")} for e in eventos],
                width="stretch", hide_index=True)
        else:
            st.success("Nenhum evento anômalo no vídeo.")
        objetos = r.get("objects") or {}
        if objetos:
            from multimodal_monitor.video.object_detection import class_pt

            st.caption("Objetos na cena (YOLOv8): "
                       + ", ".join(f"{class_pt(k)} ({v} quadros)" for k, v in objetos.items()))
        anotado = r.get("annotated_storage_path")
        if anotado:
            bucket, _, caminho = anotado.partition("/")
            assinada = (service_client().storage.from_(bucket)
                        .create_signed_url(caminho, 3600))
            url = (assinada.get("signedURL") or assinada.get("signedUrl")
                   or assinada.get("signed_url"))
            if url:
                st.markdown("**Vídeo anotado** (esqueleto + área crítica + banner):")
                st.video(url)

    elif tipo == "movement_metrics":
        linhas = {
            "Quadros analisados": r.get("frames"),
            "% com paciente detectado": r.get("valid_pct"),
            "Movimento médio (norm.)": r.get("mean_motion"),
            "Amplitude de joelho (graus)": r.get("knee_rom_deg"),
        }
        st.dataframe([{"Métrica": k, "Valor": v} for k, v in linhas.items() if v is not None],
                     width="stretch", hide_index=True)
        desvios = r.get("baseline_deviations") or {}
        if desvios:
            st.warning("Desvio vs. sessões anteriores do paciente: "
                       + ", ".join(f"{k} {v:+.0%}" for k, v in desvios.items()))

    elif tipo == "news2":
        n2 = r.get("news2", r)
        sev = n2.get("severity")
        (st.error if sev == "critical" else st.warning if sev == "warning" else st.info)(
            f"NEWS2 = **{n2.get('total')}** — "
            + {"critical": "risco alto, resposta emergencial",
               "warning": "risco médio, revisão urgente",
               "info": "risco baixo, manter monitorização"}.get(sev, "sem alteração"))
        partes = n2.get("parts") or {}
        if partes:
            st.caption("Pontuação por parâmetro: "
                       + " · ".join(f"{NEWS2_PART_PT.get(k, k)}: {v}"
                                    for k, v in partes.items() if v))

    elif tipo == "interacao_medicamentosa":
        for hit in r.get("interactions", []):
            st.error(f"⚠️ {' + '.join(hit['pair'])} — {hit['interaction']}")

    else:
        st.json(r)


@st.cache_data(ttl=30, show_spinner=False)
def _professionals() -> list[dict]:
    return (table("professionals").select("id,full_name,role")
            .eq("active", True).execute().data)


@st.cache_data(ttl=15, show_spinner=False)
def _patients() -> list[dict]:
    return (table("patients").select("*").eq("active", True)
            .order("created_at", desc=True).limit(200).execute().data)


profs = _professionals()
if not profs:
    st.error("Nenhum profissional cadastrado. Rode `uv run python "
             "scripts/seed_hospital.py` ou insira em `hospital.professionals`.")
    st.stop()

with st.sidebar:
    st.header("Profissional")
    prof = st.selectbox("Atuando como", profs,
                        format_func=lambda p: f"{p['full_name']} ({p['role']})")
    st.divider()
    with st.expander("➕ Novo paciente"):
        with st.form("novo_paciente", clear_on_submit=True):
            nome = st.text_input("Nome completo")
            mrn = st.text_input("Prontuário (MRN)")
            nasc = st.date_input("Nascimento", value=None,
                                 min_value=datetime(1900, 1, 1).date())
            sexo = st.selectbox("Sexo", ["M", "F", "O"])
            if st.form_submit_button("Cadastrar", type="primary"):
                if nome and mrn and nasc:
                    table("patients").insert({
                        "full_name": nome, "mrn": mrn,
                        "birth_date": str(nasc), "sex": sexo,
                        "created_by": prof["id"],
                    }).execute()
                    _patients.clear()
                    st.success(f"Paciente {nome} cadastrado.")
                else:
                    st.error("Preencha nome, prontuário e nascimento.")

pacientes = _patients()
if not pacientes:
    st.info("Nenhum paciente. Cadastre na barra lateral ou rode o seed "
            "(`uv run python scripts/seed_hospital.py`).")
    st.stop()

pac = st.selectbox(
    "Paciente", pacientes,
    format_func=lambda p: (f"{p['full_name']} · Prontuário {p['mrn']} · "
                           f"{ADMISSION_PT.get(p['admission_status'], '—')}"))
pid = pac["id"]

with st.expander("✏️ Editar dados do paciente"):
    with st.form("editar_paciente"):
        e1, e2, e3 = st.columns([2, 1, 1])
        ed_nome = e1.text_input("Nome completo", value=pac["full_name"])
        situacoes = list(ADMISSION_PT)
        ed_sit = e2.selectbox("Situação", situacoes,
                              index=situacoes.index(pac["admission_status"])
                              if pac["admission_status"] in situacoes else 0,
                              format_func=ADMISSION_PT.get)
        ed_sexo = e3.selectbox("Sexo", ["M", "F", "O"],
                               index=["M", "F", "O"].index(pac.get("sex") or "O"))
        b1, b2 = st.columns([1, 1])
        if b1.form_submit_button("Salvar alterações", type="primary"):
            table("patients").update({
                "full_name": ed_nome.strip(), "admission_status": ed_sit,
                "sex": ed_sexo,
            }).eq("id", pid).execute()
            _patients.clear()
            st.rerun()
        if b2.form_submit_button("🗄️ Arquivar paciente"):
            table("patients").update({"active": False}).eq("id", pid).execute()
            _patients.clear()
            st.rerun()
    st.caption("Arquivar é uma exclusão lógica: o prontuário (mídias, análises, "
               "vitais e alertas) permanece íntegro para auditoria — LGPD.")

# ── risco fundido + alertas abertos ─────────────────────────────────────────
risco = compute_patient_risk(pid)
score = risco["patient_risk_score"]
cor = "#dc2626" if score >= 0.7 else "#eab308" if score >= 0.4 else "#16a34a"
c1, c2, c3 = st.columns([1, 1.2, 2])
c1.metric("Risco multimodal", f"{score:.0%}")
with c2:
    st.caption("Situação")
    st.markdown(f"<div style='font-size:1.6rem;font-weight:600'>"
                f"{ADMISSION_PT.get(pac.get('admission_status'), '—')}</div>",
                unsafe_allow_html=True)
with c3:
    if risco["contributions"]:
        st.caption("Fontes que compõem o risco (mais recente de cada, "
                   "com decaimento de 24 h):")
        st.write("  ·  ".join(
            f"**{SOURCE_PT.get(src, src)}** {c['risk_score']:.0%}"
            for src, c in risco["contributions"].items()))
st.markdown(f"<div style='height:6px;border-radius:3px;background:{cor};"
            f"width:{max(int(score*100), 2)}%'></div>", unsafe_allow_html=True)

abertos = (table("alerts").select("*").eq("patient_id", pid)
           .is_("acknowledged_at", "null").order("created_at", desc=True)
           .limit(20).execute().data)
if abertos:
    with st.expander(f"🔔 {len(abertos)} alertas em aberto", expanded=True):
        for a in abertos:
            cols = st.columns([6, 1])
            cols[0].write(f"{SEV_ICON.get(a['severity'], '·')} **{title_pt(a['title'])}** — "
                          f"{_quando(a['created_at'])}")
            if cols[1].button("Ciente", key=f"ack{a['id']}"):
                table("alerts").update({
                    "acknowledged_by": prof["id"],
                    "acknowledged_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", a["id"]).execute()
                st.rerun()

tab_vitais, tab_midia, tab_rx, tab_hist = st.tabs(
    ["📈 Sinais vitais", "📤 Enviar mídia", "💊 Prescrições", "📜 Histórico"])

# ── sinais vitais ────────────────────────────────────────────────────────────
with tab_vitais:
    with st.form("vitais"):
        c = st.columns(6)
        fc = c[0].number_input("FC (bpm)", 20, 250, 80)
        spo2 = c[1].number_input("SpO₂ (%)", 50.0, 100.0, 97.0)
        temp = c[2].number_input("Temp (°C)", 32.0, 43.0, 36.5)
        pas = c[3].number_input("PA sist.", 50, 260, 120)
        pad = c[4].number_input("PA diast.", 30, 160, 80)
        fr = c[5].number_input("FR (irpm)", 4, 60, 16)
        if st.form_submit_button("Registrar leitura", type="primary"):
            res = ingest_vitals(
                pid,
                {"heart_rate": int(fc), "spo2": float(spo2),
                 "temperature": float(temp), "systolic_bp": int(pas),
                 "diastolic_bp": int(pad), "respiratory_rate": int(fr)},
                measured_at=datetime.now(timezone.utc).isoformat(),
            )
            n2 = res["news2"]
            (st.error if n2["severity"] == "critical"
             else st.warning if n2["severity"] == "warning"
             else st.success)(
                f"NEWS2 = {n2['total']} · {res['alerts_created']} alerta(s) · "
                f"anomalias z-score: {len(res['anomalies'])}")

    hist = (table("vital_signs").select("*").eq("patient_id", pid)
            .order("measured_at", desc=True).limit(200).execute().data)
    if hist:
        hist = list(reversed(hist))
        ts = [h["measured_at"][:16].replace("T", " ") for h in hist]
        fig = go.Figure()
        for campo, nome, cor_l in (("heart_rate", "FC", "#ef4444"),
                                   ("spo2", "SpO₂", "#3b82f6"),
                                   ("systolic_bp", "PA sist.", "#a855f7")):
            fig.add_trace(go.Scatter(x=ts, y=[h[campo] for h in hist],
                                     name=nome, line=dict(color=cor_l)))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")

# ── upload multimodal ────────────────────────────────────────────────────────
EXT_TO_MOD = {".wav": "audio", ".mp3": "audio", ".m4a": "audio", ".ogg": "audio",
              ".mp4": "video", ".mov": "video", ".webm": "video",
              ".txt": "text", ".pdf": "text"}
CAT_PADRAO = {"audio": "voz_consulta", "video": "fisioterapia", "text": "laudo"}

with tab_midia:
    st.caption("O arquivo vai para o bucket privado do Supabase Storage; a análise "
               "roda em background no worker (`uv run python -m hospital_ai.workers.runner`).")
    arq = st.file_uploader(
        "Arquivo (áudio, vídeo ou documento — o tipo é detectado pela extensão)",
        type=["wav", "mp3", "m4a", "ogg", "mp4", "mov", "webm", "txt", "pdf"])
    mod = None
    if arq is not None:
        mod = EXT_TO_MOD["." + arq.name.rsplit(".", 1)[-1].lower()]
        st.caption("Tipo detectado: "
                   + {"audio": "🎙️ Áudio (consulta)",
                      "video": "🎥 Vídeo (fisioterapia/cirurgia/monitoramento)",
                      "text": "📄 Documento (laudo/evolução)"}[mod])
        cat = st.text_input("Categoria", value=CAT_PADRAO[mod], key=f"cat_{mod}")
    if arq is not None and st.button("Enviar e analisar", type="primary"):
        with st.spinner("Enviando ao Storage e enfileirando análise..."):
            data = arq.getvalue()
            ext = "." + arq.name.rsplit(".", 1)[-1].lower()
            mime = MIME_BY_EXT.get(ext, arq.type or "application/octet-stream")
            asset = (table("media_assets").insert({
                "patient_id": pid, "uploaded_by": prof["id"], "modality": mod,
                "category": cat, "mime_type": mime,
                "size_bytes": len(data),
                "checksum_sha256": hashlib.sha256(data).hexdigest(),
                "storage_path": "pendente",
            }).execute().data)[0]
            bucket = BUCKETS[mod]
            path = f"{pid}/no-encounter/{asset['id']}{ext}"
            try:
                service_client().storage.from_(bucket).upload(
                    path, data, {"content-type": mime})
            except Exception as exc:
                # sem órfãos: se o Storage recusar, remove o registro criado
                table("media_assets").delete().eq("id", asset["id"]).execute()
                st.error(f"Upload recusado pelo Storage: {exc}")
                st.stop()
            table("media_assets").update(
                {"storage_path": f"{bucket}/{path}"}).eq("id", asset["id"]).execute()
            job_id = jobs.enqueue(JOB_BY_MODALITY[mod], {
                "asset_id": asset["id"], "patient_id": pid,
                "storage_path": f"{bucket}/{path}", "category": cat,
            })
        st.success(f"Enviado ({len(data)/1e6:.1f} MB) — análise nº {job_id} na fila. "
                   "O resultado aparece abaixo quando o processamento terminar.")

    # ── histórico de mídias com as análises de cada uma ──────────────────────
    st.divider()
    head_l, head_r = st.columns([3, 1])
    head_l.subheader("Mídias enviadas e resultados")
    if head_r.button("🔄 Atualizar resultados"):
        st.rerun()

    midias = (table("media_assets").select("*").eq("patient_id", pid)
              .order("created_at", desc=True).limit(20).execute().data)
    analises_midia = (table("analysis_results")
                      .select("media_asset_id,analysis_type,engine,risk_score,result,created_at")
                      .eq("patient_id", pid).not_.is_("media_asset_id", "null")
                      .order("created_at").execute().data)
    por_midia: dict[str, list[dict]] = {}
    for a in analises_midia:
        por_midia.setdefault(a["media_asset_id"], []).append(a)

    if not midias:
        st.caption("Nenhuma mídia enviada para este paciente ainda.")
    for m in midias:
        icone = MODALITY_ICON.get(m["modality"], "📁")
        titulo = (f"{icone} {MODALITY_PT.get(m['modality'], m['modality'])} · "
                  f"{m.get('category') or 'sem categoria'} · {_quando(m['created_at'])} — "
                  f"{STATUS_PT.get(m['processing_status'], m['processing_status'])}")
        with st.expander(titulo, expanded=(m is midias[0])):
            st.caption(f"{(m.get('size_bytes') or 0)/1e6:.1f} MB · {m.get('mime_type') or '—'}")
            if m["processing_status"] == "failed" and m.get("error_message"):
                st.error(f"Falha no processamento: {m['error_message']}")
            elif m["processing_status"] in ("pending", "processing"):
                st.info("Análise em andamento — mantenha o worker rodando "
                        "(`uv run python -m hospital_ai.workers.runner`) e "
                        "clique em **Atualizar resultados**.")
            resultados = por_midia.get(m["id"], [])
            for i, a in enumerate(resultados):
                if i:
                    st.markdown("")
                render_analysis(a)

# ── prescrições ──────────────────────────────────────────────────────────────
with tab_rx:
    with st.form("rx", clear_on_submit=True):
        c = st.columns([2, 1, 1])
        med = c[0].text_input("Medicamento (ex.: warfarina)")
        dose = c[1].text_input("Dose", value="1x")
        via = c[2].selectbox("Via", ["oral", "iv", "im", "sc"])
        if st.form_submit_button("Prescrever", type="primary") and med:
            table("prescriptions").insert({
                "patient_id": pid, "prescribed_by": prof["id"],
                "medication": med.strip().lower(), "dosage": dose, "route": via,
            }).execute()
            inter = check_drug_interactions(pid)
            if inter:
                table("alerts").insert({
                    "patient_id": pid, "source_type": "prescription",
                    "severity": "critical",
                    "title": "Interação medicamentosa detectada",
                    "details": {"interactions": inter},
                }).execute()
                st.error("⚠️ Interação de risco: " + "; ".join(
                    f"{' + '.join(i['pair'])} ({i['interaction']})" for i in inter))
            else:
                st.success("Prescrição registrada, sem interações conhecidas.")

    ativas = (table("prescriptions").select("*").eq("patient_id", pid)
              .eq("active", True).order("starts_at", desc=True).execute().data)
    if ativas:
        st.dataframe([{"Medicamento": r["medication"], "Dose": r["dosage"],
                       "Via": r["route"], "Desde": r["starts_at"][:10]}
                      for r in ativas], width="stretch", hide_index=True)

# ── histórico unificado ──────────────────────────────────────────────────────
with tab_hist:
    encontros = (table("encounters").select("*").eq("patient_id", pid)
                 .order("started_at", desc=True).limit(20).execute().data)
    hist_midias = (table("media_assets").select("*").eq("patient_id", pid)
                   .order("created_at", desc=True).limit(30).execute().data)
    analises = (table("analysis_results")
                .select("analysis_type,engine,risk_score,created_at,result")
                .eq("patient_id", pid).order("created_at", desc=True)
                .limit(30).execute().data)

    if encontros:
        st.subheader("Encontros clínicos")
        st.dataframe([{
            "Início": _quando(e["started_at"]),
            "Tipo": e["encounter_type"].capitalize(),
            "Observações": e.get("notes") or "—",
        } for e in encontros], width="stretch", hide_index=True)

    col_m, col_a = st.columns(2)
    with col_m:
        st.subheader("Mídias enviadas")
        if hist_midias:
            st.dataframe([{
                "Quando": _quando(m["created_at"]),
                "Tipo": MODALITY_PT.get(m["modality"], m["modality"]),
                "Categoria": m["category"],
                "Situação": STATUS_PT.get(m["processing_status"], m["processing_status"]),
                "MB": round((m["size_bytes"] or 0) / 1e6, 2),
            } for m in hist_midias], width="stretch", hide_index=True)
            st.caption("Os resultados detalhados de cada mídia ficam na aba "
                       "**📤 Enviar mídia**.")
        else:
            st.caption("Nenhuma mídia enviada.")
    with col_a:
        st.subheader("Análises realizadas")
        if analises:
            st.dataframe([{
                "Quando": _quando(a["created_at"]),
                "Análise": ANALYSIS_PT.get(a["analysis_type"], a["analysis_type"]),
                "Motor": ENGINE_PT.get(a["engine"], a["engine"]),
                "Risco": f"{float(a['risk_score']):.0%}" if a["risk_score"] is not None else "—",
            } for a in analises], width="stretch", hide_index=True)
        else:
            st.caption("Nenhuma análise ainda.")

    if analises:
        st.subheader("Análise mais recente")
        render_analysis(analises[0])
