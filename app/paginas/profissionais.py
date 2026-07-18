"""Página: Profissionais — CRUD da equipe (Supabase Auth + hospital.professionals).

Criar um profissional cria também o login (Supabase Auth) com senha provisória.
A exclusão é LÓGICA (desativação): registros clínicos assinados pelo profissional
permanecem íntegros para auditoria — requisito de um sistema hospitalar real.
"""

from __future__ import annotations

import streamlit as st

import monitor_ui as ui  # noqa: F401 — garante src/ no sys.path

from hospital_ai.config import settings as hsettings

st.title("🧑‍⚕️ Profissionais")
st.caption("Equipe com acesso ao sistema — criação de login, edição e desativação")

if not hsettings.supabase_configured:
    st.warning("**Supabase não configurado.** Preencha as chaves no `.env` "
               "(README → *Camada hospitalar*).")
    st.stop()

from hospital_ai.db import service_client, table  # noqa: E402

ROLE_PT = {"medico": "Médico(a)", "enfermagem": "Enfermagem",
           "fisio": "Fisioterapeuta", "admin": "Administrador(a)"}


@st.cache_data(ttl=15, show_spinner=False)
def _profissionais() -> list[dict]:
    return (table("professionals").select("*")
            .order("full_name").execute().data)


def _recarrega() -> None:
    _profissionais.clear()
    st.rerun()


# ── novo profissional ────────────────────────────────────────────────────────
with st.expander("➕ Novo profissional (cria o login de acesso)"):
    with st.form("novo_prof", clear_on_submit=True):
        c1, c2 = st.columns(2)
        nome = c1.text_input("Nome completo")
        crm = c2.text_input("Registro profissional (CRM/COREN/CREFITO)")
        c3, c4, c5 = st.columns(3)
        email = c3.text_input("E-mail (login)")
        senha = c4.text_input("Senha provisória", type="password")
        papel = c5.selectbox("Papel", list(ROLE_PT), format_func=ROLE_PT.get)
        if st.form_submit_button("Criar profissional", type="primary"):
            if not (nome and email and senha):
                st.error("Preencha nome, e-mail e senha.")
            else:
                try:
                    user = service_client().auth.admin.create_user({
                        "email": email.strip(), "password": senha,
                        "email_confirm": True,
                    })
                    table("professionals").insert({
                        "id": user.user.id, "full_name": nome.strip(),
                        "crm": crm.strip() or None, "role": papel,
                    }).execute()
                    st.success(f"{nome} criado(a) — login {email} ativo.")
                    _recarrega()
                except Exception as exc:
                    st.error(f"Não foi possível criar: {exc}")

profs = _profissionais()
if not profs:
    st.info("Nenhum profissional. Crie acima ou rode o seed.")
    st.stop()

ativos = [p for p in profs if p.get("active", True)]
inativos = [p for p in profs if not p.get("active", True)]

st.subheader(f"Equipe ativa ({len(ativos)})")
for p in ativos:
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 2, 1.2, 1.2])
        c1.markdown(f"**{p['full_name']}**")
        c2.write(f"{ROLE_PT.get(p['role'], p['role'])} · {p.get('crm') or 'sem registro'}")
        if c3.button("✏️ Editar", key=f"ed{p['id']}"):
            st.session_state["prof_editando"] = p["id"]
        if c4.button("🚫 Desativar", key=f"off{p['id']}"):
            table("professionals").update({"active": False}).eq("id", p["id"]).execute()
            _recarrega()

        if st.session_state.get("prof_editando") == p["id"]:
            with st.form(f"form_ed{p['id']}"):
                e1, e2, e3 = st.columns(3)
                novo_nome = e1.text_input("Nome", value=p["full_name"])
                novo_crm = e2.text_input("Registro", value=p.get("crm") or "")
                novo_papel = e3.selectbox("Papel", list(ROLE_PT),
                                          index=list(ROLE_PT).index(p["role"]),
                                          format_func=ROLE_PT.get)
                s1, s2 = st.columns([1, 1])
                if s1.form_submit_button("Salvar", type="primary"):
                    table("professionals").update({
                        "full_name": novo_nome.strip(), "crm": novo_crm.strip() or None,
                        "role": novo_papel,
                    }).eq("id", p["id"]).execute()
                    st.session_state.pop("prof_editando", None)
                    _recarrega()
                if s2.form_submit_button("Cancelar"):
                    st.session_state.pop("prof_editando", None)
                    st.rerun()

if inativos:
    st.subheader(f"Desativados ({len(inativos)})")
    st.caption("A desativação é lógica: uploads, prescrições e auditoria assinados "
               "pelo profissional permanecem no histórico (LGPD/rastreabilidade).")
    for p in inativos:
        c1, c2, c3 = st.columns([3, 2, 1.2])
        c1.write(f"~~{p['full_name']}~~")
        c2.write(f"{ROLE_PT.get(p['role'], p['role'])} · {p.get('crm') or '—'}")
        if c3.button("↩️ Reativar", key=f"on{p['id']}"):
            table("professionals").update({"active": True}).eq("id", p["id"]).execute()
            _recarrega()
