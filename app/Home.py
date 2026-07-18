"""Dashboard de Monitoramento Multimodal — roteador de navegação.

Rode com:  uv run streamlit run app/Home.py

Usa st.navigation para que os rótulos do menu fiquem em português, mantendo os
nomes de arquivo em ASCII (compatível com Windows).
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Monitoramento Multimodal", page_icon="🏥", layout="wide")

# Oculta os controles em inglês do Streamlit (Deploy, menu de desenvolvedor, rodapé).
st.markdown(
    """
    <style>
      [data-testid="stToolbar"] {visibility: hidden;}
      [data-testid="stDecoration"] {display: none;}
      footer {visibility: hidden;}
      #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

paginas = {
    "🏥 Hospital": [
        st.Page("paginas/visao_geral.py", title="Visão Geral", icon="📊", default=True),
        st.Page("paginas/pacientes.py", title="Pacientes", icon="🗂️"),
        st.Page("paginas/profissionais.py", title="Profissionais", icon="🧑‍⚕️"),
    ],
    "🔬 Análises de demonstração": [
        st.Page("paginas/inicio.py", title="Pipeline Multimodal", icon="🧪"),
        st.Page("paginas/video.py", title="Vídeo", icon="🎥"),
        st.Page("paginas/audio.py", title="Áudio", icon="🎙️"),
        st.Page("paginas/sinais_vitais.py", title="Sinais Vitais", icon="📈"),
        st.Page("paginas/alertas.py", title="Alertas", icon="🔔"),
    ],
}

st.navigation(paginas).run()
