"""Página: Áudio — features vocais, transcrição (Azure) e análise de texto."""

from __future__ import annotations

import streamlit as st

import monitor_ui as ui

st.title("🎙️ Análise de Áudio")
st.caption("Análise avulsa com áudios de exemplo · Parâmetros vocais (Praat/librosa) + Azure Fala "
           "+ Azure Linguagem · Para analisar a consulta de um paciente cadastrado, use "
           "🏥 Hospital → Pacientes → Enviar mídia")

audio_dir = ui.SAMPLES_DIR / "audio"
audios = sorted(audio_dir.glob("*.wav")) if audio_dir.exists() else []
if not audios:
    st.warning("Nenhum áudio em `data/samples/audio/`. Rode `python scripts/generate_audio_samples.py`.")
    st.stop()

names = [a.name for a in audios]
choice = st.selectbox("Áudio da consulta", names,
                      index=names.index("consulta_critica.wav") if "consulta_critica.wav" in names else 0)
audio_path = audio_dir / choice
st.audio(str(audio_path))

if st.button("▶️ Processar áudio", type="primary"):
    with st.spinner("Extraindo parâmetros vocais e analisando texto..."):
        st.session_state["audio_result"] = ui.process_audio(str(audio_path))
        st.session_state["audio_choice"] = choice

result = st.session_state.get("audio_result")
if result and st.session_state.get("audio_choice") == choice:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Parâmetros vocais")
        f = result["features"]
        st.dataframe(
            [{"Parâmetro": ui.FEATURE_PT.get(k, k), "Valor": round(v, 3) if isinstance(v, float) else v}
             for k, v in f.items()],
            width="stretch", hide_index=True,
        )
        lvl = result["score_level"]
        color = ui.LEVEL_COLOR[lvl]
        st.markdown(f"**Índice de alteração vocal:** "
                    f"<span style='color:{color};font-weight:700'>{result['score']:.2f} "
                    f"({ui.LEVEL_PT.get(lvl, lvl)})</span>",
                    unsafe_allow_html=True)
        for ind in result["indicators"]:
            st.write("• ", ind)

    with c2:
        st.subheader("Transcrição")
        st.caption(f"Fonte: {result['transcript_source']}")
        st.info(result["text"] or "(vazio)")

        st.subheader("Análise de texto")
        if not result["azure_configured"]:
            st.caption("⚠️ Azure Linguagem não configurado — sentimento/frases-chave indisponíveis; "
                       "termos críticos por correspondência local.")
        else:
            sent_pt = {"positive": "positivo", "neutral": "neutro", "negative": "negativo",
                       "mixed": "misto"}.get(result["sentiment"], result["sentiment"])
            st.write(f"**Sentimento:** {sent_pt}  {result.get('sentiment_scores', {})}")
            if result["key_phrases"]:
                st.write("**Frases-chave:** " + ", ".join(result["key_phrases"]))
        if result["critical_terms"]:
            azure_ok = set(result.get("critical_terms_azure", []))
            termos = [t + (" ☁️" if t in azure_ok else "") for t in result["critical_terms"]]
            st.error("**Termos críticos:** " + ", ".join(termos))
            if azure_ok:
                st.caption("☁️ = termo também destacado nas frases-chave do Azure Text Analytics.")
        else:
            st.success("Nenhum termo clínico crítico identificado.")

    st.divider()
    st.subheader("Alertas gerados")
    if result["alerts"]:
        for a in result["alerts"]:
            (st.error if a["level"] == "CRITICAL" else st.warning)(
                f"**[{a['level_pt']}]** {a['message']}")
    else:
        st.success("Nenhum alerta de áudio.")
else:
    st.info("Selecione um áudio e clique em **Processar áudio**. "
            "Compare `consulta_neutra` (sem alerta) e `consulta_critica` (crítico).")
