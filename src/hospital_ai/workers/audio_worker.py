"""Worker de áudio: reutiliza o pipeline vocal da ``multimodal_monitor``.

Etapas por asset:
  1. Features acústicas Praat/librosa (jitter, shimmer, HNR, pausas, taxa de
     fala) → score de fadiga/disartria (``analysis_type='disartria'``).
  2. Transcrição Azure Speech to Text pt-BR quando configurado
     (``analysis_type='transcricao'``).
  3. Análise do texto (Azure Text Analytics: sentimento/frases-chave; termos
     clínicos críticos pt-BR com validação cruzada) (``analysis_type='ner_clinico'``).

Cada etapa grava ``analysis_results``; achados relevantes viram ``alerts``.
"""

from __future__ import annotations

from .common import create_alert, download_asset, save_result, set_asset_status

_LEVEL_TO_SEVERITY = {"INFO": "info", "WARNING": "warning", "CRITICAL": "critical"}


def handle(payload: dict) -> None:
    from multimodal_monitor.audio.acoustic_features import extract_acoustic_features
    from multimodal_monitor.audio.azure_speech import transcribe
    from multimodal_monitor.audio.azure_text import analyze_text
    from multimodal_monitor.audio.vocal_anomaly import score_vocal_fatigue

    asset_id, patient_id = payload["asset_id"], payload["patient_id"]
    set_asset_status(asset_id, "processing")
    path = download_asset(payload["storage_path"])
    try:
        # 1) alterações vocais (fadiga/disartria)
        feats = extract_acoustic_features(path)
        vocal = score_vocal_fatigue(feats)
        save_result(patient_id, asset_id, "disartria", "praat_librosa",
                    {"features": feats.to_dict(),
                     "indicators": vocal["indicators"]},
                    vocal["score"])
        severity = _LEVEL_TO_SEVERITY[vocal["level"].label]
        if severity in ("warning", "critical"):
            create_alert(patient_id, "audio", severity,
                         "Padrão vocal alterado (possível fadiga/disartria)",
                         {"score": vocal["score"], "indicators": vocal["indicators"]},
                         asset_id)

        # 2) transcrição (Azure STT, degrada graciosamente sem chaves)
        stt = transcribe(path)
        text = stt.text if stt.configured else ""
        if text:
            save_result(patient_id, asset_id, "transcricao", "azure_speech",
                        {"text": text, "segments": stt.segments}, None)

        # 3) análise do conteúdo da fala
        if text:
            analysis = analyze_text(text)
            risk = 0.9 if analysis.critical_terms else (
                0.6 if analysis.sentiment == "negative" else 0.1)
            save_result(patient_id, asset_id, "ner_clinico",
                        "azure_language" if analysis.configured else "local_terms",
                        {"sentiment": analysis.sentiment,
                         "sentiment_scores": analysis.sentiment_scores,
                         "key_phrases": analysis.key_phrases,
                         "critical_terms": analysis.critical_terms,
                         "critical_terms_azure": analysis.critical_terms_azure},
                        risk)
            if analysis.critical_terms:
                create_alert(patient_id, "audio", "critical",
                             "Termos clínicos críticos na fala do paciente",
                             {"terms": analysis.critical_terms,
                              "azure_confirmed": analysis.critical_terms_azure},
                             asset_id)

        set_asset_status(asset_id, "done")
    except Exception as exc:
        set_asset_status(asset_id, "failed", str(exc)[:500])
        raise
    finally:
        path.unlink(missing_ok=True)
