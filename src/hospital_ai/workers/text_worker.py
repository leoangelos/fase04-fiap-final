"""Worker de texto: laudos/evoluções clínicas + interações medicamentosas.

  1. Lê o documento (txt ou pdf) e roda a análise de texto da
     ``multimodal_monitor`` (Azure Text Analytics: sentimento/frases-chave;
     termos clínicos críticos pt-BR com validação cruzada) →
     ``analysis_type='laudo_nlp'``.
  2. Checa interações entre as prescrições ATIVAS do paciente (base única em
     ``multimodal_monitor.vitals.prescriptions.RISKY_COMBINATIONS``).
"""

from __future__ import annotations

from pathlib import Path

from ..services.interactions import check_drug_interactions
from .common import create_alert, download_asset, save_result, set_asset_status


def _read_text(path: Path) -> str:
    if str(path).lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader

            return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
        except Exception:
            return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def handle(payload: dict) -> None:
    from multimodal_monitor.audio.azure_text import analyze_text

    asset_id, patient_id = payload["asset_id"], payload["patient_id"]
    set_asset_status(asset_id, "processing")
    path = download_asset(payload["storage_path"])
    try:
        text = _read_text(path)
        if text.strip():
            analysis = analyze_text(text)
            risk = 0.9 if analysis.critical_terms else (
                0.6 if analysis.sentiment == "negative" else 0.1)
            save_result(patient_id, asset_id, "laudo_nlp",
                        "azure_language" if analysis.configured else "local_terms",
                        {"sentiment": analysis.sentiment,
                         "key_phrases": analysis.key_phrases,
                         "critical_terms": analysis.critical_terms,
                         "critical_terms_azure": analysis.critical_terms_azure,
                         "chars": len(text)},
                        risk)
            if analysis.critical_terms:
                create_alert(patient_id, "nlp", "critical",
                             "Termos clínicos críticos no documento",
                             {"terms": analysis.critical_terms}, asset_id)

        interactions = check_drug_interactions(patient_id)
        if interactions:
            save_result(patient_id, None, "interacao_medicamentosa",
                        "local_rules", {"interactions": interactions}, 0.8)
            create_alert(patient_id, "prescription", "critical",
                         "Interação medicamentosa detectada",
                         {"interactions": interactions}, asset_id)
        set_asset_status(asset_id, "done")
    except Exception as exc:
        set_asset_status(asset_id, "failed", str(exc)[:500])
        raise
    finally:
        path.unlink(missing_ok=True)
