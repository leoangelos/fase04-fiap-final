"""Checagem de interação medicamentosa nas prescrições ativas do paciente.

Reutiliza a base de combinações de risco da ``multimodal_monitor``
(``vitals.prescriptions.RISKY_COMBINATIONS``) — fonte única para o pipeline
de demonstração e para a camada hospitalar. Em produção, uma base como
DrugBank/Micromedex substituiria o dicionário didático.
"""

from __future__ import annotations

from multimodal_monitor.vitals.prescriptions import RISKY_COMBINATIONS

from ..db import table


def check_drug_interactions(patient_id: str) -> list[dict]:
    """Pares com interação conhecida entre as prescrições ATIVAS do paciente."""
    res = (table("prescriptions").select("medication")
           .eq("patient_id", patient_id).eq("active", True).execute())
    meds = {r["medication"].strip().lower() for r in res.data}
    hits = []
    for pair, reason in RISKY_COMBINATIONS.items():
        if pair <= meds:
            hits.append({"pair": sorted(pair), "interaction": reason})
    return hits
