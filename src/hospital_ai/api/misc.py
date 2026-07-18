from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from ..db import table
from ..services import audit
from .deps import current_professional

router = APIRouter(tags=["encounters", "prescriptions", "alerts"])


class EncounterIn(BaseModel):
    patient_id: str
    encounter_type: str
    notes: str | None = None


@router.post("/encounters", status_code=201)
def create_encounter(body: EncounterIn, prof=Depends(current_professional)):
    res = table("encounters").insert(
        body.model_dump() | {"professional_id": prof["id"]}).execute()
    return res.data[0]


class PrescriptionIn(BaseModel):
    patient_id: str
    medication: str
    dosage: str | None = None
    frequency: str | None = None
    route: str | None = None


@router.post("/prescriptions", status_code=201)
def create_prescription(body: PrescriptionIn, prof=Depends(current_professional)):
    res = table("prescriptions").insert(
        body.model_dump() | {"prescribed_by": prof["id"]}).execute()
    rx = res.data[0]
    audit.log(prof["id"], "prescribe", "prescriptions", rx["id"])

    # checagem imediata de interação contra as prescrições ativas do paciente
    from ..services.interactions import check_drug_interactions

    interactions = check_drug_interactions(body.patient_id)
    if interactions:
        table("analysis_results").insert({
            "patient_id": body.patient_id,
            "analysis_type": "interacao_medicamentosa", "engine": "local_rules",
            "result": {"interactions": interactions}, "risk_score": 0.8,
        }).execute()
        table("alerts").insert({
            "patient_id": body.patient_id, "source_type": "prescription",
            "severity": "critical", "title": "Interação medicamentosa detectada",
            "details": {"interactions": interactions}, "source_id": rx["id"],
        }).execute()
    return rx | {"interactions": interactions}


@router.get("/alerts")
def open_alerts(prof=Depends(current_professional)):
    return (table("alerts").select("*").is_("acknowledged_at", "null")
            .order("created_at", desc=True).limit(100).execute().data)


@router.post("/alerts/{alert_id}/ack")
def acknowledge(alert_id: str, prof=Depends(current_professional)):
    res = table("alerts").update({
        "acknowledged_by": prof["id"], "acknowledged_at": "now()",
    }).eq("id", alert_id).execute()
    if not res.data:
        raise HTTPException(404, "Alerta nao encontrado")
    audit.log(prof["id"], "ack_alert", "alerts", alert_id)
    return res.data[0]
