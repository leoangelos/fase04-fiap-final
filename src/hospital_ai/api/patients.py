from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from ..db import table
from ..services import audit
from ..services.fusion import compute_patient_risk
from .deps import current_professional

router = APIRouter(prefix="/patients", tags=["patients"])


class PatientIn(BaseModel):
    mrn: str
    full_name: str
    birth_date: str
    sex: str | None = None
    cpf: str | None = Field(default=None, pattern=r"^\d{11}$")
    admission_status: str = "outpatient"


@router.post("", status_code=201)
def create_patient(body: PatientIn, prof=Depends(current_professional)):
    data = body.model_dump() | {"created_by": prof["id"]}
    res = table("patients").insert(data).execute()
    audit.log(prof["id"], "create_patient", "patients", res.data[0]["id"])
    return res.data[0]


@router.get("")
def list_patients(q: str | None = None, prof=Depends(current_professional)):
    query = table("patients").select("*").order("created_at", desc=True).limit(100)
    if q:
        query = query.ilike("full_name", f"%{q}%")
    return query.execute().data


@router.get("/{patient_id}")
def get_patient(patient_id: str, prof=Depends(current_professional)):
    res = table("patients").select("*").eq("id", patient_id).execute()
    if not res.data:
        raise HTTPException(404, "Paciente nao encontrado")
    audit.log(prof["id"], "view_patient", "patients", patient_id)
    return res.data[0]


@router.get("/{patient_id}/timeline")
def patient_timeline(patient_id: str, prof=Depends(current_professional)):
    """Historico unificado: encontros, midias, analises, vitais e alertas."""
    audit.log(prof["id"], "view_timeline", "patients", patient_id)
    return {
        "encounters": table("encounters").select("*").eq("patient_id", patient_id)
            .order("started_at", desc=True).limit(50).execute().data,
        "media": table("media_assets").select("*").eq("patient_id", patient_id)
            .order("created_at", desc=True).limit(50).execute().data,
        "analyses": table("analysis_results")
            .select("id,analysis_type,engine,risk_score,created_at,media_asset_id")
            .eq("patient_id", patient_id).order("created_at", desc=True).limit(50).execute().data,
        "vitals": table("vital_signs").select("*").eq("patient_id", patient_id)
            .order("measured_at", desc=True).limit(100).execute().data,
        "alerts": table("alerts").select("*").eq("patient_id", patient_id)
            .order("created_at", desc=True).limit(50).execute().data,
    }


@router.get("/{patient_id}/risk")
def patient_risk(patient_id: str, prof=Depends(current_professional)):
    return compute_patient_risk(patient_id)
