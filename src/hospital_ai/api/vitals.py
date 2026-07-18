from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..services import audit
from ..services.vitals_ingest import ingest_vitals
from .deps import current_professional

router = APIRouter(prefix="/patients/{patient_id}/vitals", tags=["vitals"])


class VitalsIn(BaseModel):
    measured_at: str
    heart_rate: int | None = None
    spo2: float | None = None
    temperature: float | None = None
    systolic_bp: int | None = None
    diastolic_bp: int | None = None
    respiratory_rate: int | None = None
    source: str = "manual"


@router.post("", status_code=201)
def ingest(patient_id: str, body: VitalsIn, prof=Depends(current_professional)):
    """Registra a leitura e roda NEWS2 + z-score + fusão (alertas automáticos)."""
    result = ingest_vitals(
        patient_id,
        body.model_dump(exclude={"measured_at", "source"}),
        measured_at=body.measured_at,
        source=body.source,
    )
    audit.log(prof["id"], "ingest_vitals", "vital_signs", patient_id)
    return result
