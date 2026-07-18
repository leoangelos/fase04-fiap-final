"""Upload multimodal em duas etapas:
1) POST /patients/{id}/media/upload-url -> cria registro pending + signed upload URL
2) POST /media/{asset_id}/confirm -> confirma checksum/tamanho e enfileira analise
Download sempre via signed URL de curta duracao.
"""
import uuid
from pathlib import PurePosixPath
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from ..config import get_settings
from ..db import service_client, table
from ..services import audit, jobs
from .deps import current_professional

router = APIRouter(tags=["media"])

BUCKETS = {"video": "medical-videos", "audio": "medical-audio",
           "text": "medical-docs", "image": "medical-docs"}
JOB_BY_MODALITY = {"video": "analyze_video", "audio": "analyze_audio",
                   "text": "analyze_text", "image": "analyze_text"}


class UploadRequest(BaseModel):
    modality: str
    category: str | None = None
    filename: str
    mime_type: str
    encounter_id: str | None = None


@router.post("/patients/{patient_id}/media/upload-url")
def create_upload_url(patient_id: str, body: UploadRequest,
                      prof=Depends(current_professional)):
    if body.modality not in BUCKETS:
        raise HTTPException(422, f"Modalidade invalida: {body.modality}")
    asset_id = str(uuid.uuid4())
    ext = PurePosixPath(body.filename).suffix or ""
    folder = body.encounter_id or "no-encounter"
    path = f"{patient_id}/{folder}/{asset_id}{ext}"
    bucket = BUCKETS[body.modality]

    signed = service_client().storage.from_(bucket).create_signed_upload_url(path)

    table("media_assets").insert({
        "id": asset_id, "patient_id": patient_id,
        "encounter_id": body.encounter_id, "uploaded_by": prof["id"],
        "modality": body.modality, "category": body.category,
        "storage_path": f"{bucket}/{path}", "mime_type": body.mime_type,
    }).execute()
    audit.log(prof["id"], "upload_url", "media_assets", asset_id)
    return {"asset_id": asset_id, "bucket": bucket, "path": path,
            "signed_url": signed.get("signed_url") or signed.get("signedUrl"),
            "token": signed.get("token")}


class ConfirmUpload(BaseModel):
    size_bytes: int
    checksum_sha256: str
    duration_seconds: float | None = None


@router.post("/media/{asset_id}/confirm")
def confirm_upload(asset_id: str, body: ConfirmUpload,
                   prof=Depends(current_professional)):
    res = table("media_assets").update({
        "size_bytes": body.size_bytes, "checksum_sha256": body.checksum_sha256,
        "duration_seconds": body.duration_seconds,
    }).eq("id", asset_id).execute()
    if not res.data:
        raise HTTPException(404, "Asset nao encontrado")
    asset = res.data[0]
    job_id = jobs.enqueue(JOB_BY_MODALITY[asset["modality"]], {
        "asset_id": asset_id, "patient_id": asset["patient_id"],
        "storage_path": asset["storage_path"], "category": asset["category"],
    })
    audit.log(prof["id"], "confirm_upload", "media_assets", asset_id)
    return {"asset_id": asset_id, "job_id": job_id, "status": "pending"}


@router.get("/media/{asset_id}/download-url")
def download_url(asset_id: str, prof=Depends(current_professional)):
    res = table("media_assets").select("storage_path").eq("id", asset_id).execute()
    if not res.data:
        raise HTTPException(404, "Asset nao encontrado")
    bucket, _, path = res.data[0]["storage_path"].partition("/")
    signed = service_client().storage.from_(bucket).create_signed_url(
        path, get_settings().signed_url_ttl_seconds)
    audit.log(prof["id"], "download", "media_assets", asset_id)
    return {"signed_url": signed.get("signedURL") or signed.get("signed_url")}
