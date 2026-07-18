import tempfile
from pathlib import Path, PurePosixPath
from ..db import service_client


def download_asset(storage_path: str) -> Path:
    """Baixa um asset do Storage para arquivo temporario."""
    bucket, _, path = storage_path.partition("/")
    data = service_client().storage.from_(bucket).download(path)
    suffix = Path(path).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return Path(tmp.name)


def upload_sibling(storage_path: str, local_file: Path, suffix: str,
                   mime: str) -> str:
    """Sobe um arquivo derivado ao lado do original no mesmo bucket.

    Ex.: original ``medical-videos/pid/enc/asset.mp4`` + suffix ``_annotated``
    → ``medical-videos/pid/enc/asset_annotated.mp4``. Retorna o storage_path
    completo (bucket/caminho) do derivado.
    """
    bucket, _, path = storage_path.partition("/")
    # PurePosixPath: chaves do Storage usam sempre "/" — Path viraria "\" no Windows
    p = PurePosixPath(path)
    derived = str(p.with_name(p.stem + suffix + local_file.suffix))
    service_client().storage.from_(bucket).upload(
        derived, local_file.read_bytes(),
        {"content-type": mime, "upsert": "true"},
    )
    return f"{bucket}/{derived}"


def save_result(patient_id: str, asset_id: str | None, analysis_type: str,
                engine: str, result: dict, risk_score: float | None):
    service_client().table("analysis_results").insert({
        "patient_id": patient_id, "media_asset_id": asset_id,
        "analysis_type": analysis_type, "engine": engine,
        "result": result, "risk_score": risk_score,
    }).execute()


def set_asset_status(asset_id: str, status: str, error: str | None = None):
    service_client().table("media_assets").update({
        "processing_status": status, "error_message": error,
    }).eq("id", asset_id).execute()


def create_alert(patient_id: str, source_type: str, severity: str,
                 title: str, details: dict, source_id: str | None = None):
    service_client().table("alerts").insert({
        "patient_id": patient_id, "source_type": source_type,
        "severity": severity, "title": title, "details": details,
        "source_id": source_id,
    }).execute()
