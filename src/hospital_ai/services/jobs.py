"""Fila de processamento (hospital.jobs) via RPC service_role."""
from ..db import service_client, table


def enqueue(job_type: str, payload: dict) -> int:
    res = table("jobs").insert({"job_type": job_type, "payload": payload}).execute()
    return res.data[0]["id"]


def pick(job_types: list[str] | None = None) -> dict | None:
    res = service_client().rpc("pick_job", {"p_job_types": job_types}).execute()
    return res.data[0] if res.data else None


def complete(job_id: int, success: bool, error: str | None = None) -> None:
    service_client().rpc("complete_job", {
        "p_job_id": job_id, "p_success": success, "p_error": error,
    }).execute()
