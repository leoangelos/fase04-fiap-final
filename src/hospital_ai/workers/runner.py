"""Loop principal dos workers.

Uso:
    uv run python -m hospital_ai.workers.runner              # todos os tipos
    uv run python -m hospital_ai.workers.runner analyze_audio
"""
import sys
import time
import traceback
from ..services import jobs
from ..services.fusion import maybe_alert_fusion

HANDLERS = {}


def _load_handlers():
    from . import audio_worker, text_worker, video_worker
    HANDLERS.update({
        "analyze_audio": audio_worker.handle,
        "analyze_video": video_worker.handle,
        "analyze_text": text_worker.handle,
    })


def run(job_types: list[str] | None = None, poll_seconds: float = 3.0):
    _load_handlers()
    print(f"[worker] iniciado. tipos={job_types or 'todos'}")
    while True:
        job = jobs.pick(job_types)
        if not job:
            time.sleep(poll_seconds)
            continue
        print(f"[worker] job #{job['id']} ({job['job_type']})")
        try:
            HANDLERS[job["job_type"]](job["payload"])
            jobs.complete(job["id"], True)
            maybe_alert_fusion(job["payload"]["patient_id"])
            print(f"[worker] job #{job['id']} concluido")
        except Exception:
            err = traceback.format_exc()
            print(f"[worker] job #{job['id']} falhou:\n{err}")
            jobs.complete(job["id"], False, err[-2000:])


if __name__ == "__main__":
    types = sys.argv[1:] or None
    run(types)
