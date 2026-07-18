from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from .api import patients, media, vitals, misc

app = FastAPI(
    title="Hospital AI - Monitoramento Multimodal",
    description="FIAP Fase 04: fusao de audio, video, texto e sinais vitais "
                "com deteccao de anomalias em tempo real.",
    version="0.1.0",
)
app.include_router(patients.router)
app.include_router(media.router)
app.include_router(vitals.router)
app.include_router(misc.router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Raiz → documentação interativa (Swagger)."""
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok"}
