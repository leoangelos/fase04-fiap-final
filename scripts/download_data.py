"""Baixa e prepara as amostras de dados do projeto.

Como os vídeos são grandes, não ficam versionados no Git. Este script reproduz as
amostras usadas na demo:

  1. Baixa um vídeo público com pessoa em corpo inteiro (licença de amostra de CV).
  2. Recorta ``corridor_walk.mp4`` (monitoramento de movimentação — requisito 3c).
  3. Constrói ``patient_immobility_demo.mp4`` congelando um trecho → imobilidade
     prolongada reproduzível (anomalia demonstrável, análoga aos vitais sintéticos).

Uso:
    uv run python scripts/download_data.py
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "data" / "samples" / "video"

SOURCE_URL = (
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
    "one-by-one-person-detection.mp4"
)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _have_ffmpeg() -> bool:
    try:
        _run(["ffmpeg", "-version"])
        return True
    except Exception:
        return False


def download_videos() -> None:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    if not _have_ffmpeg():
        sys.exit("ffmpeg é necessário. Instale com: brew install ffmpeg")

    src = VIDEO_DIR / "_source.mp4"
    if not src.exists():
        print(f"Baixando vídeo-fonte de {SOURCE_URL} ...")
        urllib.request.urlretrieve(SOURCE_URL, src)
    print("Vídeo-fonte pronto.")

    corridor = VIDEO_DIR / "corridor_walk.mp4"
    print("Gerando corridor_walk.mp4 (20s) ...")
    _run(["ffmpeg", "-y", "-i", str(src), "-t", "20", "-c:v", "libx264",
          "-pix_fmt", "yuv420p", str(corridor)])

    print("Gerando patient_immobility_demo.mp4 (caminhada + congelamento) ...")
    walk = VIDEO_DIR / "_walk.mp4"
    freeze = VIDEO_DIR / "_freeze.mp4"
    last = VIDEO_DIR / "_last.png"
    concat = VIDEO_DIR / "_concat.txt"
    _run(["ffmpeg", "-y", "-ss", "6", "-t", "8", "-i", str(src),
          "-vf", "fps=10,scale=768:432", "-an", str(walk)])
    _run(["ffmpeg", "-y", "-sseof", "-0.15", "-i", str(walk), "-frames:v", "1", str(last)])
    _run(["ffmpeg", "-y", "-loop", "1", "-i", str(last), "-t", "7",
          "-vf", "fps=10,scale=768:432", "-pix_fmt", "yuv420p", str(freeze)])
    concat.write_text(f"file '{walk.name}'\nfile '{freeze.name}'\n")
    _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          str(VIDEO_DIR / "patient_immobility_demo.mp4")])
    for tmp in (walk, freeze, last, concat, src):
        tmp.unlink(missing_ok=True)

    print("\nVídeos prontos em", VIDEO_DIR)
    for f in sorted(VIDEO_DIR.glob("*.mp4")):
        print("  -", f.name)


def generate_audio() -> None:
    """Delegado ao gerador de áudios (TTS pt-BR)."""
    try:
        from generate_audio_samples import main as gen_audio
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from generate_audio_samples import main as gen_audio
    gen_audio()


if __name__ == "__main__":
    download_videos()
    print("\n--- Áudio ---")
    try:
        generate_audio()
    except SystemExit as exc:
        print(exc)
