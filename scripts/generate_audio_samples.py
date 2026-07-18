"""Gera áudios de consulta sintéticos em PT-BR para a demo.

Usa o TTS do macOS (``say``, voz Luciana pt-BR) + ffmpeg. Produz dois arquivos:

  - ``consulta_neutra.wav``   : paciente estável, sem termos críticos, fala fluente.
  - ``consulta_critica.wav``  : queixas de emergência (dor no peito, falta de ar),
    fala lentificada e degradada acusticamente (tremolo + ruído) para simular
    fadiga/esforço vocal — dispara features vocais E termos críticos.

Em máquinas sem ``say`` (fora do macOS), grave você mesmo os roteiros abaixo.

Uso:
    uv run python scripts/generate_audio_samples.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "data" / "samples" / "audio"

ROTEIRO_NEUTRO = (
    "Bom dia, doutor. Estou me sentindo bem hoje. "
    "Dormi bem esta noite e não tenho nenhuma queixa importante. "
    "Continuo tomando os remédios como o senhor recomendou. "
    "Minha alimentação está regular e faço caminhada todos os dias."
)

ROTEIRO_CRITICO = (
    "Doutor, não estou bem. Estou sentindo uma dor no peito muito forte "
    "e também estou com falta de ar desde ontem à noite. "
    "Tive uma tontura quando levantei e quase desmaiei. "
    "Estou muito cansado e com o coração acelerado."
)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _say_to_wav(text: str, out: Path, rate: int = 180) -> None:
    """Gera fala com `say` e converte para WAV 16 kHz mono."""
    aiff = out.with_suffix(".aiff")
    _run(["say", "-v", "Luciana", "-r", str(rate), "-o", str(aiff), text])
    _run(["ffmpeg", "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(out)])
    aiff.unlink(missing_ok=True)


def _degrade(inp: Path, out: Path) -> None:
    """Adiciona tremolo + ruído leve + pausas → simula instabilidade/fadiga vocal."""
    _run([
        "ffmpeg", "-y", "-i", str(inp),
        "-af", "vibrato=f=6:d=0.5,aemphasis=level_in=1:level_out=2:mode=production,"
               "volume=0.9,atempo=0.9",
        "-ar", "16000", "-ac", "1", str(out),
    ])


def main() -> None:
    if sys.platform != "darwin" or shutil.which("say") is None:
        sys.exit(
            "Este gerador usa o TTS 'say' do macOS. Em outros sistemas, grave manualmente\n"
            "os roteiros (veja ROTEIRO_NEUTRO e ROTEIRO_CRITICO neste arquivo) como\n"
            "consulta_neutra.wav e consulta_critica.wav em data/samples/audio/."
        )
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg é necessário. Instale com: brew install ffmpeg")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print("Gerando consulta_neutra.wav ...")
    _say_to_wav(ROTEIRO_NEUTRO, AUDIO_DIR / "consulta_neutra.wav", rate=180)

    print("Gerando consulta_critica.wav (fala lenta + degradação) ...")
    tmp = AUDIO_DIR / "_critica_raw.wav"
    _say_to_wav(ROTEIRO_CRITICO, tmp, rate=130)          # fala mais lenta
    _degrade(tmp, AUDIO_DIR / "consulta_critica.wav")
    tmp.unlink(missing_ok=True)

    print("\nÁudios prontos em", AUDIO_DIR)
    for f in sorted(AUDIO_DIR.glob("*.wav")):
        print("  -", f.name)


if __name__ == "__main__":
    main()
