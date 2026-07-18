"""Transcrição de áudio com Azure Speech to Text (pt-BR).

Isolado atrás de ``settings.azure`` — se as chaves não estiverem configuradas, a
função devolve um resultado vazio com aviso, sem quebrar o restante do pipeline.

Requisito do edital: "Utilizar Azure Speech to Text para transcrever e analisar
os áudios".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import settings


@dataclass
class TranscriptionResult:
    text: str = ""
    segments: list[str] = field(default_factory=list)
    configured: bool = True
    error: str | None = None


def _to_wav_pcm16(path: str | Path) -> Path:
    """Garante WAV PCM 16 kHz mono (formato aceito pelo SDK). Converte via ffmpeg se preciso."""
    import subprocess

    path = Path(path)
    if path.suffix.lower() == ".wav":
        return path
    out = path.with_suffix(".16k.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(out)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out


def transcribe(path: str | Path, locale: str | None = None) -> TranscriptionResult:
    """Transcreve o áudio inteiro (reconhecimento contínuo) e devolve o texto."""
    if not settings.azure.speech_configured:
        return TranscriptionResult(
            configured=False,
            error="Azure Speech não configurado (defina AZURE_SPEECH_KEY/REGION no .env).",
        )

    import azure.cognitiveservices.speech as speechsdk

    wav = _to_wav_pcm16(path)
    speech_config = speechsdk.SpeechConfig(
        subscription=settings.azure.speech_key, region=settings.azure.speech_region
    )
    speech_config.speech_recognition_language = locale or settings.azure.speech_locale
    audio_config = speechsdk.audio.AudioConfig(filename=str(wav))
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    segments: list[str] = []
    done = False

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
            segments.append(evt.result.text)

    def on_stop(evt):
        nonlocal done
        done = True

    recognizer.recognized.connect(on_recognized)
    recognizer.session_stopped.connect(on_stop)
    recognizer.canceled.connect(on_stop)

    import time

    recognizer.start_continuous_recognition()
    while not done:
        time.sleep(0.2)
    recognizer.stop_continuous_recognition()

    return TranscriptionResult(text=" ".join(segments), segments=segments)
