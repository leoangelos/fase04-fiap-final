"""Teste de fumaça da integração Azure.

Verifica se as chaves do ``.env`` funcionam: transcreve um áudio de amostra com
Azure Speech to Text e roda sentimento + frases-chave com Azure Text Analytics.

Uso:
    uv run python scripts/smoke_azure.py [caminho_audio.wav]
"""

from __future__ import annotations

import sys
from pathlib import Path

from multimodal_monitor.audio.azure_speech import transcribe
from multimodal_monitor.audio.azure_text import analyze_text
from multimodal_monitor.config import SAMPLES_DIR, settings


def main() -> None:
    print("== Configuração ==")
    print("Speech configurado :", settings.azure.speech_configured)
    print("Language configurado:", settings.azure.language_configured)
    if not (settings.azure.speech_configured or settings.azure.language_configured):
        sys.exit("\nNenhum recurso Azure configurado. Preencha o .env (veja README).")

    audio = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLES_DIR / "audio" / "consulta_critica.wav"

    if settings.azure.speech_configured:
        print(f"\n== Speech to Text ({audio.name}) ==")
        result = transcribe(audio)
        if result.error:
            print("Erro:", result.error)
        else:
            print("Transcrição:", result.text or "(vazio)")
            text = result.text
    else:
        text = "Estou com dor no peito e falta de ar."
        print("\n(Speech não configurado — usando texto de exemplo para o Language.)")

    if settings.azure.language_configured:
        print("\n== Text Analytics ==")
        analysis = analyze_text(text)
        print("Sentimento     :", analysis.sentiment, analysis.sentiment_scores)
        print("Frases-chave   :", analysis.key_phrases)
        print("Termos críticos:", analysis.critical_terms)

    print("\nOK — integração Azure funcionando.")


if __name__ == "__main__":
    main()
