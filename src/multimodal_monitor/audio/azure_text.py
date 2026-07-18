"""Análise da transcrição com Azure Text Analytics (Azure AI Language).

Requisito do edital: "Identificar termos críticos e sentimentos com Azure Text
Analytics". Fazemos:
  - análise de sentimento (positivo/neutro/negativo + scores);
  - extração de frases-chave;
  - detecção de termos clínicos críticos por correspondência em PT-BR
    (a API de Text Analytics *for Health* só suporta inglês — documentamos a
     limitação e usamos uma lista custom em português).

Isolado atrás de ``settings.azure``; sem chaves, devolve resultado vazio marcado.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ..alerts.alert_manager import Alert, AlertLevel
from ..config import CRITICAL_TERMS_PT, settings


@dataclass
class TextAnalysisResult:
    sentiment: str = "unknown"
    sentiment_scores: dict = field(default_factory=dict)
    key_phrases: list[str] = field(default_factory=list)
    critical_terms: list[str] = field(default_factory=list)
    critical_terms_azure: list[str] = field(default_factory=list)  # confirmados nas frases-chave do Azure
    configured: bool = True
    error: str | None = None


def _normalize(text: str) -> str:
    """Minúsculas sem acento, para casar termos de forma robusta."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def find_critical_terms(text: str, terms: list[str] | None = None) -> list[str]:
    """Retorna os termos críticos presentes no texto (comparação sem acento)."""
    terms = terms or CRITICAL_TERMS_PT
    norm = _normalize(text)
    found = []
    for term in terms:
        pattern = r"\b" + re.escape(_normalize(term)) + r"\b"
        if re.search(pattern, norm):
            found.append(term)
    return found


def crosscheck_terms_with_phrases(terms: list[str], phrases: list[str]) -> list[str]:
    """Termos críticos também presentes nas frases-chave extraídas pelo Azure.

    Valida a lista clínica local contra a saída do Azure Text Analytics: um termo
    "confirmado" foi destacado pelo próprio serviço de nuvem como frase relevante
    da fala do paciente (mesma correspondência sem acento e com fronteira de
    palavra usada em ``find_critical_terms``).
    """
    if not phrases:
        return []
    joined = _normalize(" • ".join(phrases))
    confirmed = []
    for term in terms:
        pattern = r"\b" + re.escape(_normalize(term)) + r"\b"
        if re.search(pattern, joined):
            confirmed.append(term)
    return confirmed


def analyze_text(text: str) -> TextAnalysisResult:
    """Sentimento + frases-chave (Azure) e termos críticos (local)."""
    critical = find_critical_terms(text)

    if not settings.azure.language_configured:
        return TextAnalysisResult(
            configured=False,
            critical_terms=critical,
            error="Azure Language não configurado (defina AZURE_LANGUAGE_KEY/ENDPOINT no .env).",
        )

    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    client = TextAnalyticsClient(
        endpoint=settings.azure.language_endpoint,
        credential=AzureKeyCredential(settings.azure.language_key),
    )
    docs = [text[:5000]] if text.strip() else [""]

    sentiment_doc = client.analyze_sentiment(documents=docs, language="pt")[0]
    phrases_doc = client.extract_key_phrases(documents=docs, language="pt")[0]

    sentiment = getattr(sentiment_doc, "sentiment", "unknown")
    scores = {}
    if hasattr(sentiment_doc, "confidence_scores"):
        cs = sentiment_doc.confidence_scores
        scores = {"positive": cs.positive, "neutral": cs.neutral, "negative": cs.negative}
    key_phrases = list(getattr(phrases_doc, "key_phrases", []))

    return TextAnalysisResult(
        sentiment=sentiment,
        sentiment_scores=scores,
        key_phrases=key_phrases,
        critical_terms=critical,
        critical_terms_azure=crosscheck_terms_with_phrases(critical, key_phrases),
    )


def text_alerts(result: TextAnalysisResult, patient_id: str = "P001") -> list[Alert]:
    """Gera alertas a partir da análise de texto (termos críticos e sentimento negativo)."""
    alerts: list[Alert] = []
    if result.critical_terms:
        alerts.append(
            Alert(
                level=AlertLevel.CRITICAL,
                modality="audio",
                message=f"Termos clínicos críticos na fala: {', '.join(result.critical_terms)}",
                patient_id=patient_id,
                metric="critical_terms",
                value=float(len(result.critical_terms)),
                details={
                    "terms": result.critical_terms,
                    "azure_confirmed": result.critical_terms_azure,
                    "rule": "critical_terms",
                },
            )
        )
    neg = result.sentiment_scores.get("negative", 0.0)
    if result.sentiment == "negative" and neg >= 0.6:
        alerts.append(
            Alert(
                level=AlertLevel.WARNING,
                modality="audio",
                message=f"Sentimento negativo predominante na consulta (score {neg:.2f})",
                patient_id=patient_id,
                metric="sentiment_negative",
                value=float(neg),
                details={"rule": "sentiment"},
            )
        )
    return alerts
