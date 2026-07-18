"""Rótulos em português (pt-BR) da camada hospitalar.

Fonte única de tradução entre as chaves internas (inglês, estáveis no banco)
e o texto exibido a médicos/equipe — usada pela página Pacientes do dashboard
e pelos serviços que geram títulos de alertas.
"""

ADMISSION_PT = {
    "outpatient": "Ambulatorial",
    "admitted": "Internado",
    "icu": "UTI",
    "discharged": "Alta",
}

MODALITY_PT = {"audio": "Áudio", "video": "Vídeo", "text": "Documento", "image": "Imagem"}
MODALITY_ICON = {"audio": "🎙️", "video": "🎥", "text": "📄", "image": "🖼️"}

STATUS_PT = {
    "pending": "🕓 Aguardando análise",
    "processing": "⚙️ Processando",
    "done": "✅ Concluído",
    "failed": "❌ Falhou",
}

ANALYSIS_PT = {
    "disartria": "Análise vocal (fadiga/disartria)",
    "transcricao": "Transcrição da consulta",
    "ner_clinico": "Conteúdo da fala (sentimento e termos)",
    "laudo_nlp": "Análise do documento",
    "pose_anomaly": "Eventos do vídeo (movimento e cena)",
    "movement_metrics": "Métricas de movimento da sessão",
    "news2": "Escore clínico NEWS2",
    "interacao_medicamentosa": "Interação medicamentosa",
}

ENGINE_PT = {
    "praat_librosa": "Praat/librosa (local)",
    "azure_speech": "Azure Fala (nuvem)",
    "azure_language": "Azure Linguagem (nuvem)",
    "local_terms": "Termos clínicos locais",
    "yolov8": "YOLOv8",
    "yolov8_pose": "YOLOv8-pose",
    "local_rules": "Regras clínicas",
}

SOURCE_PT = {
    "vital_signs": "Sinais vitais",
    "audio": "Áudio",
    "video": "Vídeo",
    "nlp": "Texto clínico",
    "prescription": "Prescrições",
    "fusion": "Fusão multimodal",
}

VITAL_PT = {
    "heart_rate": "frequência cardíaca",
    "spo2": "SpO₂",
    "temperature": "temperatura",
    "systolic_bp": "PA sistólica",
    "diastolic_bp": "PA diastólica",
    "respiratory_rate": "frequência respiratória",
}

SENTIMENT_PT = {
    "positive": "positivo", "neutral": "neutro", "negative": "negativo",
    "mixed": "misto", "unknown": "indeterminado",
}

NEWS2_PART_PT = {
    "respiratory_rate": "Frequência respiratória",
    "spo2": "Saturação de O₂",
    "systolic_bp": "PA sistólica",
    "heart_rate": "Frequência cardíaca",
    "temperature": "Temperatura",
}


def title_pt(title: str) -> str:
    """Traduz nomes de métricas embutidos em títulos de alertas já gravados."""
    for raw, pt in VITAL_PT.items():
        title = title.replace(raw, pt)
    return title
