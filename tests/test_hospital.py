"""Testes da camada hospitalar que NÃO dependem de Supabase/Redis.

Cobrem o NEWS2 (baseline clínico), o z-score em janela deslizante (fallback em
memória) e a fusão multimodal com decaimento temporal (repositório simulado).
"""

from datetime import datetime, timedelta, timezone

import pytest

from hospital_ai.services import anomaly, fusion
from hospital_ai.services.news2 import news2_score


# ─────────────────────────── NEWS2 ───────────────────────────────────────────

def test_news2_paciente_estavel_sem_alerta():
    r = news2_score(respiratory_rate=16, spo2=97, systolic_bp=120,
                    heart_rate=75, temperature=36.8)
    assert r["total"] == 0
    assert r["severity"] is None


def test_news2_deterioracao_e_critico():
    r = news2_score(respiratory_rate=28, spo2=89, systolic_bp=85,
                    heart_rate=135, temperature=39.4)
    # 3 (FR≥25) + 3 (SpO2≤91) + 3 (PAS≤90) + 3 (FC≥131) + 2 (T≥39.1) = 14
    assert r["total"] == 14
    assert r["severity"] == "critical"
    assert r["risk_score"] == 1.0


def test_news2_parametro_unico_vermelho_vira_warning():
    # total < 5, mas um parâmetro pontuando 3 já exige revisão (regra NEWS2)
    r = news2_score(respiratory_rate=16, spo2=90, systolic_bp=120,
                    heart_rate=75, temperature=36.8)
    assert r["parts"]["spo2"] == 3
    assert r["severity"] == "warning"


def test_news2_campos_ausentes_pontuam_zero():
    r = news2_score(heart_rate=75)
    assert r["total"] == 0 and r["severity"] is None


# ─────────────────── z-score em janela deslizante ────────────────────────────

@pytest.fixture(autouse=True)
def _janela_limpa(monkeypatch):
    # força o fallback em memória mesmo se houver REDIS_URL real no ambiente —
    # testes jamais devem escrever no Redis do usuário
    monkeypatch.setattr(anomaly, "get_redis", lambda: None)
    anomaly.reset_memory_windows()
    yield
    anomaly.reset_memory_windows()


def test_zscore_sem_historico_nao_dispara():
    for _ in range(5):
        assert anomaly.push_and_score("PX", {"heart_rate": 500}) == []


def test_zscore_detecta_desvio_da_propria_baseline():
    # 15 leituras estáveis (~75 bpm) constroem a janela...
    for i in range(15):
        assert anomaly.push_and_score("PX", {"heart_rate": 75 + (i % 3)}) == []
    # ...um salto para 130 bpm desvia da baseline INDIVIDUAL
    hits = anomaly.push_and_score("PX", {"heart_rate": 130})
    assert len(hits) == 1
    assert hits[0]["metric"] == "heart_rate"
    assert abs(hits[0]["zscore"]) >= 3


def test_zscore_janelas_por_paciente_sao_independentes():
    for i in range(15):
        anomaly.push_and_score("PA", {"spo2": 97.0 + (i % 2) * 0.3})
    # paciente diferente, mesma métrica: sem histórico → sem anomalia
    assert anomaly.push_and_score("PB", {"spo2": 80.0}) == []


# ─────────────────────── fusão multimodal ────────────────────────────────────

class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows
        self.not_ = self

    def __getattr__(self, name):  # select/eq/is_/order/limit → encadeável
        return lambda *a, **k: self

    def execute(self):
        return type("R", (), {"data": self._rows})()


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def test_fusao_pondera_fontes_e_decaimento(monkeypatch):
    rows = [
        {"analysis_type": "news2", "engine": "local_rules",
         "risk_score": 0.9, "created_at": _iso(0.1)},
        {"analysis_type": "disartria", "engine": "praat_librosa",
         "risk_score": 0.5, "created_at": _iso(1.0)},
        # análise de vídeo VELHA (72 h): decaimento deve reduzir o peso a ~1/8
        {"analysis_type": "pose_anomaly", "engine": "yolov8",
         "risk_score": 1.0, "created_at": _iso(72.0)},
    ]
    monkeypatch.setattr(fusion, "table", lambda name: _FakeQuery(rows))
    r = fusion.compute_patient_risk("P1")
    assert set(r["contributions"]) == {"vital_signs", "audio", "video"}
    # dominado pelos vitais recentes (peso 0.35, decaimento ~1)
    assert 0.5 <= r["patient_risk_score"] <= 0.95
    peso_video = r["contributions"]["video"]["weight"]
    assert peso_video < 0.05  # 0.20 * 0.5**(72/24) = 0.025


def test_fusao_usa_apenas_resultado_mais_recente_por_fonte(monkeypatch):
    rows = [  # dois NEWS2: o mais recente (0.1) deve vencer o antigo (0.9)
        {"analysis_type": "news2", "engine": "local_rules",
         "risk_score": 0.1, "created_at": _iso(0.1)},
        {"analysis_type": "news2", "engine": "local_rules",
         "risk_score": 0.9, "created_at": _iso(5.0)},
    ]
    monkeypatch.setattr(fusion, "table", lambda name: _FakeQuery(rows))
    r = fusion.compute_patient_risk("P1")
    assert r["patient_risk_score"] == 0.1


def test_fusao_sem_dados_e_zero(monkeypatch):
    monkeypatch.setattr(fusion, "table", lambda name: _FakeQuery([]))
    assert fusion.compute_patient_risk("P1")["patient_risk_score"] == 0.0
