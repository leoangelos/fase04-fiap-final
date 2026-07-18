"""Configuração central: chaves Azure, caminhos e thresholds dos detectores."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
PHYSIONET_DIR = DATA_DIR / "physionet"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class AzureConfig:
    speech_key: str = os.getenv("AZURE_SPEECH_KEY", "")
    speech_region: str = os.getenv("AZURE_SPEECH_REGION", "brazilsouth")
    language_key: str = os.getenv("AZURE_LANGUAGE_KEY", "")
    language_endpoint: str = os.getenv("AZURE_LANGUAGE_ENDPOINT", "")
    speech_locale: str = "pt-BR"

    @property
    def speech_configured(self) -> bool:
        return bool(self.speech_key and self.speech_region)

    @property
    def language_configured(self) -> bool:
        return bool(self.language_key and self.language_endpoint)


@dataclass(frozen=True)
class VitalThresholds:
    """Limites clínicos duros para alertas imediatos (adulto, valores usuais de literatura)."""

    hr_low: float = 50.0        # bradicardia (bpm)
    hr_high: float = 120.0      # taquicardia (bpm)
    spo2_low: float = 90.0      # dessaturação (%)
    resp_low: float = 8.0       # bradipneia (irpm)
    resp_high: float = 25.0     # taquipneia (irpm)
    sbp_low: float = 90.0       # hipotensão sistólica (mmHg)
    sbp_high: float = 180.0     # crise hipertensiva (mmHg)


@dataclass(frozen=True)
class VideoThresholds:
    fall_hip_drop_ratio: float = 0.20   # queda do y do quadril (fração da IMAGEM) na janela de ~0.5 s
                                        # (0.20 capta quedas amortecidas reais; 0.25 só pegava tombos instantâneos)
    immobility_seconds: float = 5.0     # tempo sem movimento significativo
    movement_zscore: float = 3.0        # desvio da baseline do exercício
    min_keypoint_conf: float = 0.5


@dataclass(frozen=True)
class SceneThresholds:
    """Vigilância de cena por detecção de objetos (YOLOv8 padrão).

    A zona crítica é um retângulo em coordenadas normalizadas (0–1) da imagem —
    ex.: área restrita de acesso a medicamentos/equipamentos. Pessoa cuja base
    (ponto de contato com o chão) entra na zona gera alerta; objetos de classes
    fora de ``expected_classes`` que persistem na cena também.

    O retângulo padrão (faixa direita, próxima à câmera) está ajustado à cena dos
    vídeos de demonstração; em produção configura-se por câmera/leito.
    """

    zone_name: str = "área restrita"
    zone_x1: float = 0.62
    zone_y1: float = 0.50
    zone_x2: float = 1.0
    zone_y2: float = 1.0
    expected_classes: tuple[str, ...] = ("person",)
    min_object_frames: int = 3      # persistência mínima p/ alertar objeto inesperado
    min_conf: float = 0.35
    # o alerta de objeto inesperado exige confiança MEDIANA da classe ≥ este valor
    # (classes reais ficam ≥0.6; erros de classificação do modelo ficam abaixo)
    object_alert_conf: float = 0.60
    # modelo de detecção de objetos: o "s" (small) classifica bem melhor que o
    # nano (ex.: guarda-chuva vs. skate) e ainda roda em CPU; a pose segue no nano
    model_name: str = "yolov8s.pt"


@dataclass(frozen=True)
class AudioThresholds:
    """Valores de referência aproximados da literatura de análise vocal (Praat).

    Usados como indicativos de fadiga/disartria — não constituem diagnóstico.
    """

    jitter_local_pct: float = 1.04      # acima disso: instabilidade de frequência
    shimmer_local_pct: float = 3.81     # acima disso: instabilidade de amplitude
    hnr_db_min: float = 13.0            # abaixo disso: voz soprosa/ruidosa
    pause_ratio_max: float = 0.45       # fração de silêncio na fala
    speech_rate_min: float = 2.0        # sílabas/segundo (fala lentificada)


CRITICAL_TERMS_PT = [
    "dor no peito",
    "dor torácica",
    "falta de ar",
    "dificuldade para respirar",
    "desmaio",
    "desmaiei",
    "tontura",
    "palpitação",
    "sangramento",
    "confusão mental",
    "formigamento no braço",
    "visão embaçada",
    "não consigo andar",
    "queda",
    "caí",
    "febre alta",
    "vômito",
]


@dataclass(frozen=True)
class Settings:
    azure: AzureConfig = field(default_factory=AzureConfig)
    vitals: VitalThresholds = field(default_factory=VitalThresholds)
    video: VideoThresholds = field(default_factory=VideoThresholds)
    scene: SceneThresholds = field(default_factory=SceneThresholds)
    audio: AudioThresholds = field(default_factory=AudioThresholds)


settings = Settings()
