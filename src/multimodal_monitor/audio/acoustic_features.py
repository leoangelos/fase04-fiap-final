"""Extração de features acústicas para triagem de alterações vocais.

Marcadores clássicos da literatura de análise de voz (Praat) associados a fadiga,
disartria e esforço respiratório:

  - jitter / shimmer  → instabilidade de frequência / amplitude das pregas vocais;
  - HNR               → relação harmônico-ruído (voz soprosa/ruidosa quando baixa);
  - F0 (média/desvio) → frequência fundamental e sua variabilidade;
  - razão de pausas   → fração de silêncio (fala entrecortada / dispneia);
  - taxa de fala      → sílabas/seg aproximada (fala lentificada).

Combina Praat (via ``parselmouth``) para os parâmetros de fonte e ``librosa`` para
segmentação de pausas e envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AcousticFeatures:
    duration_s: float
    f0_mean_hz: float
    f0_std_hz: float
    jitter_local_pct: float
    shimmer_local_pct: float
    hnr_db: float
    pause_ratio: float
    speech_rate_syl_s: float
    voiced_ratio: float

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _load_mono(path: str | Path, target_sr: int = 16000):
    """Carrega o áudio em mono no sample rate alvo (formato exigido pelo Azure STT)."""
    import librosa

    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return y, sr


def extract_acoustic_features(path: str | Path) -> AcousticFeatures:
    """Extrai o conjunto de features de um arquivo de áudio (wav/mp3/m4a)."""
    import librosa
    import numpy as np
    import parselmouth
    from parselmouth.praat import call

    y, sr = _load_mono(path)
    duration = len(y) / sr

    # --- Parâmetros de fonte via Praat ---
    snd = parselmouth.Sound(values=y.astype("float64"), sampling_frequency=sr)
    pitch = snd.to_pitch(pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    f0_voiced = f0[f0 > 0]
    f0_mean = float(np.mean(f0_voiced)) if f0_voiced.size else 0.0
    f0_std = float(np.std(f0_voiced)) if f0_voiced.size else 0.0
    voiced_ratio = float(f0_voiced.size / max(f0.size, 1))

    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    try:
        jitter = call(point_process, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3) * 100
    except Exception:
        jitter = float("nan")
    try:
        shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6) * 100
    except Exception:
        shimmer = float("nan")
    try:
        harmonicity = snd.to_harmonicity_cc()
        hnr = float(call(harmonicity, "Get mean", 0, 0))
    except Exception:
        hnr = float("nan")

    # --- Pausas e taxa de fala via librosa ---
    intervals = librosa.effects.split(y, top_db=30)         # trechos com som
    speech_dur = sum((e - s) for s, e in intervals) / sr if len(intervals) else 0.0
    pause_ratio = float(1 - speech_dur / duration) if duration else 0.0

    # estimativa de sílabas por picos do envelope de energia (aproximação)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(
        onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.2, wait=5
    )
    speech_rate = float(len(peaks) / speech_dur) if speech_dur > 0 else 0.0

    return AcousticFeatures(
        duration_s=round(duration, 2),
        f0_mean_hz=f0_mean,
        f0_std_hz=f0_std,
        jitter_local_pct=float(jitter),
        shimmer_local_pct=float(shimmer),
        hnr_db=hnr,
        pause_ratio=pause_ratio,
        speech_rate_syl_s=speech_rate,
        voiced_ratio=voiced_ratio,
    )
