"""Detecção de eventos anômalos em vídeo clínico.

Combina regras interpretáveis (boas para relatório médico) com desvio estatístico
sobre a linha de base do próprio exercício:

  - Queda: queda brusca da altura do quadril entre frames próximos.
  - Imobilidade prolongada: índice de movimento ~0 por > N segundos.
  - Amplitude/postura fora do padrão: z-score dos ângulos vs. baseline.
  - Pessoa fora da zona segura: nº de pessoas ou saída da região central (YOLO).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..alerts.alert_manager import Alert, AlertLevel
from ..config import VideoThresholds, settings
from .movement_features import movement_index_timeseries


def detect_falls(features: pd.DataFrame, th: VideoThresholds, patient_id: str) -> list[Alert]:
    """Queda: aumento súbito de hip_y_norm (y cresce para baixo na imagem).

    A janela de ~0.5 s é derivada do espaçamento REAL das amostras (robusto a
    ``stride`` na extração de pose — usar o fps do vídeo dobraria a janela com
    stride 2). Meio segundo cobre tanto tombos instantâneos quanto quedas
    amortecidas, em que a pessoa desaba em fases.
    """
    alerts: list[Alert] = []
    s = features["hip_y_norm"]
    t = features["t"]
    fps = features.attrs.get("fps", 30.0)
    dt = float(t.diff().median()) if len(t) > 1 else 0.0
    if not dt or dt <= 0:
        dt = 1.0 / fps
    win = max(int(round(0.5 / dt)), 1)
    delta = s.diff(win)
    hits = delta > th.fall_hip_drop_ratio
    fired_until = -1.0
    for i in np.where(hits.fillna(False).to_numpy())[0]:
        if t.iloc[i] < fired_until:
            continue
        fired_until = t.iloc[i] + 2.0   # evita múltiplos alertas da mesma queda
        alerts.append(
            Alert(
                level=AlertLevel.CRITICAL,
                modality="video",
                message="Possível queda detectada (queda abrupta da altura do quadril)",
                patient_id=patient_id,
                timestamp=float(t.iloc[i]),
                metric="hip_y_norm",
                value=float(delta.iloc[i]),
                details={"rule": "fall"},
            )
        )
    return alerts


def detect_immobility(features: pd.DataFrame, th: VideoThresholds, patient_id: str) -> list[Alert]:
    """Imobilidade: pessoa PRESENTE, porém praticamente sem movimento, por mais de
    ``immobility_seconds``.

    O limiar de "parado" é relativo ao nível típico de ATIVIDADE do próprio vídeo
    (quantil 75 % do movimento positivo): usar a mediana subestimaria a linha de
    base quando o período imóvel domina o clipe, pois o jitter residual do encoder/
    estimador de pose puxaria o limiar para dentro do próprio ruído. O movimento é
    suavizado (~0.5 s) e toleramos lacunas curtas (jitter da estimativa de pose)
    para não fragmentar o período imóvel. Frames sem pessoa detectada (motion=NaN)
    interrompem a contagem.
    """
    alerts: list[Alert] = []
    t = features["t"]
    valid = features.get("valid", pd.Series(True, index=features.index))
    fps = features.attrs.get("fps", 30.0)

    smooth_win = max(int(round(0.5 * fps)), 3)
    motion = features["motion"].rolling(smooth_win, min_periods=1, center=True).median()
    positive = motion[motion > 0]
    if len(positive) < smooth_win:
        return alerts
    still_thresh = max(0.20 * float(positive.quantile(0.75)), 1e-4)
    still = (valid & motion.notna() & (motion <= still_thresh)).to_numpy()

    gap_tol = max(int(round(0.4 * fps)), 2)   # nº de frames "não parados" tolerados no meio
    i, n = 0, len(still)
    while i < n:
        if not still[i]:
            i += 1
            continue
        j = i
        gap = 0
        last_still = i
        while j < n:
            if still[j]:
                last_still = j
                gap = 0
            else:
                gap += 1
                if gap > gap_tol:
                    break
            j += 1
        duration = t.iloc[last_still] - t.iloc[i]
        if duration >= th.immobility_seconds:
            alerts.append(_immobility_alert(patient_id, float(t.iloc[i]), float(t.iloc[last_still])))
        i = j + 1
    return alerts


def _immobility_alert(patient_id: str, start: float, end: float) -> Alert:
    return Alert(
        level=AlertLevel.WARNING,
        modality="video",
        message=f"Imobilidade prolongada por {end - start:.1f}s (sem movimento significativo)",
        patient_id=patient_id,
        timestamp=float(start),
        metric="motion",
        value=0.0,
        details={"rule": "immobility", "end_s": float(end)},
    )


def detect_posture_deviations(
    features: pd.DataFrame, th: VideoThresholds, patient_id: str, min_valid: int = 15
) -> list[Alert]:
    """Amplitude/postura fora do padrão via z-score robusto dos ângulos articulares.

    A baseline é a estatística dos frames VÁLIDOS do próprio vídeo (mediana/IQR) —
    desvios fortes sinalizam execução irregular do exercício ou postura anômala.
    Exige um mínimo de frames válidos para não tirar conclusão de ruído.
    """
    alerts: list[Alert] = []
    t = features["t"]
    valid = features.get("valid", pd.Series(True, index=features.index))
    if valid.sum() < min_valid:
        return alerts   # amostra insuficiente para estabelecer linha de base confiável

    fps = features.attrs.get("fps", 30.0)
    smooth_win = max(int(round(0.4 * fps)), 3)      # ~0.4 s: remove glitches de estimativa
    min_consec = max(int(round(0.6 * fps)), 4)      # desvio precisa persistir ~0.6 s

    angle_cols = ["knee_left", "knee_right", "elbow_left", "elbow_right", "hip_left", "hip_right"]
    for col in angle_cols:
        if col not in features:
            continue
        # suaviza dentro dos frames válidos antes de medir o desvio
        s = features[col].where(valid).rolling(smooth_win, min_periods=1, center=True).median()
        med = s.median()
        iqr = s.quantile(0.75) - s.quantile(0.25)
        if np.isnan(iqr) or iqr < 5.0:              # ângulo estável: sem base p/ z-score confiável
            continue
        z = (s - med) / (iqr / 1.349)               # IQR→desvio-padrão robusto
        hits = (z.abs() > th.movement_zscore).fillna(False).to_numpy()

        # agrupa em corridas contíguas e só dispara se sustentada por min_consec frames
        i = 0
        n = len(hits)
        while i < n:
            if not hits[i]:
                i += 1
                continue
            j = i
            while j < n and hits[j]:
                j += 1
            if j - i >= min_consec:
                seg = slice(i, j)
                peak_idx = i + int(np.nanargmax(np.abs(z.to_numpy()[seg])))
                alerts.append(
                    Alert(
                        level=AlertLevel.WARNING,
                        modality="video",
                        message=(
                            f"Amplitude/postura fora do padrão em {col} "
                            f"(z={z.iloc[peak_idx]:.1f}) por {t.iloc[j-1]-t.iloc[i]:.1f}s"
                        ),
                        patient_id=patient_id,
                        timestamp=float(t.iloc[i]),
                        metric=col,
                        value=float(features[col].iloc[peak_idx]),
                        details={"rule": "posture", "end_s": float(t.iloc[j - 1])},
                    )
                )
            i = j
    return alerts


def detect_safe_zone(features: pd.DataFrame, patient_id: str, max_persons: int = 1) -> list[Alert]:
    """Pessoa a mais na cena (ex.: paciente deveria estar sozinho na zona monitorada)."""
    alerts: list[Alert] = []
    if "n_persons" not in features:
        return alerts
    t = features["t"]
    hits = features["n_persons"] > max_persons
    fired_until = -1.0
    for i in np.where(hits.to_numpy())[0]:
        if t.iloc[i] < fired_until:
            continue
        fired_until = t.iloc[i] + 3.0
        alerts.append(
            Alert(
                level=AlertLevel.INFO,
                modality="video",
                message=f"{int(features['n_persons'].iloc[i])} pessoas na cena (esperado ≤ {max_persons})",
                patient_id=patient_id,
                timestamp=float(t.iloc[i]),
                metric="n_persons",
                value=float(features["n_persons"].iloc[i]),
                details={"rule": "safe_zone"},
            )
        )
    return alerts


def detect_video_anomalies(
    features: pd.DataFrame,
    patient_id: str = "P001",
    thresholds: VideoThresholds | None = None,
    include_movement_timeseries: bool = True,
) -> list[Alert]:
    """Executa todas as regras de vídeo e (opcional) o detector de série temporal
    sobre o índice de movimentação (ponte com o módulo de vitais)."""
    th = thresholds or settings.video
    alerts = (
        detect_falls(features, th, patient_id)
        + detect_immobility(features, th, patient_id)
        + detect_posture_deviations(features, th, patient_id)
        + detect_safe_zone(features, patient_id)
    )

    if include_movement_timeseries:
        from ..vitals.detectors import detect_zscore_anomalies

        mv = movement_index_timeseries(features).rename(columns={"motion": "hr"})
        # reusa o z-score genérico; renomeamos p/ "hr" só para reaproveitar a função,
        # e reescrevemos a mensagem para o contexto de movimentação.
        for a in detect_zscore_anomalies(mv, patient_id, window=10, z_thresh=3.0):
            a.modality = "video"
            a.metric = "movement_index"
            a.message = "Mudança abrupta no padrão de movimentação do paciente"
            a.details["rule"] = "movement_pattern"
            alerts.append(a)

    return sorted(alerts, key=lambda al: al.timestamp)
