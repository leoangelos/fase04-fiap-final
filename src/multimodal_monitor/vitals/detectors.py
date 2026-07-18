"""Detecção de anomalias em séries temporais de sinais vitais.

Três técnicas complementares (boas para o relatório):
  1. Limites clínicos duros  → alerta imediato quando um valor sai da faixa segura.
  2. Rolling z-score          → detecta mudanças bruscas relativas à própria linha de base.
  3. Isolation Forest         → detecta padrões multivariados anômalos (combinações raras).

Todas produzem ``Alert`` objects para o ``AlertManager``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..alerts.alert_manager import Alert, AlertLevel
from ..config import VitalThresholds, settings

_METRIC_LABEL = {
    "hr": "Frequência cardíaca",
    "spo2": "Saturação de O₂",
    "resp": "Frequência respiratória",
    "sbp": "Pressão arterial sistólica",
}
_METRIC_UNIT = {"hr": "bpm", "spo2": "%", "resp": "irpm", "sbp": "mmHg"}


def _merge_events(mask: pd.Series, t: pd.Series, min_gap: float = 3.0) -> list[tuple[float, float, int]]:
    """Agrupa amostras contíguas marcadas em eventos (start_s, end_s, n_amostras)."""
    events: list[tuple[float, float, int]] = []
    idx = np.where(mask.to_numpy())[0]
    if len(idx) == 0:
        return events
    start = prev = idx[0]
    count = 1
    for i in idx[1:]:
        if t.iloc[i] - t.iloc[prev] <= min_gap + 1e-9:
            prev = i
            count += 1
        else:
            events.append((float(t.iloc[start]), float(t.iloc[prev]), count))
            start = prev = i
            count = 1
    events.append((float(t.iloc[start]), float(t.iloc[prev]), count))
    return events


def detect_threshold_violations(
    df: pd.DataFrame,
    patient_id: str = "P001",
    thresholds: VitalThresholds | None = None,
) -> list[Alert]:
    """Regra 1: valores fora dos limites clínicos → alerta CRITICAL/WARNING."""
    th = thresholds or settings.vitals
    t = df["t"]
    alerts: list[Alert] = []

    checks = [
        ("hr", df.get("hr"), th.hr_low, th.hr_high, AlertLevel.CRITICAL),
        ("spo2", df.get("spo2"), th.spo2_low, None, AlertLevel.CRITICAL),
        ("resp", df.get("resp"), th.resp_low, th.resp_high, AlertLevel.WARNING),
        ("sbp", df.get("sbp"), th.sbp_low, th.sbp_high, AlertLevel.CRITICAL),
    ]
    for metric, series, low, high, level in checks:
        if series is None:
            continue
        cond = pd.Series(False, index=series.index)
        if low is not None:
            cond |= series < low
        if high is not None:
            cond |= series > high
        for start_s, end_s, _ in _merge_events(cond, t):
            seg = series[(t >= start_s) & (t <= end_s)]
            extreme = seg.min() if (low is not None and seg.min() < (low or -np.inf)) else seg.max()
            alerts.append(
                Alert(
                    level=level,
                    modality="vitals",
                    message=(
                        f"{_METRIC_LABEL[metric]} fora do limite clínico: "
                        f"{extreme:.0f} {_METRIC_UNIT[metric]}"
                    ),
                    patient_id=patient_id,
                    timestamp=start_s,
                    metric=metric,
                    value=float(extreme),
                    details={"end_s": end_s, "rule": "threshold"},
                )
            )
    return alerts


def detect_zscore_anomalies(
    df: pd.DataFrame,
    patient_id: str = "P001",
    window: int = 30,
    z_thresh: float = 3.0,
) -> list[Alert]:
    """Regra 2: desvio brusco em relação à média móvel (mudança de tendência)."""
    t = df["t"]
    alerts: list[Alert] = []
    for metric in ("hr", "spo2", "resp", "sbp"):
        if metric not in df:
            continue
        s = df[metric]
        roll_mean = s.rolling(window, min_periods=window // 2).mean()
        roll_std = s.rolling(window, min_periods=window // 2).std().replace(0, np.nan)
        z = (s - roll_mean) / roll_std
        cond = z.abs() > z_thresh
        for start_s, end_s, n in _merge_events(cond.fillna(False), t):
            if n < 2:  # ignora spikes isolados de 1 amostra (provável ruído)
                continue
            seg_z = z[(t >= start_s) & (t <= end_s)]
            peak = seg_z.abs().max()
            alerts.append(
                Alert(
                    level=AlertLevel.WARNING,
                    modality="vitals",
                    message=(
                        f"Variação abrupta em {_METRIC_LABEL[metric].lower()} "
                        f"(z={peak:.1f}) fora da linha de base"
                    ),
                    patient_id=patient_id,
                    timestamp=start_s,
                    metric=metric,
                    value=float(peak),
                    details={"end_s": end_s, "rule": "zscore"},
                )
            )
    return alerts


def detect_multivariate_anomalies(
    df: pd.DataFrame,
    patient_id: str = "P001",
    contamination: float = 0.03,
    random_state: int = 42,
) -> list[Alert]:
    """Regra 3: Isolation Forest sobre todos os sinais → combinações raras."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    feats = [c for c in ("hr", "spo2", "resp", "sbp") if c in df]
    if len(feats) < 2:
        return []

    X = StandardScaler().fit_transform(df[feats].to_numpy())
    model = IsolationForest(contamination=contamination, random_state=random_state)
    pred = model.fit_predict(X)          # -1 = anomalia
    score = -model.score_samples(X)      # quanto maior, mais anômalo

    t = df["t"]
    cond = pd.Series(pred == -1, index=df.index)
    alerts: list[Alert] = []
    for start_s, end_s, n in _merge_events(cond, t):
        if n < 3:
            continue
        seg = (t >= start_s) & (t <= end_s)
        peak_score = float(score[seg.to_numpy()].max())
        # aponta qual sinal mais destoou nesse trecho
        seg_df = df.loc[seg, feats]
        z = ((seg_df - df[feats].mean()) / df[feats].std()).abs().max()
        driver = z.idxmax()
        alerts.append(
            Alert(
                level=AlertLevel.WARNING,
                modality="vitals",
                message=(
                    f"Padrão multivariado anômalo (Isolation Forest), "
                    f"principal sinal: {_METRIC_LABEL[driver].lower()}"
                ),
                patient_id=patient_id,
                timestamp=start_s,
                metric=driver,
                value=peak_score,
                details={"end_s": end_s, "rule": "isolation_forest", "features": feats},
            )
        )
    return alerts


def detect_all(df: pd.DataFrame, patient_id: str = "P001") -> list[Alert]:
    """Executa as três técnicas e devolve todos os alertas encontrados."""
    return (
        detect_threshold_violations(df, patient_id)
        + detect_zscore_anomalies(df, patient_id)
        + detect_multivariate_anomalies(df, patient_id)
    )
