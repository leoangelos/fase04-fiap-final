"""Features de movimento derivadas da sequência de poses.

A partir dos keypoints por frame calculamos, ao longo do tempo:
  - ângulos articulares (joelho, cotovelo, quadril) — amplitude do exercício;
  - velocidade dos punhos/tornozelos — intensidade do movimento;
  - índice global de movimento — quanto o corpo se mexe (usado para imobilidade);
  - altura do quadril normalizada — para detectar quedas;
  - simetria esquerda/direita — desvios posturais.

Tudo retorna um ``pandas.DataFrame`` indexado pelo tempo, pronto para reusar os
detectores de série temporal de ``vitals.detectors``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pose_extractor import KP, PoseSequence


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ângulo (graus) no vértice b, formado por a-b-c. NaN se faltar keypoint."""
    if np.any(np.isnan([*a, *b, *c])):
        return np.nan
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return np.nan
    cos = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _mean_point(*points: np.ndarray) -> np.ndarray:
    """Média de pontos ignorando NaN, sem disparar warning de slice vazio."""
    stack = np.array(points, dtype=float)
    if np.all(np.isnan(stack)):
        return np.array([np.nan, np.nan])
    return np.nanmean(stack, axis=0)


# Keypoints centrais que definem "pessoa presente e bem detectada" no frame.
_CORE = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]


def _torso_height(kp: np.ndarray) -> float:
    """Distância vertical ombro→tornozelo, usada para normalizar posições."""
    sh = _mean_point(kp[KP["left_shoulder"]], kp[KP["right_shoulder"]])
    ank = _mean_point(kp[KP["left_ankle"]], kp[KP["right_ankle"]])
    if np.any(np.isnan([*sh, *ank])):
        return np.nan
    return abs(ank[1] - sh[1]) or np.nan


def _is_valid_pose(kp: np.ndarray, min_core: int = 3) -> bool:
    """True quando há pessoa bem detectada (≥ min_core keypoints centrais visíveis)."""
    present = sum(0 if np.any(np.isnan(kp[KP[name]])) else 1 for name in _CORE)
    return present >= min_core


def compute_movement_features(seq: PoseSequence) -> pd.DataFrame:
    """Constrói o DataFrame de features de movimento por frame.

    Frames sem pessoa bem detectada recebem ``valid=False`` e não têm ângulos
    calculados — isso evita que detecções parciais poluam a linha de base e gerem
    z-scores absurdos ou falsa imobilidade.
    """
    rows = []
    prev_kp = None
    prev_t = None
    prev_valid = False

    for p in seq.poses:
        kp = p.keypoints
        valid = _is_valid_pose(kp)
        row = {"t": p.time_s, "n_persons": p.n_persons, "valid": valid}

        if valid:
            row["knee_left"] = _angle(kp[KP["left_hip"]], kp[KP["left_knee"]], kp[KP["left_ankle"]])
            row["knee_right"] = _angle(kp[KP["right_hip"]], kp[KP["right_knee"]], kp[KP["right_ankle"]])
            row["elbow_left"] = _angle(kp[KP["left_shoulder"]], kp[KP["left_elbow"]], kp[KP["left_wrist"]])
            row["elbow_right"] = _angle(kp[KP["right_shoulder"]], kp[KP["right_elbow"]], kp[KP["right_wrist"]])
            row["hip_left"] = _angle(kp[KP["left_shoulder"]], kp[KP["left_hip"]], kp[KP["left_knee"]])
            row["hip_right"] = _angle(kp[KP["right_shoulder"]], kp[KP["right_hip"]], kp[KP["right_knee"]])
            hip = _mean_point(kp[KP["left_hip"]], kp[KP["right_hip"]])
            th = _torso_height(kp)
            row["hip_y_norm"] = (hip[1] / seq.height) if not np.isnan(hip[1]) else np.nan
            row["torso_height"] = th
        else:
            for c in ("knee_left", "knee_right", "elbow_left", "elbow_right",
                      "hip_left", "hip_right", "hip_y_norm", "torso_height"):
                row[c] = np.nan
            th = np.nan

        # velocidade só faz sentido entre dois frames válidos consecutivos
        if valid and prev_valid and prev_t is not None and p.time_s > prev_t:
            dt = p.time_s - prev_t
            ext = ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
            disp = [np.linalg.norm(kp[KP[e]] - prev_kp[KP[e]]) for e in ext]
            v = np.nanmean(disp) / dt if not np.all(np.isnan(disp)) else np.nan
            row["motion"] = float(v / th) if (th and not np.isnan(th) and not np.isnan(v)) else (
                float(v) if not np.isnan(v) else np.nan
            )
        else:
            row["motion"] = np.nan       # NaN (não 0): "desconhecido", não "parado"

        row["knee_asymmetry"] = (
            abs(row["knee_left"] - row["knee_right"])
            if not np.isnan(row["knee_left"] + row["knee_right"]) else np.nan
        )

        rows.append(row)
        prev_kp, prev_t, prev_valid = kp, p.time_s, valid
        prev_kp, prev_t = kp, p.time_s

    df = pd.DataFrame(rows)
    # suaviza a métrica de movimento (reduz jitter da estimativa de pose)
    if "motion" in df:
        df["motion"] = df["motion"].rolling(3, min_periods=1, center=True).median()
    df.attrs["fps"] = seq.fps
    return df


def movement_index_timeseries(features: pd.DataFrame, bin_s: float = 1.0) -> pd.DataFrame:
    """Agrega o movimento em janelas de tempo → série "índice de movimentação".

    Reutilizado como sinal para os detectores de ``vitals`` (requisito 3c do
    edital: padrões de movimentação do paciente durante a internação).
    """
    df = features[["t", "motion"]].copy()
    df["bin"] = (df["t"] // bin_s).astype(int)
    agg = df.groupby("bin").agg(t=("t", "first"), motion=("motion", "mean")).reset_index(drop=True)
    return agg
