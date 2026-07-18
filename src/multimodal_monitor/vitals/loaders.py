"""Carregamento de sinais vitais.

Duas fontes:
  1. PhysioNet BIDMC (real, acesso aberto) via ``wfdb`` — FC, respiração, SpO2.
  2. Gerador sintético com anomalias injetadas — garante eventos demonstráveis
     e não depende de rede. É o fallback usado por padrão na demo.

Ambos retornam um ``pandas.DataFrame`` com a coluna ``t`` (segundos) e uma coluna
por sinal, de modo que os detectores funcionam igual para real ou sintético.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PHYSIONET_DIR

VITAL_COLUMNS = ["hr", "spo2", "resp", "sbp"]


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_synthetic_vitals(
    duration_s: int = 600,
    fs: float = 1.0,
    seed: int | None = 42,
    inject_anomalies: bool = True,
) -> pd.DataFrame:
    """Gera sinais vitais sintéticos realistas com anomalias opcionais.

    Baseline: FC ~75 bpm, SpO2 ~97%, respiração ~16 irpm, PA sistólica ~120 mmHg,
    com oscilação lenta + ruído. Quando ``inject_anomalies`` é True, insere três
    eventos clínicos clássicos em janelas conhecidas (úteis para validar detecção).

    Returns:
        DataFrame com colunas ``t, hr, spo2, resp, sbp`` e atributo
        ``.attrs["anomaly_windows"]`` descrevendo os eventos injetados.
    """
    rng = _rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs

    def baseline(mean, drift_amp, drift_period, noise_sd):
        drift = drift_amp * np.sin(2 * np.pi * t / drift_period)
        return mean + drift + rng.normal(0, noise_sd, n)

    hr = baseline(75, 4, 120, 1.5)
    spo2 = baseline(97, 0.6, 180, 0.4)
    resp = baseline(16, 1.5, 90, 0.6)
    sbp = baseline(120, 6, 150, 2.0)

    anomaly_windows: list[dict] = []

    if inject_anomalies:
        def window(frac_start, frac_end):
            return int(n * frac_start), int(n * frac_end)

        # 1) Taquicardia progressiva (ex.: dor/ansiedade/sepse inicial)
        a, b = window(0.30, 0.38)
        ramp = np.linspace(0, 70, b - a)
        hr[a:b] += ramp
        anomaly_windows.append(
            {"metric": "hr", "start_s": t[a], "end_s": t[b - 1], "type": "taquicardia"}
        )

        # 2) Dessaturação de oxigênio (evento respiratório)
        a, b = window(0.55, 0.62)
        dip = -12 * np.hanning(b - a)
        spo2[a:b] += dip
        resp[a:b] += 8 * np.hanning(b - a)  # taquipneia compensatória
        anomaly_windows.append(
            {"metric": "spo2", "start_s": t[a], "end_s": t[b - 1], "type": "dessaturacao"}
        )

        # 3) Hipotensão súbita (queda de PA)
        a, b = window(0.78, 0.85)
        drop = -40 * np.hanning(b - a)
        sbp[a:b] += drop
        hr[a:b] += 25 * np.hanning(b - a)   # taquicardia reflexa
        anomaly_windows.append(
            {"metric": "sbp", "start_s": t[a], "end_s": t[b - 1], "type": "hipotensao"}
        )

    df = pd.DataFrame({"t": t, "hr": hr, "spo2": spo2.clip(70, 100), "resp": resp, "sbp": sbp})
    df.attrs["anomaly_windows"] = anomaly_windows
    df.attrs["source"] = "synthetic"
    return df


def load_bidmc_record(record_id: str = "bidmc01", target_fs: float = 1.0) -> pd.DataFrame:
    """Carrega um registro do BIDMC PPG and Respiration (PhysioNet, acesso aberto).

    Baixa via ``wfdb`` se ainda não estiver em ``data/physionet``. Extrai os sinais
    numéricos (HR, SpO2, RESP) já derivados presentes no dataset e reamostra para
    ``target_fs`` para casar com o restante do pipeline.

    Levanta ImportError/erro de rede se indisponível — o chamador deve cair no
    gerador sintético.
    """
    import wfdb  # import tardio: só quando realmente for usar dados reais

    PHYSIONET_DIR.mkdir(parents=True, exist_ok=True)
    # O dataset numérico fica no diretório "bidmc_csv"/registros "*n"; usamos a API pn.
    record = wfdb.rdrecord(
        record_id + "n",
        pn_dir="bidmc/1.0.0",
    )
    sig = pd.DataFrame(record.p_signal, columns=record.sig_name)

    colmap = {}
    for name in sig.columns:
        low = name.lower()
        if "hr" in low or "pulse" in low:
            colmap[name] = "hr"
        elif "spo2" in low or "sao2" in low:
            colmap[name] = "spo2"
        elif "resp" in low:
            colmap[name] = "resp"
    sig = sig.rename(columns=colmap)

    keep = [c for c in ("hr", "spo2", "resp") if c in sig.columns]
    sig = sig[keep].interpolate().bfill().ffill()

    fs = record.fs or 1.0
    step = max(int(round(fs / target_fs)), 1)
    sig = sig.iloc[::step].reset_index(drop=True)
    sig.insert(0, "t", np.arange(len(sig)) / target_fs)
    sig.attrs["anomaly_windows"] = []
    sig.attrs["source"] = f"bidmc:{record_id}"
    return sig


def load_vitals(prefer_real: bool = False, **kwargs) -> pd.DataFrame:
    """Ponto de entrada único: tenta o dataset real se pedido, senão sintético."""
    if prefer_real:
        try:
            return load_bidmc_record(**{k: v for k, v in kwargs.items() if k in {"record_id", "target_fs"}})
        except Exception as exc:  # rede/lib indisponível → fallback garantido
            print(f"[vitals] BIDMC indisponível ({exc}); usando dados sintéticos.")
    return generate_synthetic_vitals(**{k: v for k, v in kwargs.items() if k in {"duration_s", "fs", "seed", "inject_anomalies"}})
