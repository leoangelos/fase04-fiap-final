"""Replay de sinais vitais em "tempo real" para a demonstração.

Percorre o DataFrame emitindo amostras uma a uma (com um atraso opcional) e,
a cada novo bloco, roda os detectores só sobre a janela mais recente — imitando
um monitor de beira de leito que dispara alertas conforme os dados chegam.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pandas as pd

from ..alerts.alert_manager import Alert
from .detectors import detect_threshold_violations, detect_zscore_anomalies


class VitalsStreamSimulator:
    """Emite amostras sequencialmente e detecta anomalias na janela corrente."""

    def __init__(self, df: pd.DataFrame, patient_id: str = "P001", window: int = 60):
        self.df = df.reset_index(drop=True)
        self.patient_id = patient_id
        self.window = window
        self._seen_keys: set[tuple] = set()

    def _new_alerts(self, window_df: pd.DataFrame) -> list[Alert]:
        """Detecta na janela e filtra os que já foram emitidos (dedupe por metric+timestamp)."""
        found = detect_threshold_violations(window_df, self.patient_id) + detect_zscore_anomalies(
            window_df, self.patient_id
        )
        fresh = []
        for a in found:
            key = (a.metric, a.details.get("rule"), round(a.timestamp, 1))
            if key not in self._seen_keys:
                self._seen_keys.add(key)
                fresh.append(a)
        return fresh

    def stream(self, realtime: bool = False, speed: float = 20.0) -> Iterator[dict]:
        """Gera dicts ``{"row", "index", "new_alerts"}`` a cada amostra.

        Args:
            realtime: se True, dorme entre amostras conforme o passo de tempo.
            speed: fator de aceleração do realtime (20x = 20s de sinal por 1s real).
        """
        prev_t = None
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            lo = max(0, i - self.window + 1)
            window_df = self.df.iloc[lo : i + 1]
            new_alerts = self._new_alerts(window_df)

            if realtime and prev_t is not None:
                dt = (row["t"] - prev_t) / max(speed, 1e-6)
                if dt > 0:
                    time.sleep(min(dt, 0.5))
            prev_t = row["t"]

            yield {"row": row.to_dict(), "index": i, "new_alerts": new_alerts}
