"""Gerenciador central de alertas — reutilizado por todas as modalidades.

Cada detector (vídeo, áudio, sinais vitais, prescrições) emite ``Alert`` objects
com um nível de severidade. O ``AlertManager`` mantém o histórico ordenado,
permite filtrar por paciente/modalidade e expõe a fila para o dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Iterable

# Rótulos em português das modalidades (chave interna → texto exibido ao usuário).
MODALITY_PT = {
    "vitals": "Sinais vitais",
    "audio": "Áudio",
    "video": "Vídeo",
    "prescription": "Prescrições",
    "fusion": "Fusão",
}


class AlertLevel(IntEnum):
    """Severidade crescente. IntEnum permite comparar e ordenar por gravidade."""

    INFO = 0
    WARNING = 1
    CRITICAL = 2

    @property
    def label(self) -> str:
        """Nome interno estável (chave de serialização) — não exibir ao usuário."""
        return {0: "INFO", 1: "WARNING", 2: "CRITICAL"}[int(self)]

    @property
    def pt(self) -> str:
        """Rótulo em português para exibição na interface e nos relatórios."""
        return {0: "INFORMATIVO", 1: "ATENÇÃO", 2: "CRÍTICO"}[int(self)]

    @property
    def emoji(self) -> str:
        return {0: "🟢", 1: "🟡", 2: "🔴"}[int(self)]


@dataclass
class Alert:
    """Um evento anômalo detectado por uma das modalidades."""

    level: AlertLevel
    modality: str          # "vitals" | "audio" | "video" | "prescription" | "fusion"
    message: str
    patient_id: str = "P001"
    timestamp: float = 0.0                 # segundos relativos ao início do stream/exame
    wall_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metric: str | None = None              # ex.: "hr", "spo2", "jitter"
    value: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.label      # chave interna (INFO/WARNING/CRITICAL)
        d["level_pt"] = self.level.pt       # rótulo para exibição (INFORMATIVO/ATENÇÃO/CRÍTICO)
        d["modality_pt"] = MODALITY_PT.get(self.modality, self.modality)
        return d

    def __str__(self) -> str:
        stamp = f"t={self.timestamp:.1f}s" if self.timestamp is not None else self.wall_time
        modality = MODALITY_PT.get(self.modality, self.modality)
        return f"{self.level.emoji} [{self.level.pt}] ({modality}) {self.message} @ {stamp}"


class AlertManager:
    """Coleta, ordena e resume alertas de todas as modalidades."""

    def __init__(self) -> None:
        self._alerts: list[Alert] = []

    def add(self, alert: Alert) -> Alert:
        self._alerts.append(alert)
        return alert

    def emit(
        self,
        level: AlertLevel,
        modality: str,
        message: str,
        **kwargs: Any,
    ) -> Alert:
        """Atalho para criar e registrar um alerta em uma linha."""
        return self.add(Alert(level=level, modality=modality, message=message, **kwargs))

    def extend(self, alerts: Iterable[Alert]) -> None:
        for a in alerts:
            self.add(a)

    def all(self) -> list[Alert]:
        """Alertas ordenados por gravidade (desc) e depois por tempo."""
        return sorted(self._alerts, key=lambda a: (-int(a.level), a.timestamp))

    def by_modality(self, modality: str) -> list[Alert]:
        return [a for a in self.all() if a.modality == modality]

    def by_patient(self, patient_id: str) -> list[Alert]:
        return [a for a in self.all() if a.patient_id == patient_id]

    def critical(self) -> list[Alert]:
        return [a for a in self.all() if a.level == AlertLevel.CRITICAL]

    def max_level(self) -> AlertLevel:
        return max((a.level for a in self._alerts), default=AlertLevel.INFO)

    def summary(self) -> dict[str, int]:
        counts = {lvl.label: 0 for lvl in AlertLevel}
        for a in self._alerts:
            counts[a.level.label] += 1
        return counts

    def __len__(self) -> int:
        return len(self._alerts)

    def clear(self) -> None:
        self._alerts.clear()
