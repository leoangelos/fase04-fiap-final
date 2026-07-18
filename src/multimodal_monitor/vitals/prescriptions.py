"""Detecção de anomalias na evolução de prescrições.

Requisito 3b do edital: "Evolução de prescrições (alterações inesperadas no
tratamento)". Trabalhamos sobre um CSV com colunas:
    day, drug, dose_mg, frequency_per_day

Regras:
  1. Variação abrupta de dose (> limiar %) para o mesmo fármaco entre dias.
  2. Descontinuação abrupta (fármaco some após dose alta).
  3. Combinações de risco (pares de fármacos com interação conhecida) no mesmo dia.
"""

from __future__ import annotations

import pandas as pd

from ..alerts.alert_manager import Alert, AlertLevel

# Interações medicamentosas conhecidas (exemplos didáticos, não exaustivo).
RISKY_COMBINATIONS: dict[frozenset[str], str] = {
    frozenset({"warfarina", "aspirina"}): "risco aumentado de sangramento",
    frozenset({"warfarina", "ibuprofeno"}): "risco aumentado de sangramento",
    frozenset({"tramadol", "fluoxetina"}): "risco de síndrome serotoninérgica",
    frozenset({"sinvastatina", "claritromicina"}): "risco de rabdomiólise",
    frozenset({"espironolactona", "enalapril"}): "risco de hipercalemia",
}


def sample_prescriptions() -> pd.DataFrame:
    """Prescrições sintéticas de exemplo com anomalias plantadas para a demo."""
    rows = [
        # dia, fármaco, dose (mg), frequência/dia
        (1, "warfarina", 5, 1),
        (1, "enalapril", 10, 2),
        (2, "warfarina", 5, 1),
        (2, "enalapril", 10, 2),
        (3, "warfarina", 5, 1),
        (3, "enalapril", 10, 2),
        (3, "aspirina", 100, 1),          # anomalia 3: combinação de risco com warfarina
        (4, "warfarina", 15, 1),          # anomalia 1: dose triplicou (5 → 15)
        (4, "enalapril", 10, 2),
        (5, "enalapril", 10, 2),          # anomalia 2: warfarina descontinuada abruptamente
        (5, "espironolactona", 25, 1),    # anomalia 3: combinação de risco com enalapril
    ]
    return pd.DataFrame(rows, columns=["day", "drug", "dose_mg", "frequency_per_day"])


def detect_prescription_anomalies(
    df: pd.DataFrame,
    patient_id: str = "P001",
    dose_change_pct: float = 50.0,
) -> list[Alert]:
    """Aplica as três regras de prescrição e devolve os alertas."""
    alerts: list[Alert] = []
    df = df.sort_values(["drug", "day"])

    # Regras 1 e 2: por fármaco, acompanhar a dose ao longo dos dias.
    for drug, g in df.groupby("drug"):
        g = g.sort_values("day")
        prev_dose = None
        prev_day = None
        for _, row in g.iterrows():
            dose = row["dose_mg"]
            if prev_dose is not None:
                change = abs(dose - prev_dose) / prev_dose * 100 if prev_dose else 0
                if change >= dose_change_pct:
                    alerts.append(
                        Alert(
                            level=AlertLevel.WARNING,
                            modality="prescription",
                            message=(
                                f"Alteração abrupta de dose de {drug}: "
                                f"{prev_dose:.0f}→{dose:.0f} mg ({change:+.0f}%) "
                                f"entre os dias {prev_day} e {row['day']}"
                            ),
                            patient_id=patient_id,
                            timestamp=float(row["day"]),
                            metric="dose_mg",
                            value=float(dose),
                            details={"drug": drug, "rule": "dose_change"},
                        )
                    )
            prev_dose, prev_day = dose, row["day"]

        # descontinuação: presente até um dia e ausente nos dias seguintes
        last_day = g["day"].max()
        all_days = sorted(df["day"].unique())
        if last_day != all_days[-1] and g.loc[g["day"] == last_day, "dose_mg"].iloc[0] > 0:
            alerts.append(
                Alert(
                    level=AlertLevel.WARNING,
                    modality="prescription",
                    message=f"Descontinuação abrupta de {drug} após o dia {last_day}",
                    patient_id=patient_id,
                    timestamp=float(last_day + 1),
                    metric="dose_mg",
                    value=0.0,
                    details={"drug": drug, "rule": "discontinuation"},
                )
            )

    # Regra 3: combinações de risco no mesmo dia.
    for day, g in df.groupby("day"):
        drugs = set(g["drug"].str.lower())
        for combo, reason in RISKY_COMBINATIONS.items():
            if combo <= drugs:
                a, b = tuple(combo)
                alerts.append(
                    Alert(
                        level=AlertLevel.CRITICAL,
                        modality="prescription",
                        message=f"Combinação de risco no dia {day}: {a} + {b} ({reason})",
                        patient_id=patient_id,
                        timestamp=float(day),
                        metric="interaction",
                        details={"drugs": sorted(combo), "reason": reason, "rule": "interaction"},
                    )
                )
    return sorted(alerts, key=lambda al: al.timestamp)
