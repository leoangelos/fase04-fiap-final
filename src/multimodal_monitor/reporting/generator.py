"""Relatório consolidado automático do monitoramento multimodal.

Recebe os alertas de todas as modalidades + a avaliação de risco e gera um
relatório em Markdown (legível e versionável) com resumo executivo, índice de
risco, tabela de eventos e detalhamento por modalidade.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ..alerts.alert_manager import Alert, AlertLevel, MODALITY_PT
from ..fusion.risk_engine import RiskAssessment


def _events_table(alerts: list[Alert]) -> str:
    if not alerts:
        return "_Nenhum evento detectado._\n"
    lines = ["| Nível | Modalidade | t (s) | Evento |", "|---|---|---|---|"]
    for a in alerts:
        lines.append(
            f"| {a.level.emoji} {a.level.pt} | {MODALITY_PT.get(a.modality, a.modality)} "
            f"| {a.timestamp:.1f} | {a.message} |"
        )
    return "\n".join(lines) + "\n"


def build_markdown_report(
    alerts: list[Alert],
    risk: RiskAssessment,
    patient_id: str = "P001",
    context: dict | None = None,
) -> str:
    """Monta o relatório em Markdown."""
    context = context or {}
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    ordered = sorted(alerts, key=lambda a: (-int(a.level), a.modality, a.timestamp))
    critical = [a for a in ordered if a.level == AlertLevel.CRITICAL]

    by_modality: dict[str, list[Alert]] = {}
    for a in ordered:
        by_modality.setdefault(a.modality, []).append(a)

    md = [
        f"# Relatório de Monitoramento Multimodal — Paciente {patient_id}",
        f"\n_Gerado automaticamente em {now}_\n",
        "## Resumo executivo\n",
        f"- **Índice de risco:** {risk.score:.0f}/100 — **{risk.level_label}** {risk.level.emoji}",
        f"- **Total de alertas:** {risk.n_alerts} "
        f"({len(critical)} críticos)",
        f"- **Modalidades contribuintes:** {', '.join(risk.contributing) or '—'}",
    ]
    if context.get("sources"):
        md.append(f"- **Fontes analisadas:** {context['sources']}")

    md.append("\n## Alertas críticos\n")
    md.append(_events_table(critical) if critical else "_Sem alertas críticos._\n")

    md.append("\n## Contribuição por modalidade\n")
    md.append("| Modalidade | Risco parcial (0–100) | Nº alertas |")
    md.append("|---|---|---|")
    for m, alist in by_modality.items():
        md.append(f"| {MODALITY_PT.get(m, m)} | {risk.per_modality.get(m, 0):.0f} | {len(alist)} |")

    md.append("\n## Todos os eventos detectados\n")
    md.append(_events_table(ordered))

    md.append("\n## Detalhamento por modalidade\n")
    emojis = {"vitals": "📈", "audio": "🎙️", "video": "🎥", "prescription": "💊", "fusion": "🔀"}
    for m, alist in by_modality.items():
        md.append(f"\n### {emojis.get(m, '')} {MODALITY_PT.get(m, m)}\n")
        for a in alist:
            md.append(f"- {a.level.emoji} **t={a.timestamp:.1f}s** — {a.message}")

    md.append(
        "\n---\n\n> Projeto acadêmico (FIAP 8IADT — Fase 4). Dados públicos ou "
        "simulados; as detecções não constituem diagnóstico médico."
    )
    return "\n".join(md) + "\n"


def save_report(markdown: str, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    return out_path
