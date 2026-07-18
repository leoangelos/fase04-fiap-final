from ..db import table


def log(professional_id: str | None, action: str, entity: str, entity_id: str) -> None:
    try:
        table("audit_logs").insert({
            "professional_id": professional_id, "action": action,
            "entity": entity, "entity_id": str(entity_id),
        }).execute()
    except Exception:
        pass  # auditoria nunca derruba o fluxo principal
