"""Seed da camada hospitalar (Supabase) para demonstração.

Cria (idempotente, via upsert):
  - 1 profissional demo (auth user + hospital.professionals);
  - 3 pacientes;
  - prescrições do paciente 1 com PAR DE INTERAÇÃO (warfarina + aspirina);
  - sinais vitais: pacientes estáveis + CENÁRIO DE DETERIORAÇÃO PROGRESSIVA
    no paciente 1 (taquicardia + dessaturação + hipotensão) — dispara NEWS2,
    z-score individual e a fusão multimodal.

Requer SUPABASE_URL + SUPABASE_SERVICE_KEY no .env.

Uso:
    uv run python scripts/seed_hospital.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# consoles Windows legados (cp1252) não têm os símbolos usados na saída
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from hospital_ai.config import settings

if not settings.supabase_configured:
    sys.exit("Configure SUPABASE_URL e SUPABASE_SERVICE_KEY no .env antes do seed.")

from hospital_ai.db import service_client, table
from hospital_ai.services.interactions import check_drug_interactions
from hospital_ai.services.vitals_ingest import ingest_vitals

DEMO_EMAIL = "medico.demo@fiap-fase4.local"
DEMO_PASSWORD = "Fiap@Fase4-demo"

PACIENTES = [
    {"mrn": "MRN-0001", "full_name": "Carlos Andrade (demo)", "birth_date": "1958-03-12",
     "sex": "M", "admission_status": "admitted"},
    {"mrn": "MRN-0002", "full_name": "Maria Helena Lopes (demo)", "birth_date": "1972-11-05",
     "sex": "F", "admission_status": "outpatient"},
    {"mrn": "MRN-0003", "full_name": "João Pereira (demo)", "birth_date": "1990-07-30",
     "sex": "M", "admission_status": "outpatient"},
]


def ensure_professional() -> str:
    """Cria (ou reaproveita) o auth user demo e o registro em professionals."""
    admin = service_client().auth.admin
    try:
        user = admin.create_user({
            "email": DEMO_EMAIL, "password": DEMO_PASSWORD, "email_confirm": True,
        })
        uid = user.user.id
        print(f"• auth user criado: {DEMO_EMAIL}")
    except Exception:
        found = [u for u in admin.list_users() if u.email == DEMO_EMAIL]
        if not found:
            raise
        uid = found[0].id
        print(f"• auth user já existia: {DEMO_EMAIL}")

    table("professionals").upsert({
        "id": uid, "full_name": "Dra. Ana Souza (demo)",
        "crm": "CRM-SP 123456", "role": "medico",
    }).execute()
    return uid


def ensure_patients(prof_id: str) -> list[str]:
    ids = []
    for p in PACIENTES:
        row = table("patients").upsert(
            p | {"created_by": prof_id}, on_conflict="mrn").execute().data[0]
        ids.append(row["id"])
    print(f"• {len(ids)} pacientes")
    return ids


def seed_prescriptions(prof_id: str, patient_id: str) -> None:
    atuais = table("prescriptions").select("medication").eq(
        "patient_id", patient_id).eq("active", True).execute().data
    existentes = {r["medication"] for r in atuais}
    for med, dose in (("warfarina", "5 mg 1x/dia"), ("enalapril", "10 mg 2x/dia"),
                      ("aspirina", "100 mg 1x/dia")):
        if med not in existentes:
            table("prescriptions").insert({
                "patient_id": patient_id, "prescribed_by": prof_id,
                "medication": med, "dosage": dose, "route": "oral",
            }).execute()
    inter = check_drug_interactions(patient_id)
    if inter:
        table("alerts").insert({
            "patient_id": patient_id, "source_type": "prescription",
            "severity": "critical", "title": "Interação medicamentosa detectada",
            "details": {"interactions": inter},
        }).execute()
        print(f"• prescrições: interação de risco detectada → alerta "
              f"({inter[0]['pair'][0]} + {inter[0]['pair'][1]})")


def seed_vitals(patient_ids: list[str]) -> None:
    """Estáveis nos pacientes 2 e 3; deterioração progressiva no paciente 1.

    Idempotente: se o paciente 1 já tem sinais vitais, pula — re-executar o
    seed não deve empilhar leituras nem duplicar alertas.
    """
    ja_tem = (table("vital_signs").select("id").eq("patient_id", patient_ids[0])
              .limit(1).execute().data)
    if ja_tem:
        print("• sinais vitais já semeados — pulando (para recomeçar do zero, "
              "apague vital_signs/analysis_results/alerts dos pacientes demo).")
        return

    t0 = datetime.now(timezone.utc) - timedelta(hours=6)

    for pid in patient_ids[1:]:
        for i in range(12):
            ingest_vitals(pid, {
                "heart_rate": 72 + (i % 5), "spo2": 97.0 - (i % 2) * 0.4,
                "temperature": 36.4 + (i % 3) * 0.1,
                "systolic_bp": 118 + (i % 7), "diastolic_bp": 76 + (i % 4),
                "respiratory_rate": 14 + (i % 3),
            }, measured_at=(t0 + timedelta(minutes=30 * i)).isoformat(),
               source="monitor")

    pid = patient_ids[0]
    print("• paciente 1: 14 leituras estáveis (linha de base da janela z-score)...")
    for i in range(14):
        ingest_vitals(pid, {
            "heart_rate": 78 + (i % 4), "spo2": 96.5 + (i % 2) * 0.5,
            "temperature": 36.6, "systolic_bp": 122 + (i % 5),
            "diastolic_bp": 78, "respiratory_rate": 15 + (i % 2),
        }, measured_at=(t0 + timedelta(minutes=20 * i)).isoformat(),
           source="monitor")

    print("• paciente 1: deterioração progressiva (6 leituras)...")
    deterioracao = [
        # FC↑, SpO2↓, FR↑, PA↓ — evolução tipo sepse
        {"heart_rate": 96, "spo2": 95.0, "temperature": 37.8,
         "systolic_bp": 116, "diastolic_bp": 74, "respiratory_rate": 19},
        {"heart_rate": 105, "spo2": 94.0, "temperature": 38.4,
         "systolic_bp": 108, "diastolic_bp": 70, "respiratory_rate": 22},
        {"heart_rate": 114, "spo2": 92.5, "temperature": 38.9,
         "systolic_bp": 101, "diastolic_bp": 66, "respiratory_rate": 24},
        {"heart_rate": 122, "spo2": 91.0, "temperature": 39.2,
         "systolic_bp": 94, "diastolic_bp": 60, "respiratory_rate": 26},
        {"heart_rate": 131, "spo2": 89.5, "temperature": 39.4,
         "systolic_bp": 88, "diastolic_bp": 55, "respiratory_rate": 28},
        {"heart_rate": 138, "spo2": 87.0, "temperature": 39.6,
         "systolic_bp": 82, "diastolic_bp": 50, "respiratory_rate": 31},
    ]
    base = t0 + timedelta(minutes=20 * 14)
    for i, leitura in enumerate(deterioracao):
        res = ingest_vitals(pid, leitura,
                            measured_at=(base + timedelta(minutes=15 * i)).isoformat(),
                            source="monitor")
        n2 = res["news2"]
        print(f"    t+{15*i:>3} min → NEWS2={n2['total']:>2} ({n2['severity'] or 'ok'}) "
              f"| z-score: {len(res['anomalies'])} anomalia(s) "
              f"| fusão: {'ALERTA' if res['fusion_alert'] else '—'}")


def main() -> None:
    print(f"Seed hospitalar → {settings.supabase_url}\n")
    prof_id = ensure_professional()
    patient_ids = ensure_patients(prof_id)
    seed_prescriptions(prof_id, patient_ids[0])
    seed_vitals(patient_ids)

    alertas = (table("alerts").select("severity").eq("patient_id", patient_ids[0])
               .execute().data)
    contagem: dict[str, int] = {}
    for a in alertas:
        contagem[a["severity"]] = contagem.get(a["severity"], 0) + 1
    print(f"\nAlertas do paciente demo: {contagem}")
    print(f"Login demo (Supabase Auth): {DEMO_EMAIL} / {DEMO_PASSWORD}")
    print("Abra o dashboard → página Pacientes, ou a API em /docs.")


if __name__ == "__main__":
    main()
