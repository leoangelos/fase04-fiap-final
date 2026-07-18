# Roteiro do Vídeo de Demonstração (≤ 15 min)

**Tech Challenge FIAP 8IADT — Fase 4 · Monitoramento Multimodal de Pacientes**

> Upload no YouTube/Vimeo como **público ou não listado**. Cobre os quatro itens exigidos: demo de áudio e vídeo, detecção/resposta a anomalias, integração Azure e fluxo final do alerta à equipe médica — mais a camada hospitalar como diferencial.

**Preparação antes de gravar:**

```bash
uv sync
uv run python scripts/download_data.py       # gera os vídeos de exemplo (áudios já vêm no repo)
uv run python scripts/run_pipeline.py         # pré-gera outputs/annotated_patient_immobility_demo.mp4
uv run python scripts/seed_hospital.py        # pacientes demo + cenário de deterioração (precisa das chaves Supabase no .env)
uv run python -m hospital_ai.workers.runner   # deixe o worker rodando em um terminal
uv run streamlit run app/Home.py              # deixe o dashboard aberto em outro
```

- Chaves Azure no `.env`: rode `uv run python scripts/smoke_azure.py` antes, para mostrar a integração ao vivo.
- Login demo do Supabase Auth (se quiser mostrar a API em `/docs`): `medico.demo@fiap-fase4.local` / `Fiap@Fase4-demo`.
- Em ambiente Supabase novo, rode antes as migrations de `db/migrations/` (ver `db/README.md`).

---

## Minutagem sugerida

### 0:00 – 1:30 · Abertura e problema
- Apresentação da equipe e do desafio: monitoramento contínuo multimodal para detectar risco precoce.
- Mostrar o **diagrama de arquitetura** do README: motores multimodais (vídeo + áudio + vitais + prescrições → fusão → alerta) **+ camada hospitalar** (Supabase: pacientes, histórico, tempo real).
- Frase-guia: *"quatro fontes de dados, um índice de risco por paciente, alertas automáticos para a equipe — com prontuário persistido."*

### 1:30 – 3:30 · 🏥 Hospital: Visão Geral e paciente em deterioração
- O dashboard abre na **📊 Visão Geral**: censo (pacientes ativos, internados/UTI), **risco multimodal de cada paciente** e alertas em aberto de todo o hospital.
- Entrar em **🗂️ Pacientes** → paciente internado (Carlos): risco 100%, alertas da **deterioração progressiva tipo sepse — NEWS2 2 → 6 → 8 → 12 → 14** com z-score contra a **baseline individual** (z=15 na primeira leitura anômala).
- **Registrar uma leitura de vitais ao vivo** (ex.: FC 135, SpO₂ 88) → NEWS2 calculado na hora + alerta criado; dar **"Ciente"** em um alerta (fluxo de acknowledge da equipe).
- Na aba **💊 Prescrições**, prescrever `aspirina` para quem já usa `warfarina` → **interação de risco detectada na hora**.

### 3:30 – 5:30 · 📈 Sinais vitais + detecção de anomalias (demonstração)
- Seção **🔬 Análises de demonstração** → **Sinais Vitais** → aba **Visão completa**: FC/SpO₂/PA com os marcadores (taquicardia, dessaturação, hipotensão).
- Explicar as **três técnicas**: limites clínicos, z-score móvel e Isolation Forest.
- Aba **Simulação em tempo real**: iniciar e mostrar os **alertas disparando no instante da detecção** (o "monitoramento em tempo real" do edital).

### 5:30 – 8:00 · 🎥 Análise de vídeo
- Página **Vídeo** → `patient_fall_demo.mp4` → **Processar vídeo**: o detector dispara **🔴 QUEDA (CRÍTICO) em t≈5 s**; exibir o **vídeo anotado** com o esqueleto sobre a pessoa no chão e o banner vermelho.
- Trocar para `patient_immobility_demo.mp4`: **imobilidade prolongada** detectada e anotada + **gráfico do índice de movimento** com o instante do evento.
- Trocar para `corridor_walk.mp4`: **múltiplas pessoas (zona segura)** e **intrusão na área crítica**; mostrar os **objetos detectados pelo YOLOv8** no resumo.

### 8:00 – 10:30 · 🎙️ Análise de áudio + Azure
- Página **Áudio** → `consulta_neutra.wav` → **Processar**: features vocais normais, **sem alerta**.
- `consulta_critica.wav` → **Processar**:
  - **Features vocais**: jitter/shimmer elevados → **score CRITICAL** (fadiga/disartria);
  - **Transcrição** ao vivo pelo **Azure Speech to Text** (pt-BR);
  - **Azure Text Analytics**: sentimento negativo + **termos críticos** ("dor no peito", "falta de ar"...) com ☁️ nos confirmados pelas frases-chave da nuvem.
- Sem Azure configurado, explicar a **degradação graciosa** (termos locais).

### 10:30 – 13:00 · 📤 Upload do médico + fluxo do alerta à equipe
- De volta a **🗂️ Pacientes** → aba **📤 Enviar mídia**: enviar um áudio de consulta para um paciente → mostrar o job na fila → **Atualizar resultados** → o cartão da mídia exibe o **resultado em linguagem clínica** (índice vocal, transcrição, sentimento/termos) gerado pelo worker.
- Enfatizar o **fluxo final**: modalidade → análise → alerta → fusão com decaimento → **painel único da equipe** (Visão Geral) + histórico persistido por paciente.
- Página **Alertas** (demonstração): timeline consolidada com filtros por modalidade/nível.
- (Opcional) abrir `outputs/relatorio_consolidado.md` — o **relatório automático** — e a API em `http://localhost:8000/docs`.

### 13:00 – 15:00 · Encerramento
- Recapitular: **multimodal + nuvem + anomalias em tempo real + alerta automático + gestão hospitalar**.
- Reprodutibilidade: **37 testes offline** e bootstrap do Supabase **do zero** com as migrations versionadas (`db/migrations/`).
- Citar **limitações e ética/LGPD** (dados públicos/simulados, apoio à decisão, não diagnóstico, RLS/auditoria/exclusão lógica).
- Apontar o repositório e os entregáveis (código + relatório técnico + este vídeo).

---

## Checklist de itens obrigatórios (edital)

- [ ] Exemplo prático da **análise de áudio** (consulta crítica vs. neutra)
- [ ] Exemplo prático da **análise de vídeo** (pose + objetos/área crítica + imobilidade anotada)
- [ ] **Detecção e resposta a anomalias** (simulação em tempo real + NEWS2/z-score ao vivo)
- [ ] **Integração dos serviços Azure** (Speech to Text + Text Analytics ao vivo)
- [ ] **Fluxo final do alerta à equipe médica** (Visão Geral + acknowledge + relatório)
- [ ] Vídeo com até **15 minutos**, público ou não listado

**Diferenciais para destacar (além do edital):** camada hospitalar completa (pacientes, profissionais, upload com fila e histórico), NEWS2 + baseline individual, LGPD em código e reprodutibilidade do zero.
