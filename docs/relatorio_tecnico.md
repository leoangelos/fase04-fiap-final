# Relatório Técnico — Monitoramento Multimodal de Pacientes

**FIAP Pós-Tech IA para Devs (8IADT) — Tech Challenge Fase 4**

**Grupo 31** — Leonardo Angelos · Vagner Lopes · Lucas Oliveira · Vinícius Silva

- 🎬 **Vídeo de demonstração (YouTube):** https://www.youtube.com/watch?v=34p9tNgTX3g
- 💻 **Repositório (GitHub):** https://github.com/leoangelos/fase04-fiap-final

---

## 1. Contexto e objetivo

Com a IA já integrada aos processos médicos (análise de exames e laudos), o hospital passa a **monitorar continuamente os pacientes por dados multimodais** — vídeo, áudio e séries temporais — para identificar **sinais precoces de risco** e alertar a equipe em tempo real.

Este projeto entrega uma solução que:

1. Analisa **vídeos clínicos** (fisioterapia / movimentação) detectando eventos fora do padrão;
2. Processa **áudios de consultas**, detectando alterações vocais e transcrevendo/analisando o conteúdo com Azure;
3. Aplica **detecção de anomalias** em sinais vitais, prescrições e padrões de movimentação;
4. **Funde** os sinais em um índice de risco e **alerta** automaticamente a equipe;
5. Opera como **sistema hospitalar** (cap. 6): cadastro de pacientes e profissionais, upload multimodal por paciente com fila de processamento, histórico clínico persistido e alertas em tempo real (Supabase).

---

## 2. Fluxo multimodal

```
 ┌─────────────┐   YOLOv8-pose      ┌──────────────────┐
 │   Vídeo     │ ─────────────────► │ Features de       │──┐
 └─────────────┘   17 keypoints     │ movimento + regras│  │
                                    └──────────────────┘  │
 ┌─────────────┐  Praat/librosa     ┌──────────────────┐  │
 │   Áudio     │ ─────────────────► │ Score vocal +     │  │   ┌───────────────┐   ┌──────────────┐
 └─────────────┘  Azure STT/Text    │ termos/sentimento │  ├──►│ Motor de Fusão│──►│ Alertas +     │
                                    └──────────────────┘  │   │ risco 0–100   │   │ Dashboard/    │
 ┌─────────────┐  limites+z-score   ┌──────────────────┐  │   └───────────────┘   │ Relatório     │
 │Sinais vitais│ ─────────────────► │ IsolationForest   │──┤                      └──────────────┘
 └─────────────┘                    └──────────────────┘  │
 ┌─────────────┐  regras clínicas   ┌──────────────────┐  │
 │ Prescrições │ ─────────────────► │ dose/interação    │──┘
 └─────────────┘                    └──────────────────┘
```

Cada modalidade produz objetos `Alert` (níveis **INFO / WARNING / CRITICAL**) enviados a um `AlertManager` comum. O `risk_engine` agrega esses alertas em um índice único.

---

## 3. Modelos e técnicas por modalidade

### 3.1. Vídeo — `src/multimodal_monitor/video/`

| Etapa | Técnica |
|---|---|
| Estimativa de pose | **YOLOv8-pose** (`ultralytics`), 17 keypoints COCO por pessoa; seleção da pessoa de maior bounding box |
| Features de movimento | Ângulos articulares (joelho, cotovelo, quadril), velocidade das extremidades normalizada pela altura do tronco, simetria E/D, altura do quadril, índice global de movimento |
| Detecção de anomalias | **Queda** (queda abrupta da altura do quadril); **imobilidade prolongada** (movimento < limiar relativo por > 5 s, com tolerância a lacunas); **desvio postural** (z-score robusto sustentado dos ângulos); **zona segura** (nº de pessoas na cena) |
| Objetos e áreas críticas | **YOLOv8** de detecção (COCO): inventário de objetos por frame; **zona crítica** configurável em frações do quadro (`config.SceneThresholds`) — pessoa cuja base entra na área restrita gera alerta; **objeto inesperado** (classe fora do esperado persistente em ≥ 3 frames) |
| Saídas | Vídeo anotado (esqueleto + retângulo da área crítica + banner de alerta) e relatório de eventos com timestamps |

Robustez: só computa features quando há **pose válida** (≥3 keypoints centrais), evitando artefatos de detecções parciais. O índice de movimento por janela alimenta o **mesmo detector de série temporal** dos sinais vitais (requisito de "padrões de movimentação do paciente").

### 3.2. Áudio — `src/multimodal_monitor/audio/`

| Etapa | Técnica |
|---|---|
| Features acústicas | **Jitter, shimmer, HNR** (Praat via `parselmouth`), F0 média/desvio, **razão de pausas** e **taxa de fala** (librosa) |
| Score vocal | Acúmulo de indicadores vs. faixas de referência da literatura; exige **≥2 indicadores** para alerta (reduz falso-positivo) |
| Transcrição | **Azure Speech to Text** (pt-BR, reconhecimento contínuo) |
| Análise de texto | **Azure Text Analytics**: sentimento + frases-chave; **termos clínicos críticos** por correspondência local em pt-BR (sem acento), com **validação cruzada nas frases-chave do Azure** (termos confirmados pela nuvem são marcados) |

### 3.3. Sinais vitais e prescrições — `src/multimodal_monitor/vitals/`

| Etapa | Técnica |
|---|---|
| Fonte | **PhysioNet BIDMC** (via `wfdb`) ou gerador **sintético** com anomalias injetadas |
| Detecção | **Limites clínicos** (bradi/taquicardia, dessaturação, hipo/hipertensão); **z-score móvel** (mudança de tendência); **Isolation Forest** multivariado (combinações raras) |
| Prescrições | Variação de dose > 50 %, descontinuação abrupta, **combinações de risco** (ex.: varfarina+aspirina → sangramento) |

### 3.4. Fusão — `src/multimodal_monitor/fusion/risk_engine.py`

Índice de risco `0–100` por saturação exponencial da soma ponderada (severidade × peso da modalidade). Qualquer alerta **CRITICAL** ou score ≥ 60 eleva o nível para **ALTO**.

---

## 4. Resultados e exemplos de anomalias detectadas

Execução de referência (`scripts/run_pipeline.py`, paciente P001, dados de exemplo):

**Índice de risco: 99/100 — ALTO** · 16 alertas (7 críticos).

| Modalidade | Exemplos de anomalias detectadas |
|---|---|
| 📈 Sinais vitais | FC 145 bpm (taquicardia); SpO₂ 83 % (dessaturação); PA 83 mmHg (hipotensão) — os 3 eventos injetados foram capturados por limite clínico e z-score |
| 🎥 Vídeo (pose) | **Queda real detectada como CRÍTICO** em t=5,2 s no `patient_fall_demo.mp4` (pessoa desabando em fases — janela de ~0,5 s sobre a altura do quadril); imobilidade prolongada de **6,2 s** detectada e anotada no `patient_immobility_demo.mp4`; no `corridor_walk.mp4`: múltiplas pessoas na cena (zona segura), imobilidade de **9,0 s** da figura parada e desvios de amplitude articular |
| 🎥 Vídeo (objetos) | **Pessoa dentro da área crítica** ("área restrita") detectada no `corridor_walk.mp4` (t=18,4 s); inventário de objetos por frame (ex.: `person`×72) com **filtro de persistência** que suprimiu falsos positivos de 1 frame (`dog`) |
| 🎙️ Áudio | `consulta_critica`: **score vocal 0,50 (CRITICAL)** — jitter e shimmer elevados — e **4 termos críticos** ("dor no peito", "falta de ar", "tontura", "desmaiei"); `consulta_neutra`: sem alerta |
| 💊 Prescrições | Varfarina 5→15 mg (variação de dose); descontinuação abrupta; **2 interações de risco** (varfarina+aspirina, enalapril+espironolactona) |

**Exemplo de saída anotada de vídeo** (esqueleto + retângulo da área crítica + banner de alerta):

> `outputs/annotated_patient_immobility_demo.mp4` — gerado automaticamente por `scripts/run_pipeline.py` via `video/report.py::annotate_video`.

O relatório consolidado completo é gravado em `outputs/relatorio_consolidado.md`.

---

## 5. Integração com serviços em nuvem (Azure)

- **Azure Speech to Text** — transcrição pt-BR dos áudios de consulta.
- **Azure Text Analytics (Azure AI Language)** — sentimento e frases-chave.

Ambos no **tier gratuito F0**. A camada Azure está isolada atrás de `config.py`: sem chaves, o pipeline **degrada graciosamente** (pula a transcrição e usa correspondência local de termos), garantindo desenvolvimento e testes offline.

**Limitação documentada:** o *Text Analytics for Health* (extração de entidades clínicas) só suporta inglês; por isso os termos críticos usam uma lista custom em português, **validada de forma cruzada contra as frases-chave que o próprio Azure extrai** da transcrição (termos confirmados pela nuvem são sinalizados na interface e nos alertas). Em produção, poder-se-ia traduzir a transcrição ou usar um modelo clínico em pt-BR.

---

## 6. Camada hospitalar: pacientes, histórico e alertas em tempo real

Para aproximar a solução de um sistema hospitalar real, o projeto inclui uma camada de gestão construída sobre **Supabase** (Postgres + Storage + Auth + Realtime), com API **FastAPI** e workers assíncronos que **reutilizam a biblioteca `multimodal_monitor`** como motor de análise.

| Componente | Implementação |
|---|---|
| Modelo de dados | Schema isolado `hospital` (10 tabelas): pacientes, encontros, mídias, análises, sinais vitais, prescrições, alertas, auditoria e fila de jobs — todas com **RLS** e grants mínimos por papel |
| Upload multimodal | Duas etapas: registro *pending* + **signed upload URL** (o binário vai direto ao Storage), confirmação com **SHA-256** (integridade do arquivo) e enfileiramento da análise |
| Processamento | Fila `hospital.jobs` (padrão `FOR UPDATE SKIP LOCKED`, *retry* com backoff) consumida por workers de áudio/vídeo/texto que rodam o mesmo pipeline do capítulo 3 e gravam `analysis_results` por paciente |
| Anomalias em vitais | **Duas camadas comparáveis**: (1) **NEWS2** (Royal College of Physicians), escore determinístico usado em hospitais reais; (2) **z-score em janela deslizante** (Redis/Upstash ou memória) — baseline estatístico do **próprio paciente**, que captura desvios individuais antes dos limites populacionais |
| Fusão multimodal | *Late fusion* ponderada com **decaimento temporal** (meia-vida 24 h) dos `risk_score` de vitais, áudio, vídeo, NLP e prescrições → `patient_risk_score` 0–1; ≥ 0,7 gera alerta `fusion` com **deduplicação** (não repete enquanto houver alerta de fusão sem ciência) |
| Tempo real | Alertas inseridos em `hospital.alerts` publicados via **Supabase Realtime** para o painel da equipe |
| Baseline individual | O histórico de sessões de vídeo do paciente vira insumo do modelo: métricas da sessão atual são comparadas às anteriores (`baseline_deviations`) |
| Prescrições | Checagem de **interação medicamentosa imediata** ao prescrever (base única compartilhada com o pipeline de demonstração) |
| Interface | Dashboard em duas seções: **🏥 Hospital** (Visão Geral com censo/risco de todos os pacientes, Pacientes com upload e resultados em **linguagem clínica pt-BR**, Profissionais com CRUD e criação de login) e **🔬 Análises de demonstração** (motores isolados com dados de exemplo) |
| LGPD | RLS + grants mínimos (anon sem acesso a dados), buckets privados com signed URL de 5 min, service key restrita ao backend, **exclusão lógica** de pacientes/profissionais (prontuário preservado, sem apagar dados clínicos) |

Cenário de demonstração (`scripts/seed_hospital.py`): paciente internado com **deterioração progressiva** tipo sepse — NEWS2 evolui 2 → 6 → 8 → 12 → 14 enquanto o z-score da FC dispara contra a baseline individual (z=15 na primeira leitura anômala), culminando em alerta de fusão com risco 100%.

---

## 7. Limitações e trabalhos futuros

- **Vídeo:** o estimador de pose sofre com pessoas distantes/parcialmente visíveis; a análise postural é mais confiável em close de exercício. Uso de `yolov8n` (nano) prioriza velocidade sobre precisão.
- **Áudio:** as amostras de demonstração são geradas por TTS (voz estável); os thresholds vocais são da literatura e, em produção, devem ser **calibrados por paciente** (baseline individual).
- **Anomalias:** os limiares são genéricos de adulto; um sistema real ajustaria por perfil clínico e histórico.
- **Fusão:** os pesos por modalidade são heurísticos; poderiam ser aprendidos com dados rotulados.
- **Camada hospitalar:** o NEWS2 está sem os parâmetros ACVPU (consciência) e O₂ suplementar, ausentes do modelo de dados; a base de interações medicamentosas é didática (em produção, DrugBank/Micromedex).

---

## 8. Ética, privacidade e LGPD

- **Nenhum dado real de paciente** é utilizado: sinais vitais são públicos (PhysioNet) ou sintéticos; áudios e vídeos são simulados/de amostra com licença aberta.
- As chaves (Azure, Supabase, Redis) ficam em `.env` (fora do versionamento).
- O sistema é de **apoio à decisão** e **não substitui avaliação médica**; toda detecção é uma triagem para revisão humana.
- **LGPD implementada em código na camada hospitalar** (cap. 6): controle de acesso por papel (RLS + grants; `anon` não lê dados clínicos), arquivos em buckets privados com URLs assinadas de curta duração e checksum SHA-256, e **exclusão lógica** de pacientes/profissionais que preserva o prontuário (registros clínicos não são apagados fisicamente).
- Em uso real, seriam ainda necessários consentimento informado, minimização/anonimização de dados e políticas de retenção — em conformidade plena com a **LGPD**.

---

## 9. Reprodutibilidade

```bash
uv sync
uv run python scripts/download_data.py     # amostras
uv run pytest                               # 37 testes
uv run python scripts/run_pipeline.py       # relatório consolidado + vídeo anotado
uv run streamlit run app/Home.py            # dashboard
```

A camada hospitalar é reproduzível **do zero** em qualquer projeto Supabase: as
migrations versionadas em `db/migrations/` (0001→0005) recriam schema, RLS,
fila de jobs, buckets, grants e Realtime por SQL, sem passos manuais no
dashboard; o `scripts/seed_hospital.py` popula o cenário de demonstração.

Stack fixada em `pyproject.toml` (Python 3.12 via `uv`). Detalhes de instalação multiplataforma no [README](https://github.com/leoangelos/fase04-fiap-final#readme).
