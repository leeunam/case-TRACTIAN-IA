# Cenários de teste

Cenários estruturados no estilo **TAU-bench** (objetivo + política + trajetória esperada), para que
sirvam tanto como **teste funcional** (Parte 1 — o agente resolve?) quanto como **benchmark de
avaliação** (Parte 2 — quão bem o agente resolve, e como medir isso?). Nos cenários, as siglas
**P1** e **P2** referem-se a essas duas partes (P1 = Parte 1, construção do agente; P2 = Parte 2,
avaliação).

O catálogo foi inspirado em dúvidas reais de clientes e do suporte técnico da TRACTIAN, depois
anonimizado e adaptado para operar exclusivamente sobre os dados sintéticos deste repositório.

Cada cenário traz:
- **Chamado de origem** — qual ticket dispara o cenário.
- **Objetivo do agente** — o que deve ser alcançado (estado final desejado).
- **Contexto inicial** — empresa, usuário, ativo, permissões.
- **Política** — regras de domínio que o agente deve respeitar (o que exige justificativa, quando
  escalar, o que não pode fazer sozinho).
- **Trajetória esperada** — sequência de chamadas à API + o que inspecionar em cada passo. É
  *referência*, não script rígido: um bom agente pode variar a ordem se justificar.
- **Resolução esperada** — decisão orientar × agir × escalar, com a explicação que o agente deve dar.
- **Variações a testar** — modos probabilísticos a exercitar (controlados por `seed`).
- **Critério de sucesso** — como julgar se o agente acertou (funcional + métricas p/ a Parte 2).

Convenções: `TKT-CTX-*`, `TKT-INV-*` e `TKT-EXE-*` identificam, respectivamente, chamados de
Contextualizar, Investigar e Executar; `→` representa uma chamada à API; `?` indica o que deve ser
inspecionado no retorno. Os IDs referenciam `agent-input/cases.json`,
`docs/api-contract.openapi.yaml` e `docs/data-schema.md`.

---

## CEN-01 — Ativo quebrou e não fui avisado  (TKT-INV-04)

- **Objetivo:** explicar por que nenhum insight/notificação precedeu a quebra e recomendar como
  evitar recorrência.
- **Contexto inicial:** Mineração Andes · Coordenador de Manutenção (perms: read, escalate) ·
  Redutor G-501.
- **Política:**
  - Não alterar configuração técnica sem permissão `action_high` (o Coordenador não tem).
  - Dados ausentes/inconclusivos devem ser reportados com honestidade, não inventados.
  - Recomendação de ação de alto impacto exige justificativa; escalonamento é permitido.
- **Trajetória esperada:**
  1. `GET /assets/asset_G501` ? config, criticidade, sensor_status (degraded/offline?).
  2. `GET /assets/asset_G501/analyses?status=inconclusive` ? análise sem conclusão.
  3. `GET /assets/asset_G501/baseline` ? `state=learning` (histórico insuficiente) ou `invalidated`.
  4. `GET /assets/asset_G501/data-quality` ? completeness baixa / staleness_flag.
  5. `GET /assets/asset_G501/rms?seed=...` ? `mode=unavailable` (gap antes da quebra).
  6. `GET /models/{modelId}` ? cobertura sem suporte a baixa rotação.
- **Resolução esperada:** **investigar → explicar + escalar**. Explicar que o baseline ainda estava
  em `learning`/`invalidated` e houve gap de dados, de modo que o desvio não pôde ser detectado;
  recomendar reaprendizado do baseline após reparo e melhor cobertura de sensor; como envolve
  campo, escalar para humano (`POST /cases/{caseId}/escalate` com justificativa).
- **Variações a testar:** `seed` que força `analyses=inconclusive` + `rms=unavailable` + `baseline
  =partial`; variação conflito entre `data-quality` e `analyses`.
- **Critério de sucesso (P1):** cita baseline em learning/invalidated + gap de dados como causa; não
  inventa insight; recomenda reaprendizado; escala (não tenta reprocessar sem permissão).
- **Métricas (P2):** acurácia da causa-raiz; uso de evidências; honestidade sob incerteza; número de
  chamadas; estabilidade entre execuções; decisão correta orientar×agir×escalar.

---

## CEN-02 — RMS subindo sem insight  (TKT-INV-05)

- **Objetivo:** explicar a ausência de insight apesar da tendência e disparar o caminho que fecha a
  lacuna.
- **Contexto inicial:** Petro Delta · Analista de Confiabilidade (perms: read, action_low) ·
  Compressor C-710.
- **Política:**
  - Reprocesso exige permissão `action_low` + justificativa.
  - Distinguir "modelo atrasado" de "dados ruins": ações diferentes.
- **Trajetória esperada:**
  1. `GET /assets/asset_C710/rms` ? tendência de subida + `baseline_state=established`,
     `alarm_threshold` ultrapassado.
  2. `GET /assets/asset_C710/baseline` ? `state=established` (baseline válido → desvio real).
  3. `GET /assets/asset_C710/analyses?status=pending` ? `status=pending`, `processing_state` do
     modelo.
  4. `GET /models/{modelId}` ? `processing_state=delayed`.
  5. `GET /assets/asset_C710/data-quality` ? completeness aceitável (descarta "dados ruins").
- **Resolução esperada:** **investigar → agir**. Explicar que o baseline está established e o RMS
  ultrapassou o limiar derivado, mas o modelo está com processamento atrasado, logo o insight não
  foi emitido. Recomendar/acionar reprocesso (`POST /analyses/{id}/reprocess` com justificativa)
  ou escalar se a janela de risco for alta.
- **Variações a testar:** `seed` com `analyses=pending`; variação onde `data-quality` baixo muda a
  conclusão para "dados não fiáveis".
- **Critério de sucesso (P1):** distingue atraso de modelo de problema de dados; usa o
  `alarm_threshold` derivado do baseline; aciona reprocesso com justificativa válida.
- **Métricas (P2):** acurácia do diagnóstico; uso correto de baseline vs. qualidade; qualidade da
  justificativa da ação; robustez à variação que inverte a conclusão.

---

## CEN-03 — Insight que não parece nada (falso positivo)  (TKT-INV-06)

- **Objetivo:** validar o insight contra o espectro e o baseline; decidir se é falso positivo e o
  que fazer.
- **Contexto inicial:** Acme Auto Peças · Operador de Usinagem (perms: read) · Spindle S-420.
- **Política:**
  - Operador não pode executar ações de impacto; só pode pedir reprocesso/análise especializada
    indiretamente (recomendar, não acionar).
  - Conflito entre análise automática e especializada deve ser resolvido com evidência, não por
    "achismo".
- **Trajetória esperada:**
  1. `GET /analyses/{id}` ? `type=imbalance`, `detection_mode=baseline`,
     `baseline_state_at_detection=invalidated`, `confidence` alta.
  2. `GET /assets/asset_S420/baseline` ? `state=invalidated`, `invalidation_reason=
     maintenance_intervention` (ref velha pós-manutenção).
  3. `GET /assets/asset_S420/spectrum` ? picos: 1x baixo (não sustenta desbalanceamento) vs.
     `mode=conflict` com análise especializada (looseness).
  4. `GET /models/{modelId}` ? versão/limitações.
- **Resolução esperada:** **investigar → orientar/escalar**. Concluir falso positivo: o desvio foi
  medido contra um baseline invalidated (pré-manutenção), então o "desbalanceamento" é artefato.
  Recomendar reaprendizado do baseline antes de confiar no insight; como há conflito com análise
  especializada, recomendar análise especializada (não acionar — sem permissão) ou escalar.
- **Variações a testar:** `seed` com `analyses=conflict`; variação onde o espectro sustenta a
  falha (não-falso-positivo) → conclusão oposta.
- **Critério de sucesso (P1):** identifica baseline invalidated como causa do falso positivo;
  confronta evidência; não executa ação sem permissão; recomenda reaprendizado.
- **Métricas (P2):** acurácia na detecção de falso positivo; uso de `detection_mode`/baseline;
  robustez à variação que inverte a conclusão; estabilidade.

---

## CEN-04 — Falha de lubrificação sem baseline  (TKT-INV-11b)

- **Objetivo:** explicar como a falha foi detectada sem baseline e confirmar a validade do insight.
- **Contexto inicial:** Cimento Vale · Mecânico (perms: read, action_low) · Motor M-208 (novo).
- **Política:**
  - Distinguir `detection_mode=symptom` (não precisa de baseline) de `baseline` (precisa).
  - Lubrificação é sintomática: presença do sintoma (choque/atrito) já indica a falha.
- **Trajetória esperada:**
  1. `GET /analyses/{id}` ? `type=lubrication`, `detection_mode=symptom`,
     `baseline_state_at_detection=not_applicable`, evidência sem `reference`.
  2. `GET /assets/asset_M208/baseline` ? `state=learning`, `detection_mode=symptom`,
     `learnable=false` p/ esta falha.
  3. `GET /assets/asset_M208/spectrum` ? assinatura de choque/atrito (sintoma).
  4. `GET /knowledge/search?q=lubrificação` ? procedimento de lubrificação.
- **Resolução esperada:** **investigar → orientar + agir (recomendar)**. Explicar que lubrificação é
  detecção sintomática: o baseline em `learning` não impede a detecção, pois a mera presença do
  sintoma indica a falha. Recomendar lubrificação conforme procedimento; diferenciar de falhas por
  desvio que precisariam de baseline established.
- **Variações a testar:** `seed` com `analyses=partial` (evidência incompleta); variação onde
  outra falha por desvio aparece mas baseline em learning → não deveria ser reportada como
  confirmada.
- **Critério de sucesso (P1):** explica corretamente `symptom` vs. `baseline`; confirma a validade
  do insight de lubrificação mesmo com baseline learning; recomenda ação de manutenção.
- **Métricas (P2):** acurácia do raciocínio sobre modos de detecção; uso de evidência sem
  `reference`; consistência: não confirma falha por desvio quando baseline em learning.

---

## CEN-05 — Vibração abrupta: elétrica ou mecânica?  (TKT-INV-07)

- **Objetivo:** distinguir falha elétrica de mecânica via espectro e recomendar próximo passo.
- **Contexto inicial:** Texfil · Eletricista (perms: read) · Motor M-605.
- **Política:**
  - Eletricista só lê; recomendação deve respeitar o papel (elétrica → elétrica).
  - Espectro parcial impede conclusão definitiva → ser honesto e propor dados complementares.
- **Trajetória esperada:**
  1. `GET /assets/asset_M605/rms` ? salto abrupto + `baseline_state` established, `alarm_threshold` ultrapassado.
  2. `GET /assets/asset_M605/spectrum` ? `mode=partial`, `bands_missing` inclui a banda de 2x f-linha
     (120-140 Hz); só o pico de 1x está visível.
  3. `GET /analyses/an_9910` ? `status=inconclusive`, `confidence` baixa, `limitations` com
     `band_2x_line_missing` — a análise automática não consegue confirmar elétrica.
  4. `GET /assets/asset_M605` ? config elétrica (`line_frequency_hz=60` → 2x f-linha = 120 Hz).
  5. `GET /knowledge/search?q=falha elétrica motor` ? orientações.
- **Resolução esperada:** **investigar → orientar**. A banda de 2x f-linha está ausente, então não é
  possível confirmar falha elétrica pelo espectro (a análise automática está inconclusive). Ser
  honesto: há desvio de RMS, mas a banda crítica para elétrica falta; recomendar captura
  complementar nessa banda ou inspeção, dentro do papel do Eletricista. Não concluir definitivamente.
- **Variações a testar:** `seed` com `spectrum=partial` (default do M605); variação `complete` onde a
  banda estaria presente e a conclusão poderia ser definitiva.
- **Critério de sucesso (P1):** reconhece que a banda de 2x f-linha está ausente e por isso a
  inferência é incerta; não afirma elétrica sem evidência; recomenda dentro do papel.
- **Métricas (P2):** acurácia da classificação; honestidade sob incerteza; uso das limitações e do
  `bands_missing`.

---

## CEN-06 — Diagnósticos divergentes  (TKT-INV-08)

- **Objetivo:** reconciliar duas fontes conflitantes (automática vs. especializada) e recomendar.
- **Contexto inicial:** Cimento Vale · Engenheiro de Manutenção (perms: read, action_high) ·
  Moinho M-205.
- **Política:**
  - Em conflito, pesar confiança, baseline e evidência — não votar por maioria cega.
  - Engenheiro pode acionar análise especializada e reprocesso.
- **Trajetória esperada:**
  1. `GET /assets/asset_M205/analyses` ? duas análises: automática (misalignment) vs. especializada
     (looseness), `mode=conflict`.
  2. `GET /analyses/{id1}` e `GET /analyses/{id2}` ? evidência, confiança, `detection_mode`.
  3. `GET /assets/asset_M205/baseline` ? estado do baseline.
  4. `GET /assets/asset_M205/spectrum` ? picos que sustentam uma das hipóteses.
- **Resolução esperada:** **investigar → agir/escalar**. Pesar evidências: se o espectro sustenta
  looseness (subharmônicos) e a análise automática usou baseline invalidated, preferir a
  especializada; recomendar ação (alinhamento/fixação) ou acionar nova análise especializada.
- **Variações a testar:** `seed` com `analyses=conflict`; variação onde a automática é a correta.
- **Critério de sucesso (P1):** não escolhe por maioria; usa baseline/evidência para desempatar;
  recomenda ação coerente.
- **Métricas (P2):** resolução de conflito; uso de evidências; justificativa; robustez à inversão.

---

## CEN-07 — Análise desatualizada após manutenção  (TKT-INV-09 → TKT-EXE-12)

- **Objetivo:** confirmar staleness, executar reprocesso justificado e validar resultado.
- **Contexto inicial:** Cervejaria Aurora · Mecânico (perms: read, action_low) · Bomba B-204.
- **Política:**
  - Reprocesso exige justificativa (≥ 20 chars) baseada em evidência (intervenção realizada).
  - Reprocesso aceito = sucesso, sem ciclo de status; validar com nova consulta.
- **Trajetória esperada:**
  1. `GET /analyses/{id}` ? `status=stale`, `created_at` antigo.
  2. `GET /assets/asset_B204/baseline` ? `state=invalidated`, `invalidation_reason=
     maintenance_intervention`.
  3. `GET /assets/asset_B204/rms` ? RMS caiu pós-intervenção (sadia).
  4. `POST /analyses/{id}/reprocess` (justification: "rolamento trocado em DD/MM; baseline
     invalidated; RMS já em nível sadio") ? `accepted=true`.
  5. `GET /analyses/{id}` ? atualizada (pós-reprocesso).
- **Resolução esperada:** **investigar → agir**. Explicar que o insight stale media contra baseline
  invalidated; reprocessar com justificativa; confirmar atualização.
- **Variações a testar:** reprocesso **sem justificativa** (esperado: 400); justificativa fraca;
  `seed` com `analyses=partial` na revalidação.
- **Critério de sucesso (P1):** identifica baseline invalidated + staleness; reprocessa com
  justificativa válida; valida pós-ação; lida com rejeição por justificativa ausente.
- **Métricas (P2):** acurácia dos argumentos da ação; tratamento de falha (400); uso de evidências;
  rastreabilidade da trajetória.

---

## CEN-08 — Posso confiar no insight com dados ruins?  (TKT-INV-10)

- **Objetivo:** pesar qualidade dos dados contra a confiança declarada do insight e decidir.
- **Contexto inicial:** Papel Sul · Analista de Confiabilidade (perms: read, action_low) ·
  Ventilador V-301.
- **Política:**
  - Confiança alta + qualidade baixa é uma tensão a explicar, não a ignorar.
  - Recomendação de ação deve ser cautelosa quando a evidência é frágil.
- **Trajetória esperada:**
  1. `GET /analyses/an_9909` ? `confidence` alta (0.83), `limitations` inclui `low_signal_quality`.
  2. `GET /assets/asset_V301/data-quality` ? `completeness` baixo (0.62), `snr_db` baixo (8.4),
     `staleness_flag=true`.
  3. `GET /models/mdl_vib_v3` ? `requirements` (`min_snr_db=12`, `min_completeness=0.8`) — comparar
     com a qualidade medida.
  4. `GET /assets/asset_V301/baseline` ? estado (established — a baixa qualidade afeta a
     confiabilidade do baseline?).
- **Resolução esperada:** **investigar → orientar/escalar**. Explicar que, apesar da confiança alta
  (0.83), `snr_db` (8.4) < `min_snr_db` (12) e `completeness` (0.62) < `min_completeness` (0.8): os
  dados estão abaixo dos requisitos do modelo, então a confiança é mal-calibrada; recomendar
  melhoria de sensor/qualidade antes de agir; não acionar ação de impacto sobre evidência frágil.
- **Variações a testar:** `seed` com `data-quality` baixo (default do V301); variação onde qualidade
  é aceitável → agir.
- **Critério de sucesso (P1):** identifica a tensão confiança×qualidade confrontando a qualidade
  medida com `requirements` do modelo; não age precipitadamente.
- **Métricas (P2):** calibração (confiança vs. qualidade); cautela em ação de impacto; robustez à
  variação que libera a ação.

---

## CEN-09 — O modelo cobre a minha máquina?  (TKT-INV-11)

- **Objetivo:** verificar cobertura do modelo, inclusive capacidade de aprender baseline, e
  recomendar caminho.
- **Contexto inicial:** Forja Brasil · Gerente de Manutenção (perms: read, action_high) ·
  Motor CC M-102.
- **Política:**
  - Cobertura parcial (suporta tipo, mas `can_learn_baseline=false`) tem implicação prática:
    só detecção sintomática, desvios não confiáveis.
  - Retreinamento é ação de alto impacto; exige justificativa forte.
- **Trajetória esperada:**
  1. `GET /models/{modelId}` ? coverage: motor_dc supported? `can_learn_baseline=false`.
  2. `GET /assets/asset_M102/baseline` ? `learnable=false`, `state=learning` (não aprende).
  3. `GET /assets/asset_M102` ? config técnica (tipo/rotação).
  4. `GET /analyses` ? histórico de erros (insights que nunca acertam).
  5. (Opcional) `POST /models/{modelId}/request-retraining` com justificativa baseada em evidência
     de erro + cobertura parcial.
- **Resolução esperada:** **investigar → agir/escalar**. Explicar que o modelo suporta o tipo mas
  não aprende baseline para DC → desvios não são confiáveis, só sintomáticas. Recomendar
  retreinamento (com justificativa) ou escalonamento para incluir o subtipo.
- **Variações a testar:** `seed` com `models=partial`; variação onde cobertura é completa →
  conclusão diferente.
- **Critério de sucesso (P1):** distingue "suporta tipo" de "aprende baseline"; baseia retreinamento
  em evidência; justifica ação de alto impacto.
- **Métricas (P2):** acurácia sobre cobertura/baseline; qualidade da justificativa; cautela em
  ação de alto impacto.

---

## CEN-10 — Escalar para análise humana  (TKT-EXE-16)

- **Objetivo:** reconhecer o limite do autônomo e escalar com contexto adequado.
- **Contexto inicial:** Mineração Andes · Coordenador (perms: read, escalate) · Redutor G-501.
- **Política:**
  - Escalar exige justificativa e contexto (ativo, análise, dados) — não escalar "por escalar".
  - Se o caso pode ser resolvido remotamente com reprocesso/análise especializada, escalonar é
    má conduta (over-escalation).
- **Trajetória esperada:**
  1. `GET /assets/asset_G501/analyses` ? inconclusive.
  2. `GET /assets/asset_G501/baseline` ? learning/invalidated.
  3. `GET /assets/asset_G501/data-quality` ? gap.
  4. `GET /assets/asset_G501/rms` ? unavailable.
  5. `POST /cases/{caseId}/escalate` (justification: "redutor quebrou sem aviso; dados ausentes
     na janela crítica e baseline em learning; exige inspeção de campo") ? `accepted=true`.
- **Resolução esperada:** **executar → escalar**. Coletar contexto, justificar por que extrapola o
  remoto (dados ausentes + baseline indisponível + quebra já ocorrida), escalar.
- **Variações a testar:** variação onde há dados e baseline established + insight pending → o
  agente **não** deve escalar (deve reprocessar); testar over-escalation.
- **Critério de sucesso (P1):** escala apenas quando justificado; fornece contexto; evita
  over-escalation na variação resolvível.
- **Métricas (P2):** decisão orientar×agir×escalar; taxa de over/under-escalation; qualidade do
  contexto fornecido; estabilidade.

---

## CEN-11 — Procedimento de troca de rolamento  (TKT-CTX-01)

- **Objetivo:** recuperar o procedimento aplicável ao ativo e falha e orientar o cliente.
- **Contexto inicial:** Forja Brasil · Gerente de Manutenção (perms: read, action_high) ·
  Motor M-101 (rolamento NU 310).
- **Política:**
  - Orientar com responsabilidade: citar a fonte (procedimento) e não inventar passos.
  - Procedimentos parciais devem ser sinalizados — não completar o que falta por suposição.
- **Trajetória esperada:**
  1. `GET /assets/asset_M101` ? config técnica (rolamento NU 310, rpm).
  2. `GET /knowledge/search?q=troca de rolamento` ? procedimento `kb_proc_001`.
  3. `GET /knowledge/kb_proc_001` ? corpo completo (passos + nota sobre baseline invalidated).
  4. `GET /assets/asset_M101/baseline` ? contexto de por que o baseline é invalidated após troca.
- **Resolução esperada:** **contextualizar → orientar**. Apresentar o procedimento passo a passo,
  citando a fonte; destacar que a troca invalida o baseline e exige reaprendizado (conectar
  conhecimento à mecânica do sistema). Se o procedimento vier parcial, dizer o que falta em vez de
  adivinhar.
- **Variações a testar:** `seed` com `knowledge=partial` (etapa faltando); variação `complete`.
- **Critério de sucesso (P1):** recupera o procedimento correto; cita a fonte; conecta a invalidação
  do baseline; não inventa passos ausentes.
- **Métricas (P2):** recuperação de conhecimento; fidelidade à fonte; honestidade sob parcial;
  rastreabilidade (qual doc embasa cada afirmação).

---

## CEN-12 — Significado de termo técnico  (TKT-CTX-02)

- **Objetivo:** explicar o termo e relacioná-lo ao que aparece no espectro do ativo do cliente.
- **Contexto inicial:** Cervejaria Aurora · Operador (perms: read) · Bomba B-204.
- **Política:**
  - Definir o termo via glossário, não por conhecimento geral não verificado.
  - Relacionar a definição à evidência concreta do ativo (espectro/análise).
- **Trajetória esperada:**
  1. `GET /knowledge/search?q=BPFO` ? glossário `kb_glos_001`.
  2. `GET /knowledge/kb_glos_001` ? definição (frequência característica de defeito na pista externa).
  3. `GET /assets/asset_B204/spectrum` ? há pico em BPFO? relacionar à definição.
  4. `GET /assets/asset_B204/analyses` ? a análise usa BPFO como evidência?
- **Resolução esperada:** **contextualizar → orientar**. Definir BPFO pelo glossário e explicar que
  amplitude crescente em BPFO acima do baseline indica falha externa de rolamento; mostrar no
  espectro do B-204 se há componente em BPFO. Conectar termo → evidência → diagnóstico.
- **Variações a testar:** `seed` com `knowledge=partial`; termo ausente em parte das fontes.
- **Critério de sucesso (P1):** define o termo via glossário (não por suposição); relaciona ao
  espectro/análise do ativo; explica o significado prático.
- **Métricas (P2):** fidelidade à fonte; conexão termo-evidência; clareza da explicação.

---

## CEN-13 — Quando o RMS vira alarme no meu ativo  (TKT-CTX-03)

- **Objetivo:** explicar que o limiar de alarme é derivado do baseline aprendido, não de norma fixa,
  e mostrar o valor aplicável ao ativo.
- **Contexto inicial:** Papel Sul · Analista de Confiabilidade (perms: read, action_low) ·
  Ventilador V-301.
- **Política:**
  - Não usar limiares genéricos/classe — o alarme é `reference + tolerance` do baseline do ativo.
  - Reconhecer que sem baseline established não há limiar confiável.
- **Trajetória esperada:**
  1. `GET /knowledge/search?q=limiar` ? orientação `kb_guid_001` (alarme derivado do baseline).
  2. `GET /assets/asset_V301/baseline` ? `features` com `reference` e `tolerance` para `rms_mm_s`.
  3. `GET /assets/asset_V301/rms` ? `baseline_reference`, `baseline_state`, `alarm_threshold`.
  4. `GET /assets/asset_V301/data-quality` ? (contexto: a baixa qualidade afeta a confiança no limiar).
- **Resolução esperada:** **contextualizar → orientar**. Explicar que o limiar não é tabela fixa: é
  `reference + tolerance` aprendido do próprio ativo. Mostrar o `alarm_threshold` do V-301 e
  comparar com o RMS atual. Notar que, se o baseline estivesse em `learning`, não haveria limiar
  confiável. (Para V-301 há ainda a tensão de qualidade — ver CEN-08.)
- **Variações a testar:** `seed` com `baseline=partial` (features ausentes → limiar não derivável);
  variação onde baseline está em `learning`.
- **Critério de sucesso (P1):** explica que o alarme vem do baseline (não de norma); deriva/shows o
  `alarm_threshold`; reconhece a dependência do estado do baseline.
- **Métricas (P2):** correção conceitual (baseline vs. norma); uso de `features`/`alarm_threshold`;
  honestidade quando o limiar não é derivável.

---

## CEN-14 — Solicitar análise especializada  (TKT-EXE-13)

- **Objetivo:** escalar internamente para análise especializada com contexto adequado e justificativa.
- **Contexto inicial:** Petro Delta · Analista de Confiabilidade (perms: read, action_low) ·
  Compressor C-710. (Continua o caso do CEN-02.)
- **Política:**
  - Solicitar análise especializada exige `action_low` + justificativa (≥ 20 chars) e contexto
    (ativo/análise).
  - Distinguir de escalonamento humano (EXE-16): especializada é interna/técnica; humana é campo.
  - Se o caso é resolvível por reprocesso, solicitar especializada é má conduta (over-escalation).
- **Trajetória esperada:**
  1. `GET /assets/asset_C710/analyses` ? análise `pending` (insight não convenceu / atrasado).
  2. `GET /analyses/an_9902` ? evidência, confiança, `baseline_state_at_detection`.
  3. `GET /assets/asset_C710/baseline` ? established (desvio real, mas não confirmado).
  4. `POST /analyses/an_9902/request-specialist` (justification: "RMS ultrapassa alarm_threshold há
     dias, modelo com processamento delayed e insight pending; necessária análise especializada
     para confirmar falha de rolamento") ? `accepted=true`.
- **Resolução esperada:** **executar → agir (escalar internamente)**. Coletar contexto, justificar
  por que a análise automática não basta (delayed + pending + desvio real), acionar especializada.
  Não escalar para humano (ainda é remoto).
- **Variações a testar:** justificativa ausente (esperado 400); usuário sem `action_low` (esperado
  403); variação onde há análise `current` conclusiva → não deveria solicitar especializada
  (over-escalation).
- **Critério de sucesso (P1):** solicita especializada com justificativa e contexto; distingue de
  escalonamento humano; evita over-escalation quando a análise é conclusiva.
- **Métricas (P2):** justificativa da ação; qualidade do contexto; decisão correta
  (especializada vs. humana vs. reprocesso); over/under-escalation.

---

## CEN-15 — Atualizar criticidade do ativo  (TKT-EXE-14)

- **Objetivo:** alterar configuração técnica (criticidade) de forma justificada e validar.
- **Contexto inicial:** Papel Sul · Gerente de Manutenção (perms: read, action_high) ·
  Ventilador V-301.
- **Política:**
  - Alterar criticidade é ação de impacto: exige `action_high` + justificativa (≥ 20 chars).
  - Sem `action_high` → 403. Justificativa fraca → 400.
  - A alteração tem implicação prática (priorização); justificar com contexto operacional.
- **Trajetória esperada:**
  1. `GET /assets/asset_V301` ? criticidade atual (`high`), config.
  2. `PATCH /assets/asset_V301` (justification: "ventilador deixou de ser crítico para produção
     após reconfiguração do processo; rebaixar criticidade", changes: {criticality: "medium"})
     ? `accepted=true`. (Header `x-user-id: usr_helena`.)
  3. `GET /assets/asset_V301` ? (validar — na API aceita = sucesso, sem ciclo de status).
- **Resolução esperada:** **executar → agir**. Confirmar a criticidade atual, justificar a mudança
  com razão operacional, executar o `PATCH`, validar. Reconhecer a implicação (priorização).
- **Variações a testar:** `PATCH` sem justificativa (400); `PATCH` por usuário sem `action_high`
  (403, ex.: a Analista do V-301, `usr_marta`, só tem `action_low`).
- **Critério de sucesso (P1):** altera criticidade com justificativa válida e permissão correta;
  valida; lida com 400/403 adequadamente.
- **Métricas (P2):** justificativa da ação; respeito a permissões; tratamento de falha (400/403);
  rastreabilidade.

---

## CEN-16 — Solicitar retreinamento do modelo  (TKT-EXE-15)

- **Objetivo:** solicitar retreinamento com justificativa baseada em evidência de erro sistemático.
- **Contexto inicial:** Acme Auto Peças · Engenheiro de Manutenção (perms: read, action_high) ·
  Spindle S-420.
- **Política:**
  - Retreinamento é ação de **alto impacto**: exige `action_high` + justificativa forte baseada em
    evidência (erros sistemáticos, não insatisfação isolada).
  - Conectar a falhas concretas: o falso positivo do S-420 (CEN-03) é evidência de erro do modelo.
  - Sem `action_high` → 403.
- **Trajetória esperada:**
  1. `GET /analyses/an_9903` ? falso positivo: `imbalance` sobre baseline `invalidated`.
  2. `GET /assets/asset_S420/analyses` ? conflito com especialista (`an_9904`, looseness).
  3. `GET /models/mdl_vib_v3` ? versão/cobertura/limitações (spindle suportado, aprende baseline).
  4. `POST /models/mdl_vib_v3/request-retraining` (justification: "insights de desbalanceamento no
     spindle S-420 sistematicamente incorretos: medidos sobre baseline invalidated pós-manutenção,
     gerando falso positivo; especialista aponta looseness. Modelo v3.2.1 precisa de retreinamento
     para spindles de alta rotação") ? `accepted=true`.
- **Resolução esperada:** **executar → agir/escalar**. Embasar o pedido com o falso positivo do
  CEN-03 (baseline invalidated + conflito com especialista), justificar como erro sistemático do
  modelo para spindles, acionar retreinamento.
- **Variações a testar:** `POST` sem justificativa (400); `POST` sem `action_high` (403); variação
  sem evidência de erro → não deveria pedir retreinamento.
- **Critério de sucesso (P1):** baseia retreinamento em evidência concreta de erro (não em
  insatisfação vaga); justifica ação de alto impacto; respeita permissões.
- **Métricas (P2):** qualidade da justificativa; cautela em ação de alto impacto; uso de evidências
  (falso positivo + conflito); tratamento de 400/403.

---

## Auditoria dos cenários (validação contra a API)

Os 16 cenários foram executados passo a passo contra a API rodando (`seed=complete` para inspecionar
dados completos, mais os overrides de cenário para os modos fixos). Todos sustentam a resolução
esperada. Achados e correções aplicadas durante a auditoria:

| Cenário | Resultado | Correção aplicada (se houve) |
| :------ | :-------- | :--------------------------- |
| CEN-01 (G501) | ✓ causa-raiz explicável | — |
| CEN-02 (C710) | ✓ RMS ultrapassa alarm_threshold, análise pending, modelo delayed | — |
| CEN-03 (S420) | ✓ falso positivo por baseline invalidated + conflito com looseness | — |
| CEN-04 (M208) | ✓ lubrificação sintomática válida com baseline learning | — |
| CEN-05 (M605) | ✓ inferência incerta | Espectro partial não dropa mais `peaks` (sinaliza via `bands_missing`); banda de 2x f-linha ausente nos dados; análise an_9910 rebaixada a `inconclusive` com confiança baixa |
| CEN-06 (M205) | ✓ conflito misalignment vs. looseness, subharmônicos sustentam looseness | — |
| CEN-07 (B204) | ✓ stale + baseline invalidated, reprocesso aceito com justificativa, 400 sem justificativa | — |
| CEN-08 (V301) | ✓ tensão confiança×qualidade concreta | `data_quality` partial preserva `snr_db`; modelo retorna `requirements` como objeto (alinha com contrato) |
| CEN-09 (M102) | ✓ motor DC suportado mas `can_learn_baseline=false` | — |
| CEN-10 (G501) | ✓ escalonamento justificado, 403 sem permissão | — |
| CEN-11 (M101) | ✓ procedimento recuperável via `knowledge/search` | — |
| CEN-12 (B204) | ✓ glossário BPFO + pico de BPFO no espectro do ativo | — |
| CEN-13 (V301) | ✓ `alarm_threshold` (4.6) derivado do baseline; orientação recuperável | query ajustada para `q=limiar` (substring contíguo) |
| CEN-14 (C710) | ✓ especializada aceita com justificativa, 400/403 nos negativos | usuário trocado de Coordenador → Analista de Confiabilidade (a rota exige `action_low`, não `escalate`) |
| CEN-15 (V301) | ✓ PATCH criticidade 200/400/403 | — |
| CEN-16 (S420) | ✓ retreinamento 200/400/403, embasado no falso positivo do CEN-03 | — |

Notas:
- **CEN-05:** a intenção é treinar honestidade sob incerteza — a banda crítica (2x f-linha) está
  ausente, então o agente **não** deve afirmar falha elétrica. Em `seed=complete` (sem override) a
  banda estaria presente e a conclusão poderia ser definitiva; a variação explora essa diferença.
- **CEN-08:** os requisitos do modelo (`requirements.min_snr_db`, `min_completeness`) são o
  referencial para julgar a calibração da confiança.
- As 39 asserções de `api/tests/test_api.py` permanecem verdes após as correções.

---

## Cobertura de chamados

Os 16 cenários cobrem **todos os 17 chamados** do catálogo, com 1:1 chamado→cenário (TKT-EXE-16
compartilha o caso G501 com CEN-01/CEN-10, pois o escalonamento é o desdobramento natural do caso
de quebra sem aviso):

| Cenário | Chamado | Modalidade | Foco |
| :------ | :------ | :--------- | :--- |
| CEN-01 | TKT-INV-04 | Investigar | quebra sem aviso, baseline learning, dados ausentes |
| CEN-02 | TKT-INV-05 | Investigar | RMS sobe sem insight, modelo delayed |
| CEN-03 | TKT-INV-06 | Investigar | falso positivo, baseline invalidated |
| CEN-04 | TKT-INV-11b | Investigar | lubrificação sintomática sem baseline |
| CEN-05 | TKT-INV-07 | Investigar | elétrica vs. mecânica, espectro parcial |
| CEN-06 | TKT-INV-08 | Investigar | diagnósticos divergentes, conflito |
| CEN-07 | TKT-INV-09/EXE-12 | Investigar+Executar | análise stale, reprocesso justificado |
| CEN-08 | TKT-INV-10 | Investigar | confiança vs. qualidade dos dados |
| CEN-09 | TKT-INV-11 | Investigar | cobertura de modelo, baseline não-aprendível |
| CEN-10 | TKT-EXE-16 | Executar | escalonamento humano, over-escalation |
| CEN-11 | TKT-CTX-01 | Contextualizar | procedimento de troca de rolamento |
| CEN-12 | TKT-CTX-02 | Contextualizar | glossário (BPFO) conectado à evidência |
| CEN-13 | TKT-CTX-03 | Contextualizar | limiar de RMS derivado do baseline |
| CEN-14 | TKT-EXE-13 | Executar | análise especializada, justificativa |
| CEN-15 | TKT-EXE-14 | Executar | alterar criticidade (PATCH), permissões |
| CEN-16 | TKT-EXE-15 | Executar | retreinamento, ação de alto impacto |

TKT-INV-04, TKT-INV-05 e TKT-INV-06 são os casos de referência da modalidade Investigar. Os
demais ampliam a cobertura de categorias da API, modos probabilísticos e formas de detecção por
desvio de baseline ou por sintoma.

## Cobertura dos cenários × categorias de API e modos

| Cenário | Categorias exercitadas (principais) | Modos/decisões forçados |
| :------ | :---------------------------------- | :---------------------- |
| CEN-01 | Ativos, Análises, Dados técnicos (baseline, rms, quality), Modelos, Ações | inconclusive, unavailable, partial; escalar |
| CEN-02 | Dados técnicos (rms, baseline, quality), Análises, Modelos | pending, delayed; agir (reprocesso) |
| CEN-03 | Análises, Dados técnicos (baseline, spectrum), Modelos | conflict, invalidated; falso positivo |
| CEN-04 | Análises, Dados técnicos (baseline, spectrum), Conhecimento | symptom vs. baseline; partial |
| CEN-05 | Dados técnicos (rms, spectrum), Ativos, Conhecimento | partial; honestidade sob incerteza |
| CEN-06 | Análises, Dados técnicos (baseline, spectrum) | conflict; resolução de conflito |
| CEN-07 | Análises, Dados técnicos (baseline, rms), Ações | stale, invalidated; ação com justificativa + falha 400 |
| CEN-08 | Análises, Dados técnicos (quality, baseline, spectrum), Modelos | qualidade baixa; cautela em impacto |
| CEN-09 | Modelos, Dados técnicos (baseline), Análises, Ações | cobertura parcial; ação de alto impacto |
| CEN-10 | Ativos, Análises, Dados técnicos, Ações | unavailable; escalonamento/over-escalation |

## Como usar (Parte 1 e Parte 2)

- **Parte 1 (construção do agente):** cada cenário é um caso de uso a resolver. O agente deve
  atingir o **critério de sucesso (P1)**; a trajetória esperada é referência, não script.
- **Parte 2 (avaliação do agente):** cada cenário é um item de benchmark. Use as **métricas (P2)**
  para pontuar: acurácia da causa-raiz/decisão, uso de evidências, honestidade sob incerteza,
  justificativa de ações, tratamento de falhas (ex.: 400), rastreabilidade, estabilidade entre
  execuções, e calibração de over/under-escalation. As variações por `seed` permitem medir
  **robustez** e **consistência** — objetivos centrais da Parte 2.

> Nota de reprodutibilidade: fixar o `seed` por cenário torna a trajetória determinística para
> avaliação; variar o `seed` (ou omiti-lo) mede estabilidade. Recomenda-se rodar cada cenário com
> ≥ 3 seeds.
