# Agente industrial TRACTIAN × Inteli

Projeto individual de engenharia de agentes para atendimento industrial, desenvolvido por [Leunam Sousa de Jesus](https://www.linkedin.com/in/leunam).

O sistema deverá receber uma solicitação, investigar dados por APIs, explicar sua decisão com evidências e executar somente ações permitidas. O foco atual é o backend e o aprendizado prático da arquitetura; não há frontend no escopo inicial.

> **Estado atual:** existem o simulador FastAPI, dados, contratos, cenários, cliente HTTP assíncrono, dez tools LangChain de leitura, cinco proposal tools sem efeito, política determinística, cinco operações HTTP fixas, estado tipado, fronteira Python, grafo LangGraph com planner e writer LLM opt-in separados, ledger determinístico, gate de liberação, revisão humana retomável e checkpointer SQLite de desenvolvimento. Em 03/09/2026, `make test` passou com 99 testes da API e 1.681 do agente (1.780 no total); permanece somente o `PendingDeprecationWarning` conhecido de `python_multipart`. As Fases 1 a 9 estão concluídas. O grafo atual ainda não é um agente de produção: Logfire e runner Pydantic Evals continuam planejados em [`TASKS.md`](./TASKS.md).

## Problema

O agente atende três tipos de solicitação:

- **Contextualizar:** consultar conhecimento e explicar de forma responsável;
- **Investigar:** consultar ativos, análises e dados técnicos para recomendar próximos passos;
- **Executar:** propor ou realizar uma ação permitida, ou encaminhar para revisão humana.

Se faltarem evidências, houver conflito ou a ação ultrapassar a permissão do usuário, o agente não inventa uma conclusão. Ele solicita dados, confirmação ou revisão humana.

## Arquitetura atual e evolução planejada

```mermaid
flowchart TD
    A[Solicitação + identidade] --> B[Fronteira de entrada do agente]
    B --> C[Estado persistente do LangGraph]
    C --> D[Planner: decide o próximo passo]
    D --> E[Tools com validação em código]
    E --> F[API industrial]
    F --> G[Normalização e ledger de evidências]
    G --> D
    D --> H[Writer: redige com base no ledger]
    H --> I[Gate determinístico de segurança]
    I --> J[Resposta ou pedido seguro]
    I --> L[Revisão humana retomável]
    L --> I
    C -. traces e métricas .-> K[Logfire]
```

Existe **um agente lógico com dois papéis de LLM**:

1. o **planner** escolhe entre investigar, pedir informação, propor uma ação ou encerrar;
2. o **writer** recebe somente a decisão e as evidências necessárias para produzir a resposta.

Essa separação reduz contexto, facilita testes e impede que a redação altere silenciosamente a decisão. LangGraph controla estado, nós, transições, interrupções e retomadas. Código determinístico continua responsável por identidade, permissão, argumentos, justificativa, idempotência e liberação da resposta.

No MVP, planner e writer podem usar o mesmo modelo com prompts e contratos diferentes. A separação é de responsabilidade, não uma obrigação de contratar dois modelos.

O writer usa o prompt `writer-v1`, não recebe tools e devolve somente um draft
estruturado com a decisão imutável, IDs ordenados e próximo passo enumerado. Seu
contexto usa um orçamento total de 64 referências e somente IDs e categorias
fechadas: alvos, valores, timestamps e caminhos técnicos permanecem no ledger.
O gate deriva a decisão e o alvo da ação da request confiável, da proposal, da
intenção, da aprovação e do recibo atuais; recompõe IDs e conflitos e exige
`read` sempre que o draft cita um fato de tool. Apenas um atestado `release`
permite ao renderer buscar os valores no ledger; a mensagem técnica não vem do
modelo e é revalidada ao restaurar o checkpoint. Uma saída de formato inválido
admite somente um repair; contador, âncora e próximo nó impedem uma terceira
chamada. Duas falhas, erro de provider ou qualquer incerteza terminam em aviso
sanitizado ou, quando aplicável, revisão humana.

A revisão humana é uma fronteira persistente e retomável: o pedido sanitizado é
gravado antes da interrupção e a identidade da pessoa revisora chega separada da
resposta. A aprovação só remove o motivo fechado `HUMAN_DISPOSITION_REQUIRED`
de um draft `GUIDE` que já passou pelas demais verificações e pediu disposição
humana como próximo passo; ela nunca transforma a decisão explícita
`REQUIRE_HUMAN_REVIEW` do planner em orientação. Edição limita-se à seleção e à
ordem das evidências atuais e ao próximo passo enumerado, e só é oferecida para
`HUMAN_DISPOSITION_REQUIRED`, `WRITER_FAILURE`,
`EVIDENCE_REFERENCE_MISMATCH` ou `NEXT_STEP_MISMATCH`; demais motivos aceitam
somente rejeição. Para `ACT` e `ESCALATE`, a seleção revisada precisa preservar
o fato `accepted` do recibo ligado à intenção atual. Toda retomada revalida
ledger, permissões e intenções; bloqueios duros, rejeição, expiração ou uma
segunda necessidade de revisão encerram de forma segura, sem executar ação.
O ID da revisão inclui o escopo do thread; contratos rejeitam coerções Python e
o envelope confiável de autoria, empresa, permissão, horário e reply fica ligado
à auditoria. Perda da permissão `read` não impede o fechamento interno seguro,
mas bloqueia qualquer retorno público do estado técnico, inclusive em replay.
O orçamento reserva o interrupt e o segundo gate antes de iniciar caminhos
revisáveis; checkpoints legados sem essa reserva terminam por contrato próprio.
No vencimento, a expiração precede a semântica da operação; uma nova solicitação
encerra primeiro a revisão vencida, sem reexecutar trabalho antigo.

O **ledger de evidências** associa fatos às fontes consultadas no estado da execução. Ele recebe somente observações de leitura validadas e recibos tipados de intenções terminais; texto livre de LLM, proposals e mensagens de recibo não viram fatos. O Logfire receberá traces e métricas para consulta humana e operação; ele não será o banco principal do ledger nem uma fonte que o agente consulta durante o atendimento. Logfire ainda não está implementado.

### Persistência e idempotência

Há dois armazenamentos SQLite independentes no desenvolvimento. A API usa `IDEMPOTENCY_DB_PATH` (padrão `.run/idempotency.sqlite3`) para a idempotência de reprocesso; `IDEMPOTENCY_PROCESSING_TIMEOUT_SECONDS` altera seu limite de processamento, cujo padrão é 300 segundos. O grafo usa `.run/agent-checkpoints.sqlite3` por `AsyncSqliteSaver`, com serializer restrito e sem remoção automática de threads; a exclusão é explícita. PostgreSQL continua sendo a evolução futura, não uma implementação atual.

- `thread_id` identifica a linha persistida; um thread pode receber novos `request_id`, e cada execução ou retomada recebe novo `execution_id`. Mudança de caso, empresa, pessoa usuária ou alvo confiável falha fechada.
- A intenção persistida registra ID, request, escopo imutável, hash, decisão/status, origem da aprovação autorizadora, tentativas, execução preparadora e recibo ou erro; runtime, cliente, credenciais, seed, golden set, resposta HTTP bruta e raciocínio não entram no checkpoint.
- O estado persiste somente os três alvos confiáveis necessários à escrita — ativo central, caso atual e modelo industrial configurado — e os vincula novamente à request, à intenção e ao atestado; esse contexto nunca é enviado ao writer.
- A assinatura estrutural da intenção inclui ação, alvo canônico, parâmetros materiais e justificativa, e o runtime precisa coincidir com o contexto persistido em cada retomada. Esse hash detecta divergência, mas não é um MAC nem prova autenticidade contra alguém com escrita arbitrária no SQLite e capacidade de recalcular todo o checkpoint; por isso o banco de checkpoints permanece uma fronteira confiável e deve ter acesso controlado.
- Criação e retomada que podem escrever usam `durability="sync"`; `prepare_intent` fica em superstep distinto e é persistido antes de `execute_action`. Confirmações usam `interrupt()` estruturado e `Command` pelo ID na fronteira confiável.
- O primeiro alvo de idempotência é `POST /analyses/{analysisId}/reprocess`.
- A API exige uma `Idempotency-Key` de 1 a 255 caracteres sem espaços, reserva a intenção antes da ação, persiste respostas concluídas em SQLite e faz replay mesmo após recriar o armazenamento; mesma chave com payload diferente retorna `409 Conflict`.
- O fluxo determinístico de reprocesso cria `tractian-agent:<uuid>` uma única vez após `allow`, persiste-a antes do HTTP e a reutiliza somente em, no máximo, um retry com o mesmo corpo. Cliente, proposal tools e operações não criam retries.
- A reserva é atômica: enquanto a primeira chamada está em execução, uma chamada concorrente com a mesma intenção recebe `409 IDEMPOTENCY_IN_PROGRESS` e não cria outra ação. Uma falha inesperada durante a ação marca o resultado como `uncertain`; retries recebem `409 IDEMPOTENCY_OUTCOME_UNKNOWN` em vez de repetir a ação.
- Um registro `processing` com mais de 300 segundos, ou o limite definido em `IDEMPOTENCY_PROCESSING_TIMEOUT_SECONDS`, muda para `uncertain` sem repetir a ação.
- Registros vencidos são removidos sob demanda depois de 7 dias; a mesma chave, após esse prazo, inicia uma nova execução. O horário de criação identifica cada geração e impede que um trabalho antigo altere a reserva nova.
- Se a resposta se perde depois do commit, o retry recupera a resposta persistida sem repetir a ação.
- Especialista, criticidade, retreinamento e escalonamento não usam chave nem retry automático: têm no máximo um despacho. Em retomada de uma intenção `prepared` por outro `execution_id`, terminam conservadoramente em `uncertain/0` sem tocar a rede, antes de revalidar escopo, política ou preflight; somente a transição exata `prepare_intent → execute_action` recebe essa exceção na fronteira, mesmo se ativo, caso ou modelo do runtime mudaram. Isso pode produzir falso incerto e `attempts` pode subcontar um crash pós-efeito; o lock por `thread_id` é local ao processo/event loop e o preflight do especialista é opaco.
- No grafo com planner, esse único terminal de retomada é derivado do shape estrutural completo — ação não idempotente, intenção `uncertain/0` preparada por outra execução, sem recibo e com âncora `execute_action` — e exige erro `NON_IDEMPOTENT_OUTCOME_UNKNOWN_AFTER_RESUME` e resultado final canônicos. Somente ele segue diretamente para `END`, com mensagem determinística e sem writer/gate; qualquer outro erro ou resultado continua no writer e na porta de segurança.
- A API aplica uma segunda barreira de empresa nas ações ligadas a ativo, análise e chamado. Os cinco endpoints de ação retornam recibos sem reescrever os fixtures; o PATCH aceita somente o formulário técnico documentado e falha fechado para campos, tipos ou valores inválidos.
- O processo da API abre somente uma allowlist de Parquets operacionais e lê chamados do pacote público sanitizado em `agent-input/`. O `data/cases.parquet`, o gabarito em `eval/` e os cenários de teste não entram no runtime.

### Modelos

`ModelProvider` é a interface comum para construir `BaseChatModel` a partir do
`ModelConfig` estrito. O adapter inicial é `GroqModelProvider`, com
`openai/gpt-oss-120b`, temperatura zero, timeout de 30 segundos, no máximo 512
tokens e sem retry oculto. O planner usa `bind_tools` para uma escolha por
turno e uma chamada Pydantic separada para encerrar. No adapter Groq, essa
finalização usa o JSON Schema nativo estrito (`method="json_schema"`,
`strict=True`); a correção evita o `tool_use_failed` observado quando o default
`function_calling` criava uma tool sintética que GPT-OSS podia não chamar. Para
schemas Pydantic, o adapter valida diretamente o texto JSON com
`model_validate_json`: valores de Enum mantêm a semântica do wire JSON, enquanto
os validators e as regras de coerência do contrato continuam falhando fechados.
O planner e seu contrato permanecem independentes desse detalhe de transporte.
O modelo, credenciais e respostas brutas não entram no estado. IDs persistidos
de tool call são derivados pelo runtime de `request_id` e do ordinal, nunca do
ID externo do provider. NVIDIA NIM continua alternativa futura; o writer pode
usar outro modelo sem mudar as regras de negócio ou o gate determinístico.

O smoke opt-in `make smoke-groq` compara `openai/gpt-oss-120b` e
`openai/gpt-oss-20b` com dados sintéticos, sem retry e com o mesmo orçamento de
512 tokens da configuração inicial. A finalização usa o contrato real
`PlannerTerminalDecision` e exige `guide`, `sufficient_evidence` e informação
ausente nula. Por padrão há uma rodada por modelo e a estabilidade é declarada
como `not_measured`; defina
`GROQ_SMOKE_RUNS=2` ou maior para comparar as assinaturas dos contratos entre
rodadas. Ele exige `GROQ_API_KEY` já disponível no ambiente, não lê `.env` e só
imprime métricas agregadas seguras. Sem a chave, sai com código zero e
`status=skipped reason=missing_groq_api_key`, sem tocar a rede. No aceite desta
entrega, o smoke ao vivo inicialmente ficou **skipped** porque a chave não
estava disponível. Na verificação manual posterior, o 120b passou os dois
contratos; o 20b passou a seleção e falhou na finalização com o orçamento antigo
de 128 tokens. Um probe isolado confirmou `json_validate_failed` em 128 e
conclusão válida em 512, com `finish_reason=stop`, 171 tokens de saída, 150 deles
de raciocínio e Pydantic válido. Depois da correção de orçamento e parser, a
repetição completa aprovou ambos os modelos com português, tool, argumentos e
Pydantic válidos em duas chamadas. Como houve somente uma rodada, a estabilidade
permanece corretamente `not_measured`; as latências observadas não constituem
benchmark para escolher o modelo.

## Avaliação offline planejada

Avaliação não participa do atendimento ao cliente nem provoca retries automáticos no runtime.

```mermaid
flowchart LR
    A[Casos de benchmark] --> B[Pydantic Evals]
    B --> C[Executa o agente]
    C --> D[Checks programáticos]
    C --> E[Juiz cego do resultado]
    C --> F[Juiz da trajetória]
    G[Golden set oculto] --> D
    G --> E
    G --> F
    H[Rótulos humanos] --> I[Calibração]
    D --> I
    E --> I
    F --> I
```

- **Checks programáticos:** contratos, tools, argumentos, permissões, erros e regras críticas.
- **Juiz do resultado:** relevância, fidelidade às evidências, honestidade, decisão, clareza e tom.
- **Juiz da trajetória:** escolha e ordem das tools, falhas, repetições e momento de parada.
- **Humano:** cria 20–30 rótulos de referência e analisa concordância, falsos aprovados e falsos reprovados.

Os limiares `0.7`, `0.8` e `0.9` serão comparados; `0.8` é apenas o candidato inicial. Como haverá um único avaliador humano, Cohen's kappa mede humano × juiz. Krippendorff alpha fica fora do escopo atual.

O golden set nunca entra no runtime e não é consultado por RAG. Ele é visível apenas aos avaliadores depois que a execução termina.

## Tecnologias

| Responsabilidade | Tecnologia | Motivo |
|---|---|---|
| API simulada | FastAPI | Contrato HTTP e testes rápidos |
| Orquestração | LangGraph + LangChain | Estado, ciclos, tools e retomada |
| Contratos | Pydantic | Tipagem e validação explícitas |
| Cliente HTTP | httpx | Chamadas assíncronas e testáveis |
| Modelos | Groq; NVIDIA NIM candidato | Início gratuito e troca por adapter |
| Persistência | SQLite; PostgreSQL futuro | Simplicidade local e caminho de escala |
| Observabilidade | Pydantic Logfire | Traces e investigação sem criar dashboard |
| Testes | pytest | Unidade, integração e regressão |
| Avaliação | Pydantic Evals | Casos, avaliadores e experimentos tipados |

`promptfoo` poderá comparar prompts e modelos quando o pipeline estiver estável. `Ragas` só será adotado se o produto ganhar uma etapa real de RAG.

## Executar o que já existe

Requisitos: Python 3.10 ou superior e [`uv`](https://docs.astral.sh/uv/).

```bash
make setup   # instala API/agente e gera os dados
make up      # inicia a API em http://localhost:8000
make test    # executa as suítes da API e do agente
make logs    # acompanha o log da API
make stop    # encerra a API iniciada pelo Makefile
```

O Swagger fica em `http://localhost:8000/docs`. Ações usam o header `x-user-id`. O parâmetro `seed` reproduz variações; `seed=complete` pede retornos completos quando o cenário não possui override fixo.

## Desenvolvimento com uma IA copiloto

1. Abra [`AGENTS.md`](./AGENTS.md) para dar contexto e regras à IA.
2. Escolha a primeira fase incompleta e sem dependências em [`TASKS.md`](./TASKS.md).
3. Estude o conceito correspondente em [`LEARNING-GUIDE.md`](./LEARNING-GUIDE.md).
4. Peça à IA uma explicação curta, um micro-objetivo e critérios de teste — não a implementação inteira.
5. Implemente você mesmo uma mudança pequena; use a IA para revisar, explicar erros e sugerir testes.
6. Marque a tarefa concluída somente quando seus critérios de aceite passarem.

Prompt curto recomendado:

> Leia `AGENTS.md`, `TASKS.md` e a etapa relevante de `LEARNING-GUIDE.md`. Atue como professor e copiloto. Explique os conceitos sem assumir conhecimento prévio, defina um único micro-objetivo e seus testes. Não escreva a solução completa antes da minha tentativa; revise o código que eu produzir.

## Documentação

| Arquivo | Função |
|---|---|
| [`AGENTS.md`](./AGENTS.md) | Regras obrigatórias para qualquer IA ou contribuidor |
| [`TASKS.md`](./TASKS.md) | Backlog linear, decisões pendentes e critérios de aceite |
| [`LEARNING-GUIDE.md`](./LEARNING-GUIDE.md) | Conceitos a aprender na ordem de construção |
| [`CONTEXT.md`](./CONTEXT.md) | Vocabulário canônico do domínio |
| [`docs/api-contract.openapi.yaml`](./docs/api-contract.openapi.yaml) | Contrato HTTP fornecido |
| [`docs/data-schema.md`](./docs/data-schema.md) | Estrutura dos dados fornecidos |
| [`docs/test-scenarios.md`](./docs/test-scenarios.md) | Cenários comentados do benchmark |

## Estrutura

```text
.
├── README.md
├── AGENTS.md
├── TASKS.md
├── LEARNING-GUIDE.md
├── CONTEXT.md
├── LICENSE
├── Makefile
├── agent/                   # contratos, tools, planner, writer, gate, grafo e checkpointer
├── agent-input/             # entradas permitidas ao agente
├── api/                     # simulador FastAPI e testes
├── data/                    # dados do simulador
├── docs/                    # contrato, schema e cenários
└── eval/                    # gabarito restrito aos avaliadores
```

## Limitações atuais

- Existe um grafo LangGraph com planner e writer LLM opt-in separados, ledger de evidências, gate determinístico, revisão humana retomável, fluxos de escrita e checkpointer. Ele ainda não possui Logfire nem runner Pydantic Evals; portanto não é um agente de produção.
- As cinco proposal tools apenas propõem (`effect_executed=false`). Somente o fluxo determinístico, após política, confirmação quando necessária e checkpoint, acessa as cinco operações HTTP fixas.
- O simulador não representa todas as garantias transacionais de produção.
- As rotas de ação do simulador devolvem recibos, mas não alteram os recursos Parquet; um novo GET não comprova a mutação solicitada.

## Autor

**Leunam Sousa de Jesus** — [LinkedIn](https://www.linkedin.com/in/leunam)

## Licença

O código original deste repositório é distribuído sob a [Licença MIT](./LICENSE). Dados, marcas e materiais originados da TRACTIAN, do Inteli ou de terceiros continuam sujeitos aos direitos de seus respectivos titulares.
