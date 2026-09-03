# Backlog linear de construção

Este é o documento operacional do projeto. O [`LEARNING-GUIDE.md`](./LEARNING-GUIDE.md) ensina os conceitos; este arquivo registra **o que construir, quando decidir e como saber que terminou**.

## Como usar em cada sessão

1. Escolha a primeira etapa incompleta cujas dependências estejam concluídas.
2. Peça à IA que explique os conceitos e divida a etapa em um único micro-objetivo.
3. Antes do código, escreva o contrato esperado e os testes do micro-objetivo.
4. Implemente você mesmo; a IA revisa, explica falhas e ajuda apenas onde solicitado.
5. Execute os testes e marque itens somente com evidência do critério de aceite.
6. Registre decisões novas na própria etapa; altere a arquitetura do README apenas se a decisão for global.

Não antecipe frontend, banco vetorial, RAG, multiagentes, fine-tuning, `promptfoo` ou `Ragas`. Eles só entram mediante necessidade demonstrada.

## Mapa de estudos

Antes de iniciar uma fase, estude estas etapas do `LEARNING-GUIDE.md`:

| Fase de construção | Etapas de aprendizagem |
|---|---|
| 1 — idempotência | 1, 2, 4, 5 e 10 |
| 2 — contratos e cliente | 1, 2, 4 e 5 |
| 3 — tools de leitura | 6 |
| 4 — tools de escrita | 6 e 10 |
| 5 — estado e SQLite | 8 |
| 6 — provider e planner | 7 e 8 |
| 7 — ledger | 9 |
| 8 — writer e segurança | 9 e 10 |
| 9 — revisão humana | 10 |
| 10 — Logfire | 11 |
| 11 — runner | 12 |
| 12 — juízes | 13 |
| 13 — calibração | 14 |
| 14 — providers | 7 e 15 |
| 15 — entrega | 15 |

## Fase 0 — material-base

**Estado:** concluída.

- [x] Simulador FastAPI, dados e empacotamento do benchmark.
- [x] Contrato, schema, cenários e vocabulário reunidos.
- [x] Suíte-base inicial com 39 testes.
- [x] Arquitetura e ordem de aprendizagem definidas.

## Fase 1 — idempotência do reprocessamento

**Aprender:** idempotência, hash canônico, transação, condição de corrida e códigos HTTP.

**Decidir durante a etapa:** tabela SQLite, formato da chave, retenção, representação da resposta persistida e ponto exato usado para simular timeout.

**Decisões implementadas:** a API exige `Idempotency-Key` de 1 a 255 caracteres sem espaços; valores ausentes ou inválidos retornam `400 VALIDATION_ERROR`. Uma nova intenção usa uma nova chave, enquanto retries do mesmo pedido reutilizam a chave. O desenho abaixo descreve a persistência entregue.

**Persistência entregue:**

- Criar `api/app/idempotency.py` com `sqlite3`, separado do armazenamento Parquet e sem ORM.
- Usar `IDEMPOTENCY_DB_PATH`; na ausência da variável, gravar em `.run/idempotency.sqlite3`. Testes usam um arquivo temporário isolado. SQLite continua sendo a opção de desenvolvimento; PostgreSQL permanece a evolução futura.
- Manter registros por 7 dias desde a primeira solicitação e remover vencidos sob demanda, dentro de uma nova operação.
- Aceitar chaves de 1 a 255 caracteres sem espaços e diferenciar maiúsculas de minúsculas. A camada de execução de escritas, não a pessoa usuária nem o LLM, gera e persiste antes da chamada uma chave no formato `tractian-agent:<uuid>`; o cliente HTTP apenas a valida e propaga.
- Tratar a chave como protocolo, não como credencial. Autenticação e permissão protegem a ação; no simulador, `x-user-id` continua sendo apenas uma aproximação da identidade de produção.
- Identificar unicamente o pedido por pessoa autenticada, método, endpoint completo e chave idempotente, com restrição `UNIQUE` no SQLite sobre essa combinação.
- Calcular `payload_hash` como `sha256:v1:<digest>` sobre JSON canônico com chaves ordenadas, sem alterar valores nem persistir o payload original.
- Persistir `idempotency_key`, `user_id`, `method`, `endpoint`, `payload_hash`, `status`, `response_status`, `response_body`, `created_at`, `updated_at` e `expires_at`, com instantes em UTC.
- Usar os estados `processing`, `completed` e `uncertain`. Reservar e confirmar `processing` antes da ação; confirmar `completed` e a resposta antes de responder ao cliente.
- Considerar `processing` vencido após `IDEMPOTENCY_PROCESSING_TIMEOUT_SECONDS`, com padrão de 300 segundos, e convertê-lo para `uncertain` sem repetir a ação.
- Repetir status HTTP e corpo JSON originais quando chave, escopo e hash coincidirem em um registro `completed`.
- Retornar `409 IDEMPOTENCY_PAYLOAD_CONFLICT` para mesma chave e escopo com hash diferente, `409 IDEMPOTENCY_IN_PROGRESS` para processamento ativo e `409 IDEMPOTENCY_OUTCOME_UNKNOWN` para resultado incerto.
- Em falha inesperada durante a ação, retornar `500`, marcar `uncertain` e bloquear repetição automática. Erros anteriores à reserva não são persistidos.
- Simular perda de resposta somente nos testes, depois do commit e antes da entrega ao cliente; não expor flag ou header de falha em produção.
- Construir em fatias verticais: primeira execução/replay, conflito, reinicialização, concorrência, resultado incerto, expiração e timeout após commit, sempre um ciclo `RED → GREEN` por comportamento.

- [x] Exigir `Idempotency-Key` em `POST /analyses/{analysisId}/reprocess`.
- [x] Persistir chave, usuário, método, endpoint, hash do payload, status e resposta original.
- [x] Retornar a resposta original para mesma chave e mesmo payload.
- [x] Retornar `409 Conflict` para mesma chave e payload diferente.
- [x] Garantir atomicidade para duas requisições concorrentes.
- [x] Testar primeira execução, replay, conflito, concorrência e timeout após a ação/commit antes da resposta.

**Aceite:** durante a janela de retenção de 7 dias, nenhum retry ou acesso concorrente cria dois trabalhos para a mesma intenção. Após esse prazo, a chave pode iniciar uma nova execução, mas uma geração antiga não pode alterar o registro da nova.

## Fase 2 — contratos e cliente da API

**Aprender:** modelos Pydantic, cliente assíncrono, timeout e fronteira entre erro de transporte e erro de domínio.

**Decidir:** timeouts, categorias de erro, envelope normalizado e propagação da identidade.

**Decisões implementadas:**

- Manter o consumidor em um projeto Python próprio, `agent/`, separado do simulador em `api/`.
- Usar contratos Pydantic estritos (`extra="forbid"`) para solicitação, identidade, chamada de tool, resultado e erro; argumentos de tool são modelos Pydantic específicos e não recebem identidade. `asset_id` é obrigatório, mas anulável em chamados sem ativo central, conforme o esquema dos casos.
- Exigir identidade confiável em cada chamada e propagar somente `user_id` no header `x-user-id`; `company_id` permanece no contexto para validações determinísticas posteriores e não vira header inventado.
- Separar consultas envelopadas de respostas JSON diretas. Em `mode=complete`, validar `data` pelo modelo solicitado; em `partial`, `inconclusive`, `conflict` ou `unavailable`, preservar `mode`, `notes` e o JSON degradado sem fingir que o recurso completo existe.
- Classificar erros como `api`, `server`, `timeout`, `transport` ou `invalid_response`, preservando status e códigos fornecidos pela API quando o payload é válido.
- Usar timeouts configuráveis de 2 s para conexão, 10 s para leitura, 5 s para escrita e 2 s para obtenção de conexão do pool.
- Não fazer retry nem decidir se um erro é repetível dentro do cliente. A criação e a persistência da chave pertencem à intenção de escrita e ao estado persistido do grafo; o cliente valida o protocolo e apenas propaga a chave recebida.
- Recusar URLs externas antes de enviar identidade e não seguir redirects automaticamente.
- Testar o transporte somente com `httpx.MockTransport`, sem servidor, porta, espera real ou chamada de modelo.

- [x] Criar contratos tipados para solicitação, identidade, chamada de tool, resultado e erro.
- [x] Criar cliente `httpx` sem lógica de decisão do agente.
- [x] Normalizar respostas `2xx`, `4xx`, `5xx`, timeout e payload inválido.
- [x] Testar o cliente sem depender de servidor externo.

**Aceite:** toda resposta da API vira um resultado tipado ou erro explícito.

## Fase 3 — tools de leitura

**Aprender:** tool calling, descrição de ferramenta, validação de argumentos e princípio do menor acesso.

**Decidir:** agrupamento das tools, limites de tamanho, retries seguros e quais campos entram no retorno normalizado.

**Decisões implementadas:**

- Expor tools LangChain reais por intenção de consulta, sem uma tool genérica que aceite URL, método ou caminho. O catálogo inicial tem `get_asset`, `list_asset_analyses`, `get_analysis`, `get_baseline`, `get_rms_series`, `get_spectrum`, `get_data_quality`, `get_model`, `search_knowledge` e `get_knowledge_document`.
- Manter cada adapter LangChain fino: nome, descrição, schema público, contexto injetado e conversão do retorno. A operação determinística em Python valida escopo e significado, chama o `IndustrialApiClient` e normaliza a observação; ela não recria transporte nem autorização da API.
- Obter identidade, empresa, permissões, ativo central, cliente HTTP, `seed` de avaliação e modelo industrial configurado por contexto confiável. Esses dados não fazem parte dos argumentos visíveis ao modelo. `get_current_user` é uma consulta interna da fronteira de entrada, não uma tool do LLM.
- Usar argumentos Pydantic específicos e restritos. IDs aceitam somente o prefixo e os caracteres esperados; filtros usam valores fechados. Paths, método, headers e modelo de resposta ficam fixos em código.
- Aplicar menor acesso antes da chamada: consultas de ativo ficam limitadas ao ativo central; respostas completas também confirmam empresa e relação com o recurso pai antes de serem expostas. Conhecimento é global no escopo atual, e o modelo consultado é o configurado no runtime.
- Retornar conteúdo JSON compacto e normalizado para o modelo e um artifact JSON serializável para código, trace, ledger e avaliações futuras. O artifact não contém headers, identidade, cliente ou resposta HTTP bruta. Qualquer redução declara `truncated=true` e a quantidade omitida; não há truncamento silencioso.
- Não executar retry dentro das tools. Cada tentativa e cada falha permanecem explícitas para a política determinística do LangGraph.
- Preservar `mode`, `notes` e todo `ApiError`; validação inválida impede HTTP e exceção inesperada de programação não é convertida em sucesso.
- Construir em fatias verticais `RED → GREEN`, começando por `get_asset`, e testar contrato público, escopo, chamada fixa, modos, erros, conteúdo/artifact e integração mínima com `ToolNode` sem criar ainda o agente completo.
- Para baseline, RMS, espectro e qualidade, expor somente `asset_id` e `point_id` opcional; identidade, permissões, ativo central, cliente e seed continuam exclusivamente no `ReadToolRuntime`.
- Normalizar séries RMS em ordem cronológica estável e projetar conteúdo/artifact com espaçamento uniforme e extremos preservados para no máximo 100/1.000 amostras. Para espectro, usar somente os campos reais (`asset_id`, `point_id`, `collected_at`, `peaks`, `bands_missing`), ordenar picos por frequência de modo estável e limitar conteúdo/artifact a 20/200. Toda redução declara a quantidade omitida.
- Nas listagens completas de análises, validar todas as linhas antes de qualquer corte, devolver somente resumos no conteúdo do modelo (até 20) e guardar análises normalizadas completas no artifact (até 200). Em respostas degradadas, somente os campos conhecidos dos resumos ocupam esses mesmos limites. Todo corte declara contagens e truncamento.
- Derivar `alarm_threshold` apenas da feature `rms_mm_s` do baseline (`reference + tolerance`), sem inferir validade a partir do estado. Em dados degradados, aplicar o guard compartilhado de JSON parcial e recusar IDs de ativo ou ponto que contradigam o escopo já conhecido.
- Exigir ponto não nulo e verificável em payload completo técnico; validar `point_id` também no início de cada operação Python. Datas completas exigem hora e timezone; números de wire devem ser finitos antes de entrar em qualquer artifact JSON.
- Aplicar a mesma garantia JSON-safe a qualquer resposta degradada: o guard compartilhado recusa recursivamente NaN e infinitos antes de compor `content` ou artifact. Campos nullable obrigatórios no wire precisam aparecer explicitamente, ainda que com `null`.
- No cadastro completo do ativo, recusar NaN e infinitos em rotação e em todas as frequências técnicas antes da normalização. Em cadastro degradado, verificar recursivamente campos de ativo e empresa, inclusive variantes snake, kebab, camel e separadas por espaço.
- Tratar nomes de campos degradados de forma sensível a segmentos: combinações que representam identidade, cliente, autenticação, credenciais, runtime, seed, avaliação/gabarito ou envelope HTTP são recusadas em qualquer profundidade, sem bloquear nomes industriais como `baseline_reference`, `processing_state`, `response_time`, `bearing_authenticity` e `machine_runtime_hours`.
- Para `get_model`, usar exclusivamente `configured_model_id` imutável no `ReadToolRuntime` (padrão explícito `mdl_vib_v3`), sem expor o ID no schema público; o retorno completo confirma esse ID e a cobertura não aceita tipos de máquina duplicados.
- Para conhecimento global, `search_knowledge` aceita somente consulta sem espaços externos, de 2 a 200 caracteres, com conteúdo Unicode visível após normalização NFKD e remoção de marcas combinantes; a consulta válida original não é alterada. O filtro é fechado em `procedure`/`glossary`/`guidance` e, em resposta degradada, uma linha sem `type` não satisfaz filtro solicitado. A busca normaliza todos os documentos antes do corte de 10 metadados/snippets de 240 caracteres; o detalhe limita o corpo a 8.000 caracteres no contexto do modelo e 32.000 no artifact, sempre declarando caracteres retornados, omitidos e truncamento.
- Modelos e conhecimento preservam os modos e notas sem expor JSON HTTP bruto: em resposta degradada, projetam somente campos de domínio conhecidos, flags permitidas e limites aplicáveis. O validador de timestamps com data, hora e timezone passou a ser compartilhado pelas tools técnicas, de análises e de modelo.
- Publicar as dez tools em um catálogo estático e imutável `READ_TOOLS`, sem descoberta dinâmica. Um grafo mínimo somente de teste comprova a injeção do contexto confiável pelo `ToolNode`; o grafo de produção, planner, writer, checkpointer e ledger continuam fora desta fase.

- [x] Implementar tools de consulta necessárias aos cenários.
- [x] Validar IDs, filtros e identidade antes da chamada.
- [x] Manter resposta bruta fora do prompt quando a forma normalizada for suficiente.
- [x] Testar escolha isolada, argumentos e tratamento de cada erro relevante.

**Evidência de aceite:** em 30/08/2026, os 181 testes focados de ativo, catálogo e integração, os 398 testes do agente e os 59 testes da API passaram. O `make test` totalizou 457 testes; a única mensagem adicional foi o aviso de depreciação já existente do `python_multipart`. Os testes cobrem os dez schemas públicos, uma resposta completa e uma chamada GET por tool através de `ToolNode`, rejeição antes de HTTP, finitude numérica, escopo degradado recursivo, cinco modos de resposta, cinco categorias de erro e artifacts JSON. As saídas não serializam o objeto de identidade nem os valores ocultos de pessoa usuária, seed, cliente, gabarito ou envelope HTTP; o `company_id` validado do ativo permanece como dado legítimo do domínio.

**Aceite verificado:** tools de leitura são determinísticas nas bordas e não escondem falhas da API.

## Fase 4 — tools de escrita e política

**Aprender:** autorização, confirmação, proposta versus execução e impacto reversível/irreversível.

**Decidir:** matriz de permissões, formato da justificativa, quando pedir confirmação e quando expandir idempotência.

**Decisões implementadas:**

- As cinco proposal tools são união imutável por `action`, expõem schemas públicos fechados e retornam artifact com `effect_executed=false`; não recebem runtime, cliente nem fazem HTTP. Política, identidade, empresa, ativo, caso, modelo, aprovação e URLs permanecem na fronteira confiável.
- A política determinística exige pedido explícito, permissão, justificativa válida e escopo integral (ação, alvo e parâmetros materiais; criticidade é material). Ausência ou divergência de aprovação pede confirmação; permissão ou justificativa inválida nega; somente `allow` pode preparar uma intenção.
- Confirmação usa o escopo persistido e aprovação estruturada pela fronteira. O nó determinístico, em vez de texto livre ou da proposal tool, é o único ponto que recebe as cinco operações HTTP fixas após o checkpoint.
- Reprocesso é a única ação suscetível a retry: recebe a chave persistida `tractian-agent:<uuid>`, com TTL de sete dias, e no máximo um retry com mesma chave e mesmo corpo. Especialista, criticidade, retreinamento e escalonamento não recebem chave, têm no máximo um despacho e jamais fazem retry automático.
- Para as quatro ações não idempotentes, timeout, transporte, `5xx`, resposta inválida ou queda no despacho tornam o resultado `uncertain`; `4xx` torna-o `failed`. Retomada `prepared` por outro `execution_id` termina `uncertain/0` sem rede, em vez de repetir o efeito.

**Evidência histórica da primeira fatia:** os 6 testes focados da política, os 404 testes do agente e os 59 testes da API passaram; `make test` totalizou 463 testes, com apenas o aviso de depreciação já conhecido do `python_multipart`.

**Evidência final de aceite (30/08/2026):** a suíte focada de `test_write_policy.py`, `test_write_policy_matrix.py`, `test_write_proposal_contracts.py`, `test_write_proposal_tools.py`, `test_write_operations.py`, `test_write_contracts.py`, `test_checkpoint.py`, `test_graph_entrypoint.py`, `test_reprocess_flow.py` e `test_non_idempotent_flow.py` passou com **389 testes**. `uv lock --check` resolveu 47 pacotes. `make test` passou com **59 testes da API + 1.053 do agente = 1.112 testes**, e somente o `PendingDeprecationWarning` já conhecido de `python_multipart`. A revisão independente da Task 7 foi aprovada, sem achados Critical/Important; os casos cobrem permitido, negado, ambíguo, conflito de escopo, repetição/replay e retomada conservadora.

- [x] Separar proposta de ação e execução efetiva.
- [x] Exigir pedido explícito, permissão, escopo claro e justificativa.
- [x] Pedir confirmação para ação inferida, ampliada ou ambígua.
- [x] Aplicar idempotência a cada nova ação suscetível a retry: somente reprocesso é retryable; as outras quatro ações não repetem e retomam conservadoramente.
- [x] Testar permitido, proibido, ambíguo, repetido e conflitante.

**Aceite verificado:** nenhuma escrita ocorre apenas porque o LLM a sugeriu.

## Fase 5 — estado LangGraph e SQLite

**Aprender:** grafo de estados, nó, aresta condicional, checkpointer, `thread_id`, interrupção e retomada.

**Decidir:** schema do estado, política de retenção, relação entre `request_id`, `thread_id` e execução e primeira fronteira de entrada do agente (função, CLI ou endpoint HTTP).

**Decisões implementadas na Fase 5 (registro histórico anterior à integração do
planner na Fase 6):**

- `thread_id` identifica a linha persistida e pode receber novos `request_id`; toda invocação ou retomada usa novo `execution_id`. Caso, empresa, pessoa usuária ou alvo confiável divergentes falham fechados. O estado tipado mantém somente valores JSON-safe e observáveis, incluindo mensagens, chamadas, evidências, proposta, decisão, intenções, passos, resultado e revisão.
- A fronteira é a função Python assíncrona `invoke_agent`, que exige runtime autenticado e `thread_id`. Runtime, cliente, credenciais, seed, golden set, resposta HTTP bruta e raciocínio não são checkpointados.
- O desenvolvimento usa `AsyncSqliteSaver` em `.run/agent-checkpoints.sqlite3`, serializer restrito e ciclo de vida fechado. Não há expiração ou remoção automática de threads; `adelete_thread(thread_id)` é explícito. Isso é distinto do SQLite idempotente da API e do TTL de sete dias da chave de reprocesso. PostgreSQL continua futuro.
- Criação e retomada que podem escrever usam `durability="sync"`; `prepare_intent` e `execute_action` ocupam supersteps distintos. `interrupt()` estruturado e `Command` pelo ID retomam uma confirmação sem efeito anterior.
- Ao encerrar a Fase 5, o grafo era determinístico e sem LLM: leitura percorria
  `ingest → route → finish`, e escritas passavam por política, confirmação quando
  aplicável, preparação persistida e execução. A Fase 6 integrou depois o planner
  opt-in sem retirar as fronteiras determinísticas de escrita. Locks por
  `thread_id` continuam locais ao processo/event loop; não há lease distribuído.

**Evidência final de aceite (30/08/2026):** a suíte focada de estado, checkpoint, entrada, reprocesso e ações não idempotentes passou com **389 testes**; `uv lock --check` resolveu 47 pacotes; `make test` passou com **59 testes da API + 1.053 do agente = 1.112 testes**, mantendo somente o `PendingDeprecationWarning` conhecido. A Task 6 provou SQLite temporário real, fechamento/reabertura do saver, `prepared` antes do HTTP, replay de recibo e ausência de segundo efeito; a Task 7, revisada sem achados Critical/Important, provou que a retomada de uma ação não idempotente por novo `execution_id` não toca a rede.

- [x] Definir estado tipado com solicitação, identidade, mensagens, chamadas, evidências, decisão, passos, chaves idempotentes e revisão.
- [x] Montar um grafo mínimo sem LLM para provar as transições.
- [x] Configurar checkpointer SQLite.
- [x] Testar persistência e retomada após reinício.

**Aceite verificado naquela fase:** uma execução interrompida retomava seu estado
persistido sem repetir uma ação confirmada. Planner ainda não existia nesse
marco e foi integrado na Fase 6; writer, ledger completo, gate de liberação,
Logfire e avaliação continuam ausentes.

## Fase 6 — provider e planner

**Aprender:** adapter, structured output, prompt de sistema, seleção de tools e orçamento de contexto.

**Decidir:** modelo Groq inicial, temperatura, timeout, limites e contrato comum de provider.

**Decisões implementadas:**

- `ModelProvider.create_chat_model(ModelConfig)` isola o adapter do domínio. O
  primeiro adapter é `GroqModelProvider`, com `openai/gpt-oss-120b`, temperatura
  zero, timeout de 30 segundos, máximo de 512 tokens e `max_retries=0`.
- O planner versionado usa uma seleção via `bind_tools` e uma finalização
  Pydantic separada. No adapter Groq, a finalização é traduzida para JSON Schema
  estrito nativo (`method="json_schema"`, `strict=True`), em vez do default
  `function_calling`: o smoke anterior observou `tool_use_failed` porque a tool
  sintética desse default podia não ser chamada pelos GPT-OSS. Schemas Pydantic
  são revalidados pelo adapter diretamente a partir do texto JSON com
  `model_validate_json`, preservando a semântica JSON dos Enums estritos e todos
  os validators de coerência; o contrato do planner não conhece esse detalhe do
  provider. A seleção continua usando somente `bind_tools`, em requisição
  distinta e sem retry. O grafo só oferece catálogo autorizado pelo estado e
  runtime, limita sete tools, oito seleções, uma finalização, 48 mil caracteres
  e 20 passos; tool inválida, repetida ou fora de escopo falha fechada.
- O estado nunca recebe modelo, credencial, resposta HTTP bruta, texto livre do
  seletor nem ID externo de provider. Cada `PersistedToolCall` recebe
  `call_planner_<24 hex>` derivado por SHA-256 de
  `planner-v1\0<request_id>\0<ordinal>` com ordinal one-based; repetição continua definida pelo
  fingerprint canônico de tool e argumentos.
- A proposal escolhida pelo planner continua indo para a política determinística
  antes de confirmação, checkpoint e qualquer HTTP. O provider não pode criar
  aprovação, identidade, permissão ou efeito de escrita.
- `make smoke-groq` é opt-in e fica fora de `make test`: compara os dois modelos
  Groq com dados sintéticos, sem retry e com 512 tokens de saída, emitindo apenas
  status e métricas agregadas. A finalização usa o contrato real
  `PlannerTerminalDecision` e exige `guide`, `sufficient_evidence` e
  `missing_information=null`. Uma rodada declara estabilidade `not_measured`; com
  `GROQ_SMOKE_RUNS>=2`, compara assinaturas de contratos entre rodadas. Sem
  `GROQ_API_KEY`, não constrói provider nem toca a rede, imprime
  `status=skipped reason=missing_groq_api_key` e sai com zero.

**Evidência final de aceite (01/09/2026):** os testes focados de provider,
planner, grafo e smoke passaram, incluindo dois `ModelProvider` falsos pelo
caminho público `create_chat_model → Planner → build_agent_graph → invoke_agent`.
Eles receberam IDs externos distintos e produziram os mesmos schemas e ordem
de catálogo, tool e argumentos persistidos, decisão `request_confirmation` e
resultado de política `require_confirmation/explicit_approval_required`, sem
HTTP. A suíte completa do agente passou com **1.434 testes**; os
locks offline resolveram 49 pacotes do agente e 55 da API. O smoke real
`make smoke-groq` foi **skipped**, nunca passed, pois `GROQ_API_KEY` não estava
disponível; isso não tocou a rede e deixa a compatibilidade ao vivo da conta
como verificação futura opt-in naquele momento.

**Evidência da correção de aceite (02/09/2026):** o teste RED/GREEN sem rede
provou que a interface pública de saída estruturada do modelo devolvido pelo
adapter usa `json_schema` estrito. Um segundo ciclo RED/GREEN comprovou que, se
a seleção terminou e a finalização falha, o smoke emite somente métricas seguras
com `tool=true`, `arguments=true`, `pydantic=false` e `calls=1`; falha de
construção ou da primeira chamada mantém métricas falsas e zero chamadas. Os 26
testes focados de provider e smoke passaram; os locks offline resolveram 49
pacotes do agente e 55 da API; e `make test` passou com 59 testes da API e 1.445
do agente (1.504 no total). Sem chave no ambiente automatizado, o smoke saiu
como **skipped** sem tocar a rede; a execução manual final está registrada abaixo.

**Evidência manual diagnóstica e final (02/09/2026):** com chave fornecida somente no
terminal, o smoke mostrou o 120b aprovado (`portuguese`, tool, argumentos e
Pydantic verdadeiros, duas chamadas) e o 20b com seleção aprovada, mas
finalização reprovada após uma chamada. O probe isolado do 20b identificou
`BadRequestError` HTTP 400 com código `json_validate_failed` no orçamento antigo
de 128 tokens. Com 512, terminou com `finish_reason=stop`, 171 tokens de saída,
150 tokens de raciocínio, sem erro de parser e com Pydantic válido. Isso demonstra
insuficiência do orçamento antigo, não incompatibilidade do modelo. O parser do
adapter e o smoke foram corrigidos e cobertos sem rede. Na repetição completa
pós-correção, 120b e 20b retornaram `status=passed`, português, tool, argumentos
e Pydantic verdadeiros, com duas chamadas; as latências da única rodada foram
1.733 ms e 804 ms, respectivamente. `stable=not_measured` é esperado com
`runs=1`, portanto essa execução confirma compatibilidade funcional, mas não
mede estabilidade nem fundamenta a escolha de modelo por desempenho.

- [x] Criar interface de modelo independente do provedor.
- [x] Implementar adapter inicial do Groq.
- [x] Criar prompt e saída estruturada do planner.
- [x] Expor apenas tools pertinentes ao estado atual.
- [x] Limitar passos, repetição e consumo de contexto.

**Aceite verificado:** trocar o adapter não altera estado, tools ou regras de
segurança; a prova de independência usa providers falsos, e o smoke ao vivo
pós-correção aprovou os dois modelos Groq candidatos pelo mesmo contrato.

**Auditoria de prontidão para push (02/09/2026):** os sete documentos Markdown
não têm links locais quebrados; a superfície dos 18 pares método/caminho FastAPI
coincide com a do OpenAPI após normalizar os nomes internos dos parâmetros de
path, sem chave de rota duplicada. O formulário e as respostas do PATCH também
coincidem com o Swagger dinâmico; operation IDs e nomes camelCase do contrato
estático não são tratados como equivalência textual com os nomes Python. Os 15
artefatos de dados regeneram de forma determinística, e `agent-input/cases.json`
permanece sem campos do golden set.
O runtime abre somente a allowlist operacional e o pacote público de chamados;
um teste espiona os acessos e prova que o golden set não é lido. O gabarito passou
a preservar incerteza em S-420/M-605, separar investigação de escrita e tratar as
cinco ações do simulador como recibos sem mutação dos fixtures. Datas de análise e
invalidação de baseline, harmônicos do M-205, isolamento por empresa e schema
estrito do PATCH também têm regressões. Ruff e os dois locks passaram; `make test`
confirmou 99 testes da API e 1.445 do agente (1.544 no total), mantendo somente o
warning conhecido de `python_multipart`.

## Fase 7 — ledger de evidências

**Aprender:** proveniência, evidência suficiente, conflito e diferença entre dado, inferência e conclusão.

**Decidir:** schema de `EvidenceItem`, regras mínimas de suficiência e representação de conflitos.

- [x] Preencher o ledger em código com fonte, valor, instante, limitações e referência ao retorno.
- [x] Impedir que texto livre do LLM seja registrado como fato da API.
- [x] Detectar evidência ausente, parcial, vencida ou conflitante.
- [x] Testar fundamentação e suficiência por cenário representativo.

**Aceite verificado (02/09/2026):** o `AgentState` persiste um `ledger`
tipado da solicitação atual e mantém `ledger_history` por `request_id`. O nó de
tool só o atualiza após `validate_planner_read_observation`; recibos de escrita
aceitos viram fatos vinculados exclusivamente à intenção terminal, enquanto
falhas e resultado incerto viram lacunas sanitizadas. A validação do estado
recusa adulteração de request, call, recurso e valor. A reabertura do SQLite
preserva o ledger e a suíte focada passou com 598 testes; a suíte completa do
agente passou com 1.463 testes e `uv lock --check --offline` resolveu 49
pacotes. Writer e gate continuam pendentes na Fase 8.

## Fase 8 — writer e gate de segurança

**Aprender:** separação de responsabilidades, contexto mínimo e validação pós-geração.

**Decidir:** schema da resposta, regras críticas e no máximo uma tentativa de reparo de formato.

- [x] Criar prompt separado para o writer.
- [x] Enviar ao writer somente decisão e evidências necessárias.
- [x] Validar formato, afirmações críticas, permissões e incerteza em código.
- [x] Pedir informação ou revisão humana quando a resposta não puder ser liberada com segurança.

**Aceite:** o writer não consegue criar nova decisão, tool call ou fato não registrado.

**Aceite verificado (02/09/2026):** o `Writer` usa o prompt versionado
`writer-v1`, somente `with_structured_output` e uma projeção positiva com
orçamento total de 64 referências. O modelo recebe apenas decisão, IDs e categorias fechadas de
fato/limitação, além da informação ausente já decidida; resource, target,
`source_at`, valor e `fact_path` permanecem no ledger. Seu draft strict não
contém prosa, valores ou tool calls. Uma falha de formato permite uma única
tentativa adicional em superstep persistível; erro de provider não ganha retry
e sua exceção sanitizada não retém saída bruta em causa, contexto ou traceback.
O `ReleaseGate` puro deriva ACT/ESCALATE e o alvo do escopo confiável persistido,
da intenção, proposal e terminal do planner atuais; recompõe cada ID, valida
coerência mode/quality/obsolescence, reconstrói conflitos e recompila
integralmente a evidência de ação contra intent/aprovação/recibo. Fatos de tool
citados exigem permissão `read`, inclusive em respostas de ação; recibo isolado
usa somente a permissão da ação. Intenção negada, falha, incerta ou não terminal nunca pode ser
escondida por uma decisão GUIDE. O renderer resolve todo texto técnico somente
no ledger. Draft, atestado e resposta são recompostos ao restaurar o estado, de
modo que adulteração do checkpoint falha fechada mesmo com digests recalculados.

**Decisão local de segurança:** enquanto não existir claim-scope tipado, o gate
trata qualquer lacuna, item parcial ou obsoleto e qualquer conflito da request
atual como bloqueante para uma resposta técnica. Isso pode pedir revisão em
casos conservadores, mas impede que o writer esconda uma lacuna fora da seleção.
IDs de limitações derivam de tipo, request, todas as fontes/referências, razão e
detalhe canônicos; IDs de conflito são únicos, ordenados e deduplicados antes do
hash. Lacunas também precisam pertencer à request atual. O planner passa a reservar writer, um possível repair e gate
dentro do teto de 24 passos; o fallback sem planner conserva os budgets exatos
3/5 e a resposta determinística legada. O contador, o resultado, a âncora e o
próximo nó do writer são validados juntos, de modo que nenhum checkpoint permita
uma terceira chamada ao modelo.

**Decisão local de segurança — Ruling 9:** a assinatura `write-scope-v1`
vincula estruturalmente ação, alvo canônico, parâmetros materiais e justificativa
à intenção. O contexto confiável é persistido no mesmo superstep que aceita a
primeira proposal e deve coincidir com o runtime atual antes de política,
confirmação, preparação, efeito e replay. Esses hashes detectam divergências e
corrupção acidental; não são MACs e não autenticam o checkpoint contra alguém
capaz de reescrever coordenadamente o banco e recalcular todos os hashes. Não foi
inventada chave criptográfica: o armazenamento do checkpoint continua sendo uma
fronteira confiável que precisa de controle de acesso operacional.

**Decisão local de segurança — round 4:** qualquer resultado técnico de ação
concluída, tanto no terminal legado `EXECUTE_ACTION` quanto em `RELEASE_GATE`,
exige uma aprovação cuja ação, alvo, parâmetros materiais e origem coincidam
com proposal, intenção e contexto confiável. A origem autorizadora é persistida
na intenção para não ser autoatestada pela própria aprovação. Nas quatro ações
não idempotentes, uma intenção `PREPARED` por outro `execution_id` termina
primeiro em `uncertain/0`, inclusive se ativo, caso ou modelo atuais divergirem.
A exceção da fronteira vale somente para a aresta pendente
`prepare_intent → execute_action`, sem nova proposal, aprovação ou confirmação;
execução normal, confirmação, mesma execução e replay terminal continuam
revalidando todo o escopo.

**Decisão local de segurança — round 5:** no grafo com planner, somente o
terminal criado pelo código exato
`NON_IDEMPOTENT_OUTCOME_UNKNOWN_AFTER_RESUME` preserva a resposta determinística
de `execute_action` e segue diretamente para `END`. Esse caminho não consulta
writer, repair ou gate e não contém afirmação técnica. Falhas, incertezas com
qualquer outro código e resultados normais continuam obrigatoriamente em
`writer → release_gate`.

**Decisão local de segurança — escalação da Task 16:** o terminal conservador
é reconhecido primeiro pela estrutura independente do código — ação não
idempotente, intenção `UNCERTAIN` preparada por outra execução, zero tentativas,
nenhum recibo e âncora `EXECUTE_ACTION`. Um único contrato puro compartilhado
pelo estado e pela aresta do grafo exige então o erro local completo e o
`FinalResult` sanitizado exato, sem referências, próximo passo, status HTTP ou
afirmação técnica. O fast replay rejeita qualquer divergência antes de invocar
o grafo, inclusive ausência do resultado final. Contadores e o discriminador
booleano do erro são estritos no wire, de modo que strings, inteiros e booleanos
não sejam normalizados antes da igualdade canônica. Essa validação continua
sendo integridade estrutural dentro da fronteira confiável do checkpoint; ela
não acrescenta MAC nem autenticação criptográfica.

As cinco ações passaram por interrupção após o efeito, fechamento/reabertura do
SQLite, retomada somente em writer/gate e replay sem novo HTTP ou modelo. A
confirmação estruturada também terminou no gate sem repetir efeito. As suítes
focadas da Fase 8 cobrem writer, gate, estado, contratos, planner e fronteira;
as regressões dos fluxos de escrita somam 235 testes. A suíte completa do agente
tem 1.624 testes e o `make test` confirmou 99 testes da API + 1.624 do agente =
1.723 testes. `uv lock --check --offline` resolveu 49
pacotes; permaneceu apenas o warning conhecido de `python_multipart`.

## Fase 9 — revisão humana

**Aprender:** human-in-the-loop, fila de revisão, estado suspenso e trilha de auditoria.

**Decidir:** autenticação provisória, operações de aprovar/editar/rejeitar e expiração.

- [x] Interromper o grafo com motivo, ponto de dúvida e evidências disponíveis.
- [x] Permitir retomada após decisão do revisor.
- [x] Registrar autor, horário e alteração do humano.
- [x] Testar aprovação, edição, rejeição e retomada.

**Aceite:** caso ambíguo não chega ao cliente como certeza e pode ser retomado sem recomeçar.

**Decisão local de segurança — Ruling 10:** a única razão que uma aprovação
humana pode limpar é `human_disposition_required`: uma decisão `guide` cujo
draft, proveniência, suficiência, permissões e estado de ação já passaram por
todas as regras determinísticas, mas cujo próximo passo fechado pede disposição
humana. A aprovação cria um `ReviewedDraft` separado e troca somente esse
próximo passo por `monitor`; ela não altera a decisão do planner. Pedido
explícito `require_human_review`, falha de integridade, lacuna, conflito,
obsolescência, overflow, permissão incompatível, informação ausente inválida,
aprovação de ação divergente e intenção ausente, incerta ou não concluída são
bloqueios duros. Uma edição pode selecionar/reordenar IDs elegíveis e escolher
um próximo passo enumerado, inclusive reconstruir o draft após duas falhas do
writer, mas sempre volta ao mesmo gate; se continuar bloqueada, encerra com
aviso seguro e não abre uma segunda revisão.

**Evidência de aceite (03/09/2026):** `make test` passou com 99 testes da API e
1.689 do agente; os 235 fluxos de escrita passaram separadamente, assim como
Ruff e `uv lock --check --offline`. A retomada com SQLite reaberto preservou
planner, writer, tools e ações já concluídos sem nova chamada. Testes adversariais
adicionais cobrem perda de `read` em todas as operações e replay, terminais
canônicos, base/auditoria adulteradas, drift de permissão, edição sem fatos para
pedido de informação, coerções, orçamento, concorrência e isolamento entre
threads. A allowlist fechada oferece edição apenas para disposição humana,
falha do writer, seleção de evidência ou próximo passo; motivos duros aceitam
somente rejeição. Expiração tem precedência sobre a operação e é fechada antes
de uma nova solicitação, enquanto `ACT`/`ESCALATE` revisado preserva o recibo
`accepted` da intenção atual. A autorização read-only de thread, tenant, caso,
alvo central, modelo configurado, request e execução ocorre antes de qualquer
escrita no checkpoint, inclusive quando a request omite o ativo. O gate-base
original nunca é removido durante drift de permissão: um marcador estrito e
derivado vincula permissões de base/atuais, decisão e a fase exata anterior ao
julgamento ou posterior à auditoria, até o segundo gate.

## Fase 10 — Logfire

**Aprender:** trace, span, atributo, correlação e sanitização.

**Decidir:** atributos permitidos, pseudonimização, retenção e limites do plano.

- [x] Instrumentar requisição, nós, LLMs, tools, decisão e resultado.
- [x] Correlacionar `request_id`, `trace_id`, `experiment_id` e `case_id` quando aplicável.
- [x] Remover tokens, credenciais e payloads sensíveis.
- [x] Testar a sanitização antes de enviar dados.

**Aceite:** um funcionário autorizado localiza a execução pelo ID sem expor segredo ou golden set.

**Aceite verificado (03/09/2026):** o runtime usa uma fachada nula por padrão,
recorder in-memory e adapter Logfire carregado somente após o opt-in triplo. Em
subprocessos limpos, importar contratos, estado e entrypoint não carregou módulo
`logfire*` nem o plugin Pydantic, inclusive sob configuração automática
adversarial. O exporter real em memória foi percorrido recursivamente e não
conteve sentinelas de token, autorização, mensagem, artifact, evidência,
provider ou golden set; referências opcionais ausentes não viraram a string
`null`. Request, nós, planner, writer, tool, policy, POST/PATCH de ação, gate,
revisão e resposta compartilham o `trace_id` técnico da execução, enquanto IDs
de domínio são HMAC por tipo. A resposta registra somente decisão fechada e
resultado operacional; o trace não entra no `AgentState`, resultado ou SQLite.
O runtime continua emitindo zero spans de avaliação.

A equivalência Null/Recording cobriu fallback, consulta completa, os cinco
fluxos de escrita, preflight, retry, recovery, repair, revisão com SQLite
reaberto, replay, concorrência, cancelamento e erros. A auditoria final adicionou
RED/GREEN para valores de configuração que falham durante validação e para uma
fachada injetada que adultera seus próprios metadados; ambos agora degradam sem
alterar o negócio. Passaram 71 testes focados integrados, 235 testes de escrita e
`make test` com 99 testes da API + 1.759 do agente = 1.858. Ruff passou, e os
locks offline resolveram 64 pacotes do agente e 55 da API. Permaneceu somente o
warning conhecido de `python_multipart`. Retenção de 30 dias e limites de volume
continuam controles externos da conta, não propriedades do ledger ou do código.

## Fase 11 — runner e checks programáticos

**Aprender:** Pydantic Evals, isolamento de benchmark, repetição e reprodutibilidade.

**Decidir:** divisão desenvolvimento/calibração/oculto, número de repetições e versionamento de casos, prompts e modelos.

- [x] Executar casos sem disponibilizar o gabarito ao runtime.
- [x] Validar resposta, tools, argumentos, permissões, erros e regras críticas.
- [x] Comparar trajetória observada com a esperada.
- [x] Gerar relatório por caso, dimensão e experimento.

**Aceite:** o mesmo experimento pode ser reproduzido com versões e configuração registradas.

**Aceite verificado (03/09/2026):** `tractian-eval-v1` executou as 17 entradas
públicas duas vezes pela fronteira real do agente antes de abrir o gabarito. A
configuração, os hashes dos dois datasets, a revisão do Git, prompts, rubricas,
modelos e pacotes ficaram registrados no manifesto. O relatório Pydantic Evals
contém 34 runs e as dez dimensões por caso; o perfil determinístico sem rede
também é executável por `make eval`.

## Fase 12 — juízes offline

**Aprender:** rubrica, juiz cego, juiz de trajetória, viés e variância de LLM.

**Decidir:** modelo de cada juiz, rubricas, exemplos rotulados e regras críticas de reprovação.

- [x] Implementar juiz cego do resultado sem acesso ao trace.
- [x] Implementar juiz de trajetória com acesso às chamadas e falhas.
- [x] Separar clareza/tom das dimensões industriais críticas.
- [x] Retornar score, aprovação e motivo estruturados.
- [x] Repetir casos para detectar acerto por acaso e instabilidade.

**Aceite:** feedback de avaliação nunca retorna ao agente durante o atendimento.

**Aceite verificado (03/09/2026):** os juízes receberam somente artefatos de
uma execução encerrada. O cego omite steps/call IDs; o de trajetória recebe
steps e falhas sanitizados. Ambos retornam `pass`, `score` e `reason` por
dimensão, e os cortes 0,7/0,8/0,9 são aplicados sem nova chamada. Um teste prova
que uma resposta correta por acaso pode passar no resultado e falhar na
trajetória. No experimento real, 68 chamadas julgaram 34 runs e os 17 casos
foram instáveis entre repetições; nenhuma nota entrou no grafo ou acionou retry.

## Fase 13 — calibração humana

**Aprender:** golden set, concordância, Cohen's kappa, falsos positivos/negativos e escolha de limiar.

- [ ] Rotular 20–30 respostas sem ver a nota do juiz.
- [ ] Medir concordância bruta, kappa, falso aprovado e falso reprovado.
- [ ] Comparar limiares `0.7`, `0.8` e `0.9` com as mesmas execuções.
- [ ] Refinar rubricas, não o gabarito, quando a discordância revelar ambiguidade.
- [x] Registrar a limitação de existir um único avaliador humano.

**Aceite:** o limiar escolhido é justificado pelos erros observados, não por preferência.

**Status em 03/09/2026 — skipped:** o autor não é especialista industrial da
TRACTIAN e não produzirá rótulos por suposição. O lote cego de 24 respostas e o
template sem scores estão prontos, assim como cálculo de concordância bruta,
Cohen's kappa, falsos aprovados/reprovados e taxa de revisão nos três limiares.
As quatro tarefas de rotulagem/calibração permanecem abertas até uma pessoa da
TRACTIAN avaliar o lote sem consultar antes as notas dos juízes. Nenhum limiar
foi escolhido e o golden set não foi alterado.

## Fase 14 — comparação Groq × NVIDIA NIM

**Aprender:** modelo versus provedor, API compatível, hosted versus self-hosted, latência e custo.

**Decidir:** modelos disponíveis no momento do experimento e orçamento de repetições.

- [x] Implementar adapter NVIDIA NIM sem alterar a lógica central.
- [x] Comparar português, tool calling, saída estruturada, estabilidade, contexto, latência e custo.
- [x] Avaliar planner e writer separadamente.

**Aceite:** a recomendação se baseia no benchmark versionado e nas condições reais do teste.

**Aceite verificado (03/09/2026):** os dois providers executaram duas vezes os
mesmos probes, modelo `openai/gpt-oss-20b`, contexto de 8.000 caracteres e
contratos reais. Groq passou planner e writer com estabilidade; NVIDIA NIM
passou writer, mas falhou saída estruturada/estabilidade do planner. As
latências totais foram 1.735/1.039 ms na Groq e 6.361/21.335 ms na NVIDIA para
planner/writer. Groq foi recomendado nessas condições. Tokens foram registrados;
custo ficou indisponível, sem estimativa inventada, pois a configuração não
congela uma tarifa confiável.

## Fase 15 — experimento e entrega

- [x] Congelar versões dos dados, prompts, código, modelos e rubricas.
- [x] Rodar experimento final e documentar resultados e limitações.
- [x] Adicionar comandos reais de agente e avaliação ao `Makefile` e README.
- [x] Adotar `promptfoo` somente se houver comparação recorrente que ele simplifique.
- [x] Adotar `Ragas` somente se existir um pipeline RAG para medir.
- [x] Revisar segurança, reprodutibilidade e isolamento do golden set.

**Aceite:** outra pessoa consegue reproduzir o experimento seguindo apenas o repositório.

**Decisões:** `promptfoo` não foi adotado porque a comparação atual já é
versionada e não é uma rotina recorrente. `Ragas` não foi adotado porque não há
RAG. LangSmith e Phoenix também não entram nesta fase: duplicariam Pydantic
Evals + Logfire e ampliariam a superfície de traces; podem ser reconsiderados
somente diante de uma necessidade operacional concreta. A comparação
programático versus programático + juízes usa exatamente os mesmos runs. No
resultado atual, ambos rejeitaram todos os 34 e o ganho incremental foi zero;
isso não autoriza escolher um limiar sem a calibração humana adiada acima.

**Aceite verificado (03/09/2026):** `make eval` executou o perfil local com 17
casos × 2 e produziu manifesto, relatório e lote cego. O experimento real
produziu os 34 runs, os dois juízes avaliaram cada run e o benchmark de
providers comparou os dois papéis separadamente. A auditoria confirmou que
somente o pacote `evaluation/` importa gabarito, juízes ou rubricas; o runtime
não recebe esses objetos. `ruff check src tests`, `git diff --check`, os locks
offline (80 pacotes no agente e 55 na API) e os 44 testes focados passaram.
`make test` passou com 99 testes da API e 1.803 do agente, 1.902 no total,
mantendo apenas o warning conhecido de `python_multipart`. A calibração humana
continua skipped e explicitamente fora deste aceite automático.

## Registro histórico do plano SDD — Fases 5 e 4

Esta seção preserva o plano executado e suas evidências. Afirmações sobre o
grafo sem LLM descrevem o estado anterior à Fase 6 e não substituem o estado
atual registrado na seção da Fase 6 acima.

**Especificação vinculante:** `AGENTS.md`, as Fases 4 e 5 deste arquivo, as etapas 6, 8 e 10 do `LEARNING-GUIDE.md` e a decisão aprovada de persistir intenções somente no estado/checkpointer do LangGraph.

**Ordem deliberada:** `5A → 4A/reprocesso → aceite da Fase 5 → restante da Fase 4`. A escrita real depende de um checkpoint durável anterior ao HTTP, enquanto o aceite de retomada da Fase 5 precisa de uma escrita real para provar que não há duplicação.

### Restrições globais

- Não criar SQLite, tabela ou repositório paralelo para intenções do agente. A intenção vive no estado persistido pelo checkpointer; a API mantém seu SQLite idempotente independente.
- Usar SQLite como checkpointer de desenvolvimento e manter PostgreSQL como evolução futura.
- Usar `durability="sync"` na entrada e na retomada de qualquer fluxo que possa escrever. `prepare_intent` e `execute_action` ocupam nós e supersteps distintos.
- Somente `POST /analyses/{id}/reprocess` admite retry automático, sempre com a mesma chave e o mesmo corpo. As outras quatro ações fazem no máximo uma tentativa automática e ficam `uncertain` se o resultado puder ter sido aplicado.
- Identidade, empresa, permissões, aprovação, `thread_id`, `request_id`, `execution_id`, cliente e chave idempotente pertencem ao contexto/estado confiável; nunca são argumentos públicos do modelo.
- Aprovação vincula ação, alvo e parâmetros materiais. Justificativa estruturalmente válida não deve ser tratada como evidência suficiente.
- Validar escopo antes do HTTP: ativo central, análise pertencente ao ativo, modelo configurado e caso atual. A API simulada não substitui essa proteção.
- Propostas LangChain não produzem efeito. Somente o nó determinístico de execução, depois de `allow` e do checkpoint, recebe acesso à operação HTTP.
- Não executar retry dentro de cliente, proposal tool ou operação. Preservar todo `ApiError`; `IN_PROGRESS`, `OUTCOME_UNKNOWN` e conflito de payload bloqueiam nova execução.
- O checkpoint guarda somente valores observáveis e serializáveis. Não recebe cliente, transporte, resposta HTTP bruta, credencial, token, seed, golden set, trace de raciocínio ou arquivos restritos de avaliação.
- Naquela entrega, o grafo era determinístico e sem LLM. Planner ficou fora do
  escopo e foi integrado posteriormente na Fase 6; writer, ledger completo,
  Logfire e runner de avaliação continuam fora do estado atual.
- O MVP declara processo local único por `thread_id`; não promete lease distribuído com SQLite.
- Cada tarefa segue TDD em fatias verticais, executa testes focados e `make test`, recebe commit próprio e passa por revisão independente de especificação e qualidade.

### Task 1 — consolidar a política inicial existente

**Objetivo:** incorporar e verificar a primeira fatia já presente no worktree, criando uma base revisada para as tarefas seguintes.

**Arquivos:** `agent/src/tractian_agent/write_policy.py`, `agent/tests/test_write_policy.py` e esta seção de `TASKS.md`.

**Contrato e testes:**

- Manter a política pura de reprocesso com `allow`, `require_confirmation` e `deny` e códigos estáveis.
- Confirmar `action_low`, justificativa de 20 caracteres após `strip`, aprovação ausente, alvo divergente e rejeição de campos extras.
- Não adicionar HTTP, idempotência, grafo ou generalização nesta tarefa.
- Executar o teste focado e `make test`; fazer self-review e commit.

### Task 2 — contratos tipados do estado e das intenções

**Objetivo:** definir valores observáveis e serializáveis que o LangGraph poderá persistir, sem ainda construir o grafo.

**Decisões implementadas nesta fatia:** o estado vincula cada `thread_id` a caso, empresa e pessoa usuária e exige novo `execution_id` em cada continuação; um mesmo thread aceita novos `request_id`. Mensagens, chamadas, observações, evidências, intenções, resultado e revisão usam contratos JSON-safe e imutáveis, sem objetos do runtime. O orçamento é positivo e validado antes de cada avanço. A intenção inicial cobre somente reprocesso; a união das cinco propostas permanece na Task 4.

**Arquivos:** criar `agent/src/tractian_agent/state.py`, `agent/src/tractian_agent/write_contracts.py` e testes focados correspondentes; alterar contratos compartilhados somente quando necessário.

**Contrato e testes:**

- Estado tipado contém solicitação, identidade confiável, `request_id`, `thread_id`, `execution_id`, mensagens, chamadas/observações, evidências inicialmente vazias e tipadas, decisão, contador/limite de passos, proposta pendente, aprovação, intenções, resultado final e revisão.
- `thread_id` identifica a linha persistente de um caso; um thread pode receber vários `request_id`; cada invocação/retomada usa novo `execution_id`. Reuso do thread para outro caso, empresa ou pessoa falha fechado.
- Intenção registra ID, escopo imutável, hash canônico, decisão, status, chave/expiração opcional, execução que a preparou, tentativas explícitas e recibo/erro tipado.
- Status permitidos: `proposed`, `awaiting_confirmation`, `prepared`, `completed`, `denied`, `failed` e `uncertain`.
- Modelos rejeitam campos extras e mutação; serialização não contém objetos ou nomes restritos.
- O limite de passos é positivo e impede progresso além do orçamento.

### Task 3 — checkpointer SQLite, grafo mínimo e fronteira Python

**Objetivo:** concluir a infraestrutura 5A sem LLM e provar persistência/reabertura do estado.

**Decisões implementadas nesta fatia:** o desenvolvimento usa
`AsyncSqliteSaver.from_conn_string` em contexto assíncrono, no caminho padrão
da raiz do projeto `.run/agent-checkpoints.sqlite3`, com serializer sem fallback
de pickle nem módulos JSON/MsgPack arbitrários. A fronteira Python exige
`ReadToolRuntime` e `thread_id`; o grafo declara esse runtime como
`context_schema`, recebe o mesmo objeto em `context` e nunca persiste cliente,
seed ou contexto. O trecho `aget_state → decisão/update → ainvoke` é serializado
por lock local efêmero de cada `thread_id`, pertencente ao owner ativo do
checkpointer. Todo wrapper deriva obrigatoriamente o owner do checkpointer do
grafo compilado, sem aceitar owner injetado, e wrappers do mesmo saver
compartilham o pool. Uma segunda abertura do mesmo namespace no processo é
recusada, e o registro é liberado no fechamento. Threads distintos não se
bloqueiam. A garantia pressupõe um único event loop por owner e não promete
lease multiprocesso.
Criação e retomada executável usam `durability="sync"`. Nova `request_id` zera o
progresso e aplica novo `step_limit`; a mesma request parcial retoma somente os
nós pendentes e preserva orçamento. Antes de toda retomada, o único
`snapshot.next` seleciona explicitamente seu predecessor
(`ingest←START`, `route←ingest`, `finish←route`); forma ausente, múltipla ou
desconhecida falha com erro de protocolo, sem inferência do histórico. Orçamento
insuficiente também produz erro de protocolo. A mesma request já terminal é
replay imutável de entrega, sem novo `ainvoke` nem alteração do checkpoint. O
grafo acíclico
`ingest → route → finish` consome três passos e somente produz um resultado
determinístico explícito, sem LLM, tool produtiva ou efeito. Threads não expiram
nem são removidos automaticamente nesta fatia; a remoção disponível é
`adelete_thread(thread_id)`. Intenções preexistentes são apenas preservadas,
inclusive `expires_at` e chave, sem geração ou reuso.

**Evidência desta fatia:** os 30 testes de checkpoint, grafo e fronteira, os
331 testes focados com os contratos herdados e o `make test` completo passaram;
a execução global totalizou 59 testes da API e 736 do agente, com
somente o aviso de depreciação já conhecido do `python_multipart`.

**Arquivos:** adicionar `langgraph-checkpoint-sqlite>=3.1.1,<3.2`; criar `agent/src/tractian_agent/checkpoint.py`, `agent/src/tractian_agent/graph.py`, `agent/src/tractian_agent/entrypoint.py` e testes focados; atualizar lock.

**Contrato e testes:**

- Usar `AsyncSqliteSaver` com contexto assíncrono, conexão sempre fechada e serializer restrito a tipos seguros.
- O caminho padrão de desenvolvimento é `.run/agent-checkpoints.sqlite3`, ancorado na raiz do projeto; testes usam arquivo temporário.
- A primeira fronteira é função Python assíncrona; ela exige `thread_id`, contexto autenticado e sempre invoca/retoma com `durability="sync"`.
- Grafo mínimo determinístico prova `ingest → route → finish`, encerra um caso simples de leitura e respeita limite de passos; não finge possuir planner ou writer.
- Estado sobrevive ao fechamento e reabertura do saver; threads distintos não compartilham estado; contexto não é serializado.
- Retenção do MVP: sem remoção automática, com exclusão explícita do thread; uma chave de reprocesso expira em sete dias e não é reutilizada silenciosamente depois disso.

### Task 4 — matriz completa e cinco proposal tools

**Objetivo:** generalizar a política fechada e expor ao modelo apenas propostas sem efeito.

**Decisões implementadas nesta fatia:** as cinco propostas formam uma união
imutável discriminada por `action`, sem quebrar a construção existente de
`ReprocessProposal`. A política resolve o escopo canônico pela ação, alvo e
parâmetros materiais; somente a criticidade é material nesta entrega. Ativo
central, caso atual e modelo configurado ficam em `TrustedWriteContext`, fora
dos schemas públicos. A ordem fechada é permissão, justificativa, presença de
aprovação e igualdade integral do escopo. `AgentState.pending_proposal`
persiste e restaura as cinco variantes. Nesta fatia, `WriteIntent` e
`ReprocessIntentScope` ainda cobriam somente reprocesso; a Task 7 generalizou
`WriteIntent.scope` para uma união discriminada das cinco ações, preservando o
wire e o nome de `ReprocessIntentScope`.

As cinco tools LangChain retornam conteúdo de proposta e artifact próprio com
`effect_executed=false`, sem runtime, cliente ou HTTP. O catálogo
`WRITE_PROPOSAL_TOOLS` é uma tupla ordenada e única; seus schemas públicos
rejeitam campos extras e expõem somente os argumentos definidos nesta task.

**Evidência desta fatia:** 46 casos de teste foram acrescentados. Os 384 testes
focados de política, contratos, estado, checkpoint, grafo e proposal tools
passaram. O `make test` completo passou com 59 testes da API e 782 do agente
(841 no total), além somente do aviso de depreciação já conhecido do
`python_multipart`. `uv lock --check` também passou para o projeto do agente.

**Arquivos:** evoluir `write_policy.py` e `write_contracts.py`; criar `agent/src/tractian_agent/tools/writes.py`; atualizar catálogos/exportações e testes.

**Contrato e testes:**

- Propostas: reprocessar análise, solicitar especialista, atualizar somente criticidade, solicitar retreinamento do modelo configurado e escalar o caso atual.
- Permissões: `action_low`, `action_low`, `action_high`, `action_high` e `escalate`, respectivamente.
- Aprovação compara ação, alvo e parâmetros materiais; divergência pede confirmação, ausência de permissão ou justificativa inválida nega.
- Schemas públicos não aceitam identidade, permissão, aprovação, chave, URL, método, headers, IDs ocultos ou campos arbitrários.
- As cinco tools apenas devolvem proposta/conteúdo e artifact tipados; nenhuma recebe cliente nem chama HTTP.
- Publicar catálogo estático e imutável `WRITE_PROPOSAL_TOOLS`.

### Task 5 — runtime confiável e cinco operações HTTP fixas

**Objetivo:** implementar efeitos isolados, ainda sem conectá-los automaticamente ao grafo.

**Decisões implementadas nesta fatia:** `WriteToolRuntime` estende o runtime de
leitura e exige `current_case_id`, sem tornar o caso obrigatório para as dez
tools existentes. O módulo de operações publica somente cinco funções Python;
método, caminho e bodies Pydantic fechados permanecem privados e fixos. O
preflight de reprocesso e especialista reutiliza `execute_get_analysis`: somente
resposta `complete`, validada e vinculada ao ativo central libera o write. Erros
da API são preservados; resposta degradada, divergência de escopo ou falta de
prova produz `ANALYSIS_SCOPE_UNCONFIRMED`, e chave persistida inválida produz
`INVALID_IDEMPOTENCY_KEY` antes de qualquer HTTP. Não há loop, backoff ou retry;
somente reprocesso recebe e propaga `Idempotency-Key`.

**Evidência desta fatia:** os 20 testes novos de runtime e operações passaram;
os 158 testes focados relacionados também passaram. O `make test` completo
passou com 59 testes da API e 810 do agente (869 no total), além somente do
aviso de depreciação já conhecido do `python_multipart`. `uv lock --check`
também passou para o projeto do agente.

**Arquivos:** evoluir `tools/runtime.py`; criar `agent/src/tractian_agent/write_operations.py` e testes focados.

**Contrato e testes:**

- Runtime confiável fornece identidade, permissões, ativo central, caso atual, modelo configurado e cliente; nenhum desses valores aparece em schema público.
- Cada operação fixa método, path e body em código e devolve `ActionReceipt` ou `ApiError` sem esconder falha.
- Reprocesso e especialista verificam por leitura completa que a análise pertence ao ativo central antes do POST; resposta degradada não autoriza escrita.
- Criticidade só aceita `low`, `medium`, `high` ou `critical` e somente o ativo central; retreinamento usa apenas o modelo configurado; escalonamento usa apenas o caso atual.
- Apenas reprocesso aceita chave persistida e é retryable. As demais operações não enviam chave e nunca fazem retry.
- Testar sucesso, cinco categorias de `ApiError`, resposta inválida e rejeição de escopo antes do HTTP de escrita.

### Task 6 — reprocesso vertical com intenção persistida

**Objetivo:** fechar a dependência circular com o primeiro fluxo de escrita seguro e retomável.

**Decisões implementadas:** o fluxo determinístico percorre `proposal → policy
→ confirmation/deny → prepare_intent → checkpoint sync → execute_action`.
Após `allow`, a chave `tractian-agent:<uuid>` nasce uma vez, é vinculada ao
escopo e hash canônico e expira em sete dias; ela é checkpointada antes do
HTTP. A confirmação interrompe sem efeito anterior e retorna por `Command`
estruturado. Timeout, transporte, `5xx` e resposta inválida podem receber no
máximo um retry com a mesma chave e corpo; `4xx`, conflito, `IN_PROGRESS`,
`OUTCOME_UNKNOWN` e chave vencida não criam chave ou retry novo. `completed`
é replay imutável, sem novo HTTP. A precisão de `attempts` pode subcontar um
crash pós-efeito; o lock é local ao processo/event loop.

**Evidência final:** a Task 6 foi aprovada após re-review independente. SQLite
temporário real comprovou fechamento/reabertura após `prepared`, a mesma chave
na retomada, perda de resposta após commit com um único efeito e replay do
recibo, e falha de checkpoint antes do HTTP sem chamada. A evidência histórica
mais recente da Task 6 foi 59 testes da API + 882 do agente = 941; a execução
final da Task 8 atualizou a evidência integrada para 59 + 1.053 = 1.112.

**Arquivos:** evoluir grafo, entrypoint e contratos; criar testes de integração de reprocesso com SQLite real temporário e transporte HTTP simulado.

**Contrato e testes:**

- Fluxo: `proposal → policy → confirmation/deny → prepare_intent → checkpoint sync → execute_action → checkpoint do resultado`.
- A chave `tractian-agent:<uuid>` é criada por código exatamente uma vez, somente após `allow`, e fica checkpointada antes do primeiro HTTP.
- Ausência de aprovação usa `interrupt()` sem efeito anterior; retomada recebe aprovação estruturada pela fronteira confiável e reutiliza o mesmo `thread_id`.
- Fechar o saver depois de `prepare_intent`, recriar saver/grafo e retomar reutiliza a mesma chave.
- Perda da resposta depois do commit remoto produz replay do mesmo recibo e somente uma ação real.
- Estado `completed` não chama HTTP novamente; chave vencida, conflito, `IN_PROGRESS` e `OUTCOME_UNKNOWN` não geram chave nem retry novo.
- Falha antes de confirmar o checkpoint impede o HTTP.

### Task 7 — integrar as quatro ações não idempotentes

**Objetivo:** concluir o caminho agir/escalar sem prometer garantias que a API não oferece.

**Decisões implementadas:** `WriteIntent.scope` é união discriminada por
`action`: especialista guarda `analysis_id`, criticidade guarda `asset_id` e
`criticality`, retreinamento guarda `model_id` e escalonamento usa o `case_id`.
As quatro ações persistem `prepared_execution_id`, sem chave ou expiração. A
primeira execução pode fazer um único dispatch tipado; com outro
`execution_id`, o guard ocorre antes de policy, preflight ou rede e termina em
`uncertain/0`. Na mesma execução, drift de runtime, escopo, hash ou autorização
falha sem rede. Receipt aceito conclui; receipt rejeitado ou `4xx` falha;
timeout, transporte, `5xx` ou resposta inválida ficam incertos, sem retry.
Uma nova ação exige nova intenção e nova aprovação. Especialista tem preflight
opaco; o runtime de restart precisa realmente gerar novo `execution_id`.

**Evidência final:** a Task 7 foi aprovada na revisão independente, sem
achados Critical/Important. Os testes com SQLite temporário real cobriram as
quatro ações para policy, confirmação, escopo, reinício, falhas, replay e
concorrência; a última execução da Task 7 foi 285 focados, 1.053 do agente e
59 da API (1.112 no total). A suíte final integrada da Task 8 repetiu os 389
testes focados relevantes, `uv lock --check` e `make test` com os mesmos totais
globais.

**Arquivos:** evoluir o grafo/entrypoint e testes de integração parametrizados.

**Contrato e testes:**

- Especialista, criticidade, retreinamento e escalonamento passam pela mesma política, confirmação e checkpoint de preparação.
- A intenção registra o `execution_id` que a preparou. A primeira execução pode despachar; retomada com outro `execution_id` não repete e marca `uncertain`/revisão.
- Timeout, transporte, `5xx` ou queda no ponto de despacho ficam `uncertain`; `4xx` fica `failed`; nenhum caso dispara retry automático.
- Uma nova ação exige nova intenção e nova aprovação; não reutilizar silenciosamente intenção concluída ou incerta.
- Testar permitido, proibido, ambíguo, conflito de escopo e retomada conservadora para as quatro ações.

### Task 8 — aceite, documentação e estado real

**Objetivo:** provar os critérios de aceite das duas fases e alinhar a documentação sem descrever componentes futuros como funcionais.

**Arquivos:** completar cobertura faltante da API somente se necessária; atualizar `TASKS.md` e `README.md`; atualizar `Makefile` apenas se existir runner real utilizável.

**Contrato e testes:**

- Executar todas as suites focadas e `make test` com saída registrada.
- Registrar decisões sobre IDs, retenção, fronteira Python, durabilidade, retry e resultado incerto.
- Marcar Fases 4 e 5 somente se todos os itens e aceites estiverem demonstrados.
- README deve distinguir grafo determinístico/checkpointer e proposal tools reais de planner, writer, ledger, Logfire e avaliação ainda planejados.
- Não ampliar o contrato da API com idempotência para os outros quatro endpoints nesta entrega.

## Registro histórico do plano de implementação SDD — Fase 6

Este plano executado é preservado como trilha das decisões da Fase 6. Ele
concluiu somente provider e planner; writer, resposta gerada ao cliente, ledger
completo, porta de segurança, Logfire e avaliações permanecem fora do escopo.
Cada tarefa seguiu uma fatia vertical `RED → GREEN → refactor`, recebeu revisão
independente e só avançou depois de aprovada.

### Task 9 — consolidar o provider comum e o adapter Groq

**Objetivo:** transformar a base local já iniciada em uma fronteira completa,
testável e sem credenciais no contrato comum.

**Decisões:** usar `openai/gpt-oss-120b` como modelo inicial do planner,
`temperature=0`, timeout de 30 segundos, no máximo 512 tokens de saída e
`max_retries=0`. A escolha privilegia qualidade multilíngue e tool use; o
`openai/gpt-oss-20b` permanece candidato de menor latência. A chave vem somente
de `GROQ_API_KEY`; `.env` pode fornecê-la ao processo, mas nunca é lido pelo
runtime do agente nem persistido. NVIDIA NIM continua comparação futura.

**Arquivos:** consolidar `agent/src/tractian_agent/model_provider.py`,
`agent/src/tractian_agent/groq_provider.py`, `agent/pyproject.toml`,
`agent/uv.lock` e seus testes; adicionar apenas uma amostra segura de ambiente
se ela for necessária para tornar o uso inequívoco.

**Contrato e testes:**

- `ModelConfig` é estrito, congelado e contém somente ID, temperatura, timeout e limite de saída; não contém provider, chave ou segredo.
- `model_id` é um identificador opaco sem espaços; limites rejeitam bool, string, zero, negativos, NaN, infinito e campos extras.
- `ModelProvider.create_chat_model(config)` devolve `BaseChatModel` e aceita implementações estruturais falsas.
- `GroqModelProvider` traduz o contrato para `ChatGroq`, desativa retries ocultos e oferece construção explícita a partir de `GROQ_API_KEY` com erro claro quando ausente ou vazia.
- Testes não acessam a rede e demonstram que a chave não aparece em `repr`, configuração comum, estado ou mensagens de erro.
- Executar testes focados, `uv lock --check`, self-review e commit próprio.

### Task 10 — primeira fatia vertical do planner

**Objetivo:** provar, fora do grafo de produção, o ciclo tipado
`pedido → seleção de tool → observação → decisão`.

**Decisões:** a Groq não combina tool calling e JSON Schema estrito na mesma
requisição. O planner primeiro usa `bind_tools`; se não houver tool, descarta o
texto livre e faz uma segunda requisição sem tools com saída Pydantic. A saída
terminal aceita somente `guide`, `request_information` ou
`require_human_review`, com motivo de parada enumerado e informação ausente
curta quando aplicável. `act`, `escalate` e `request_confirmation` continuam
exclusivos da política determinística após uma proposal tool.

**Decisões implementadas nesta fatia:** o prompt `planner-v1` e a classe
`Planner` permanecem isolados do grafo. A seleção usa somente `bind_tools` com
o catálogo recebido pelo chamador, valida no máximo uma chamada pelo schema
público da tool e falha fechada para nome, argumentos ou forma inválidos. A
ausência de chamada descarta o texto do seletor e inicia outra requisição no
modelo original com `with_structured_output(PlannerTerminalDecision)`. A
decisão terminal não contém resposta do writer; seu motivo é coerente com a
decisão e `missing_information` é limitado a 300 caracteres e existe somente
para `request_information`. `ToolObservation.content` guarda o JSON entregue
ao próximo turno sem expor o artifact técnico, e o wire de `JsonSnapshot` é
restaurável após fechar e reabrir o checkpointer SQLite. Nenhum `BaseChatModel`,
`AIMessage`, runtime, segredo, resposta HTTP bruta ou texto livre do seletor
entra no estado.

**Arquivos:** criar `agent/src/tractian_agent/planner.py` e testes focados;
evoluir `state.py` e seus testes somente para persistir valores observáveis e
JSON-safe necessários ao ciclo.

**Contrato e testes:**

- O prompt de sistema versionado separa planner de writer, manda usar uma tool por turno, proíbe inventar evidência e esclarece que proposal tool não executa efeito.
- Um modelo falso demonstra que seleção com `bind_tools` e finalização com `with_structured_output` acontecem em requisições distintas.
- Exatamente uma tool oferecida é aceita por turno; tool desconhecida, múltiplas chamadas ou saída terminal inválida falham fechadas antes do HTTP.
- Texto livre da chamada de seleção não vira decisão, resposta ao cliente, evidência ou raciocínio persistido.
- Chamadas, conteúdo entregue ao próximo turno e artifact validado são persistíveis; runtime, `BaseChatModel`, `AIMessage`, segredo e resposta HTTP bruta não entram no estado.
- Fechar e reabrir um SQLite temporário preserva chamada e observação sem implementar o futuro ledger.
- Construir cada comportamento em um ciclo TDD vertical e fazer commit próprio.

### Task 11 — catálogo pertinente e limites determinísticos

**Objetivo:** limitar o que o modelo pode escolher e quanto trabalho uma
solicitação pode consumir, sem classificador baseado em palavras-chave.

**Decisões:** permitir no máximo sete tool calls, oito chamadas de seleção,
uma finalização estruturada, nenhuma repetição da mesma combinação canônica
`tool + argumentos`, 48 mil caracteres de contexto e 20 passos no caminho do
planner. O orçamento em caracteres é deliberadamente independente do tokenizer
do provider; excedê-lo corta contexto antigo de modo explícito ou encerra antes
de nova chamada, nunca aumenta o limite silenciosamente.

**Decisões implementadas nesta fatia:** `select_planner_tools` reutiliza os
catálogos estáticos e filtra somente por escopo confiável, permissões, tipo do
runtime e IDs tipados presentes no pedido ou em campos estruturais específicos
das tools que realmente os produzem, sempre na mesma `request_id`; notas,
erros, snippets e texto livre incidental não concedem acesso. A chamada
selecionada é validada novamente contra esses conjuntos, o ativo central e os
pontos explícitos ou observados; pares atuais que contradizem esses alvos são
histórico inválido e falham antes do modelo. Chamadas, observações e
`PlannerUsage` ficam
vinculados à solicitação; histórico legado sem `request_id` permanece
auditável como não atribuído e não entra no contexto, fingerprint ou orçamento
atuais, enquanto uma nova solicitação zera somente os contadores transitórios.
A fronteira exige `request_id` e `PlannerUsage` e bloqueia antes de `bind_tools`
quando já há sete chamadas concluídas, oito seleções ou uma finalização; saídas
inválidas
do modelo consomem e devolvem o uso atualizado no erro de protocolo. O
fingerprint canônico usa somente tool e argumentos, sem o `call_id` do provider.
O preflight também rejeita como histórico inválido dois fingerprints canônicos
já persistidos na solicitação atual, antes de expor qualquer payload ao modelo.
Cada call atual precisa pertencer aos catálogos estáticos e ter argumentos
equivalentes ao wire validado pelo schema público; o dump canônico validado é
usado no fingerprint, na validação de alvo e no contexto.
Antes de qualquer catálogo, fingerprint ou contexto, uma única validação
sequencial compartilhada pelo seletor e pelo planner reidrata o artifact pelo
modelo concreto de cada uma das dez read tools, confere `source.resource`,
escopo, metadata e a projeção exata `outcome → content`. Erros aceitam somente
o envelope sanitizado e não concedem IDs; resultados completos ou degradados
só concedem IDs presentes nos campos tipados que a tool realmente projeta.
Artifacts persistidos conservam essa projeção especializada de forma JSON-safe
e a revalidam pelo modelo exato após reabertura do checkpoint, sem guardar HTTP
bruto ou objetos de runtime.
Para RMS e espectro truncados, o artifact especializado conserva ainda uma
projeção limitada e tipada idêntica ao conteúdo do modelo (no máximo 100
amostras ou 20 picos). O preflight compara todo o conteúdo contra essa forma
autoritativa após JSON/SQLite, além de conferir os índices compartilhados com
a projeção técnica principal; artifacts truncados antigos sem prova suficiente
falham fechados. A validação semântica também reaplica as restrições reais dos
wires normalizados de ativo, análise/lista, baseline, RMS, espectro, qualidade
e documento. Em respostas degradadas, `point_id: null` significa somente
ausência de evidência para cadastro de ativo, lista genérica de análises e
detalhe de análise; não autoriza ponto e continua inválido nas quatro tools
técnicas.
O orçamento de contexto mede o wire OpenAI-compatible de mensagens e schemas
das tools, remove apenas pares completos antigos com marcador explícito e não
remove a observação mais recente nem erros/modos degradados; se o conjunto
protegido não couber, falha antes do modelo. Antes da segunda requisição, o
contexto é recalculado com o schema real de `PlannerTerminalDecision`; falha de
espaço preserva a seleção consumida sem iniciar a finalização. O limite de 20
passos permanece no `AgentState.step_limit` e será integrado ao caminho do
planner somente na Task
12, sem alterar grafo ou entrypoint nesta fatia.

**Arquivos:** evoluir `planner.py`, contratos persistíveis estritamente
necessários e testes; reutilizar `READ_TOOLS` e `WRITE_PROPOSAL_TOOLS` sem
recriar catálogos paralelos.

**Contrato e testes:**

- `select_planner_tools(state, runtime)` é puro: leitura exige `read`; propostas exigem `WriteToolRuntime`, a permissão correspondente e os pré-requisitos observáveis.
- Tools dependentes de IDs só aparecem depois que o ID está no pedido ou em uma observação validada; o runtime não lê cenários, golden set nem arquivos de avaliação.
- Histórico de outra `request_id` não entra no prompt nem conta como repetição da solicitação atual.
- Duas chamadas com argumentos distintos são possíveis; uma chamada canonicamente idêntica não executa uma segunda vez.
- O oitavo tool call, a nona seleção, falta de espaço para o pedido atual ou contexto excedido param deterministicamente antes de LLM/HTTP.
- Erros e modos degradados permanecem visíveis para a decisão seguinte, sem serem convertidos em sucesso.
- Limites e fingerprints têm testes de fronteira e a tarefa termina em commit próprio.

**Evidência desta fatia (01/09/2026):** os 339 testes focados de planner,
estado e checkpoint passaram; `uv lock --check --offline` resolveu 49 pacotes;
a suíte completa do agente passou com 1.184 testes; e `make test` passou com 59
testes da API e 1.184 do agente (1.243 no total), mantendo somente o
`PendingDeprecationWarning` conhecido de `python_multipart`.
No quinto ciclo corretivo, os **395 testes** de planner e estado passaram
(393 no sandbox e os dois casos SQLite em execução externa), incluindo a
matriz dos dez artifacts, erros sanitizados, modos degradados e round-trip JSON;
os locks offline de API e agente e `git diff --check` também passaram.
No sexto ciclo corretivo, a suíte focada completa passou externamente com
**752 testes**, incluindo a reabertura SQLite das projeções técnicas; dois
casos adicionais de adulteração exatamente nos cortes passaram localmente,
assim como **751 testes sem SQLite**. Os locks offline continuaram resolvendo
49 pacotes no agente e 55 na API.
No sétimo ciclo corretivo, a validação degradada de `get_asset` passou a
reutilizar a mesma regra do executor para o `id` primário, aliases recursivos
normalizados de ativo/empresa e estrutura de pontos, preservando
`point_id: null` apenas como ausência de evidência. O planner eliminou seu
parser divergente de timestamps e reutiliza `parse_aware_iso_timestamp` em
análises, baseline, RMS, espectro e modelo. Passaram **782 testes sem SQLite**;
o novo restore SQLite passou externamente, e os locks offline permaneceram em
49 pacotes do agente e 55 da API.
No oitavo ciclo corretivo, `CompanyId` tornou-se o tipo compartilhado entre o
wire completo de ativo e o preflight do planner. Assim, um artifact correlato
com empresa fora de `comp_*` falha como histórico inválido antes do catálogo ou
modelo, mesmo quando pedido e runtime carregam o mesmo tenant amplo. O wire de
ponto também reutiliza `PointId`; a auditoria dos identificadores confirmou
que ativo, empresa, pai e ponto preservam a autoridade do wire, enquanto o
`point.asset_id` descartado continua validado pelo executor antes da
normalização. Passaram **784 testes focados sem SQLite**, e os locks offline
permaneceram em 49 pacotes do agente e 55 da API.

### Task 12 — integrar o planner ao LangGraph e preservar as escritas

**Objetivo:** substituir o caminho de leitura fictício por um loop real do
planner, mantendo intacta a fronteira determinística dos cinco efeitos.

**Decisões:** o grafo recebe o planner/modelo como dependência de construção,
nunca como estado. O caminho novo é
`ingest → planner_select → planner_tool → planner_select`; ausência de tool
leva a `planner_finalize`, e uma proposal tool leva imediatamente a
`write_policy`. Um `resume_anchor` observável registra o último nó concluído e
a fronteira valida pares permitidos `(anchor, next)` antes de retomar ciclos.
O builder sem planner preserva o pequeno fluxo determinístico apenas para
compatibilidade dos testes existentes; a construção Groq explícita habilita o
planner real.

**Arquivos:** evoluir `graph.py`, `entrypoint.py`, `state.py` e testes de grafo,
checkpoint, reprocesso e ações não idempotentes.

**Contrato e testes:**

- Um caso simples de leitura faz uma tool real via `ToolNode`, retorna ao modelo, obtém decisão Pydantic e encerra sem loop infinito.
- Cada proposal tool escolhida pelo planner somente preenche `pending_proposal`; política, confirmação, preparação, checkpoint e execução continuam em código.
- Resultado do modelo nunca cria aprovação, idempotency key, identidade, permissão ou chamada HTTP de escrita diretamente.
- Retomada após cada nó novo usa `resume_anchor` validado; anchor ausente, impossível ou divergente falha fechado sem inferir trace.
- O limite usa passos restantes antes de entrar no trecho fixo de escrita; os fluxos diretos existentes mantêm o mesmo comportamento e não repetem efeitos.
- Repetição, limite e falha de tool terminam com decisão segura e registro observável, sem retry oculto.
- Rodar testes focados dos cinco fluxos e fazer commit próprio.

### Task 13 — troca de adapter, smoke Groq e aceite documental

**Objetivo:** demonstrar o aceite da Fase 6 no caminho público e alinhar a
documentação ao estado real, sem promover componentes das fases seguintes.

**Arquivos:** adicionar somente testes/runner de smoke necessários; atualizar
`Makefile`, `README.md` e a Fase 6 deste arquivo depois das evidências.

**Contrato e testes:**

- Dois providers falsos, por meio do mesmo contrato, percorrem a mesma solicitação e produzem os mesmos schemas, catálogo oferecido, tool/argumentos, decisão e resultado de política.
- IDs internos de chamada são gerados ou normalizados pelo runtime e não tornam o estado dependente do provider.
- Um smoke opt-in, fora de `make test`, compara `openai/gpt-oss-120b` e `openai/gpt-oss-20b` na conta disponível quanto a português, tool, argumentos, Pydantic separado, latência e estabilidade; imprime somente métricas e status seguros.
- O smoke lê `GROQ_API_KEY` do ambiente, não imprime prompt/resposta brutos, headers, trace ou segredo e não faz retry automático.
- Executar testes focados, `uv lock --check` e `make test`; registrar totais e o resultado real do smoke.
- Marcar a Fase 6 somente após todo o aceite. README passa a declarar provider/planner reais, mas mantém writer, resposta ao cliente, ledger completo, gate, Logfire e Pydantic Evals como ausentes.
- Fazer self-review, commit próprio e uma revisão final independente de toda a faixa da Fase 6.

## Plano de implementação SDD — Fases 7 a 10

Este plano implementa, em ordem, o ledger de evidências, o writer com porta de
segurança, a revisão humana e a observabilidade Logfire. Cada task segue uma
fatia vertical `RED → GREEN → refactor`, termina com commit próprio e recebe
revisão independente antes da próxima. O aceite documental ocorre somente após
os testes integrados das quatro fases.

### Restrições globais

- O ledger é derivado em código de observações e recibos já validados; texto
  livre de modelo nunca se torna fato da API.
- Toda afirmação técnica liberada é renderizada a partir de uma evidência
  persistida e vinculada à solicitação atual. Erros não sustentam fatos;
  `partial`, `conflict`, `inconclusive`, `unavailable`, truncamento, limitações e
  obsolescência permanecem explícitos.
- Planner e writer são papéis separados. Modelo, runtime, credenciais, resposta
  HTTP bruta, prompt completo, raciocínio e IDs externos do provider não entram
  no checkpoint.
- A porta de segurança é determinística. Writer, revisor e Logfire não criam
  permissão, aprovação, tool call, retry, chave idempotente nem efeito HTTP.
- Revisão humana não é aprovação de ação nem escalonamento humano. Qualquer
  edição retorna à mesma porta de segurança, e uma retomada nunca repete efeito.
- Logfire recebe somente nomes fixos e atributos de allowlist. Tokens, chaves,
  credenciais, headers, mensagens, justificativas, valores industriais, golden
  set, gabarito, tracebacks e identidades literais não são exportados.
- O runtime continua sem acesso a `eval/expected-paths.json`,
  `docs/test-scenarios.md` e `data/cases.parquet`. Logfire não é o banco do
  ledger e avaliações não alteram o atendimento.
- SQLite continua sendo o checkpointer de desenvolvimento; PostgreSQL permanece
  futuro. O caminho sem planner permanece somente como fallback determinístico
  de compatibilidade e não é descrito como o agente completo.

### Task 14 — contratos e compilador determinístico do ledger

**Objetivo:** substituir o placeholder de evidência por contratos completos e
um compilador puro que transforma observações tipadas em fatos, lacunas e
conflitos auditáveis.

**Decisões:** `EvidenceItem` registra ID determinístico, `request_id`, `call_id`,
tool, recurso, caminho canônico do fato, valor JSON observado, modo, instante da
fonte quando existir, instante de registro, limitações e qualidade. IDs são
derivados por SHA-256 de campos canônicos, nunca pelo LLM. Obsolescência usa
somente sinais semânticos explícitos do domínio (`analysis.status=stale`,
`baseline.state=invalidated`, `data_quality.staleness_flag=true` e expiração
persistida de recibo/intenção); não inventa TTL industrial genérico. Um fato
completo e não obsoleto pode ser `claimable`; fatos parciais ou truncados são
preservados com limitação, mas não liberam afirmação crítica. Modos
`conflict`, `inconclusive` e `unavailable` produzem lacunas bloqueantes; erro
produz lacuna sanitizada e nenhum fato. Valores diferentes para a mesma chave
canônica permanecem como itens separados e criam um conflito explícito.

**Arquivos:** criar `agent/src/tractian_agent/evidence.py`; evoluir somente os
contratos necessários em `state.py`; criar `agent/tests/test_evidence.py` e
ajustar testes de estado pertinentes.

**Contrato e testes:**

- Compilar uma observação completa gera fatos com fonte, recurso, valor,
  instante e referência exata ao `call_id` sem aceitar conteúdo livre do modelo.
- Erro, parcial, truncamento, indisponibilidade, inconclusão, conflito e os três
  sinais explícitos de obsolescência têm testes RED/GREEN próprios.
- Duas fontes divergentes conservam ambos os itens e um conflito; uma repetição
  idêntica é determinística e não duplica o ledger.
- `EvidenceAssessment` só é `sufficient` quando existe fato `claimable` e não há
  lacuna ou conflito bloqueante para a conclusão; a causa insuficiente é
  enumerada e não contém texto bruto externo.
- Contratos são estritos, congelados, JSON-safe e sobrevivem a round-trip JSON.
- Rodar testes focados, self-review e commit próprio.

### Task 15 — integrar o ledger ao estado, grafo e checkpoint

**Objetivo:** preencher o ledger no caminho real imediatamente após uma read
tool validada e após um recibo terminal de escrita, preservando histórico por
`request_id` e coerência de retomada.

**Decisões:** o nó de tool acrescenta evidência somente depois de
`validate_planner_read_observation`. O caminho de ação registra como evidência
apenas recibo tipado aceito ou falha/lacuna sanitizada; proposal nunca é efeito.
O estado valida que cada evidência de tool referencia uma observação da mesma
solicitação e que cada evidência de ação referencia uma intenção terminal da
mesma solicitação. Histórico anterior continua auditável, mas não fundamenta a
resposta atual.

**Arquivos:** evoluir `graph.py`, `state.py`, testes de planner/grafo/checkpoint
e os fluxos de escrita; atualizar a Fase 7 somente após o aceite.

**Contrato e testes:**

- O caminho público `build_agent_graph → invoke_agent` preenche evidências sem
  permitir que planner ou writer forneçam o ledger.
- Tool com erro preserva a lacuna e não cria fato; modos degradados e conflito
  atravessam o checkpoint sem virar sucesso.
- Recibo de ação concluída fica ligado à intenção; resultado incerto não afirma
  conclusão e nunca provoca novo HTTP na retomada.
- Fechar e reabrir SQLite preserva fatos, lacunas, conflitos e seus vínculos;
  adulteração de `request_id`, `call_id`, recurso ou valor falha fechada.
- Casos representativos demonstram suficiência, insuficiência, conflito e
  obsolescência pelo caminho público.
- Rodar testes focados, `uv lock --check --offline`, self-review e commit.

### Task 16 — writer mínimo e porta determinística de segurança

**Objetivo:** gerar uma resposta em português a partir da decisão e do ledger,
sem conceder ao modelo poder para criar fatos ou mudar a decisão.

**Decisões:** `Writer` usa prompt `writer-v1`, `with_structured_output` e um
contrato sem tool calls. Recebe somente decisão, fatos canônicos utilizáveis,
lacunas necessárias e informação ausente já decidida; não recebe mensagem
original, identidade, permissões, artifacts, proposta, recibo bruto nem runtime.
O draft contém a decisão imutável, IDs ordenados de evidência, IDs de limitações
e um próximo passo enumerado. Frases técnicas finais são renderizadas em código
a partir dos itens do ledger; o modelo não fornece valores nem afirmações
livres. Uma falha de formato permite exatamente uma nova chamada de reparo com
o mesmo contexto mínimo e sem reutilizar a saída inválida; nova falha encaminha
para revisão.

`ReleaseGate` valida decisão, solicitação, suficiência, referências, limitações,
permissões e estado da intenção. Suas saídas fechadas são `release`,
`request_information`, `request_confirmation` ou `require_human_review`.
Somente `release` cria resposta técnica ao cliente.

**Arquivos:** criar `writer.py` e `release_gate.py`; evoluir `state.py`,
`graph.py`, `entrypoint.py` e testes públicos. O builder exige writer quando o
planner estiver habilitado; planner/modelo e writer/modelo continuam
dependências de construção separadas.

**Contrato e testes:**

- Modelo falso comprova o contexto mínimo recebido e a separação entre planner
  e writer; `bind_tools` nunca é usado pelo writer.
- Decisão divergente, ID inexistente, fato parcial/obsoleto/conflitante,
  limitação omitida, permissão incompatível ou intenção incerta bloqueiam a
  liberação antes de `FinalResult` técnico.
- Formato inválido faz uma única tentativa de reparo; duas falhas encerram em
  revisão sem retry oculto.
- Respostas de orientar, agir, escalar e pedir informação são renderizadas com
  proveniência; confirmação continua pertencendo à política já existente.
- Novas âncoras, orçamento de passos e retomadas são validados, sem alterar as
  garantias de checkpoint/idempotência dos cinco fluxos.
- Rodar testes focados, suíte dos fluxos de escrita, self-review e commit.

### Task 17 — revisão humana persistida e retomável

**Objetivo:** suspender respostas bloqueadas, aceitar decisão humana confiável e
retomar sem reiniciar a investigação ou repetir ação.

**Decisões:** `ReviewRequest` persiste ID, solicitação, motivo enumerado, ponto de
dúvida, IDs de evidência, draft, criação e expiração em UTC. Expira após 24
horas. `ReviewerIdentity` é contexto confiável separado e exige permissão
`review`; o reply não escolhe o próprio autor. Operações são `approve`, `edit`
e `reject`. A edição altera somente seleção/ordem de evidências e próximo passo
do contrato fechado, nunca texto técnico livre, e sempre volta ao
`ReleaseGate`. Aprovação também passa pelo gate contra o estado atual. Rejeição
produz somente aviso seguro de não liberação. Expiração bloqueia a retomada e
exige nova solicitação; nenhum caminho repete efeito já terminal.

**Arquivos:** criar `human_review.py`; evoluir estado, grafo, entrypoint e testes
de checkpoint/fluxos; atualizar a Fase 9 somente após o aceite.

**Contrato e testes:**

- `interrupt()` expõe motivo, dúvida e referências disponíveis sem credenciais,
  runtime ou raciocínio; `Command` retoma pelo ID persistido.
- Aprovação, edição válida, edição inválida, rejeição, autor sem permissão,
  reply obsoleto e expiração têm ciclos RED/GREEN separados.
- Auditoria registra autor confiável, horário, decisão e mudança estrutural;
  adulteração ou segundo julgamento divergente falha fechada.
- Reinício real do SQLite retoma no nó correto, reaplica o gate e não repete
  tool, planner, writer ou ação HTTP já concluídos.
- Rodar testes focados, regressões dos cinco fluxos, self-review e commit.

### Task 18 — observabilidade Logfire segura e opt-in

**Objetivo:** reconstruir uma execução por `trace_id` usando spans e métricas de
baixa cardinalidade, sem exportar conteúdo de atendimento ou segredo.

**Estado implementado e aceito na Task 19:** fachada nula,
recorder seguro e adapter Logfire manual; opt-in triplo validado antes do import;
correlações HMAC por domínio; envelope público sem persistência do `trace_id`;
spans de request, nós, planner, writer, tool, política, tentativas de ação, gate,
revisão e resposta; métricas sem IDs. O runtime não emite avaliação. Retenção
inicial de 30 dias e limites de volume/cardinalidade pertencem à operação externa
da conta Logfire, não ao ledger nem ao código do atendimento.

**Decisões:** adicionar o SDK Logfire ao agente e criar uma fachada injetável
com implementação nula por padrão. A exportação exige simultaneamente
`TRACTIAN_LOGFIRE_ENABLED=true`, `LOGFIRE_TOKEN` explícito e
`TRACTIAN_LOGFIRE_PSEUDONYM_KEY`; configuração ausente, vazia ou inválida não
configura o SDK nem toca a rede. A fronteira lê uma única vez os três valores e
configura o SDK com esse snapshot imutável. Usar apenas spans manuais;
instrumentações automáticas de LangChain/LangGraph, HTTPX, FastAPI e Pydantic
ficam desligadas. O pacote acrescenta precocemente `logfire-plugin` a
`PYDANTIC_DISABLE_PLUGINS`, preservando entradas existentes e sentinelas
globais; por isso o agente requer Pydantic 2.7 ou posterior.
Um `trace_id` opaco é criado na fronteira para cada execução, correlacionado por
atributo e retornável como único identificador técnico. IDs de request, thread,
execução, caso, empresa, pessoa e, quando aplicável, experimento/caso de
benchmark são HMAC-SHA256 truncados antes do export.

Tentativas de ação são contexto efêmero e só abrem span ao redor do `POST` ou
`PATCH` modificador real, depois de qualquer GET de preflight. Toda operação da
fachada e do backend é fail-open, inclusive para `BaseException`, mas uma
exceção ou `CancelledError` do negócio é reemitida sem troca de objeto, causa ou
traceback. Resultados de policy, gate, ação, revisão e tool são classificados
somente por enums e contratos validados; mensagens e payloads não participam da
decisão observacional.

**Allowlist:** nomes fixos para request raiz, nó, planner, writer, tool,
política, ação, gate, revisão, resposta e avaliação; atributos limitados a
versão, enums, flags, contadores, duração, nomes de catálogo e referências
pseudonimizadas. Métricas não recebem IDs. Exceções são reemitidas ao chamador e
registradas apenas por código fechado, sem mensagem ou traceback.

**Arquivos:** criar `observability.py` e testes; evoluir dependências/lock,
grafo, entrypoint e contratos mínimos de resultado. Não instrumentar o
simulador FastAPI nem criar runner de avaliação; a fachada oferece o span de
avaliação e correlação opcional que a Fase 11 consumirá.

**Contrato e testes:**

- Implementação nula prova zero configuração/rede sem o opt-in completo.
- Recorder falso cobre request, nós, LLMs, tools, erro, política, ação, gate,
  revisão, resposta e avaliação pelo mesmo contrato público.
- Sentinelas em token, header, mensagem, justificativa, artifact, evidência,
  provider e golden set nunca aparecem em nomes ou atributos exportáveis.
- Pseudônimos são determinísticos por chave, diferentes entre chaves e não
  revelam o ID literal; `trace_id` permite localizar todos os spans da execução.
- Telemetria não altera decisão, retry, exceção, resultado ou checkpoint.
- Rodar testes focados, `uv lock --check`, self-review e commit próprio.

### Task 19 — aceite integrado e documentação real

**Objetivo:** provar as quatro fases em conjunto e alinhar a documentação ao que
foi realmente executado, sem antecipar Pydantic Evals ou juízes.

**Arquivos:** atualizar `README.md`, `AGENTS.md`, as Fases 7–10 e este plano em
`TASKS.md`; ajustar `Makefile` somente se existir novo comando operacional real.

**Contrato e testes:**

- Um cenário completo percorre consulta, ledger, writer e liberação; cenários
  parcial, conflitante, obsoleto e falho terminam no estado seguro correto.
- Um cenário bloqueado interrompe, fecha/reabre SQLite, recebe aprovação ou
  edição humana, reaplica o gate e termina sem repetir tool ou efeito.
- Uma execução instrumentada por recorder seguro compartilha `trace_id` entre
  os estágios sem registrar valores proibidos.
- Rodar testes focados, locks offline quando aplicável, `git diff --check` e
  `make test`; registrar contagens e warning conhecido.
- Marcar as Fases 7, 8, 9 e 10 somente após todos os critérios acima. README e
  AGENTS passam a declarar ledger, writer, gate, revisão e Logfire reais, mas
  mantêm Pydantic Evals, juízes e calibração como ausentes.
- Fazer self-review, commit próprio e revisão final independente de toda a faixa
  antes da integração na `main`.

**Concluída em 03/09/2026:** os cenários integrados e as regressões acima
comprovaram as quatro fases juntas. README e AGENTS refletem somente os
componentes executados; Pydantic Evals, juízes, calibração e a Fase 11 permanecem
ausentes. Não foi criado comando novo no Makefile porque a observabilidade é uma
dependência injetável do runtime, não um processo independente.
