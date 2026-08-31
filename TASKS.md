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
- Aceitar chaves de 1 a 255 caracteres sem espaços e diferenciar maiúsculas de minúsculas. A futura camada de execução de escritas, não a pessoa usuária nem o LLM, gera e persiste antes da chamada uma chave no formato `tractian-agent:<uuid>`; o cliente HTTP apenas a valida e propaga.
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
- Não fazer retry nem decidir se um erro é repetível dentro do cliente. A criação e a persistência da chave pertencem à intenção de escrita e ao estado das fases posteriores; o cliente valida o protocolo e apenas propaga a chave recebida.
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

- Expor tools LangChain reais por intenção de consulta, sem uma tool genérica que aceite URL, método ou caminho. O catálogo inicial terá `get_asset`, `list_asset_analyses`, `get_analysis`, `get_baseline`, `get_rms_series`, `get_spectrum`, `get_data_quality`, `get_model`, `search_knowledge` e `get_knowledge_document`.
- Manter cada adapter LangChain fino: nome, descrição, schema público, contexto injetado e conversão do retorno. A operação determinística em Python valida escopo e significado, chama o `IndustrialApiClient` e normaliza a observação; ela não recria transporte nem autorização da API.
- Obter identidade, empresa, permissões, ativo central, cliente HTTP, `seed` de avaliação e modelo industrial configurado por contexto confiável. Esses dados não fazem parte dos argumentos visíveis ao modelo. `get_current_user` é uma consulta interna da fronteira de entrada, não uma tool do LLM.
- Usar argumentos Pydantic específicos e restritos. IDs aceitam somente o prefixo e os caracteres esperados; filtros usam valores fechados. Paths, método, headers e modelo de resposta ficam fixos em código.
- Aplicar menor acesso antes da chamada: consultas de ativo ficam limitadas ao ativo central; respostas completas também confirmam empresa e relação com o recurso pai antes de serem expostas. Conhecimento é global no escopo atual, e o modelo consultado é o configurado no runtime.
- Retornar conteúdo JSON compacto e normalizado para o modelo e um artifact JSON serializável para código, trace, ledger e avaliações futuras. O artifact não contém headers, identidade, cliente ou resposta HTTP bruta. Qualquer redução declara `truncated=true` e a quantidade omitida; não há truncamento silencioso.
- Não executar retry dentro das tools. Cada tentativa e cada falha permanecem explícitas para a futura política do LangGraph.
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

**Decisões implementadas:**

- `thread_id` identifica a linha persistida e pode receber novos `request_id`; toda invocação ou retomada usa novo `execution_id`. Caso, empresa, pessoa usuária ou alvo confiável divergentes falham fechados. O estado tipado mantém somente valores JSON-safe e observáveis, incluindo mensagens, chamadas, evidências, proposta, decisão, intenções, passos, resultado e revisão.
- A fronteira é a função Python assíncrona `invoke_agent`, que exige runtime autenticado e `thread_id`. Runtime, cliente, credenciais, seed, golden set, resposta HTTP bruta e raciocínio não são checkpointados.
- O desenvolvimento usa `AsyncSqliteSaver` em `.run/agent-checkpoints.sqlite3`, serializer restrito e ciclo de vida fechado. Não há expiração ou remoção automática de threads; `adelete_thread(thread_id)` é explícito. Isso é distinto do SQLite idempotente da API e do TTL de sete dias da chave de reprocesso. PostgreSQL continua futuro.
- Criação e retomada que podem escrever usam `durability="sync"`; `prepare_intent` e `execute_action` ocupam supersteps distintos. `interrupt()` estruturado e `Command` pelo ID retomam uma confirmação sem efeito anterior.
- O grafo é determinístico e sem LLM: leitura percorre `ingest → route → finish`, e escritas passam por política, confirmação quando aplicável, preparação persistida e execução. Locks por `thread_id` são locais ao processo/event loop; não há lease distribuído.

**Evidência final de aceite (30/08/2026):** a suíte focada de estado, checkpoint, entrada, reprocesso e ações não idempotentes passou com **389 testes**; `uv lock --check` resolveu 47 pacotes; `make test` passou com **59 testes da API + 1.053 do agente = 1.112 testes**, mantendo somente o `PendingDeprecationWarning` conhecido. A Task 6 provou SQLite temporário real, fechamento/reabertura do saver, `prepared` antes do HTTP, replay de recibo e ausência de segundo efeito; a Task 7, revisada sem achados Critical/Important, provou que a retomada de uma ação não idempotente por novo `execution_id` não toca a rede.

- [x] Definir estado tipado com solicitação, identidade, mensagens, chamadas, evidências, decisão, passos, chaves idempotentes e revisão.
- [x] Montar um grafo mínimo sem LLM para provar as transições.
- [x] Configurar checkpointer SQLite.
- [x] Testar persistência e retomada após reinício.

**Aceite verificado:** uma execução interrompida retoma seu estado persistido sem repetir uma ação confirmada; o grafo não implementa planner, writer, ledger completo, gate de liberação, Logfire nem avaliação.

## Fase 6 — provider e planner

**Aprender:** adapter, structured output, prompt de sistema, seleção de tools e orçamento de contexto.

**Decidir:** modelo Groq inicial, temperatura, timeout, limites e contrato comum de provider.

- [ ] Criar interface de modelo independente do provedor.
- [ ] Implementar adapter inicial do Groq.
- [ ] Criar prompt e saída estruturada do planner.
- [ ] Expor apenas tools pertinentes ao estado atual.
- [ ] Limitar passos, repetição e consumo de contexto.

**Aceite:** trocar o adapter não altera estado, tools ou regras de segurança.

## Fase 7 — ledger de evidências

**Aprender:** proveniência, evidência suficiente, conflito e diferença entre dado, inferência e conclusão.

**Decidir:** schema de `EvidenceItem`, regras mínimas de suficiência e representação de conflitos.

- [ ] Preencher o ledger em código com fonte, valor, instante, limitações e referência ao retorno.
- [ ] Impedir que texto livre do LLM seja registrado como fato da API.
- [ ] Detectar evidência ausente, parcial, vencida ou conflitante.
- [ ] Testar fundamentação e suficiência por cenário representativo.

**Aceite:** toda afirmação crítica liberada pode ser ligada a evidência real.

## Fase 8 — writer e gate de segurança

**Aprender:** separação de responsabilidades, contexto mínimo e validação pós-geração.

**Decidir:** schema da resposta, regras críticas e no máximo uma tentativa de reparo de formato.

- [ ] Criar prompt separado para o writer.
- [ ] Enviar ao writer somente decisão e evidências necessárias.
- [ ] Validar formato, afirmações críticas, permissões e incerteza em código.
- [ ] Pedir informação ou revisão humana quando a resposta não puder ser liberada com segurança.

**Aceite:** o writer não consegue criar nova decisão, tool call ou fato não registrado.

## Fase 9 — revisão humana

**Aprender:** human-in-the-loop, fila de revisão, estado suspenso e trilha de auditoria.

**Decidir:** autenticação provisória, operações de aprovar/editar/rejeitar e expiração.

- [ ] Interromper o grafo com motivo, ponto de dúvida e evidências disponíveis.
- [ ] Permitir retomada após decisão do revisor.
- [ ] Registrar autor, horário e alteração do humano.
- [ ] Testar aprovação, edição, rejeição e retomada.

**Aceite:** caso ambíguo não chega ao cliente como certeza e pode ser retomado sem recomeçar.

## Fase 10 — Logfire

**Aprender:** trace, span, atributo, correlação e sanitização.

**Decidir:** atributos permitidos, pseudonimização, retenção e limites do plano.

- [ ] Instrumentar requisição, nós, LLMs, tools, decisão e resultado.
- [ ] Correlacionar `request_id`, `trace_id`, `experiment_id` e `case_id` quando aplicável.
- [ ] Remover tokens, credenciais e payloads sensíveis.
- [ ] Testar a sanitização antes de enviar dados.

**Aceite:** um funcionário autorizado localiza a execução pelo ID sem expor segredo ou golden set.

## Fase 11 — runner e checks programáticos

**Aprender:** Pydantic Evals, isolamento de benchmark, repetição e reprodutibilidade.

**Decidir:** divisão desenvolvimento/calibração/oculto, número de repetições e versionamento de casos, prompts e modelos.

- [ ] Executar casos sem disponibilizar o gabarito ao runtime.
- [ ] Validar resposta, tools, argumentos, permissões, erros e regras críticas.
- [ ] Comparar trajetória observada com a esperada.
- [ ] Gerar relatório por caso, dimensão e experimento.

**Aceite:** o mesmo experimento pode ser reproduzido com versões e configuração registradas.

## Fase 12 — juízes offline

**Aprender:** rubrica, juiz cego, juiz de trajetória, viés e variância de LLM.

**Decidir:** modelo de cada juiz, rubricas, exemplos rotulados e regras críticas de reprovação.

- [ ] Implementar juiz cego do resultado sem acesso ao trace.
- [ ] Implementar juiz de trajetória com acesso às chamadas e falhas.
- [ ] Separar clareza/tom das dimensões industriais críticas.
- [ ] Retornar score, aprovação e motivo estruturados.
- [ ] Repetir casos para detectar acerto por acaso e instabilidade.

**Aceite:** feedback de avaliação nunca retorna ao agente durante o atendimento.

## Fase 13 — calibração humana

**Aprender:** golden set, concordância, Cohen's kappa, falsos positivos/negativos e escolha de limiar.

- [ ] Rotular 20–30 respostas sem ver a nota do juiz.
- [ ] Medir concordância bruta, kappa, falso aprovado e falso reprovado.
- [ ] Comparar limiares `0.7`, `0.8` e `0.9` com as mesmas execuções.
- [ ] Refinar rubricas, não o gabarito, quando a discordância revelar ambiguidade.
- [ ] Registrar a limitação de existir um único avaliador humano.

**Aceite:** o limiar escolhido é justificado pelos erros observados, não por preferência.

## Fase 14 — comparação Groq × NVIDIA NIM

**Aprender:** modelo versus provedor, API compatível, hosted versus self-hosted, latência e custo.

**Decidir:** modelos disponíveis no momento do experimento e orçamento de repetições.

- [ ] Implementar adapter NVIDIA NIM sem alterar a lógica central.
- [ ] Comparar português, tool calling, saída estruturada, estabilidade, contexto, latência e custo.
- [ ] Avaliar planner e writer separadamente.

**Aceite:** a recomendação se baseia no benchmark versionado e nas condições reais do teste.

## Fase 15 — experimento e entrega

- [ ] Congelar versões dos dados, prompts, código, modelos e rubricas.
- [ ] Rodar experimento final e documentar resultados e limitações.
- [ ] Adicionar comandos reais de agente e avaliação ao `Makefile` e README.
- [ ] Adotar `promptfoo` somente se houver comparação recorrente que ele simplifique.
- [ ] Adotar `Ragas` somente se existir um pipeline RAG para medir.
- [ ] Revisar segurança, reprodutibilidade e isolamento do golden set.

**Aceite:** outra pessoa consegue reproduzir o experimento seguindo apenas o repositório.

## Plano SDD — concluir as Fases 5 e 4

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
- O grafo desta entrega é determinístico e sem LLM. Planner, writer, ledger completo, Logfire e runner de avaliação continuam fora do escopo.
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
