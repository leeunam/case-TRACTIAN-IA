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

**Decisões aprovadas:**

- Expor tools LangChain reais por intenção de consulta, sem uma tool genérica que aceite URL, método ou caminho. O catálogo inicial terá `get_asset`, `list_asset_analyses`, `get_analysis`, `get_baseline`, `get_rms_series`, `get_spectrum`, `get_data_quality`, `get_model`, `search_knowledge` e `get_knowledge_document`.
- Manter cada adapter LangChain fino: nome, descrição, schema público, contexto injetado e conversão do retorno. A operação determinística em Python valida escopo e significado, chama o `IndustrialApiClient` e normaliza a observação; ela não recria transporte nem autorização da API.
- Obter identidade, empresa, permissões, ativo central, cliente HTTP, `seed` de avaliação e modelo industrial configurado por contexto confiável. Esses dados não fazem parte dos argumentos visíveis ao modelo. `get_current_user` é uma consulta interna da fronteira de entrada, não uma tool do LLM.
- Usar argumentos Pydantic específicos e restritos. IDs aceitam somente o prefixo e os caracteres esperados; filtros usam valores fechados. Paths, método, headers e modelo de resposta ficam fixos em código.
- Aplicar menor acesso antes da chamada: consultas de ativo ficam limitadas ao ativo central; respostas completas também confirmam empresa e relação com o recurso pai antes de serem expostas. Conhecimento é global no escopo atual, e o modelo consultado é o configurado no runtime.
- Retornar conteúdo JSON compacto e normalizado para o modelo e um artifact JSON serializável para código, trace, ledger e avaliações futuras. O artifact não contém headers, identidade, cliente ou resposta HTTP bruta. Qualquer redução declara `truncated=true` e a quantidade omitida; não há truncamento silencioso.
- Não executar retry dentro das tools. Cada tentativa e cada falha permanecem explícitas para a futura política do LangGraph.
- Preservar `mode`, `notes` e todo `ApiError`; validação inválida impede HTTP e exceção inesperada de programação não é convertida em sucesso.
- Construir em fatias verticais `RED → GREEN`, começando por `get_asset`, e testar contrato público, escopo, chamada fixa, modos, erros, conteúdo/artifact e integração mínima com `ToolNode` sem criar ainda o agente completo.

- [ ] Implementar tools de consulta necessárias aos cenários.
- [ ] Validar IDs, filtros e identidade antes da chamada.
- [ ] Manter resposta bruta fora do prompt quando a forma normalizada for suficiente.
- [ ] Testar escolha isolada, argumentos e tratamento de cada erro relevante.

**Aceite:** tools de leitura são determinísticas nas bordas e não escondem falhas da API.

## Fase 4 — tools de escrita e política

**Aprender:** autorização, confirmação, proposta versus execução e impacto reversível/irreversível.

**Decidir:** matriz de permissões, formato da justificativa, quando pedir confirmação e quando expandir idempotência.

- [ ] Separar proposta de ação e execução efetiva.
- [ ] Exigir pedido explícito, permissão, escopo claro e justificativa.
- [ ] Pedir confirmação para ação inferida, ampliada ou ambígua.
- [ ] Aplicar idempotência a cada nova ação suscetível a retry.
- [ ] Testar permitido, proibido, ambíguo, repetido e conflitante.

**Aceite:** nenhuma escrita ocorre apenas porque o LLM a sugeriu.

## Fase 5 — estado LangGraph e SQLite

**Aprender:** grafo de estados, nó, aresta condicional, checkpointer, `thread_id`, interrupção e retomada.

**Decidir:** schema do estado, política de retenção, relação entre `request_id`, `thread_id` e execução e primeira fronteira de entrada do agente (função, CLI ou endpoint HTTP).

- [ ] Definir estado tipado com solicitação, identidade, mensagens, chamadas, evidências, decisão, passos, chaves idempotentes e revisão.
- [ ] Montar um grafo mínimo sem LLM para provar as transições.
- [ ] Configurar checkpointer SQLite.
- [ ] Testar persistência e retomada após reinício.

**Aceite:** uma execução interrompida retoma sem perder evidências nem repetir uma ação confirmada.

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
