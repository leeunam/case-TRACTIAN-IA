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
- [x] Suíte-base com 39 testes.
- [x] Arquitetura e ordem de aprendizagem definidas.

## Fase 1 — idempotência do reprocessamento

**Aprender:** idempotência, hash canônico, transação, condição de corrida e códigos HTTP.

**Decidir durante a etapa:** tabela SQLite, formato da chave, retenção, representação da resposta persistida e ponto exato usado para simular timeout.

- [ ] Exigir `Idempotency-Key` em `POST /analyses/{analysisId}/reprocess`.
- [ ] Persistir chave, usuário, método, endpoint, hash do payload, status e resposta original.
- [ ] Retornar a resposta original para mesma chave e mesmo payload.
- [ ] Retornar `409 Conflict` para mesma chave e payload diferente.
- [ ] Garantir atomicidade para duas requisições concorrentes.
- [ ] Testar primeira execução, replay, conflito, concorrência e timeout após a ação/commit antes da resposta.

**Aceite:** nenhum retry ou acesso concorrente cria dois trabalhos para a mesma intenção.

## Fase 2 — contratos e cliente da API

**Aprender:** modelos Pydantic, cliente assíncrono, timeout e fronteira entre erro de transporte e erro de domínio.

**Decidir:** timeouts, categorias de erro, envelope normalizado e propagação da identidade.

- [ ] Criar contratos tipados para solicitação, identidade, chamada de tool, resultado e erro.
- [ ] Criar cliente `httpx` sem lógica de decisão do agente.
- [ ] Normalizar respostas `2xx`, `4xx`, `5xx`, timeout e payload inválido.
- [ ] Testar o cliente sem depender de servidor externo.

**Aceite:** toda resposta da API vira um resultado tipado ou erro explícito.

## Fase 3 — tools de leitura

**Aprender:** tool calling, descrição de ferramenta, validação de argumentos e princípio do menor acesso.

**Decidir:** agrupamento das tools, limites de tamanho, retries seguros e quais campos entram no retorno normalizado.

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
