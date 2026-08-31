# Relatório da Task 4 — matriz completa e proposal tools

## Commit

- Mensagem convencional: `feat(agent): add write proposal policy tools`.
- Este relatório pertence ao mesmo e único commit da Task 4. O hash final é
  informado na entrega, pois um commit não pode registrar o próprio hash em seu
  conteúdo sem alterá-lo.

## Resultado

A política de escrita agora cobre cinco propostas sem executar efeitos:
reprocessar uma análise, solicitar análise especializada, atualizar somente a
criticidade do ativo central, solicitar retreinamento do modelo configurado e
escalar o caso atual. `ReprocessProposal(analysis_id=..., justification=...)` e
`evaluate_reprocess_policy(...)` continuam compatíveis.

As propostas são modelos Pydantic imutáveis, fechados e reunidos na união
discriminada `WriteProposal`. O discriminador `action` tem valor fixo em cada
variante e é criado pela tool; ele não aparece nos argumentos públicos do
modelo. `AgentState.pending_proposal` aceita e restaura as cinco variantes sem
generalizar `WriteIntent` ou `ReprocessIntentScope`, que continuam exclusivos
do reprocesso.

## Interfaces implementadas

- `TrustedWriteContext`: ativo central, caso atual e modelo configurado.
- `WriteMaterialParameters`: somente a nova criticidade nesta entrega.
- `TrustedActionApproval`: ação, alvo, parâmetros materiais e origem da
  aprovação, com validação específica do tipo de ID.
- `CanonicalActionScope` e `resolve_action_scope(...)`: escopo comparável
  `(action, target_id, material_parameters)`.
- `evaluate_write_policy(...)`: matriz geral de permissões e aprovação.
- Wrapper compatível `evaluate_reprocess_policy(...)`.
- Conteúdo `WriteProposalContent` e artifact `WriteProposalArtifact`, cujo campo
  `effect_executed` é sempre `false`.
- Catálogo imutável `WRITE_PROPOSAL_TOOLS`, nesta ordem:
  1. `propose_reprocess_analysis(analysis_id, justification)`;
  2. `propose_request_specialist_analysis(analysis_id, justification)`;
  3. `propose_update_asset_criticality(criticality, justification)`;
  4. `propose_request_model_retraining(justification)`;
  5. `propose_escalate_case(justification)`.

A política aplica a ordem determinística exigida: permissão ausente nega;
justificativa com menos de 20 caracteres após `strip` nega; aprovação ausente
pede confirmação; qualquer divergência de ação, alvo ou parâmetros materiais
pede confirmação; escopo idêntico permite. A justificativa não integra os
parâmetros materiais aprovados.

## TDD — RED

Os testes foram criados em fatias verticais e executados antes da implementação
correspondente. Falhas observadas:

1. contrato de criticidade: coleta falhou com `ImportError` de
   `TrustedWriteContext`;
2. matriz de cinco propostas: coleta falhou com `ImportError` de
   `EscalateCaseProposal`;
3. validação confiável de alvos: cinco casos falharam porque IDs de outro tipo
   ainda eram aceitos;
4. persistência no estado: quatro variantes falharam porque
   `pending_proposal` aceitava somente `ReprocessProposal`;
5. primeira execução real: coleta falhou com `ModuleNotFoundError` de
   `tractian_agent.tools.writes`;
6. quatro tools restantes: coleta falhou por exports ainda ausentes;
7. catálogo: coleta falhou porque `WRITE_PROPOSAL_TOOLS` ainda não existia;
8. schema público: o teste falhou com `KeyError: additionalProperties`, pois o
   subset automático do LangChain removia `extra="forbid"` do schema exposto;
9. regressão do contrato antigo detectou o novo discriminador interno no schema
   de `ReprocessProposal`; o teste foi atualizado para distinguir o contrato da
   proposta do schema público exato da tool.

## TDD — GREEN e verificações

- ciclos focados incrementais: todos passaram após cada implementação mínima;
- política + contratos + estado + checkpoint + grafo + tools: `384 passed`;
- suíte completa do agente: `782 passed`;
- `rtk proxy make test`: `59 passed` na API e `782 passed` no agente, `841` no
  total; somente o `PendingDeprecationWarning` já conhecido do
  `python_multipart`;
- `uv lock --check` no projeto `agent`: `Resolved 47 packages` e sucesso, com
  cache temporário porque o cache padrão do ambiente é somente leitura.

Foram acrescentados 46 casos: matriz de permissões, ordem de decisão, alvo e
parâmetro divergentes, mudança posterior dos alvos confiáveis, limite 19/20,
IDs e criticidade inválidos, JSON round-trip, modelos frozen/extra-forbid,
schemas públicos exatos, catálogo ordenado/único, ToolNode real sem runtime e
ausência de mutação das entradas.

## Arquivos

- `agent/src/tractian_agent/write_policy.py`;
- `agent/src/tractian_agent/state.py`;
- `agent/src/tractian_agent/tools/identifiers.py`;
- `agent/src/tractian_agent/tools/writes.py`;
- `agent/src/tractian_agent/tools/__init__.py`;
- `agent/tests/test_write_policy_matrix.py`;
- `agent/tests/test_write_proposal_contracts.py`;
- `agent/tests/test_write_proposal_tools.py`;
- `agent/tests/test_state.py`;
- `agent/tests/test_write_contracts.py`;
- `TASKS.md`;
- este relatório.

`write_contracts.py` foi deliberadamente preservado: generalizar a intenção
preparada nesta task violaria a fronteira vinculante que reserva essa mudança
para a Task 7.

## Self-review e preocupações

- As proposal tools não importam cliente, runtime ou HTTP e não recebem contexto
  injetado. A execução por `ToolNode` funciona sem runtime.
- Nenhuma operação, idempotency key, retry, relógio, UUID, grafo de escrita ou
  estado global mutável foi criado.
- Os schemas enviados ao modelo têm `additionalProperties: false`. Foi usado um
  `StructuredTool` específico porque o subset automático da versão atual do
  LangChain descartava essa garantia mesmo quando o `args_schema` Pydantic era
  fechado.
- Os exports de escrita são carregados de forma tardia em `tools.__init__` para
  evitar o ciclo `write_policy → tools → writes → write_policy`; o catálogo em
  `tools.writes` continua uma tupla estática.
- O artifact registra proposta, não origem `industrial_api`, e torna explícito
  que nenhum efeito foi executado.
- A generalização do runtime, operações HTTP, idempotência e intenções das
  demais ações permanece pendente para as tasks previstas. Não há falha aberta
  nesta fatia.
