# Task 3 — Analysis tools

## Status

Concluída. As tools `list_asset_analyses` e `get_analysis` foram adicionadas
sem importar dados de seed, documentação, cenários ou gabarito para o runtime.

## RED → GREEN

O primeiro teste da fatia vertical foi criado em
`agent/tests/test_analysis_tools.py`. Antes da implementação, o comando abaixo
falhou na coleta porque `tractian_agent.tools.analyses` não existia:

```text
rtk uv run --project agent pytest agent/tests/test_analysis_tools.py -q
ModuleNotFoundError: No module named 'tractian_agent.tools.analyses'
```

Em seguida, a implementação mínima dos modelos wire estritos, operações
determinísticas e adapters LangChain tornou a fatia verde. Foram adicionados
incrementalmente testes para escopo, modos degradados, erros e limites.

## Implementação

- `list_asset_analyses(asset_id, status=None)` exige `read`, valida o ativo
  central e o filtro fechado antes de uma única chamada
  `GET /assets/{asset_id}/analyses`. `status` só entra nos parâmetros quando
  fornecido; `seed` vem exclusivamente do runtime confiável.
- `get_analysis(analysis_id)` valida o padrão de ID e `read` antes de uma única
  chamada `GET /analyses/{analysis_id}`. No modo completo, confirma tanto o ID
  solicitado quanto o ativo central antes de expor a análise.
- O payload completo usa modelos Pydantic estritos (`extra="forbid"`): envelope
  de lista com `analyses`, evidência com `reference` nullable obrigatório,
  floats finitos, confiança em `[0, 1]`, enums canônicos e timestamps com hora
  e timezone.
- A lista é ordenada de forma estável do `created_at` mais recente para o mais
  antigo; cada análise é conferida contra o ativo central, IDs duplicados são
  recusados e o filtro de status é revalidado na resposta.
- O prompt recebe no máximo 20 resumos contendo somente ID, ativo/ponto, tipo,
  severidade, confiança, status, data e limitações. Ele declara total,
  retornados, omitidos e truncamento. O artifact mantém no máximo 200 análises
  completas e declara `truncated`/`omitted_items` no nível superior.
- Em modos degradados, o guard comum valida JSON seguro e a nova inspeção
  recursiva recusa `asset_id` nulo ou contraditório; o detalhe também recusa
  `id`/`analysis_id` nulo, inválido ou diferente quando presentes. `ApiError`
  é preservado sem alteração e não há retry.

## Testes e verificações

| Comando | Resultado |
| --- | --- |
| `rtk uv run --project agent pytest agent/tests/test_analysis_tools.py -q` | 31 passed |
| `rtk uv run --project agent pytest agent/tests -q` | 192 passed |
| `rtk make test` | API: 59 passed (1 aviso externo de depreciação); agente: 192 passed |
| `rtk git diff --check` | passed |

Os testes usam somente `httpx.MockTransport`, sem LLM, servidor ou retry. Eles
cobrem schemas públicos e campos ocultos, IDs/status inválidos com zero HTTP,
permissão e escopo, path/parâmetros fixos, payload real, inconsistências de
ativo/ID/filtro/duplicata, campos ausentes, timezone, não finitude, cortes em
21/201 itens, ordenação, contagens, degradação segura, `ApiError` e os dois
adapters.

## Arquivos

- `agent/src/tractian_agent/tools/analyses.py`
- `agent/src/tractian_agent/tools/__init__.py`
- `agent/tests/test_analysis_tools.py`

## Decisões e preocupações

A ordenação da lista usa `created_at` descendente e conserva a ordem original
quando há empate. O artifact de lista inclui as contagens também no outcome,
enquanto os indicadores universais de truncamento permanecem no nível superior
do artifact já definido pela fundação. Não há preocupação bloqueante conhecida.

## Correção após revisão independente

### RED → GREEN

A revisão identificou que o simulador mantém `data.analyses` mesmo em
`partial` e `conflict`; a versão inicial preservava essa lista bruta em
`partial_data`. Foram adicionadas regressões para a forma real do envelope,
cortes degradados 21/201, validação de linhas fora das janelas, campos
ausentes, escopo/status/duplicatas, `reference=null` sintomático, o código 404
real `NOT_FOUND` e `model.id` aninhado no detalhe parcial. Antes da correção,
o foco terminou com `6 failed, 33 passed`.

### Implementação corrigida

- Em modo degradado com `analyses`, a tool valida e projeta cada linha antes de
  qualquer corte. O prompt recebe no máximo 20 resumos; o artifact recebe no
  máximo 200 resumos, ambos com contagens e truncamento explícitos. Evidências,
  versão de modelo e qualquer outro campo bruto não são copiados.
- Flags seguros de topo, como `conflict` e `inconclusive`, continuam em
  `partial_data`, mas sem a chave `analyses`. Linhas realmente incompletas
  preservam somente os campos de resumo presentes — a ferramenta não cria
  `null` ou defaults.
- `asset_id` segue validado recursivamente no dado degradado. Em detalhe,
  `id` e `analysis_id` são conferidos apenas no objeto raiz para não confundir
  IDs de objetos aninhados, como `model.id`, com o recurso solicitado.
- `AnalysisId` passou a ser reexportado por `tools/identifiers.py`, mantendo o
  mesmo schema público restrito e preparando o reuso em tools de escrita.

| Comando | Resultado da correção |
| --- | --- |
| `rtk uv run --project agent pytest agent/tests/test_analysis_tools.py -q` | 43 passed |
| `rtk uv run --project agent pytest agent/tests -q` | 204 passed |
| `rtk make test` | API: 59 passed (1 aviso externo de depreciação); agente: 204 passed |
| `rtk git diff --check` | passed |
