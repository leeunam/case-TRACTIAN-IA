# Task 2 — baseline, RMS, spectrum e qualidade dos dados

## Status

Concluída. As quatro tools de leitura foram adicionadas sobre a fundação da
Task 1, sem importar dados, fixtures, cenários ou gabarito no runtime.

## RED → GREEN vertical

| Fatia | RED registrado | GREEN |
| --- | --- | --- |
| Baseline | `rtk uv run --project agent pytest agent/tests/test_technical_tools.py -q` falhou na coleta com `ModuleNotFoundError: tractian_agent.tools.technical`. | A primeira tool passou (`1 passed`). |
| RMS | O mesmo comando falhou com `ImportError: cannot import name 'execute_get_rms_series'`. | Baseline e RMS passaram (`2 passed`). |
| Espectro | O mesmo comando falhou com `ImportError: cannot import name 'execute_get_spectrum'`. | As três tools passaram (`3 passed`). |
| Qualidade | O mesmo comando falhou com `ImportError: cannot import name 'execute_get_data_quality'`. | As quatro tools passaram (`4 passed`). |

Após as quatro fatias, os testes de contrato, adapters, escopo, degradação,
erros e limites foram acrescentados incrementalmente; o teste focado final
terminou com `37 passed`.

## Implementação

- `get_baseline`, `get_rms_series`, `get_spectrum` e `get_data_quality` são
  tools LangChain reais com `ToolRuntime[ReadToolRuntime]` injetado. Seus
  schemas públicos aceitam somente `asset_id` estrito e `point_id` opcional
  estrito.
- Cada operação Python valida `read` e o ativo central antes do HTTP, usa uma
  única chamada `GET` de caminho fixo e transmite somente `point_id` quando
  informado e `seed` vindo do runtime. Não há retry.
- Os payloads completos passam por wire models Pydantic estritos, com
  validação do ativo retornado e do ponto solicitado. Resultados degradados
  preservam `mode`, `notes` e JSON parcial apenas após
  `assert_safe_partial_json`; IDs conhecidos que contradizem escopo falham.
  `ApiError` é retornado sem alteração.
- O limiar do baseline é derivado exclusivamente da feature exatamente
  `rms_mm_s`, como `reference + tolerance`; ausência dessa feature mantém
  `alarm_threshold=None`, independentemente do estado do baseline.
- RMS é normalizado em ordem cronológica estável, preserva duplicatas e usa
  projeção evenly-spaced com extremos para até 100 amostras no conteúdo. O
  artifact retém até 1.000; reduções expõem contagens.
- Picos do espectro são ordenados estavelmente por frequência, sem deduzir
  novos rótulos, preservam `bands_missing`, e usam limites de 20 no conteúdo e
  200 no artifact. Qualidade preserva completude, frescor, SNR e obsolescência.

## Arquivos

- `agent/src/tractian_agent/tools/technical.py`
- `agent/src/tractian_agent/tools/identifiers.py`
- `agent/src/tractian_agent/tools/__init__.py`
- `agent/tests/test_technical_tools.py`
- `TASKS.md`

## Verificação

| Comando | Resultado |
| --- | --- |
| `rtk uv run --project agent pytest agent/tests/test_technical_tools.py -q` | 37 passed |
| `rtk uv run --project agent pytest agent/tests -q` | 116 passed |
| `rtk make test` | API: 59 passed (1 aviso externo de depreciação); agent: 116 passed |
| `rtk git diff --check --cached` | passed |

## Decisões e preocupações

As decisões de limite, projeção e derivação foram registradas em `TASKS.md`.
`get_asset` não foi alterada: a nova implementação compartilha helpers apenas
entre as quatro tools técnicas, preservando os contratos aprovados da Task 1.
O wire model de espectro aceita `note` e `bands_missing` ausentes como valores
vazios/nulos compatíveis com o contrato documentado; qualquer campo extra em
resposta completa continua inválido. Não há preocupação bloqueante conhecida.
