# Task 14 — contratos e compilador determinístico do ledger

## Implementação

- Criado `agent/src/tractian_agent/evidence.py` com contratos estritos e
  congelados para `EvidenceItem`, lacunas, conflitos, ledger e avaliação de
  suficiência.
- O compilador puro aceita somente `ToolObservation` e reidrata o artifact
  tipado; ele ignora deliberadamente `content`, que é contexto do planner e
  não fonte de fatos.
- IDs usam SHA-256 sobre campos canônicos. Fatos repetidos idênticos são
  deduplicados; valores distintos com a mesma chave canônica preservam ambos os
  itens e geram conflito.
- Observações sem `request_id`, artifacts legados/não reidratáveis e erros não
  produzem fatos claimable. Erros geram somente código enumerado sanitizado.
- `partial`, truncamento, `unavailable`, `inconclusive`, `conflict` e os sinais
  explícitos de obsolescência são preservados como lacunas bloqueantes. Não há
  TTL industrial genérico. A expiração persistida é uma entrada explícita por
  `expired_call_ids`.
- `StateEvidence` recebeu `request_id` opcional e documentação de legado para
  manter checkpoints existentes desserializáveis sem promovê-los a fatos.

## Arquivos

- `agent/src/tractian_agent/evidence.py`
- `agent/src/tractian_agent/state.py`
- `agent/tests/test_evidence.py`
- `agent/tests/test_state.py`

## Evidência TDD

RED, antes de criar o módulo:

```text
$ PYTHONPATH=src ... python -m pytest -q tests/test_evidence.py
ImportError: No module named 'tractian_agent.evidence'
```

GREEN após a primeira fatia (observação completa, artefato tipado e
proveniência):

```text
1 passed in 0.80s
```

GREEN focado após os casos de erro, parcial, truncamento, modos degradados,
obsolescência, conflito, deduplicação, JSON round-trip e compatibilidade do
estado:

```text
337 passed in 1.12s
```

Suíte completa do agente:

```text
1456 passed in 45.44s
```

## Self-review

- Verifiquei que o compilador não lê `ToolObservation.content` nem qualquer
  arquivo de runtime, cenário ou golden set.
- Verifiquei que toda ausência de proveniência e todo artifact legado falham
  fechados, sem item claimable.
- Verifiquei que os únicos sinais de obsolescência são status de análise
  `stale`, baseline `invalidated`, flag de qualidade e expiração explícita.
- Verifiquei `git diff --check` antes do commit e o round-trip JSON dos novos
  contratos.

## Preocupações conhecidas

- Esta task não integra o compilador ao grafo, writer ou gate; isso pertence às
  tarefas seguintes. Assim, o campo `AgentState.evidence` legado permanece
  preservado e não é promovido automaticamente.
- A expiração de recibo/intenção é recebida como informação persistida explícita
  (`expired_call_ids`); a futura integração deve derivá-la dos contratos de
  intenção, sem acrescentar relógio/TTL industrial ao compilador.
