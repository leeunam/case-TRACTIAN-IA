# Instruções para agentes de IA

## Antes de alterar

Leia:

1. `README.md` para objetivo, arquitetura, comandos e estado real;
2. `CONTEXT.md` para o vocabulário canônico;
3. `docs/data-schema.md` ou `docs/test-scenarios.md` somente quando a mudança
   tocar dados ou avaliação.

## Forma de trabalho

- Explique termos em português simples e não presuma conhecimento técnico.
- Apresente contrato e testes antes da implementação.
- Trabalhe em mudanças pequenas e verificáveis.
- Atualize o README apenas quando escopo, arquitetura, operação ou estado real
  mudar.
- Não descreva componente planejado, integração não repetida ou qualidade não
  medida como funcional.

## Estado e invariantes

O projeto contém API simulada, agente LangGraph, Pydantic Evals, backend da
demonstração, central React, decisões delegadas, Slack MCP e adapters Groq/NIM.
A entrega técnica foi aceita; o benchmark de qualidade e a calibração humana
ainda impedem a classificação para produção.

- Existe um agente lógico com planner e writer separados.
- Pydantic valida contratos; LangGraph controla fluxo e estado.
- Segurança do runtime é determinística. Juízes não participam do atendimento.
- O runtime nunca recebe `eval/expected-paths.json`,
  `docs/test-scenarios.md` ou `data/cases.parquet`.
- O golden set não é RAG e só fica disponível após a execução avaliada.
- Evidências vivem no estado; Logfire não é o banco do ledger.
- Consultas são autônomas. Escritas exigem pedido explícito e autorização.
- Retry de ação exige idempotência persistida antes do HTTP.
- Não envie tokens, chaves, golden set, prompts ou conteúdo sensível ao Logfire,
  Slack, frontend ou relatórios públicos.
- Não exponha rubricas, notas de juiz ou trace completo ao cliente.
- Não introduza RAG, banco vetorial, multiagentes, fine-tuning ou nova
  infraestrutura sem decisão arquitetural explícita.

## Arquivos e verificação

Preserve alterações do usuário e evite mudanças fora do escopo. Não crie novos
Markdown quando a informação pertencer ao README, ao contexto ou aos contratos
de `docs/`.

Verificação mínima:

```bash
make accept
```

Use `make accept-live` somente quando a tarefa exigir repetir providers e Slack
reais; o comando consome cota e envia notificações.
