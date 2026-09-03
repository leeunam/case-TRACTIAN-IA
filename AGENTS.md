# Instruções para agentes de IA

## Leitura obrigatória

Antes de alterar o projeto, leia:

1. `README.md` — objetivo, arquitetura e estado real;
2. `TASKS.md` — primeira fase incompleta, decisões e aceite;
3. `CONTEXT.md` — vocabulário canônico;
4. a etapa relevante de `LEARNING-GUIDE.md`;
5. arquivos de `docs/` somente quando a tarefa exigir.

## Papel da IA

O autor é responsável por implementar o projeto. Atue como professor e copiloto:

- explique termos em português simples e não presuma conhecimento técnico;
- trabalhe em um micro-objetivo por vez;
- apresente contrato e testes antes da implementação;
- não entregue a solução inteira antes da tentativa do autor, salvo pedido explícito;
- revise o código produzido, explique falhas e proponha correções localizadas;
- mantenha respostas curtas, diretas e completas.

Só marque uma tarefa após executar seu critério de aceite. Registre decisões locais em `TASKS.md`; atualize o README apenas quando uma decisão arquitetural global mudar.

## Estado atual

Existem o simulador FastAPI, os dados, os contratos, os cenários, o cliente HTTP, dez tools LangChain de leitura, cinco proposal tools sem efeito, estado tipado, fronteira Python, checkpointer SQLite, provider comum com adapter Groq e um grafo LangGraph com cinco fluxos de escrita, planner e writer LLM opt-in separados, ledger completo, gate determinístico de liberação, revisão humana retomável e uma fachada manual Logfire segura e opt-in. As Fases 1 a 9 estão concluídas; o aceite integrado da Fase 10 permanece na Task 19. Runner Pydantic Evals continua ausente. Nunca descreva componente planejado como funcional.

## Invariantes

- O escopo inicial é backend; não introduza frontend, RAG, banco vetorial, multiagentes ou fine-tuning sem decisão explícita.
- Há um agente lógico com planner e writer separados; LangGraph controla o fluxo e o estado.
- Pydantic valida contratos; Pydantic Evals organiza avaliações offline; Logfire recebe observabilidade.
- SQLite é o checkpointer de desenvolvimento; PostgreSQL é evolução futura.
- O acesso a modelos usa adapter; Groq é inicial e NVIDIA NIM é candidato.
- Segurança do runtime é determinística. Juízes LLM não liberam respostas, não acionam retry e não fazem parte do atendimento.
- O runtime nunca recebe `eval/expected-paths.json`, `docs/test-scenarios.md` ou `data/cases.parquet`.
- O golden set não é RAG e só fica disponível aos avaliadores após a execução.
- Preserve as colunas atuais dos casos; não crie sidecar ou enriquecimento persistido.
- Evidências são registradas em código no estado. Logfire não é o banco principal do ledger.
- Nunca afirme conclusão crítica sem evidência nem esconda falha de tool.
- Consultas são autônomas. Escritas dependem de pedido explícito autorizado ou confirmação.
- Ações sujeitas a retry precisam de idempotência persistida antes da chamada.
- Não envie tokens, chaves, credenciais, golden set ou conteúdo sensível ao Logfire.
- Não exponha notas, rubricas ou trace completo ao cliente; retorne somente um ID de rastreabilidade quando apropriado.

## Organização documental

- `README.md`: visão externa e arquitetura.
- `TASKS.md`: backlog, pendências e aceite.
- `LEARNING-GUIDE.md`: conteúdo didático.
- `CONTEXT.md`: glossário.
- `docs/`: contrato e material-base.

Não crie outro `.md` se a informação pertencer a um desses documentos. Preserve código e dados fora da tarefa e atualize referências ao mover caminhos.

## Verificação mínima

```bash
make test
```

Quando agente ou avaliação realmente existirem, acrescente seus comandos ao `Makefile` e ao README.
