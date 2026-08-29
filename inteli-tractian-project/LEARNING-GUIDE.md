# Guia linear de aprendizagem e construção

Este guia é para o estudante implementar o projeto. O copiloto explica conceitos, revisa decisões e ajuda com código quando solicitado; o estudante escreve e executa a solução.

A arquitetura não é redefinida aqui. Consulte [`README.md`](./README.md) para decisões, [`TASKS.md`](./TASKS.md) para o backlog executável e [`CONTEXT.md`](./CONTEXT.md) para vocabulário.

Em cada sessão, selecione uma única tarefa em `TASKS.md`, peça ao copiloto que ensine o conceito e defina os testes, implemente a sua tentativa e então solicite revisão. Este guia determina a sequência de aprendizagem; não substitui os critérios de aceite do backlog.

## Regra de progressão

Não avance enquanto não conseguir:

1. explicar o conceito com suas palavras;
2. executar um exemplo mínimo;
3. testar o comportamento;
4. relacioná-lo a um cenário do projeto.

Não avance apenas porque o código executou: você deve conseguir explicar por que a solução é segura e como os testes demonstram o comportamento.

## Etapa 0 — conhecer o material

Leia, nesta ordem:

1. `README.md`;
2. `CONTEXT.md`;
3. `docs/data-schema.md`;
4. `docs/api-contract.openapi.yaml`;
5. dois cenários simples em `docs/test-scenarios.md`.

Depois execute:

```bash
make setup
make up
make test
```

Explore `http://localhost:8000/docs` e faça manualmente uma consulta de ativo, baseline, análise e conhecimento.

Aprenda:

- estrutura do repositório;
- diferença entre input do agente e gabarito;
- por que o agente não pode receber `eval/expected-paths.json`;
- diferença entre Contextualizar, Investigar e Executar.

## Etapa 1 — Python aplicado ao projeto

Aprenda:

- módulos e pacotes;
- funções síncronas e assíncronas;
- type hints;
- `dataclass` e `Enum`;
- tratamento de exceções;
- variáveis de ambiente;
- leitura de JSON e Parquet.

Pratique lendo `agent-input/cases.json` e os Parquets sem modificá-los. Liste casos, usuários, permissões, ativos e análises.

Critério de conclusão: explicar como uma linha de caso se relaciona com dados de usuário, ativo e trajetória esperada.

## Etapa 2 — Pydantic Validation

Aprenda:

- `BaseModel`;
- `Field`;
- campos opcionais;
- listas e objetos aninhados;
- `Literal` e enums;
- validators;
- serialização;
- JSON Schema;
- erros de validação.

Modele, sem alterar o dataset:

- entrada do atendimento;
- argumentos de uma tool;
- evidência;
- decisão final;
- solicitação de revisão humana;
- registro de chamada de tool.

Pydantic confirma estrutura, não verdade. Um campo válido ainda pode conter um diagnóstico errado.

Critério de conclusão: uma entrada inválida falha antes de alcançar a API e o erro é compreensível.

## Etapa 3 — domínio industrial mínimo

Use `CONTEXT.md` e os dados para aprender:

- empresa, usuário, ativo, ponto e sensor;
- baseline `learning`, `established` e `invalidated`;
- RMS, referência, tolerância e limiar;
- espectro e frequências características;
- qualidade, completude, SNR e frescor;
- análise automática, evidência, confiança e limitações;
- detecção por baseline e detecção sintomática;
- análise especializada, revisão humana e escalonamento.

Pratique explicando três cenários:

1. `TKT-CTX-02`, para contextualização;
2. `TKT-INV-10`, para investigação;
3. `TKT-EXE-12`, para execução.

Critério de conclusão: explicar por que confiança alta do modelo não compensa dado abaixo do requisito.

## Etapa 4 — HTTP, REST e OpenAPI

Aprenda:

- URL, rota, método, query, header e body;
- GET, POST e PATCH;
- JSON de request e response;
- códigos `200`, `400`, `401`, `403`, `404` e `422`;
- timeout e retry;
- idempotência;
- autenticação e autorização;
- leitura de OpenAPI.

Implemente um cliente HTTP sem LLM. Ele deve:

- usar uma URL-base configurável;
- aplicar timeout;
- enviar `x-user-id` quando necessário;
- gerar uma nova `Idempotency-Key` para cada intenção de escrita e reutilizá-la somente em
  retries do mesmo pedido;
- converter respostas em modelos Pydantic;
- distinguir erro de transporte de rejeição da API;
- nunca registrar credenciais.

Critério de conclusão: conseguir reproduzir manualmente uma trajetória esperada apenas com o cliente.

## Etapa 5 — pytest e testes determinísticos

Aprenda:

- Arrange, Act, Assert;
- fixtures;
- parametrização;
- mocks apenas quando necessários;
- teste unitário e integração;
- teste de contrato;
- regressão.

Teste primeiro:

- parsing dos casos;
- validação dos argumentos;
- tratamento de status HTTP;
- permissões;
- justificativa mínima;
- conversão de retornos parciais.

Critério de conclusão: erros óbvios são encontrados sem gastar chamadas de LLM.

## Etapa 6 — tools

Uma tool é uma função oferecida ao modelo com nome, descrição e schema de argumentos.

Comece com tools de leitura:

- contexto e ativos;
- análises;
- baseline;
- RMS;
- espectro;
- qualidade;
- modelo;
- conhecimento.

Cada tool deve:

- ter responsabilidade única;
- receber apenas os argumentos necessários;
- obter identidade do contexto confiável;
- usar o cliente HTTP;
- devolver resultado estruturado;
- preservar erro e modo da resposta;
- ser testável sem agente.

Depois implemente tools de ação:

- reprocessar;
- solicitar especialista;
- atualizar criticidade;
- solicitar retreinamento;
- escalar caso.

Critério de conclusão: todas as tools funcionam isoladamente e não dependem de texto livre para validar permissão.

## Etapa 7 — modelo Groq e tool calling

Aprenda:

- mensagem de sistema, usuário e tool;
- temperatura;
- janela de contexto;
- structured output;
- tool calling;
- limites de requisição e tokens;
- diferença entre modelo do agente e modelo juiz.

Faça um smoke test com os modelos disponíveis na conta Groq. Compare:

- português;
- escolha de tool;
- argumentos válidos;
- saída estruturada;
- latência;
- estabilidade;
- limites gratuitos.

Registre modelo e configuração escolhidos. IDs e limites mudam; confirme-os no momento da implementação.

## Etapa 8 — LangGraph mínimo

Aprenda:

- estado;
- nó;
- aresta;
- roteamento condicional;
- `StateGraph`;
- `ToolNode`;
- checkpointer;
- interrupção e retomada.

Construa primeiro:

```text
entrada → modelo → tool → modelo → resposta
```

O estado deve conter somente dados observáveis, como mensagens, identidade, chamadas, observações, evidências, decisão, contador de passos e status de revisão. Não armazene monólogo interno do modelo como fonte de verdade.

Critério de conclusão: o grafo resolve um caso simples de leitura e para sem loop infinito.

## Etapa 9 — ledger de evidências

Registre cada fato usado na conclusão:

```text
afirmação
fonte/tool
recurso consultado
valor observado
limitação
instante
```

O ledger deve permitir verificar se uma afirmação final veio de uma observação real.

Regras:

- não usar retorno com erro como evidência;
- preservar `partial`, `conflict`, `inconclusive` e `unavailable`;
- registrar fontes conflitantes separadamente;
- declarar o que falta;
- parar quando a evidência é suficiente ou a lacuna é objetiva.

Critério de conclusão: cada afirmação técnica da resposta pode ser ligada a uma fonte.

## Etapa 10 — decisão, ações e revisão humana

A saída deve distinguir:

- orientar;
- agir;
- escalar;
- pedir informação;
- pedir confirmação;
- sinalizar revisão humana.

Implemente a política fechada:

- leitura é autônoma;
- pedido explícito autorizado pode executar no mesmo fluxo;
- ação inferida, ampliada ou ambígua pede confirmação;
- falta de permissão impede a ação;
- falta de evidência impede a conclusão;
- `requires_human_review=true` bloqueia a liberação, mas não chama a API parceira.

Use checkpointer para retomar uma execução que aguardou confirmação.

Critério de conclusão: cobrir sucesso, `403`, justificativa inválida, ambiguidade e revisão humana.

## Etapa 11 — Logfire

Aprenda trace, span e atributos. Instrumente:

- request raiz;
- chamadas do modelo;
- tools;
- erros;
- política;
- resposta;
- avaliação.

Use:

- `request_id` para atendimento;
- `trace_id` para a execução técnica;
- `experiment_id` para uma rodada do benchmark;
- `case_id` para um item dessa rodada.

Envie somente campos permitidos. Nunca envie chaves, tokens, headers de autorização ou credenciais. Identidades devem ser pseudonimizadas quando o valor literal não for necessário.

Critério de conclusão: abrir um `trace_id` e reconstruir a execução sem consultar logs dispersos.

## Etapa 12 — Pydantic Evals e checks programáticos

Aprenda:

- `Dataset`;
- `Case`;
- task avaliada;
- evaluator;
- relatório;
- experimento;
- separação entre agente e gabarito.

Use somente as colunas atuais dos casos. O runner reúne as fontes existentes sem modificar o schema.

Checks programáticos devem cobrir:

- formato;
- decisão;
- tool e argumentos;
- IDs e permissões;
- justificativa;
- erros tratados;
- chamadas obrigatórias e proibidas;
- repetições e limite de passos.

Critério de conclusão: executar os 17 casos e produzir resultados reproduzíveis sem juiz LLM.

## Etapa 13 — juízes virtuais

Crie juízes separados.

Resultado cego:

- relevância;
- fidelidade;
- honestidade sobre incerteza;
- qualidade da decisão;
- comunicação.

Trajetória:

- estratégia de investigação;
- fundamentação nos retornos reais;
- tratamento de falhas;
- qualidade da parada.

Cada juiz devolve `pass`, `score` e `reason`. Comunicação tem peso candidato de 10%; dimensões críticas não são compensadas por média.

Rode o juiz uma vez e aplique `0.7`, `0.8` e `0.9` aos mesmos scores.

Critério de conclusão: uma resposta correta por acaso passa no resultado cego e falha na trajetória.

## Etapa 14 — rótulos humanos e calibração

Você será a única pessoa avaliadora. Rotule 20–30 saídas sem ver antes a nota do juiz.

Calcule:

- concordância bruta;
- Cohen's kappa;
- falsos aprovados;
- falsos reprovados;
- taxa de revisão humana por limiar.

Krippendorff alpha não faz parte desta entrega.

Casos artificiais mantêm o schema existente e podem variar mensagem, usuário, ativo, modo e trajetória. Use comportamentos representáveis pela API. Timeout ou `500` forçado ficam em testes de componente se o dataset não puder representá-los.

Critério de conclusão: justificar o limiar escolhido e mostrar onde o juiz ainda diverge da referência humana.

## Etapa 15 — experimento e extensões

Congele agente, prompts, rubricas e dataset antes do resultado final.

Compare:

```text
checks programáticos
versus
checks programáticos + juízes virtuais
```

Analise ganho de detecção, falsos resultados, custo e latência. Não conclua além dos dados sintéticos e da cobertura disponível.

Depois do núcleo:

- use Ragas se uma métrica pronta acrescentar sinal;
- use promptfoo para comparar prompts e modelos;
- não adicione ferramenta apenas por disponibilidade.

## Sequência prática de cenários

1. `TKT-CTX-02`: conhecimento e explicação;
2. `TKT-INV-11b`: baseline versus sintoma;
3. `TKT-INV-05`: múltiplas consultas;
4. `TKT-INV-10`: qualidade e requisito;
5. `TKT-INV-08`: conflito;
6. `TKT-EXE-12`: escrita autorizada;
7. `TKT-EXE-14`: `action_high` e justificativa;
8. `TKT-EXE-16`: escalonamento;
9. demais casos e variações.

## Plano de quatro semanas

### Semana 1

- Etapas 0–5;
- API explorada;
- cliente HTTP e modelos Pydantic;
- testes básicos.

### Semana 2

- Etapas 6–10;
- tools;
- grafo;
- evidências;
- políticas e ações.

### Semana 3

- Etapas 11–14;
- Logfire;
- runner;
- checks;
- juízes e rótulos.

### Semana 4

- Etapa 15;
- calibração;
- experimento final;
- análise de limitações;
- atualização do README e demonstração.
