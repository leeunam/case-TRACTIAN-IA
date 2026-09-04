# Suporte Industrial com Agentes

Este contexto descreve a linguagem do atendimento industrial simulado e de sua avaliação. Ele existe para que solicitações, evidências, ações e resultados sejam discutidos sem misturar conceitos de domínio com detalhes da implementação.

## Estrutura industrial

**Empresa**:
Organização industrial à qual pertencem pessoas usuárias e ativos.
_Evitar_: Cliente, conta

**Pessoa usuária**:
Pessoa que abre um caso e cujo perfil determina as permissões disponíveis naquele atendimento.
_Evitar_: Usuário autenticado, operador, quando o papel específico não for conhecido

**Permissão**:
Capacidade concedida à pessoa usuária para leitura, ação de baixo impacto, ação de alto impacto ou escalonamento humano.
_Evitar_: Papel, autorização, como sinônimos exatos

**Ativo**:
Máquina industrial monitorada, identificada por configuração técnica, criticidade e posição na planta.
_Evitar_: Equipamento, máquina, quando a referência for ao recurso cadastrado

**Ponto de medição**:
Local de um ativo no qual um sensor coleta sinais de condição.
_Evitar_: Sensor, ativo

**Criticidade**:
Importância operacional de um ativo para fins de priorização.
_Evitar_: Severidade

## Condição e diagnóstico

**Baseline**:
Referência do estado normal aprendida para um ativo e ponto de medição. Seu estado é `learning`, `established` ou `invalidated`.
_Evitar_: Limite fixo, norma ISO, média global

**Detecção por desvio**:
Modo de detecção que compara o sinal atual com um baseline `established`.
_Evitar_: Detecção por baseline, quando puder ser confundida com o próprio baseline

**Detecção sintomática**:
Modo de detecção em que a presença de uma assinatura característica sustenta a falha sem depender de baseline.
_Evitar_: Detecção sem dados, detecção por desvio

**Análise**:
Diagnóstico produzido por um modelo ou especialista, contendo tipo, severidade, confiança, evidências, limitações e estado no tempo.
_Evitar_: Insight, relatório, diagnóstico, quando se referirem ao recurso da plataforma

**Evidência**:
Observação mensurável que sustenta ou enfraquece uma hipótese de falha.
_Evitar_: Confiança, conclusão

**Limitação**:
Condição conhecida que reduz o alcance ou a confiabilidade de uma análise.
_Evitar_: Erro, indisponibilidade, como sinônimos genéricos

**Severidade**:
Intensidade da condição indicada por uma análise.
_Evitar_: Criticidade, confiança

**Confiança**:
Grau declarado de certeza de uma análise, que deve ser confrontado com a qualidade dos dados e os requisitos do modelo.
_Evitar_: Probabilidade de falha, qualidade do sinal

**Série RMS**:
Sequência temporal da velocidade global de vibração de um ponto, expressa em mm/s neste projeto.
_Evitar_: Espectro, waveform

**Limiar de alarme**:
Valor derivado da referência mais a tolerância do baseline do próprio ativo.
_Evitar_: Limite ISO, tabela fixa

**Espectro**:
Representação simplificada das componentes de frequência do sinal, usada para relacionar picos a assinaturas de falha.
_Evitar_: RMS, série temporal

**Qualidade dos dados**:
Conjunto formado por completude, frescor, relação sinal-ruído e indicação de obsolescência dos dados.
_Evitar_: Confiança da análise

**Cobertura do modelo**:
Declaração de quais tipos de máquina o modelo suporta e se consegue aprender baseline para cada tipo.
_Evitar_: Qualidade do modelo, estado de processamento

## Atendimento e ações

**Caso**:
Solicitação de atendimento que reúne mensagem, empresa, pessoa usuária e ativo central.
_Evitar_: Cenário, ticket, quando se referirem ao objeto executado pelo agente

**Modalidade de atendimento**:
Natureza da solicitação: contextualizar, investigar ou executar.
_Evitar_: Decisão do agente

**Orientar**:
Decisão de explicar ou recomendar sem alterar a plataforma nem encaminhar o caso.
_Evitar_: Contextualizar, quando a referência for à decisão final

**Agir**:
Decisão de executar uma operação permitida e justificada na plataforma.
_Evitar_: Executar, quando a referência for à modalidade do caso

**Análise especializada**:
Revisão técnica interna solicitada para aprofundar uma análise que ainda pode ser resolvida remotamente.
_Evitar_: Escalonamento humano, inspeção de campo

**Escalonamento humano**:
Encaminhamento do caso para uma pessoa quando ele ultrapassa o atendimento remoto.
_Evitar_: Análise especializada

**Revisão humana**:
Inspeção de uma resposta ou ação proposta antes de sua liberação, usada quando a evidência ou a avaliação de segurança não é suficiente.
_Evitar_: Escalonamento humano, análise especializada

**Sinalização de revisão**:
Classificação interna que impede a liberação e encaminha uma resposta ou proposta para revisão humana, sem executar uma ação industrial na plataforma.
_Evitar_: Escalonamento humano, aprovação de ação

**Aprovação de ação**:
Autorização explícita de uma pessoa com permissão para uma escrita ou acionamento de escopo definido; pode estar no pedido original ou em uma confirmação posterior.
_Evitar_: Revisão humana, permissão de acesso

**Incerteza declarada**:
Registro explícito do que não pôde ser concluído, das evidências ausentes ou conflitantes e do efeito dessa lacuna sobre a decisão.
_Evitar_: Confiança do modelo, pontuação do avaliador

**Reprocesso**:
Solicitação para executar novamente uma análise existente após mudança de contexto ou de dados.
_Evitar_: Retreinamento

**Retreinamento**:
Solicitação de mudança no aprendizado de um modelo, sustentada por evidência de erro sistemático.
_Evitar_: Reprocesso, reaprendizado do baseline

## Avaliação

**Modo de resposta**:
Condição do envelope de uma consulta: `complete`, `partial`, `inconclusive`, `conflict` ou `unavailable`.
_Evitar_: Status da análise

**Status da análise**:
Estado temporal ou operacional de uma análise: `current`, `stale`, `pending` ou `inconclusive`.
_Evitar_: Modo de resposta

**Cenário**:
Item de benchmark que combina objetivo, contexto, política, trajetória de referência, resolução esperada e critérios de sucesso.
_Evitar_: Caso

**Trajetória de referência**:
Sequência esperada de consultas e ações de um cenário, aceita como referência sem exigir ordem rígida quando caminhos equivalentes preservam a política.
_Evitar_: Script obrigatório, resposta do agente

**Trace**:
Registro cronológico de entradas, chamadas de ferramenta, argumentos, observações, decisão e resposta final de uma execução.
_Evitar_: Log bruto, trajetória de referência

**Verificação programática**:
Regra determinística que aprova ou reprova uma propriedade observável da execução, como schema, permissão, argumento ou chamada realizada.
_Evitar_: Juiz virtual, revisão humana

**Juiz virtual**:
Modelo separado que aplica uma rubrica a uma saída ou trajetória e devolve veredito, pontuação e justificativa.
_Evitar_: Agente avaliado, verdade de referência

**Avaliador de resultado**:
Componente que julga a decisão e a resposta final sem acesso ao trace da execução.
_Evitar_: Avaliador de trajetória, verificação programática

**Avaliador de trajetória**:
Componente que julga a sequência de tools, observações, erros, evidências e decisões registrada no trace.
_Evitar_: Avaliador de resultado, trajetória de referência

**Métrica de concordância**:
Cálculo aplicado a rótulos já produzidos para medir quanto avaliadores humanos ou virtuais concordam entre si; não produz o rótulo original.
_Evitar_: Juiz virtual, rubrica, pontuação do avaliador

**Referência humana individual**:
Conjunto de rótulos produzido pela única pessoa avaliadora nesta versão, usado para calibrar o juiz virtual sem representar consenso entre especialistas.
_Evitar_: Concordância entre humanos, verdade absoluta

**Golden set**:
Conjunto versionado de casos com referência resolvida, fatos obrigatórios, decisão esperada e rótulos humanos, reservado para calibrar ou validar os avaliadores.
_Evitar_: Lista bruta de tickets, qualquer saída antiga

**Proveniência da evidência**:
Vínculo verificável entre uma afirmação da resposta e a observação registrada que a sustenta.
_Evitar_: Semelhança textual, confiança do modelo

**Porta de segurança**:
Decisão anterior à liberação que permite responder, solicita aprovação ou encaminha para revisão humana conforme evidência, permissão e risco.
_Evitar_: Avaliação offline, juiz virtual

**Avaliação offline**:
Execução controlada sobre um dataset para medir o sistema e comparar versões, sem alterar retroativamente a resposta avaliada.
_Evitar_: Porta de segurança, monitoramento de produção

**Pontuação do avaliador**:
Resultado de uma rubrica aplicado a uma dimensão da execução. Não representa probabilidade de verdade sem um processo adicional de calibração.
_Evitar_: Percentual de certeza, confiança da análise

**Limiar de revisão**:
Valor de corte calibrado por dimensão crítica que transforma uma pontuação do avaliador em liberação ou encaminhamento para revisão humana.
_Evitar_: Percentual de certeza, média geral

**Verdade de referência**:
Rótulo ou resultado aceito para comparação, proveniente do gabarito do cenário ou de julgamento humano resolvido.
_Evitar_: Resposta do juiz virtual, opinião isolada

## Central de demonstração

**Persona simulada**:
Identidade escolhida na central para demonstrar visibilidade e permissões; o backend a resolve em fixtures confiáveis e nunca aceita permissões enviadas pelo navegador.
_Evitar_: Usuário autenticado de produção

**Pedido de decisão**:
Registro persistente de uma escolha humana pendente, ligado a um caso, público, escopo exato, validade e operações permitidas.
_Evitar_: Intenção de escrita, mensagem do Slack

**Autorização delegada**:
Atestado consumível uma vez que liga decisor, empresa, permissão, horário e digest do escopo; não troca a pessoa dona do thread.
_Evitar_: Troca de identidade, revisão técnica

**Outbox**:
Fila transacional de notificações externas persistidas antes da tentativa de entrega.
_Evitar_: Aprovação, fila de execução do agente

**Notificação Slack**:
Aviso sanitizado com link para a central. Nunca resolve a decisão nem executa uma ação.
_Evitar_: Decisão Slack, comando de aprovação
