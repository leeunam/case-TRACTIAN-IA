# Schema de dados (parquet)

Dados sintéticos que populam a API. Tudo anonimizado, sem PII. Formato **parquet** (didático,
compacto, leitura fácil com pandas/duckdb). Os arquivos ficam em `data/`; a API carrega apenas a
allowlist de tabelas operacionais. `cases.parquet`, que contém o gabarito, fica fora do runtime, e
os chamados entram somente pelo pacote sanitizado `agent-input/cases.json`. `seed.json` controla os
modos de variação das respostas.

Os dados foram desenhados para contextualizar os chamados, inclusive com observações conflitantes
ou incompletas. M-605 não possui a banda necessária para concluir falha elétrica, e S-420 combina
baseline invalidated com hipóteses espectrais divergentes; esses casos exigem incerteza, não uma
conclusão categórica. Nos demais casos, o ativo reúne dados relacionados à pergunta raiz (ex.: o
ativo que "quebrou sem aviso" tem uma janela ausente antes da quebra e um modelo sem cobertura
para baixa rotação).

## Tabelas

### `companies.parquet`
| coluna       | tipo     | descrição                          |
| :----------- | :------- | :--------------------------------- |
| id           | string   | `comp_forja_br`                    |
| name         | string   | Forja Brasil                       |
| segment      | string   | metalurgia/papel/cimento/...       |
| timezone     | string   | America/Sao_Paulo                  |

### `users.parquet`
| coluna        | tipo    | descrição                                          |
| :------------ | :------ | :------------------------------------------------- |
| id            | string  | `usr_ana`                                          |
| name          | string  | Ana Mantovani                                      |
| role          | string  | operator/mechanic/reliability_analyst/...          |
| permissions   | string  | JSON array (`["read","action_high","escalate"]`)   |
| company_id    | string  | FK companies.id                                    |

### `assets.parquet`
| coluna            | tipo    | descrição                                                       |
| :---------------- | :------ | :-------------------------------------------------------------- |
| id                | string  | `asset_M101`                                                    |
| name              | string  | Motor principal da forja                                        |
| company_id        | string  | FK                                                              |
| criticality       | string  | low/medium/high/critical                                        |
| plant             | string  | Planta 1                                                        |
| line              | string  | Linha de forjamento                                             |
| parent_asset_id   | string  | nullable; hierarquia                                            |
| machine_type      | string  | compressor/fan/gearbox/mill/motor_dc/motor_induction/pump/spindle    |
| rotation_rpm      | int64   | 1780                                                            |
| bearing_pn        | string  | NU 310 (nullable)                                               |
| bpfo_hz           | double  | frequência de defeito externo (nullable)                        |
| bpfi_hz           | double  | interno (nullable)                                              |
| bsf_hz            | double  | ball spin (nullable)                                            |
| ftf_hz            | double  | cage (nullable)                                                 |
| line_frequency_hz | double  | 60 (motores; nullable)                                          |
| sensor_status     | string  | online/offline/degraded (estado do ponto principal)            |

### `points.parquet`
| coluna          | tipo   | descrição                                   |
| :-------------- | :----- | :------------------------------------------ |
| id              | string | `pt_M101_de`                                |
| asset_id        | string | FK                                          |
| location        | string | DE/NDE/axial/horizontal/vertical            |
| sensor_status   | string | online/offline/degraded                     |

### `analyses.parquet`
| coluna                    | tipo    | descrição                                              |
| :------------------------ | :------ | :----------------------------------------------------- |
| id                        | string  | `an_9901`                                              |
| asset_id                  | string  | FK                                                     |
| point_id                  | string  | FK                                                     |
| type                      | string  | imbalance/misalignment/bearing_fault/electrical_fault/looseness/lubrication |
| detection_mode            | string  | baseline/symptom                                       |
| severity                  | string  | none/low/medium/high/critical                          |
| confidence                | double  | 0.0–1.0                                                |
| baseline_state_at_detection | string | learning/established/invalidated/not_applicable     |
| evidence                  | string  | JSON array de {metric,value,reference?,note}           |
| limitations               | string  | JSON array (`["low_signal_quality","baseline_learning",...]`) |
| model_version             | string  | 3.2.1                                                  |
| created_at                | string  | ISO datetime                                           |
| status                    | string  | current/stale/pending/inconclusive                     |

### `baselines.parquet`
| coluna              | tipo    | descrição                                                       |
| :------------------ | :------ | :-------------------------------------------------------------- |
| id                  | string  | `bs_M101_de`                                                    |
| asset_id            | string  | FK                                                              |
| point_id            | string  | FK                                                              |
| state               | string  | learning/established/invalidated                                |
| detection_mode      | string  | baseline/symptom (symptom = não usa baseline, ex.: lubrificação)|
| learnable           | boolean | se o modelo consegue aprender baseline p/ este ativo/tipo       |
| established_at      | string  | ISO datetime, nullable                                          |
| invalidated_at      | string  | ISO datetime, nullable                                          |
| invalidation_reason | string  | maintenance_intervention/config_change (nullable)              |
| features            | string  | JSON array de {feature, reference, tolerance}                   |

### `rms.parquet` (long)
| coluna   | tipo   | descrição                                   |
| :------- | :----- | :------------------------------------------ |
| asset_id | string | FK                                          |
| point_id | string | FK                                          |
| ts       | string | ISO datetime (amostragem horária/diária)    |
| value    | double | mm/s                                        |

> Os limiares de alarme **não** vêm de norma ISO; são derivados do baseline aprendido
> (`baselines.features` → `reference + tolerance`). Falhas sintomáticas (lubrificação) não têm
> baseline (`detection_mode=symptom`, `learnable=false`).

### `spectra.parquet`
| coluna          | tipo   | descrição                                         |
| :-------------- | :----- | :------------------------------------------------ |
| asset_id        | string | FK                                                |
| point_id        | string | FK                                                |
| collected_at    | string | ISO datetime                                      |
| peaks           | string | JSON array de {freq_hz, amplitude_mm_s, note?}    |
| bands_missing   | string | JSON array de bandas ausentes na medição; independente do modo do envelope |

### `data_quality.parquet`
| coluna             | tipo    | descrição                              |
| :----------------- | :------ | :------------------------------------- |
| asset_id           | string  | FK                                     |
| point_id           | string  | FK                                     |
| completeness       | double  | 0.0–1.0                                |
| freshness_minutes | integer | minutos desde última amostra válida    |
| snr_db             | double  | relação sinal-ruído                    |
| staleness_flag     | boolean | dados obsoletos                        |

### `models.parquet`
| coluna            | tipo    | descrição                                            |
| :---------------- | :------ | :--------------------------------------------------- |
| id                | string  | `mdl_vib_v3`                                         |
| version           | string  | 3.2.1                                                |
| coverage          | string  | JSON array de {machine_type, supported, can_learn_baseline?, note?} |
| min_completeness  | double  | requisito                                             |
| min_snr_db        | double  | requisito                                             |
| min_rotation_rpm  | double  | nullable                                              |
| processing_state  | string  | idle/running/pending/delayed/failed                   |
| last_run_at       | string  | ISO datetime, nullable                                |

> `coverage.can_learn_baseline` indica se o modelo consegue aprender baseline para aquele tipo
> (false → só detecção sintomática, ex.: certos subtipos).

### `knowledge.parquet`
| coluna | tipo    | descrição                            |
| :----- | :------ | :----------------------------------- |
| id     | string  | `kb_proc_001`                        |
| type   | string  | procedure/glossary/guidance          |
| title  | string  | Procedimento de troca de rolamento   |
| body   | string  | markdown                             |
| tags   | string  | JSON array                           |

### `cases.parquet` (unifica chamados + contexto)
| coluna          | tipo    | descrição                                              |
| :-------------- | :------ | :----------------------------------------------------- |
| id              | string  | `case_tkt_inv_04`                                      |
| ticket_id       | string  | TKT-INV-04                                             |
| company_id      | string  | FK                                                     |
| user_id         | string  | FK (quem abriu)                                        |
| asset_id        | string  | FK (ativo central; nullable p/ chamados só de conhecimento) |
| message         | string  | texto do cliente                                       |
| root_question   | string  | pergunta raiz do analista                              |
| mode            | string  | rótulo esperado: cinco modos do envelope ou `pending`/`stale`, derivados do status do cenário |
| expected_path   | string  | JSON array de passos (referência p/ cenários)          |

## `seed.json` (modos de resposta)

Controla overrides por recurso e categoria e os pesos usados pelo hash determinístico. Este é o
conteúdo vigente:

```json
{
  "overrides": {
    "asset_G501": {
      "analyses": "inconclusive",
      "rms": "unavailable",
      "data_quality": "partial",
      "baseline": "partial"
    },
    "asset_C710": {
      "rms": "complete"
    },
    "asset_S420": {
      "analyses": "conflict"
    },
    "asset_M208": {
      "analyses": "partial"
    },
    "asset_M605": {
      "spectrum": "partial"
    },
    "asset_V301": {
      "data_quality": "partial"
    },
    "asset_M205": {
      "analyses": "conflict"
    }
  },
  "distribution": {
    "complete": 0.60, "partial": 0.15, "inconclusive": 0.10,
    "conflict": 0.08, "unavailable": 0.07
  }
}
```

Sem `seed`, o servidor deriva um modo estável de `noseed + recurso + categoria` e dos pesos de
`distribution`; ele não sorteia de novo a cada chamada. Uma semente comum (`seed=<x>`) troca
`noseed` pelo valor informado e continua determinística. Duas sentinelas são exceções:
`seed=complete` força o envelope completo e `seed=degraded` força o parcial. Os overrides fixos
vencem qualquer seed. O modo altera o envelope ou omite campos definidos pela API, mas não cria
outra linha, outro diagnóstico ou outra medição.

## Mapeamento chamado → dados (sanity check)

| Ticket   | Ativo     | Por que os dados sustentam a pergunta raiz                                     |
| :------- | :-------- | :------------------------------------------------------------------------------ |
| INV-04   | G-501     | `rms` indisponível e dados com gap; `baselines.state = learning`; gearbox tem suporte apenas sintomático (`can_learn_baseline=false`); `analyses` inconclusive |
| INV-05   | C-710     | `rms` com tendência de subida; `baselines.state = established` mas `models.processing_state = delayed`; `analyses.status = pending` |
| INV-06   | S-420     | `baselines.state = invalidated` torna a referência antiga; ambas as análises ocorreram após essa invalidação e os picos espectrais sustentam hipóteses divergentes, sem provar falso positivo |
| INV-07   | M-605     | série RMS aproximadamente estável (`max=2,158 < alarm_threshold=2,7`) não corrobora o salto relatado; `an_9910` registra RMS 2,7 e diverge da série; espectro parcial omite a banda elétrica de 2x f-linha |
| INV-08   | M-205     | em 240 rpm, o pico de 8 Hz representa 2× e sustenta `misalignment`; o de 2 Hz representa 0,5× e sustenta `looseness`, preservando o conflito |
| INV-09   | B-204     | a análise foi criada com `baseline_state_at_detection=established`, antes da manutenção, e ficou `stale`; o baseline atual está `invalidated`, a série pós-intervenção caiu abaixo da referência histórica e `data_quality.staleness_flag = false` |
| INV-10   | V-301     | `data_quality` baixo + `analyses.confidence` alta (tensão)                     |
| INV-11   | M-102     | `models.coverage` suporta tipo mas `can_learn_baseline=false` p/ motor DC       |
| INV-11b  | M-208     | `baselines.state = learning` (ativo novo) mas `analyses.detection_mode = symptom` (lubrificação) válida |
| EXE-12   | B-204     | pós-intervenção: `rms` caiu, a análise anterior ficou stale e o baseline passou a invalidated → gatilho de reprocesso/reaprendizado |
