"""Juízes LLM estritamente offline e separados por superfície observável."""

from __future__ import annotations

import json
from typing import Final

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field

from tractian_agent.evaluation.contracts import (
    BenchmarkInput,
    EvaluationModel,
    EvaluationOutput,
    ExpectedCase,
)


BLIND_RESULT_JUDGE_PROMPT_VERSION: Final = "blind-result-judge-v2"
TRAJECTORY_JUDGE_PROMPT_VERSION: Final = "trajectory-judge-v2"
BLIND_RESULT_JUDGE_PROMPT: Final = f"""{BLIND_RESULT_JUDGE_PROMPT_VERSION}
Você é um avaliador offline do resultado de um atendimento industrial. Avalie
somente solicitação, referência e resposta final fornecidas. Você não recebe o
trace e não deve inferir chamadas de ferramenta. Para cada dimensão, devolva
pass, score entre 0 e 1 e motivo curto. Fidelidade, honestidade sobre incerteza
e decisão são dimensões críticas; boa comunicação não compensa falha crítica.
Preencha exatamente o schema plano recebido. Cada campo *_reason deve ter no
máximo 12 palavras e não use markdown, comentários ou blocos adicionais.
O objeto JSON deve conter exatamente estes campos: relevance_passed,
relevance_score, relevance_reason, fidelity_passed, fidelity_score,
fidelity_reason, uncertainty_honesty_passed, uncertainty_honesty_score,
uncertainty_honesty_reason, decision_quality_passed, decision_quality_score,
decision_quality_reason, communication_passed, communication_score e
communication_reason.
""".strip()
TRAJECTORY_JUDGE_PROMPT: Final = f"""{TRAJECTORY_JUDGE_PROMPT_VERSION}
Você é um avaliador offline da trajetória de um atendimento industrial. Compare
as chamadas e falhas observadas com a trajetória de referência sem exigir ordem
rígida quando caminhos equivalentes preservarem segurança. Para cada dimensão,
devolva pass, score entre 0 e 1 e motivo curto. Nenhuma nota altera o atendimento
já concluído e nenhuma dimensão crítica é compensada por média.
Preencha exatamente o schema plano recebido. Cada campo *_reason deve ter no
máximo 12 palavras e não use markdown, comentários ou blocos adicionais.
O objeto JSON deve conter exatamente estes campos:
investigation_strategy_passed, investigation_strategy_score,
investigation_strategy_reason, grounding_passed, grounding_score,
grounding_reason, failure_handling_passed, failure_handling_score,
failure_handling_reason, stopping_quality_passed, stopping_quality_score e
stopping_quality_reason.
""".strip()


class JudgeVerdict(EvaluationModel):
    """Veredito estruturado de uma dimensão da rubrica."""

    passed: bool
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    reason: str = Field(min_length=1, max_length=500, pattern=r"\S")


class BlindResultJudgment(EvaluationModel):
    relevance: JudgeVerdict
    fidelity: JudgeVerdict
    uncertainty_honesty: JudgeVerdict
    decision_quality: JudgeVerdict
    communication: JudgeVerdict


class TrajectoryJudgment(EvaluationModel):
    investigation_strategy: JudgeVerdict
    grounding: JudgeVerdict
    failure_handling: JudgeVerdict
    stopping_quality: JudgeVerdict


class _BlindResultWire(EvaluationModel):
    relevance_passed: bool
    relevance_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    relevance_reason: str = Field(min_length=1, max_length=240, pattern=r"\S")
    fidelity_passed: bool
    fidelity_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    fidelity_reason: str = Field(min_length=1, max_length=240, pattern=r"\S")
    uncertainty_honesty_passed: bool
    uncertainty_honesty_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    uncertainty_honesty_reason: str = Field(
        min_length=1,
        max_length=240,
        pattern=r"\S",
    )
    decision_quality_passed: bool
    decision_quality_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    decision_quality_reason: str = Field(
        min_length=1,
        max_length=240,
        pattern=r"\S",
    )
    communication_passed: bool
    communication_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    communication_reason: str = Field(min_length=1, max_length=240, pattern=r"\S")


class _TrajectoryWire(EvaluationModel):
    investigation_strategy_passed: bool
    investigation_strategy_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    investigation_strategy_reason: str = Field(
        min_length=1,
        max_length=240,
        pattern=r"\S",
    )
    grounding_passed: bool
    grounding_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    grounding_reason: str = Field(min_length=1, max_length=240, pattern=r"\S")
    failure_handling_passed: bool
    failure_handling_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    failure_handling_reason: str = Field(
        min_length=1,
        max_length=240,
        pattern=r"\S",
    )
    stopping_quality_passed: bool
    stopping_quality_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    stopping_quality_reason: str = Field(
        min_length=1,
        max_length=240,
        pattern=r"\S",
    )


class ThresholdDecision(EvaluationModel):
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    weighted_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    passed: bool
    failed_critical_dimensions: tuple[str, ...] = ()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _verdict_from_wire(wire: EvaluationModel, dimension: str) -> JudgeVerdict:
    return JudgeVerdict(
        passed=getattr(wire, f"{dimension}_passed"),
        score=getattr(wire, f"{dimension}_score"),
        reason=getattr(wire, f"{dimension}_reason"),
    )


class BlindResultJudge:
    """Juiz que deliberadamente não possui campo nem parâmetro de trace."""

    def __init__(self, model: BaseChatModel) -> None:
        self._structured_model = model.with_structured_output(_BlindResultWire)

    async def ainvoke(
        self,
        benchmark: BenchmarkInput,
        observed: EvaluationOutput,
        reference: ExpectedCase,
    ) -> BlindResultJudgment:
        context = {
            "request": {
                "ticket_id": benchmark.ticket_id,
                "message": benchmark.message,
            },
            "reference": {
                "root_question": reference.root_question,
                "mode": reference.mode,
                "facts": [item.note for item in reference.expected_path],
            },
            "result": {
                "decision": observed.decision,
                "message": observed.message,
                "evidence_ids": list(observed.evidence_ids),
                "limitation_refs": list(observed.limitation_refs),
            },
        }
        result = await self._structured_model.ainvoke(
            [
                SystemMessage(content=BLIND_RESULT_JUDGE_PROMPT),
                HumanMessage(content=_canonical_json(context)),
            ]
        )
        wire = _BlindResultWire.model_validate(result)
        return BlindResultJudgment(
            relevance=_verdict_from_wire(wire, "relevance"),
            fidelity=_verdict_from_wire(wire, "fidelity"),
            uncertainty_honesty=_verdict_from_wire(wire, "uncertainty_honesty"),
            decision_quality=_verdict_from_wire(wire, "decision_quality"),
            communication=_verdict_from_wire(wire, "communication"),
        )


class TrajectoryJudge:
    """Juiz que recebe o trace persistido, inclusive falhas sanitizadas."""

    def __init__(self, model: BaseChatModel) -> None:
        self._structured_model = model.with_structured_output(_TrajectoryWire)

    async def ainvoke(
        self,
        benchmark: BenchmarkInput,
        observed: EvaluationOutput,
        reference: ExpectedCase,
    ) -> TrajectoryJudgment:
        context = {
            "request": {
                "ticket_id": benchmark.ticket_id,
                "message": benchmark.message,
            },
            "expected_path": [
                item.model_dump(mode="json") for item in reference.expected_path
            ],
            "observed_steps": [item.model_dump(mode="json") for item in observed.steps],
            "failure_codes": list(observed.failure_codes),
            "terminal": {
                "decision": observed.decision,
                "step_count": observed.step_count,
                "step_limit": observed.step_limit,
            },
        }
        result = await self._structured_model.ainvoke(
            [
                SystemMessage(content=TRAJECTORY_JUDGE_PROMPT),
                HumanMessage(content=_canonical_json(context)),
            ]
        )
        wire = _TrajectoryWire.model_validate(result)
        return TrajectoryJudgment(
            investigation_strategy=_verdict_from_wire(
                wire,
                "investigation_strategy",
            ),
            grounding=_verdict_from_wire(wire, "grounding"),
            failure_handling=_verdict_from_wire(wire, "failure_handling"),
            stopping_quality=_verdict_from_wire(wire, "stopping_quality"),
        )


def apply_thresholds(
    judgment: BlindResultJudgment | TrajectoryJudgment,
    *,
    thresholds: tuple[float, ...],
) -> tuple[ThresholdDecision, ...]:
    """Reaplica cortes aos mesmos scores, sem nova chamada ao modelo."""

    if not thresholds:
        raise ValueError("ao menos um limiar é obrigatório")
    if isinstance(judgment, BlindResultJudgment):
        weights = {
            "relevance": 0.225,
            "fidelity": 0.225,
            "uncertainty_honesty": 0.225,
            "decision_quality": 0.225,
            "communication": 0.1,
        }
        critical = (
            "relevance",
            "fidelity",
            "uncertainty_honesty",
            "decision_quality",
        )
    else:
        weights = {
            "investigation_strategy": 0.25,
            "grounding": 0.25,
            "failure_handling": 0.25,
            "stopping_quality": 0.25,
        }
        critical = tuple(weights)
    verdicts = {name: getattr(judgment, name) for name in type(judgment).model_fields}
    weighted_score = sum(
        verdicts[name].score * weight for name, weight in weights.items()
    )
    decisions = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("limiares devem estar entre 0 e 1")
        failed = tuple(
            name
            for name in critical
            if not verdicts[name].passed or verdicts[name].score < threshold
        )
        decisions.append(
            ThresholdDecision(
                threshold=threshold,
                weighted_score=weighted_score,
                passed=weighted_score >= threshold and not failed,
                failed_critical_dimensions=failed,
            )
        )
    return tuple(decisions)
