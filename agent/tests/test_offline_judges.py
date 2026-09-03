import asyncio
import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import PrivateAttr

from tractian_agent.evaluation.contracts import (
    BenchmarkInput,
    EvaluationOutput,
    ExpectedCase,
    ObservedStep,
)
from tractian_agent.evaluation.judges import (
    BlindResultJudgment,
    BlindResultJudge,
    JudgeVerdict,
    TrajectoryJudgment,
    TrajectoryJudge,
    apply_thresholds,
)


class _RecordingJudgeModel(BaseChatModel):
    response: object
    _messages: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _schemas: list[object] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording-judge-model"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise AssertionError("juízes devem usar saída estruturada")

    def bind_tools(self, *args: Any, **kwargs: Any) -> RunnableLambda:
        raise AssertionError("juízes offline não recebem tools")

    def with_structured_output(
        self,
        schema: object,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> RunnableLambda:
        assert include_raw is False
        self._schemas.append(schema)

        async def invoke(messages: list[BaseMessage]) -> object:
            self._messages.append(list(messages))
            return self.response

        return RunnableLambda(invoke)


def _verdict(score: float, *, passed: bool = True) -> JudgeVerdict:
    return JudgeVerdict(passed=passed, score=score, reason="rubrica aplicada")


def _input() -> BenchmarkInput:
    return BenchmarkInput(
        id="case_tkt_ctx_02",
        ticket_id="TKT-CTX-02",
        company_id="comp_aurora",
        user_id="usr_lucas",
        asset_id="asset_B204",
        message="O que significa BPFO?",
    )


def _reference() -> ExpectedCase:
    return ExpectedCase(
        id="case_tkt_ctx_02",
        ticket_id="TKT-CTX-02",
        root_question="Definir BPFO sem concluir falha atual.",
        mode="partial",
        expected_path=(
            {
                "step": "GET /knowledge/search?q=BPFO",
                "note": "definição do glossário",
            },
        ),
    )


def _output() -> EvaluationOutput:
    return EvaluationOutput(
        case_id="case_tkt_ctx_02",
        ticket_id="TKT-CTX-02",
        decision="guide",
        message="BPFO é uma frequência característica; isoladamente não prova falha.",
        permissions=("read",),
        steps=(
            ObservedStep(
                ordinal=1,
                call_id="call_secret_trace",
                tool_name="search_knowledge",
                arguments={"query": "BPFO"},
                method="GET",
                resource="/secret-trace",
                outcome="success",
            ),
        ),
        step_limit=12,
    )


def _flat_wire(judgment) -> dict[str, object]:
    values: dict[str, object] = {}
    for dimension in type(judgment).model_fields:
        verdict = getattr(judgment, dimension)
        values[f"{dimension}_passed"] = verdict.passed
        values[f"{dimension}_score"] = verdict.score
        values[f"{dimension}_reason"] = verdict.reason
    return values


def test_blind_result_judge_never_receives_the_execution_trace() -> None:
    response = BlindResultJudgment(
        relevance=_verdict(0.9),
        fidelity=_verdict(0.9),
        uncertainty_honesty=_verdict(0.9),
        decision_quality=_verdict(0.9),
        communication=_verdict(0.9),
    )
    model = _RecordingJudgeModel(response=_flat_wire(response))

    result = asyncio.run(
        BlindResultJudge(model).ainvoke(_input(), _output(), _reference())
    )

    assert result == response
    assert model._schemas[0].__name__ == "_BlindResultWire"
    wire = json.dumps(
        [message.content for message in model._messages[0]], ensure_ascii=False
    )
    assert "/secret-trace" not in wire
    assert "call_secret_trace" not in wire
    assert "steps" not in wire


def test_trajectory_judge_receives_calls_failures_and_expected_path() -> None:
    response = TrajectoryJudgment(
        investigation_strategy=_verdict(0.9),
        grounding=_verdict(0.9),
        failure_handling=_verdict(0.9),
        stopping_quality=_verdict(0.9),
    )
    model = _RecordingJudgeModel(response=_flat_wire(response))

    result = asyncio.run(
        TrajectoryJudge(model).ainvoke(_input(), _output(), _reference())
    )

    assert result == response
    assert model._schemas[0].__name__ == "_TrajectoryWire"
    wire = json.dumps(
        [message.content for message in model._messages[0]], ensure_ascii=False
    )
    assert "/secret-trace" in wire
    assert "GET /knowledge/search?q=BPFO" in wire


def test_same_judgment_scores_are_reused_for_all_thresholds() -> None:
    judgment = BlindResultJudgment(
        relevance=_verdict(0.95),
        fidelity=_verdict(0.85),
        uncertainty_honesty=_verdict(0.92),
        decision_quality=_verdict(0.88),
        communication=_verdict(0.99),
    )

    decisions = apply_thresholds(judgment, thresholds=(0.7, 0.8, 0.9))

    assert [item.threshold for item in decisions] == [0.7, 0.8, 0.9]
    assert [item.passed for item in decisions] == [True, True, False]
    assert decisions[0].weighted_score == decisions[1].weighted_score
    assert decisions[1].weighted_score == decisions[2].weighted_score


def test_correct_by_chance_passes_blind_result_but_fails_trajectory() -> None:
    blind = BlindResultJudgment(
        relevance=_verdict(0.95),
        fidelity=_verdict(0.95),
        uncertainty_honesty=_verdict(0.95),
        decision_quality=_verdict(0.95),
        communication=_verdict(0.95),
    )
    trajectory = TrajectoryJudgment(
        investigation_strategy=_verdict(0.2, passed=False),
        grounding=_verdict(0.1, passed=False),
        failure_handling=_verdict(0.9),
        stopping_quality=_verdict(0.3, passed=False),
    )

    assert apply_thresholds(blind, thresholds=(0.8,))[0].passed is True
    assert apply_thresholds(trajectory, thresholds=(0.8,))[0].passed is False
