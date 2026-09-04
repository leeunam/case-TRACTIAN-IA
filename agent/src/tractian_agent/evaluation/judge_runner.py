"""Execução repetida e análise de estabilidade dos dois juízes offline."""

from __future__ import annotations

from collections import defaultdict
import asyncio
from time import perf_counter
from typing import Protocol

from pydantic import Field

from tractian_agent.evaluation.contracts import (
    BenchmarkInput,
    EvaluationModel,
    EvaluationOutput,
    ExpectedCase,
)
from tractian_agent.evaluation.judges import (
    BlindResultJudgment,
    ThresholdDecision,
    TrajectoryJudgment,
    apply_thresholds,
)
from tractian_agent.evaluation.runner import CompletedExecutionBatch


class BlindJudgeProtocol(Protocol):
    async def ainvoke(
        self,
        benchmark: BenchmarkInput,
        observed: EvaluationOutput,
        reference: ExpectedCase,
    ) -> BlindResultJudgment: ...


class TrajectoryJudgeProtocol(Protocol):
    async def ainvoke(
        self,
        benchmark: BenchmarkInput,
        observed: EvaluationOutput,
        reference: ExpectedCase,
    ) -> TrajectoryJudgment: ...


class JudgeCaseReport(EvaluationModel):
    run_id: str = Field(min_length=1, pattern=r"\S")
    case_id: str = Field(min_length=1, pattern=r"^case_[a-z0-9_]+$")
    blind_result: BlindResultJudgment
    trajectory: TrajectoryJudgment
    blind_thresholds: tuple[ThresholdDecision, ...]
    trajectory_thresholds: tuple[ThresholdDecision, ...]
    duration_ms: float = Field(ge=0, allow_inf_nan=False)


class JudgeStability(EvaluationModel):
    case_id: str = Field(min_length=1, pattern=r"^case_[a-z0-9_]+$")
    repetitions: int = Field(ge=1, strict=True)
    stable: bool
    max_score_delta: float = Field(ge=0, le=1, allow_inf_nan=False)


class OfflineJudgeReport(EvaluationModel):
    version: str = "offline-judges-v1"
    provider: str = Field(min_length=1, pattern=r"^\S+$")
    blind_model_id: str = Field(min_length=1, pattern=r"^\S+$")
    trajectory_model_id: str = Field(min_length=1, pattern=r"^\S+$")
    thresholds: tuple[float, ...]
    cases: tuple[JudgeCaseReport, ...]
    stability: tuple[JudgeStability, ...]


class JudgeInputRecord(EvaluationModel):
    run_id: str = Field(min_length=1, pattern=r"\S")
    benchmark: BenchmarkInput
    observed: EvaluationOutput
    reference: ExpectedCase


def conservative_calibration_score(
    blind: BlindResultJudgment,
    trajectory: TrajectoryJudgment,
) -> float:
    """Reduz as dimensões críticas sem permitir compensação pela comunicação."""

    return min(
        blind.relevance.score,
        blind.fidelity.score,
        blind.uncertainty_honesty.score,
        blind.decision_quality.score,
        trajectory.investigation_strategy.score,
        trajectory.grounding.score,
        trajectory.failure_handling.score,
        trajectory.stopping_quality.score,
    )


def _scores(result: JudgeCaseReport) -> tuple[float, ...]:
    return tuple(
        verdict.score
        for judgment in (result.blind_result, result.trajectory)
        for verdict in (
            getattr(judgment, field_name) for field_name in type(judgment).model_fields
        )
    )


async def run_offline_judges(
    batch: CompletedExecutionBatch,
    *,
    blind_result_judge: BlindJudgeProtocol,
    trajectory_judge: TrajectoryJudgeProtocol,
    thresholds: tuple[float, ...],
    provider: str = "unspecified",
    blind_model_id: str = "unspecified",
    trajectory_model_id: str = "unspecified",
    pacing_seconds: float = 0.0,
) -> OfflineJudgeReport:
    """Avalia somente saídas concluídas; não possui referência ao grafo do agente."""

    if batch.execution_report.failures:
        raise ValueError("juízes não podem ocultar execuções que falharam")
    records = tuple(
        JudgeInputRecord(
            run_id=result.name,
            benchmark=result.inputs,
            observed=result.output,
            reference=batch.references[result.inputs.id],
        )
        for result in batch.execution_report.cases
    )
    return await run_offline_judge_records(
        records,
        blind_result_judge=blind_result_judge,
        trajectory_judge=trajectory_judge,
        thresholds=thresholds,
        provider=provider,
        blind_model_id=blind_model_id,
        trajectory_model_id=trajectory_model_id,
        pacing_seconds=pacing_seconds,
    )


async def run_offline_judge_records(
    records: tuple[JudgeInputRecord, ...],
    *,
    blind_result_judge: BlindJudgeProtocol,
    trajectory_judge: TrajectoryJudgeProtocol,
    thresholds: tuple[float, ...],
    provider: str = "unspecified",
    blind_model_id: str = "unspecified",
    trajectory_model_id: str = "unspecified",
    pacing_seconds: float = 0.0,
) -> OfflineJudgeReport:
    """Avalia registros serializados de um experimento já encerrado."""

    if not 0 <= pacing_seconds <= 60:
        raise ValueError("pacing_seconds deve estar entre 0 e 60")

    case_results = []
    for record in records:
        started = perf_counter()
        blind = await blind_result_judge.ainvoke(
            record.benchmark,
            record.observed,
            record.reference,
        )
        if pacing_seconds:
            await asyncio.sleep(pacing_seconds)
        trajectory = await trajectory_judge.ainvoke(
            record.benchmark,
            record.observed,
            record.reference,
        )
        case_results.append(
            JudgeCaseReport(
                run_id=record.run_id,
                case_id=record.benchmark.id,
                blind_result=blind,
                trajectory=trajectory,
                blind_thresholds=apply_thresholds(blind, thresholds=thresholds),
                trajectory_thresholds=apply_thresholds(
                    trajectory, thresholds=thresholds
                ),
                duration_ms=(perf_counter() - started) * 1_000,
            )
        )
        if pacing_seconds:
            await asyncio.sleep(pacing_seconds)

    grouped: dict[str, list[JudgeCaseReport]] = defaultdict(list)
    for result in case_results:
        grouped[result.case_id].append(result)
    stability = []
    for case_id, repetitions in grouped.items():
        score_rows = tuple(_scores(result) for result in repetitions)
        max_delta = max(
            (max(values) - min(values) for values in zip(*score_rows, strict=True)),
            default=0.0,
        )
        threshold_signatures = {
            tuple(
                item.passed
                for item in (
                    *result.blind_thresholds,
                    *result.trajectory_thresholds,
                )
            )
            for result in repetitions
        }
        stability.append(
            JudgeStability(
                case_id=case_id,
                repetitions=len(repetitions),
                stable=max_delta <= 0.1 and len(threshold_signatures) == 1,
                max_score_delta=round(max_delta, 12),
            )
        )
    return OfflineJudgeReport(
        provider=provider,
        blind_model_id=blind_model_id,
        trajectory_model_id=trajectory_model_id,
        thresholds=thresholds,
        cases=tuple(case_results),
        stability=tuple(stability),
    )
