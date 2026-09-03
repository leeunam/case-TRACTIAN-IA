"""Comparação determinística entre checks e checks acrescidos de juízes."""

from __future__ import annotations

from pydantic import Field

from tractian_agent.evaluation.artifacts import ProgrammaticArtifact
from tractian_agent.evaluation.calibration import CalibrationReport
from tractian_agent.evaluation.contracts import EvaluationModel
from tractian_agent.evaluation.judge_runner import OfflineJudgeReport


class ExperimentComparison(EvaluationModel):
    version: str = "experiment-comparison-v1"
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    total_runs: int = Field(ge=0, strict=True)
    programmatic_rejections: int = Field(ge=0, strict=True)
    combined_rejections: int = Field(ge=0, strict=True)
    additional_judge_detections: int = Field(ge=0, strict=True)
    judge_calls: int = Field(ge=0, strict=True)
    judge_latency_ms: float = Field(ge=0, allow_inf_nan=False)
    estimated_judge_cost_usd: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
    )
    unstable_cases: int = Field(ge=0, strict=True)
    false_approved: int | None = Field(default=None, ge=0, strict=True)
    false_rejected: int | None = Field(default=None, ge=0, strict=True)
    limitations: tuple[str, ...]


class EvaluationLayerArtifact(EvaluationModel):
    version: str = "evaluation-layers-v1"
    comparisons: tuple[ExperimentComparison, ...]


def compare_all_thresholds(
    programmatic: ProgrammaticArtifact,
    judges: OfflineJudgeReport,
    *,
    calibration: CalibrationReport | None = None,
) -> EvaluationLayerArtifact:
    return EvaluationLayerArtifact(
        comparisons=tuple(
            compare_evaluation_layers(
                programmatic,
                judges,
                threshold=threshold,
                calibration=calibration,
            )
            for threshold in judges.thresholds
        )
    )


def compare_evaluation_layers(
    programmatic: ProgrammaticArtifact,
    judges: OfflineJudgeReport,
    *,
    threshold: float,
    calibration: CalibrationReport | None = None,
) -> ExperimentComparison:
    """Compara exatamente os mesmos runs, sem reinterpretar scores do juiz."""

    programmatic_by_run = {item.run_id: item for item in programmatic.cases}
    judges_by_run = {item.run_id: item for item in judges.cases}
    if set(programmatic_by_run) != set(judges_by_run):
        raise ValueError("checks e juízes devem cobrir exatamente os mesmos runs")
    if threshold not in judges.thresholds:
        raise ValueError("limiar não foi aplicado aos scores congelados")

    def passed_at_threshold(run_id: str) -> bool:
        case = judges_by_run[run_id]
        blind = next(item for item in case.blind_thresholds if item.threshold == threshold)
        trajectory = next(
            item for item in case.trajectory_thresholds if item.threshold == threshold
        )
        return blind.passed and trajectory.passed

    rejected_programmatic = {
        run_id for run_id, item in programmatic_by_run.items() if not item.passed
    }
    rejected_by_judges = {
        run_id for run_id in judges_by_run if not passed_at_threshold(run_id)
    }
    combined = rejected_programmatic | rejected_by_judges
    false_approved = false_rejected = None
    limitations = []
    if calibration is None:
        limitations.append(
            "Falsos aprovados e reprovados dependem dos rótulos humanos cegos e ainda não estão disponíveis."
        )
    else:
        metric = next(
            (item for item in calibration.metrics if item.threshold == threshold),
            None,
        )
        if metric is None:
            raise ValueError("calibração não contém o limiar comparado")
        false_approved = metric.false_approved
        false_rejected = metric.false_rejected
    limitations.append(
        "Custo permanece indisponível quando o provider não fornece tarifa versionada na configuração."
    )
    return ExperimentComparison(
        threshold=threshold,
        total_runs=len(programmatic_by_run),
        programmatic_rejections=len(rejected_programmatic),
        combined_rejections=len(combined),
        additional_judge_detections=len(rejected_by_judges - rejected_programmatic),
        judge_calls=2 * len(judges.cases),
        judge_latency_ms=sum(item.duration_ms for item in judges.cases),
        unstable_cases=sum(not item.stable for item in judges.stability),
        false_approved=false_approved,
        false_rejected=false_rejected,
        limitations=tuple(limitations),
    )
