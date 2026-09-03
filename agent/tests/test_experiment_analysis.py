from tractian_agent.evaluation.artifacts import (
    CaseProgrammaticReport,
    ProgrammaticArtifact,
)
from tractian_agent.evaluation.contracts import EvaluationOutput
from tractian_agent.evaluation.experiment_analysis import (
    compare_all_thresholds,
    compare_evaluation_layers,
)
from tractian_agent.evaluation.judge_runner import (
    JudgeCaseReport,
    JudgeStability,
    OfflineJudgeReport,
)
from tractian_agent.evaluation.judges import (
    BlindResultJudgment,
    JudgeVerdict,
    TrajectoryJudgment,
    apply_thresholds,
)


def _verdict(score: float) -> JudgeVerdict:
    return JudgeVerdict(passed=score >= 0.8, score=score, reason="rubrica")


def _judge_case(run_id: str, score: float) -> JudgeCaseReport:
    verdict = _verdict(score)
    blind = BlindResultJudgment(
        relevance=verdict,
        fidelity=verdict,
        uncertainty_honesty=verdict,
        decision_quality=verdict,
        communication=verdict,
    )
    trajectory = TrajectoryJudgment(
        investigation_strategy=verdict,
        grounding=verdict,
        failure_handling=verdict,
        stopping_quality=verdict,
    )
    return JudgeCaseReport(
        run_id=run_id,
        case_id="case_tkt_ctx_02",
        blind_result=blind,
        trajectory=trajectory,
        blind_thresholds=apply_thresholds(blind, thresholds=(0.8,)),
        trajectory_thresholds=apply_thresholds(trajectory, thresholds=(0.8,)),
        duration_ms=10.0,
    )


def test_comparison_counts_only_additional_rejections_on_the_same_runs() -> None:
    output = EvaluationOutput(case_id="case_tkt_ctx_02")
    programmatic = ProgrammaticArtifact(
        experiment_id="experiment",
        config_digest="sha256:v1:" + "a" * 64,
        total_runs=2,
        passed_runs=1,
        cases=(
            CaseProgrammaticReport(
                run_id="run_001",
                case_id="case_tkt_ctx_02",
                output=output,
                checks=(),
                passed=True,
            ),
            CaseProgrammaticReport(
                run_id="run_002",
                case_id="case_tkt_ctx_02",
                output=output,
                checks=(),
                passed=False,
            ),
        ),
        dimensions=(),
    )
    judges = OfflineJudgeReport(
        provider="groq",
        blind_model_id="judge",
        trajectory_model_id="judge",
        thresholds=(0.8,),
        cases=(_judge_case("run_001", 0.2), _judge_case("run_002", 0.9)),
        stability=(
            JudgeStability(
                case_id="case_tkt_ctx_02",
                repetitions=2,
                stable=False,
                max_score_delta=0.7,
            ),
        ),
    )

    comparison = compare_evaluation_layers(
        programmatic,
        judges,
        threshold=0.8,
    )

    assert comparison.programmatic_rejections == 1
    assert comparison.combined_rejections == 2
    assert comparison.additional_judge_detections == 1
    assert comparison.judge_calls == 4
    assert comparison.judge_latency_ms == 20.0
    assert comparison.false_approved is None
    assert comparison.unstable_cases == 1
    assert len(compare_all_thresholds(programmatic, judges).comparisons) == 1
