import asyncio
from pathlib import Path

from tractian_agent.evaluation.contracts import EvaluationOutput
from tractian_agent.evaluation.dataset import load_public_dataset
from tractian_agent.evaluation.judge_runner import (
    conservative_calibration_score,
    run_offline_judges,
)
from tractian_agent.evaluation.judges import (
    BlindResultJudgment,
    JudgeVerdict,
    TrajectoryJudgment,
)
from tractian_agent.evaluation.runner import execute_before_loading_references


def _verdict(score: float) -> JudgeVerdict:
    return JudgeVerdict(
        passed=score >= 0.8,
        score=score,
        reason="rubrica aplicada",
    )


class _BlindJudge:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, benchmark, observed, reference):
        self.calls += 1
        score = 0.9 if self.calls == 1 else 0.6
        verdict = _verdict(score)
        return BlindResultJudgment(
            relevance=verdict,
            fidelity=verdict,
            uncertainty_honesty=verdict,
            decision_quality=verdict,
            communication=verdict,
        )


class _TrajectoryJudge:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, benchmark, observed, reference):
        self.calls += 1
        verdict = _verdict(0.9)
        return TrajectoryJudgment(
            investigation_strategy=verdict,
            grounding=verdict,
            failure_handling=verdict,
            stopping_quality=verdict,
        )


def test_offline_judges_run_once_per_completed_repetition_and_flag_instability(
    tmp_path: Path,
) -> None:
    public_path = tmp_path / "cases.json"
    public_path.write_text(
        """[{"id":"case_tkt_ctx_02","ticket_id":"TKT-CTX-02",\
"company_id":"comp_aurora","user_id":"usr_lucas",\
"asset_id":"asset_B204","message":"O que significa BPFO?"}]""",
        encoding="utf-8",
    )
    reference_path = tmp_path / "expected-paths.json"
    reference_path.write_text(
        """[{"id":"case_tkt_ctx_02","ticket_id":"TKT-CTX-02",\
"root_question":"Definir BPFO.","mode":"partial",\
"expected_path":[{"step":"GET /knowledge/search?q=BPFO",\
"note":"localizar glossário"}]}]""",
        encoding="utf-8",
    )

    async def execute(inputs):
        return EvaluationOutput(
            case_id=inputs.id,
            ticket_id=inputs.ticket_id,
            decision="guide",
            message="BPFO é uma frequência característica.",
        )

    batch = asyncio.run(
        execute_before_loading_references(
            load_public_dataset(public_path),
            execute,
            reference_path=reference_path,
            repeat=2,
        )
    )
    blind = _BlindJudge()
    trajectory = _TrajectoryJudge()

    report = asyncio.run(
        run_offline_judges(
            batch,
            blind_result_judge=blind,
            trajectory_judge=trajectory,
            thresholds=(0.7, 0.8, 0.9),
            provider="groq",
            blind_model_id="judge-blind-v1",
            trajectory_model_id="judge-trajectory-v1",
        )
    )

    assert blind.calls == trajectory.calls == 2
    assert report.provider == "groq"
    assert report.blind_model_id == "judge-blind-v1"
    assert report.trajectory_model_id == "judge-trajectory-v1"
    assert len(report.cases) == 2
    assert report.stability[0].case_id == "case_tkt_ctx_02"
    assert report.stability[0].repetitions == 2
    assert report.stability[0].stable is False
    assert report.stability[0].max_score_delta == 0.3
    assert report.cases[0].blind_thresholds[1].passed is True
    assert report.cases[1].blind_thresholds[1].passed is False


def test_calibration_score_uses_lowest_critical_dimension_not_communication() -> None:
    blind = BlindResultJudgment(
        relevance=_verdict(0.8),
        fidelity=_verdict(0.7),
        uncertainty_honesty=_verdict(0.9),
        decision_quality=_verdict(0.85),
        communication=_verdict(0.1),
    )
    trajectory = TrajectoryJudgment(
        investigation_strategy=_verdict(0.75),
        grounding=_verdict(0.95),
        failure_handling=_verdict(0.8),
        stopping_quality=_verdict(0.9),
    )

    assert conservative_calibration_score(blind, trajectory) == 0.7
