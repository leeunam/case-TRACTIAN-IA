import asyncio
from pathlib import Path

from tractian_agent.evaluation.checks import run_programmatic_checks
from tractian_agent.evaluation.contracts import EvaluationOutput, ObservedStep
from tractian_agent.evaluation.dataset import load_public_dataset
from tractian_agent.evaluation.runner import execute_before_loading_references


def _write_public_case(path: Path) -> None:
    path.write_text(
        """[{"id":"case_tkt_ctx_02","ticket_id":"TKT-CTX-02",\
"company_id":"comp_aurora","user_id":"usr_lucas",\
"asset_id":"asset_B204","message":"O que significa BPFO?"}]""",
        encoding="utf-8",
    )


def _write_reference(path: Path) -> None:
    path.write_text(
        """[{"id":"case_tkt_ctx_02","ticket_id":"TKT-CTX-02",\
"root_question":"Definir BPFO.","mode":"partial",\
"expected_path":[{"step":"GET /knowledge/search?q=BPFO",\
"note":"localizar glossário"}]}]""",
        encoding="utf-8",
    )


def test_programmatic_checks_report_each_critical_dimension(tmp_path: Path) -> None:
    input_path = tmp_path / "cases.json"
    reference_path = tmp_path / "expected-paths.json"
    _write_public_case(input_path)
    _write_reference(reference_path)

    async def execute(inputs):
        return EvaluationOutput(
            case_id=inputs.id,
            ticket_id=inputs.ticket_id,
            decision="guide",
            message="BPFO é uma frequência de passagem dos elementos rolantes.",
            permissions=("read", "action_low"),
            steps=(
                ObservedStep(
                    ordinal=1,
                    call_id="call_01",
                    tool_name="search_knowledge",
                    arguments={"query": "BPFO"},
                    method="GET",
                    resource="/knowledge/search?q=BPFO",
                    outcome="success",
                ),
            ),
            step_count=5,
            step_limit=12,
            planner_selection_count=1,
            planner_finalization_count=1,
            writer_attempts=1,
            gate_outcome="release",
        )

    batch = asyncio.run(
        execute_before_loading_references(
            load_public_dataset(input_path),
            execute,
            reference_path=reference_path,
        )
    )
    report = asyncio.run(run_programmatic_checks(batch))

    result = report.cases[0]
    assert result.assertions.keys() == {
        "format",
        "decision",
        "tools",
        "arguments",
        "ids",
        "trajectory",
        "permissions",
        "justification",
        "errors",
        "step_limit",
    }
    assert all(assertion.value for assertion in result.assertions.values())


def test_correct_looking_result_fails_when_required_trajectory_is_missing(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "cases.json"
    reference_path = tmp_path / "expected-paths.json"
    _write_public_case(input_path)
    _write_reference(reference_path)

    async def execute(inputs):
        return EvaluationOutput(
            case_id=inputs.id,
            ticket_id=inputs.ticket_id,
            decision="guide",
            message="BPFO é a frequência de passagem dos elementos rolantes.",
            permissions=("read",),
            step_count=4,
            step_limit=12,
            planner_finalization_count=1,
            writer_attempts=1,
            gate_outcome="release",
        )

    batch = asyncio.run(
        execute_before_loading_references(
            load_public_dataset(input_path),
            execute,
            reference_path=reference_path,
        )
    )
    result = asyncio.run(run_programmatic_checks(batch)).cases[0]

    assert result.assertions["format"].value is True
    assert result.assertions["decision"].value is True
    assert result.assertions["trajectory"].value is False
    assert "GET /knowledge/search?q=BPFO" in (
        result.assertions["trajectory"].reason or ""
    )


def test_programmatic_checks_reject_unknown_tool_extra_argument_and_mismatched_id(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "cases.json"
    reference_path = tmp_path / "expected-paths.json"
    _write_public_case(input_path)
    _write_reference(reference_path)

    async def execute(inputs):
        return EvaluationOutput(
            case_id=inputs.id,
            ticket_id=inputs.ticket_id,
            decision="guide",
            message="Resposta plausível, mas a chamada observada é inválida.",
            permissions=("read",),
            steps=(
                ObservedStep(
                    ordinal=1,
                    call_id="call_unknown",
                    tool_name="read_everything",
                    arguments={"asset_id": "asset_B204"},
                    method="GET",
                    resource="/assets/asset_OTHER",
                    outcome="success",
                ),
                ObservedStep(
                    ordinal=2,
                    call_id="call_extra",
                    tool_name="search_knowledge",
                    arguments={"query": "BPFO", "unexpected": True},
                    method="GET",
                    resource="/knowledge/search?q=BPFO",
                    outcome="success",
                ),
            ),
            step_count=6,
            step_limit=12,
            planner_selection_count=2,
            planner_finalization_count=1,
            writer_attempts=1,
            gate_outcome="release",
        )

    batch = asyncio.run(
        execute_before_loading_references(
            load_public_dataset(input_path),
            execute,
            reference_path=reference_path,
        )
    )
    result = asyncio.run(run_programmatic_checks(batch)).cases[0]

    assert result.assertions["tools"].value is False
    assert result.assertions["arguments"].value is False
    assert result.assertions["ids"].value is False
