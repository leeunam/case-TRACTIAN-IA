import asyncio
import json
from pathlib import Path

from tractian_agent.evaluation.artifacts import (
    build_programmatic_artifact,
    write_json_artifact,
)
from tractian_agent.evaluation.checks import run_programmatic_checks
from tractian_agent.evaluation.contracts import EvaluationOutput, ObservedStep
from tractian_agent.evaluation.dataset import load_public_dataset
from tractian_agent.evaluation.runner import execute_before_loading_references


def test_programmatic_artifact_reports_case_dimension_and_experiment(
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
            permissions=("read",),
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
        )

    batch = asyncio.run(
        execute_before_loading_references(
            load_public_dataset(public_path),
            execute,
            reference_path=reference_path,
        )
    )
    checks = asyncio.run(run_programmatic_checks(batch))

    artifact = build_programmatic_artifact(
        checks,
        experiment_id="experiment_test",
        config_digest="sha256:v1:" + "a" * 64,
    )

    assert artifact.experiment_id == "experiment_test"
    assert artifact.total_runs == 1
    assert artifact.cases[0].run_id == "run_001"
    assert artifact.cases[0].case_id == "case_tkt_ctx_02"
    assert {item.dimension for item in artifact.cases[0].checks} == {
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
    assert sum(item.total for item in artifact.dimensions) == 10

    output_path = tmp_path / "reports" / "programmatic.json"
    write_json_artifact(output_path, artifact)
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == artifact.model_dump(mode="json")
