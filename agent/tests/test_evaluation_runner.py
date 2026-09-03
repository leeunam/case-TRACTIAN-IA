import asyncio
from pathlib import Path

import httpx
import pytest

from tractian_agent.checkpoint import open_checkpointer
from tractian_agent.client import IndustrialApiClient
from tractian_agent.evaluation.contracts import EvaluationOutput, ExpectedCase
from tractian_agent.evaluation.dataset import load_public_dataset
from tractian_agent.evaluation.runner import (
    AgentCaseExecutor,
    execute_before_loading_references,
)
from tractian_agent.graph import build_agent_graph
from tractian_agent.tools.runtime import ReadToolRuntime


def test_runner_finishes_agent_execution_before_loading_the_golden_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "cases.json"
    input_path.write_text(
        """[{"id":"case_tkt_ctx_02","ticket_id":"TKT-CTX-02",\
"company_id":"comp_aurora","user_id":"usr_lucas",\
"asset_id":"asset_B204","message":"O que significa BPFO?"}]""",
        encoding="utf-8",
    )
    events: list[str] = []

    async def execute(inputs):
        events.append(f"execute:{inputs.id}")
        return EvaluationOutput(case_id=inputs.id)

    expected = ExpectedCase(
        id="case_tkt_ctx_02",
        ticket_id="TKT-CTX-02",
        root_question="Definir BPFO.",
        mode="partial",
        expected_path=(
            {
                "step": "GET /knowledge/search?q=BPFO",
                "note": "localizar glossário",
            },
        ),
    )

    def load_after_execution(path: Path) -> dict[str, ExpectedCase]:
        events.append(f"golden:{path.name}")
        return {expected.id: expected}

    monkeypatch.setattr(
        "tractian_agent.evaluation.runner.load_reference_cases",
        load_after_execution,
    )

    batch = asyncio.run(
        execute_before_loading_references(
            load_public_dataset(input_path),
            execute,
            reference_path=tmp_path / "expected-paths.json",
        )
    )

    assert events == ["execute:case_tkt_ctx_02", "golden:expected-paths.json"]
    assert batch.execution_report.cases[0].output.case_id == expected.id
    assert batch.references == {expected.id: expected}


def test_runner_rejects_reference_case_set_that_differs_from_executed_cases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "cases.json"
    input_path.write_text(
        """[{"id":"case_tkt_ctx_02","ticket_id":"TKT-CTX-02",\
"company_id":"comp_aurora","user_id":"usr_lucas",\
"asset_id":"asset_B204","message":"O que significa BPFO?"}]""",
        encoding="utf-8",
    )

    async def execute(inputs):
        return EvaluationOutput(case_id=inputs.id)

    monkeypatch.setattr(
        "tractian_agent.evaluation.runner.load_reference_cases",
        lambda _: {},
    )

    with pytest.raises(ValueError, match="conjunto de casos"):
        asyncio.run(
            execute_before_loading_references(
                load_public_dataset(input_path),
                execute,
                reference_path=tmp_path / "expected-paths.json",
            )
        )


def test_agent_case_executor_uses_the_public_agent_boundary(tmp_path: Path) -> None:
    async def scenario() -> EvaluationOutput:
        def forbidden_http(request: httpx.Request) -> httpx.Response:
            raise AssertionError(f"fallback não deveria chamar HTTP: {request.url}")

        async with open_checkpointer(tmp_path / "eval.sqlite3") as saver:
            graph = build_agent_graph(saver)
            async with IndustrialApiClient(
                "https://simulator.test",
                transport=httpx.MockTransport(forbidden_http),
            ) as client:
                executor = AgentCaseExecutor(
                    graph=graph,
                    runtime_factory=lambda case: ReadToolRuntime.create(
                        user_id=case.user_id,
                        company_id=case.company_id,
                        permissions=frozenset({"read"}),
                        central_asset_id=case.asset_id,
                        client=client,
                    ),
                    experiment_id="experiment_contract_test",
                    step_limit=3,
                )
                return await executor(
                    load_public_dataset_from_single_case(tmp_path).cases[0].inputs
                )

    output = asyncio.run(scenario())

    assert output.case_id == "case_tkt_ctx_02"
    assert output.ticket_id == "TKT-CTX-02"
    assert output.decision == "guide"
    assert output.step_count == 3
    assert output.step_limit == 3


def load_public_dataset_from_single_case(tmp_path: Path):
    input_path = tmp_path / "single-case.json"
    input_path.write_text(
        """[{"id":"case_tkt_ctx_02","ticket_id":"TKT-CTX-02",\
"company_id":"comp_aurora","user_id":"usr_lucas",\
"asset_id":"asset_B204","message":"O que significa BPFO?"}]""",
        encoding="utf-8",
    )
    return load_public_dataset(input_path)


def test_agent_case_executor_accepts_async_runtime_factory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    benchmark = load_public_dataset_from_single_case(tmp_path).cases[0].inputs
    runtime = object()
    observed: list[object] = []

    async def runtime_factory(inputs):
        observed.append(inputs.id)
        return runtime

    async def fake_invoke_agent(graph, **kwargs):
        observed.append(kwargs["runtime"])
        return object()

    monkeypatch.setattr(
        "tractian_agent.evaluation.runner.invoke_agent",
        fake_invoke_agent,
    )
    monkeypatch.setattr(
        "tractian_agent.evaluation.runner.output_from_agent_state",
        lambda state, *, duration_ms: EvaluationOutput(
            case_id="case_tkt_ctx_02"
        ),
    )
    executor = AgentCaseExecutor(
        graph=object(),
        runtime_factory=runtime_factory,
        experiment_id="experiment-v1",
        step_limit=64,
    )

    result = asyncio.run(executor.execute(benchmark))

    assert result.case_id == "case_tkt_ctx_02"
    assert observed == ["case_tkt_ctx_02", runtime]
