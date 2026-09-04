from pathlib import Path

import pytest

from tractian_demo.contracts import AgentRunProjection, CreateCaseRequest
from tractian_demo.repository import DemoRepository
from tractian_demo.worker import DemoWorker

from test_repository import PUBLIC_CASE


class SuccessfulExecutor:
    async def execute(self, *, case, message, execution):
        return AgentRunProjection(
            assistant_message="A análise está consistente com os dados disponíveis.",
            decision="guide",
            trace_id="trace_safe_1",
            provider="groq",
            fallback_reason=None,
            evidence_count=2,
            limitation_count=1,
            tool_names=("get_asset", "list_asset_analyses"),
        )


class BrokenExecutor:
    async def execute(self, *, case, message, execution):
        raise RuntimeError("GROQ_API_KEY=secret raw provider output")


def queued(repository: DemoRepository):
    case = repository.create_case(CreateCaseRequest(source_case_id="case_public_1"))
    _, execution = repository.enqueue_message(
        case_id=case.id, persona_id="usr_1", content="analise",
        idempotency_key="worker-test",
    )
    return case, execution


@pytest.mark.anyio
async def test_worker_persists_sanitized_projection_and_completes(tmp_path: Path) -> None:
    repository = DemoRepository(tmp_path / "demo.sqlite3")
    repository.open(public_cases=[PUBLIC_CASE])
    case, execution = queued(repository)

    worked = await DemoWorker(repository, SuccessfulExecutor(), worker_id="w1").run_once()

    assert worked is True
    completed = repository.get_execution(execution.id)
    assert completed.status.value == "completed"
    assert completed.provider == "groq"
    assert completed.trace_id == "trace_safe_1"
    messages = repository.list_messages(case.id)
    assert messages[-1].role == "assistant"
    assert messages[-1].content.startswith("A análise")
    payloads = [event.payload for event in repository.list_events(case.id)]
    assert {"tool_names": ["get_asset", "list_asset_analyses"]} in payloads
    repository.close()


@pytest.mark.anyio
async def test_worker_maps_exception_to_closed_error_without_leaking(tmp_path: Path) -> None:
    repository = DemoRepository(tmp_path / "demo.sqlite3")
    repository.open(public_cases=[PUBLIC_CASE])
    case, execution = queued(repository)

    await DemoWorker(repository, BrokenExecutor(), worker_id="w1").run_once()

    failed = repository.get_execution(execution.id)
    assert failed.status.value == "failed"
    assert failed.error_code == "AGENT_EXECUTION_FAILED"
    exposed = str(repository.list_events(case.id))
    assert "secret" not in exposed
    assert "GROQ_API_KEY" not in exposed
    repository.close()
