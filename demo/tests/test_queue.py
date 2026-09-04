from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tractian_demo.contracts import CreateCaseRequest
from tractian_demo.repository import DemoRepository

from test_repository import PUBLIC_CASE


def test_only_one_worker_claims_a_queued_execution(tmp_path: Path) -> None:
    database = tmp_path / "demo.sqlite3"
    seed = DemoRepository(database)
    seed.open(public_cases=[PUBLIC_CASE])
    case = seed.create_case(CreateCaseRequest(source_case_id="case_public_1"))
    _, execution = seed.enqueue_message(
        case_id=case.id, persona_id="usr_1", content="trabalhe",
        idempotency_key="queue-1",
    )
    seed.close()

    def claim(worker: str):
        repository = DemoRepository(database)
        repository.open()
        try:
            return repository.claim_execution(worker_id=worker)
        finally:
            repository.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ("worker-a", "worker-b")))

    claimed = [item for item in results if item is not None]
    assert len(claimed) == 1
    assert claimed[0].id == execution.id
    assert claimed[0].attempt == 1


def test_event_replay_uses_persisted_id(tmp_path: Path) -> None:
    repository = DemoRepository(tmp_path / "demo.sqlite3")
    repository.open(public_cases=[PUBLIC_CASE])
    case = repository.create_case(CreateCaseRequest(source_case_id="case_public_1"))
    first = repository.append_event(case.id, "planner.started", {"safe": True})
    second = repository.append_event(case.id, "tool.completed", {"tool": "get_asset"})

    assert repository.list_events(case.id, after_id=first.id) == (second,)
    repository.close()
