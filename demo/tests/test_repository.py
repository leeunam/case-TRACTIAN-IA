from pathlib import Path

import pytest

from tractian_demo.contracts import CreateCaseRequest
from tractian_demo.repository import DemoRepository


PUBLIC_CASE = {
    "id": "case_public_1",
    "ticket_id": "TKT-1",
    "company_id": "comp_1",
    "user_id": "usr_1",
    "asset_id": "asset_1",
    "message": "Mensagem original",
}


def test_duplicate_and_enqueue_survive_sqlite_reopen(tmp_path: Path) -> None:
    database = tmp_path / "demo.sqlite3"
    repository = DemoRepository(database)
    repository.open(public_cases=[PUBLIC_CASE])

    copied = repository.create_case(CreateCaseRequest(source_case_id="case_public_1"))
    message, execution = repository.enqueue_message(
        case_id=copied.id,
        persona_id="usr_1",
        content="Execute a análise novamente",
        idempotency_key="message-1",
    )
    repository.close()

    reopened = DemoRepository(database)
    reopened.open(public_cases=[PUBLIC_CASE])
    assert reopened.get_case(copied.id) == copied
    assert reopened.list_messages(copied.id) == (message,)
    assert reopened.get_execution(execution.id) == execution
    reopened.close()


def test_message_and_execution_are_atomic(tmp_path: Path) -> None:
    repository = DemoRepository(tmp_path / "demo.sqlite3")
    repository.open(public_cases=[PUBLIC_CASE])
    copied = repository.create_case(CreateCaseRequest(source_case_id="case_public_1"))

    with pytest.raises(RuntimeError, match="injected transaction failure"):
        repository.enqueue_message(
            case_id=copied.id,
            persona_id="usr_1",
            content="não pode ficar pela metade",
            idempotency_key="message-broken",
            _fail_after_message_for_test=True,
        )

    assert repository.list_messages(copied.id) == ()
    assert repository.list_executions(copied.id) == ()
    repository.close()


def test_idempotent_message_replays_same_execution(tmp_path: Path) -> None:
    repository = DemoRepository(tmp_path / "demo.sqlite3")
    repository.open(public_cases=[PUBLIC_CASE])
    copied = repository.create_case(CreateCaseRequest(source_case_id="case_public_1"))

    first = repository.enqueue_message(
        case_id=copied.id,
        persona_id="usr_1",
        content="mensagem",
        idempotency_key="same-key",
    )
    replay = repository.enqueue_message(
        case_id=copied.id,
        persona_id="usr_1",
        content="mensagem",
        idempotency_key="same-key",
    )

    assert replay == first
    with pytest.raises(ValueError, match="IDEMPOTENCY_CONFLICT"):
        repository.enqueue_message(
            case_id=copied.id,
            persona_id="usr_1",
            content="conteúdo diferente",
            idempotency_key="same-key",
        )
    repository.close()
