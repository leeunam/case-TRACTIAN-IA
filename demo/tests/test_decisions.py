from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tractian_demo.contracts import (
    AgentRunProjection,
    CreateCaseRequest,
    DecisionCandidate,
    Persona,
)
from tractian_demo.decisions import route_decision
from tractian_demo.repository import DemoRepository

from test_repository import PUBLIC_CASE


def persona(id: str, company: str | None, profile: str, *permissions: str) -> Persona:
    return Persona(
        id=id,
        name=id,
        profile=profile,
        company_id=company,
        permissions=frozenset(permissions),
    )


def prepared(repository: DemoRepository):
    case = repository.create_case(CreateCaseRequest(source_case_id="case_public_1"))
    _, execution = repository.enqueue_message(
        case_id=case.id,
        persona_id="usr_1",
        content="mudar criticidade",
        idempotency_key="decision-setup",
    )
    execution = repository.claim_execution(worker_id="worker")
    assert execution is not None
    return case, execution


def test_routing_keeps_technical_and_company_decisions_distinct() -> None:
    assert (
        route_decision(
            action="request_specialist_analysis",
            requester_permissions=frozenset({"read"}),
        ).audience
        == "tractian"
    )
    assert (
        route_decision(
            action="request_model_retraining", requester_permissions=frozenset({"read"})
        ).audience
        == "tractian"
    )
    assert (
        route_decision(
            action="update_asset_criticality", requester_permissions=frozenset({"read"})
        ).audience
        == "authority"
    )
    assert (
        route_decision(
            action="reprocess_analysis", requester_permissions=frozenset({"read"})
        ).audience
        == "authority"
    )
    assert (
        route_decision(
            action="reprocess_analysis",
            requester_permissions=frozenset({"read", "action_low"}),
        ).audience
        == "requester"
    )
    assert (
        route_decision(
            action=None,
            requester_permissions=frozenset({"read"}),
            technical_review=True,
        ).audience
        == "tractian"
    )


def test_decision_visibility_and_outbox_are_atomic(tmp_path: Path) -> None:
    repository = DemoRepository(tmp_path / "demo.sqlite3")
    repository.open(public_cases=[PUBLIC_CASE])
    case, execution = prepared(repository)
    decision = repository.create_decision(
        case=case,
        execution=execution,
        candidate=DecisionCandidate(
            audience="authority",
            kind="action_authorization",
            summary="Autorizar criticidade critical para asset_1",
            scope={
                "action": "update_asset_criticality",
                "target_id": "asset_1",
                "material_parameters": {"criticality": "critical"},
            },
            required_permission="action_high",
            resume_kind="delegated_action",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        ),
    )
    assert (
        repository.list_decisions(persona("usr_1", "comp_1", "requester", "read")) == ()
    )
    assert repository.list_decisions(
        persona("boss", "comp_1", "authority", "read", "action_high")
    ) == (decision,)
    assert (
        repository.list_decisions(
            persona("other", "comp_2", "authority", "action_high")
        )
        == ()
    )
    outbox = repository.claim_outbox(worker_id="slack-worker")
    assert outbox is not None
    assert outbox.decision_id == decision.id
    assert outbox.audience == "authority"
    repository.close()


def test_first_resolution_wins_and_identical_retry_is_replay(tmp_path: Path) -> None:
    database = tmp_path / "demo.sqlite3"
    repository = DemoRepository(database)
    repository.open(public_cases=[PUBLIC_CASE])
    case, execution = prepared(repository)
    decision = repository.create_decision(
        case=case,
        execution=execution,
        candidate=DecisionCandidate(
            audience="authority",
            kind="action_authorization",
            summary="Autorizar ação",
            scope={
                "action": "update_asset_criticality",
                "target_id": "asset_1",
                "material_parameters": {"criticality": "high"},
            },
            required_permission="action_high",
            resume_kind="delegated_action",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        ),
    )
    repository.close()
    actor = persona("boss", "comp_1", "authority", "action_high")

    def resolve(value: str):
        repo = DemoRepository(database)
        repo.open()
        try:
            try:
                return repo.resolve_decision(
                    decision.id, persona=actor, resolution=value
                )
            except ValueError as error:
                return error
        finally:
            repo.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(resolve, ("approve", "reject")))
    winners = [item for item in results if not isinstance(item, Exception)]
    assert len(winners) == 1
    winner = winners[0]
    winning_resolution = "approve" if winner.status.value == "approved" else "reject"
    assert resolve(winning_resolution) == winner
    losing_resolution = "reject" if winning_resolution == "approve" else "approve"
    assert "DECISION_ALREADY_RESOLVED" in str(resolve(losing_resolution))


def test_wait_for_decision_is_atomic_and_replays_by_execution(tmp_path: Path) -> None:
    repository = DemoRepository(tmp_path / "demo.sqlite3")
    repository.open(public_cases=[PUBLIC_CASE])
    case, execution = prepared(repository)
    candidate = DecisionCandidate(
        audience="tractian",
        kind="technical_review",
        summary="Revisar análise inconclusiva",
        scope={"case_id": case.id},
        resume_kind="acknowledgement",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    projection = AgentRunProjection(
        assistant_message="A análise requer revisão técnica.",
        decision="require_human_review",
        trace_id="trace_atomic_1",
        provider="groq",
        fallback_reason=None,
        evidence_count=1,
        limitation_count=1,
        tool_names=("get_asset",),
        decision_candidate=candidate,
    )

    first = repository.wait_for_decision(execution.id, projection)
    replay = repository.wait_for_decision(execution.id, projection)

    assert replay == first
    assert len(repository.list_messages(case.id)) == 2
    assert repository.claim_outbox(worker_id="slack-worker") is not None
    assert repository.claim_outbox(worker_id="slack-worker") is None
    repository.close()
