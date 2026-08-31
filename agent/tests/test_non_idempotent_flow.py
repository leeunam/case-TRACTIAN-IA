from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import tractian_agent.graph as graph_module
from tractian_agent.checkpoint import open_checkpointer
from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import (
    ActionReceipt,
    ApiErrorCategory,
    Identity,
    SupportRequest,
)
from tractian_agent.entrypoint import AgentInvocationProtocolError, invoke_agent
from tractian_agent.graph import build_agent_graph
from tractian_agent.state import AgentDecision, AgentState
from tractian_agent.tools.runtime import WriteToolRuntime
from tractian_agent.write_contracts import (
    ConfirmationReply,
    EscalateCaseIntentScope,
    IntentStatus,
    RequestModelRetrainingIntentScope,
    RequestSpecialistAnalysisIntentScope,
    UpdateAssetCriticalityIntentScope,
    WriteIntent,
)
from tractian_agent.write_policy import (
    ApprovalSource,
    EscalateCaseProposal,
    RequestModelRetrainingProposal,
    RequestSpecialistAnalysisProposal,
    TrustedActionApproval,
    UpdateAssetCriticalityProposal,
    WriteMaterialParameters,
)


JUSTIFICATION = "A condição exige uma ação industrial explicitamente justificada."


@dataclass(frozen=True)
class ActionCase:
    slug: str
    proposal: object
    permission: str
    approval: TrustedActionApproval
    scope_type: type
    target_id: str
    method: str
    path: str
    body: dict[str, object]
    final_decision: AgentDecision


ACTION_CASES = (
    ActionCase(
        slug="specialist",
        proposal=RequestSpecialistAnalysisProposal(
            analysis_id="an_9901",
            justification=JUSTIFICATION,
        ),
        permission="action_low",
        approval=TrustedActionApproval(
            action="request_specialist_analysis",
            target_id="an_9901",
            source=ApprovalSource.ORIGINAL_REQUEST,
        ),
        scope_type=RequestSpecialistAnalysisIntentScope,
        target_id="an_9901",
        method="POST",
        path="/analyses/an_9901/request-specialist",
        body={"justification": JUSTIFICATION},
        final_decision=AgentDecision.ACT,
    ),
    ActionCase(
        slug="criticality",
        proposal=UpdateAssetCriticalityProposal(
            criticality="critical",
            justification=JUSTIFICATION,
        ),
        permission="action_high",
        approval=TrustedActionApproval(
            action="update_asset_criticality",
            target_id="asset_M101",
            material_parameters=WriteMaterialParameters(criticality="critical"),
            source=ApprovalSource.ORIGINAL_REQUEST,
        ),
        scope_type=UpdateAssetCriticalityIntentScope,
        target_id="asset_M101",
        method="PATCH",
        path="/assets/asset_M101",
        body={
            "changes": {"criticality": "critical"},
            "justification": JUSTIFICATION,
        },
        final_decision=AgentDecision.ACT,
    ),
    ActionCase(
        slug="retraining",
        proposal=RequestModelRetrainingProposal(justification=JUSTIFICATION),
        permission="action_high",
        approval=TrustedActionApproval(
            action="request_model_retraining",
            target_id="mdl_vib_v3",
            source=ApprovalSource.ORIGINAL_REQUEST,
        ),
        scope_type=RequestModelRetrainingIntentScope,
        target_id="mdl_vib_v3",
        method="POST",
        path="/models/mdl_vib_v3/request-retraining",
        body={"justification": JUSTIFICATION},
        final_decision=AgentDecision.ACT,
    ),
    ActionCase(
        slug="escalation",
        proposal=EscalateCaseProposal(justification=JUSTIFICATION),
        permission="escalate",
        approval=TrustedActionApproval(
            action="escalate_case",
            target_id="case_tkt_exe_12",
            source=ApprovalSource.ORIGINAL_REQUEST,
        ),
        scope_type=EscalateCaseIntentScope,
        target_id="case_tkt_exe_12",
        method="POST",
        path="/cases/case_tkt_exe_12/escalate",
        body={"justification": JUSTIFICATION},
        final_decision=AgentDecision.ESCALATE,
    ),
)

_OPERATION_NAMES = {
    "specialist": "execute_request_specialist_analysis",
    "criticality": "execute_update_asset_criticality",
    "retraining": "execute_request_model_retraining",
    "escalation": "execute_escalate_case",
}


def _conflicting_approval(case: ActionCase) -> TrustedActionApproval:
    if case.slug == "criticality":
        return TrustedActionApproval(
            action=case.approval.action,
            target_id=case.approval.target_id,
            material_parameters=WriteMaterialParameters(criticality="low"),
            source=ApprovalSource.ORIGINAL_REQUEST,
        )
    wrong_target = {
        "specialist": "an_9902",
        "retraining": "mdl_other",
        "escalation": "case_other",
    }[case.slug]
    return TrustedActionApproval(
        action=case.approval.action,
        target_id=wrong_target,
        source=ApprovalSource.ORIGINAL_REQUEST,
    )


def _request() -> SupportRequest:
    return SupportRequest(
        case_id="case_tkt_exe_12",
        ticket_id="TKT-EXE-12",
        asset_id="asset_M101",
        message="Execute a ação industrial solicitada.",
        identity=Identity(user_id="usr_ana", company_id="comp_forja_br"),
    )


def _analysis_payload() -> dict[str, object]:
    return {
        "id": "an_9901",
        "asset_id": "asset_M101",
        "point_id": "pt_M101_de",
        "type": "bearing_fault",
        "detection_mode": "baseline",
        "severity": "high",
        "confidence": 0.78,
        "baseline_state_at_detection": "established",
        "evidence": [],
        "limitations": [],
        "model_version": "3.2.1",
        "created_at": "2026-01-02T03:04:05+00:00",
        "status": "current",
    }


def _runtime(
    handler: Any,
    case: ActionCase,
    *,
    permissions: frozenset[str] | None = None,
    configured_model_id: str = "mdl_vib_v3",
    central_asset_id: str = "asset_M101",
    current_case_id: str = "case_tkt_exe_12",
) -> WriteToolRuntime:
    return WriteToolRuntime.create(
        user_id="usr_ana",
        company_id="comp_forja_br",
        permissions=(
            frozenset({"read", case.permission})
            if permissions is None
            else permissions
        ),
        central_asset_id=central_asset_id,
        current_case_id=current_case_id,
        configured_model_id=configured_model_id,
        client=IndustrialApiClient(
            "https://simulator.test",
            transport=httpx.MockTransport(handler),
        ),
    )


def _canonical_hash(body: dict[str, object]) -> str:
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:v1:{hashlib.sha256(encoded).hexdigest()}"


def _success_handler(
    requests: list[httpx.Request],
    *,
    accepted: bool = True,
):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"mode": "complete", "notes": None, "data": _analysis_payload()},
            )
        return httpx.Response(
            200,
            json={
                "accepted": accepted,
                "action_id": "act_non_idempotent_01",
                "message": "Ação processada pela plataforma.",
            },
        )

    return handler


def _write_requests(
    requests: list[httpx.Request],
    case: ActionCase,
) -> list[httpx.Request]:
    return [
        request
        for request in requests
        if request.method == case.method and request.url.path == case.path
    ]


def _checkpoint_intent_statuses(checkpoint: dict[str, object]) -> set[str]:
    channel_values = checkpoint.get("channel_values", {})
    if not isinstance(channel_values, dict):
        return set()
    raw_intents = channel_values.get("intents", ())
    return {
        intent.status.value
        if isinstance(intent, WriteIntent)
        else intent["status"]
        for intent in raw_intents
        if isinstance(intent, WriteIntent)
        or (isinstance(intent, dict) and isinstance(intent.get("status"), str))
    }


def _write_intent_statuses(writes: list[tuple[str, object]]) -> set[str]:
    statuses: set[str] = set()
    for channel, value in writes:
        if channel != "intents":
            continue
        for intent in value:
            if isinstance(intent, WriteIntent):
                statuses.add(intent.status.value)
            elif isinstance(intent, dict) and isinstance(intent.get("status"), str):
                statuses.add(intent["status"])
    return statuses


def _replace_current_intent_for_test(
    state: AgentState,
    replacement: WriteIntent,
) -> AgentState:
    data = state.model_dump(mode="python")
    data["intents"] = tuple(
        replacement if intent.intent_id == replacement.intent_id else intent
        for intent in state.intents
    )
    return AgentState.model_validate(data)


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_original_approval_executes_each_non_idempotent_action_once(
    tmp_path: Path,
    case: ActionCase,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_success_handler(requests), case)
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                return await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_allowed_{case.slug}",
                    request_id=f"req_allowed_{case.slug}",
                    execution_id=f"exec_allowed_{case.slug}",
                    proposal=case.proposal,
                    original_approval=case.approval,
                )
        finally:
            await runtime.client.aclose()

    state = asyncio.run(scenario())
    intent = state.intents[0]
    writes = _write_requests(requests, case)

    assert state.step_count == 5
    assert state.step_limit == 5
    assert state.decision is case.final_decision
    assert intent.status is IntentStatus.COMPLETED
    assert type(intent.scope) is case.scope_type
    assert intent.payload_hash == _canonical_hash(case.body)
    assert intent.prepared_execution_id == f"exec_allowed_{case.slug}"
    assert intent.idempotency_key is None
    assert intent.expires_at is None
    assert intent.attempts == 1
    assert len(writes) == 1
    assert json.loads(writes[0].content) == case.body
    assert "idempotency-key" not in writes[0].headers
    if case.slug == "specialist":
        assert [(request.method, request.url.path) for request in requests] == [
            ("GET", "/analyses/an_9901"),
            (case.method, case.path),
        ]


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
@pytest.mark.parametrize("denial", ["permission", "justification"])
def test_policy_denies_non_idempotent_action_before_prepare_or_http(
    tmp_path: Path,
    case: ActionCase,
    denial: str,
):
    requests: list[httpx.Request] = []
    proposal_data = case.proposal.model_dump(mode="python")
    if denial == "justification":
        proposal_data["justification"] = "curta"
    proposal = type(case.proposal).model_validate(proposal_data)
    permissions = (
        frozenset({"read"})
        if denial == "permission"
        else frozenset({"read", case.permission})
    )

    async def scenario():
        runtime = _runtime(
            lambda request: requests.append(request),
            case,
            permissions=permissions,
        )
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}-{denial}.sqlite3"
            ) as saver:
                return await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_deny_{case.slug}_{denial}",
                    request_id=f"req_deny_{case.slug}_{denial}",
                    execution_id=f"exec_deny_{case.slug}_{denial}",
                    proposal=proposal,
                    original_approval=case.approval,
                )
        finally:
            await runtime.client.aclose()

    state = asyncio.run(scenario())

    assert state.step_count == 2
    assert state.intents[0].status is IntentStatus.DENIED
    assert state.intents[0].prepared_execution_id is None
    assert state.intents[0].idempotency_key is None
    assert requests == []


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_scope_mismatch_interrupts_without_rebinding_or_http(
    tmp_path: Path,
    case: ActionCase,
):
    requests: list[httpx.Request] = []
    wrong_approval = _conflicting_approval(case)

    async def scenario():
        runtime = _runtime(lambda request: requests.append(request), case)
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                state = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_scope_{case.slug}",
                    request_id=f"req_scope_{case.slug}",
                    execution_id=f"exec_scope_{case.slug}",
                    proposal=case.proposal,
                    original_approval=wrong_approval,
                )
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": f"thread_scope_{case.slug}"}}
                )
                return state, snapshot
        finally:
            await runtime.client.aclose()

    state, snapshot = asyncio.run(scenario())

    assert state.step_count == 2
    assert state.intents[0].status is IntentStatus.AWAITING_CONFIRMATION
    assert snapshot.next == ("confirmation_gate",)
    assert requests == []


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_approval_action_mismatch_interrupts_without_http(
    tmp_path: Path,
    case: ActionCase,
):
    requests: list[httpx.Request] = []
    wrong_approval = (
        TrustedActionApproval(
            action="request_specialist_analysis",
            target_id="an_9901",
            source=ApprovalSource.ORIGINAL_REQUEST,
        )
        if case.slug == "escalation"
        else TrustedActionApproval(
            action="escalate_case",
            target_id="case_tkt_exe_12",
            source=ApprovalSource.ORIGINAL_REQUEST,
        )
    )

    async def scenario():
        runtime = _runtime(lambda request: requests.append(request), case)
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                state = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_action_mismatch_{case.slug}",
                    request_id=f"req_action_mismatch_{case.slug}",
                    execution_id=f"exec_action_mismatch_{case.slug}",
                    proposal=case.proposal,
                    original_approval=wrong_approval,
                )
                return state
        finally:
            await runtime.client.aclose()

    state = asyncio.run(scenario())

    assert state.intents[0].status is IntentStatus.AWAITING_CONFIRMATION
    assert state.intents[0].prepared_execution_id is None
    assert requests == []


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_interrupt_approve_uses_persisted_scope_and_flat_prompt(
    tmp_path: Path,
    case: ActionCase,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_success_handler(requests), case)
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                waiting = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_confirm_{case.slug}",
                    request_id=f"req_confirm_{case.slug}",
                    execution_id=f"exec_wait_{case.slug}",
                    proposal=case.proposal,
                )
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": f"thread_confirm_{case.slug}"}}
                )
                prompt = snapshot.interrupts[0].value

                async def forbidden_update(*args, **kwargs):
                    raise AssertionError("interrupt deve retomar por Command")

                graph.aupdate_state = forbidden_update
                completed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_confirm_{case.slug}",
                    request_id=f"req_confirm_{case.slug}",
                    execution_id=f"exec_approve_{case.slug}",
                    confirmation=ConfirmationReply(
                        intent_id=waiting.intents[0].intent_id,
                        decision="approve",
                    ),
                )
                return waiting, prompt, completed
        finally:
            await runtime.client.aclose()

    waiting, prompt, completed = asyncio.run(scenario())

    expected_prompt = {
        "intent_id": waiting.intents[0].intent_id,
        "action": case.approval.action,
        "target_id": case.target_id,
        "justification": JUSTIFICATION,
        "payload_hash": _canonical_hash(case.body),
    }
    if case.slug == "criticality":
        expected_prompt["criticality"] = "critical"
    assert prompt == expected_prompt
    assert completed.step_count == 5
    assert completed.approval is not None
    assert completed.approval.source is ApprovalSource.CONFIRMATION
    assert completed.approval.action == case.approval.action
    assert completed.approval.target_id == case.target_id
    assert completed.approval.material_parameters == case.approval.material_parameters
    assert completed.intents[0].status is IntentStatus.COMPLETED


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_interrupt_deny_finishes_in_three_steps_without_http(
    tmp_path: Path,
    case: ActionCase,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(lambda request: requests.append(request), case)
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                waiting = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_deny_confirm_{case.slug}",
                    request_id=f"req_deny_confirm_{case.slug}",
                    execution_id=f"exec_wait_deny_{case.slug}",
                    proposal=case.proposal,
                )
                return await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_deny_confirm_{case.slug}",
                    request_id=f"req_deny_confirm_{case.slug}",
                    execution_id=f"exec_deny_confirm_{case.slug}",
                    confirmation=ConfirmationReply(
                        intent_id=waiting.intents[0].intent_id,
                        decision="deny",
                    ),
                )
        finally:
            await runtime.client.aclose()

    state = asyncio.run(scenario())

    assert state.step_count == 3
    assert state.intents[0].status is IntentStatus.DENIED
    assert state.intents[0].prepared_execution_id is None
    assert requests == []


_RESULT_CASES = (
    ("rejected", IntentStatus.FAILED, False),
    ("http_400", IntentStatus.FAILED, False),
    ("http_409", IntentStatus.FAILED, False),
    ("timeout", IntentStatus.UNCERTAIN, True),
    ("transport", IntentStatus.UNCERTAIN, True),
    ("server", IntentStatus.UNCERTAIN, True),
    ("invalid_response", IntentStatus.UNCERTAIN, True),
)


def _result_handler(
    requests: list[httpx.Request],
    result_kind: str,
):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"mode": "complete", "notes": None, "data": _analysis_payload()},
            )
        if result_kind == "rejected":
            return httpx.Response(
                200,
                json={
                    "accepted": False,
                    "action_id": "act_rejected_01",
                    "message": "Ação rejeitada.",
                },
            )
        if result_kind == "http_400":
            return httpx.Response(
                400,
                json={"code": "VALIDATION_ERROR", "message": "Pedido inválido."},
            )
        if result_kind == "http_409":
            return httpx.Response(
                409,
                json={
                    "code": "IDEMPOTENCY_OUTCOME_UNKNOWN",
                    "message": "Código idempotente não muda a capacidade do endpoint.",
                },
            )
        if result_kind == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if result_kind == "transport":
            raise httpx.ConnectError("transport", request=request)
        if result_kind == "server":
            return httpx.Response(
                500,
                json={"code": "INTERNAL_ERROR", "message": "Falha remota."},
            )
        return httpx.Response(200, json={"accepted": "not-a-boolean"})

    return handler


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
@pytest.mark.parametrize(
    ("result_kind", "expected_status", "requires_review"),
    _RESULT_CASES,
)
def test_non_idempotent_result_matrix_never_retries(
    tmp_path: Path,
    case: ActionCase,
    result_kind: str,
    expected_status: IntentStatus,
    requires_review: bool,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_result_handler(requests, result_kind), case)
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}-{result_kind}.sqlite3"
            ) as saver:
                return await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_result_{case.slug}_{result_kind}",
                    request_id=f"req_result_{case.slug}_{result_kind}",
                    execution_id=f"exec_result_{case.slug}_{result_kind}",
                    proposal=case.proposal,
                    original_approval=case.approval,
                )
        finally:
            await runtime.client.aclose()

    state = asyncio.run(scenario())
    intent = state.intents[0]
    writes = _write_requests(requests, case)

    assert intent.status is expected_status
    assert intent.attempts == 1
    assert len(writes) == 1
    assert "idempotency-key" not in writes[0].headers
    assert (intent.receipt is not None) is (result_kind == "rejected")
    assert (intent.error is not None) is (result_kind != "rejected")
    assert (state.decision is AgentDecision.REQUIRE_HUMAN_REVIEW) is requires_review
    assert state.review is None


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
@pytest.mark.parametrize(
    ("status_code", "expected_status"),
    [
        pytest.param(400, IntentStatus.FAILED, id="400"),
        pytest.param(403, IntentStatus.FAILED, id="403"),
        pytest.param(409, IntentStatus.FAILED, id="409"),
        pytest.param(200, IntentStatus.UNCERTAIN, id="2xx"),
    ],
)
def test_non_idempotent_malformed_http_response_never_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: ActionCase,
    status_code: int,
    expected_status: IntentStatus,
):
    requests: list[httpx.Request] = []
    operation_calls: list[object] = []
    operation_name = _OPERATION_NAMES[case.slug]
    original_operation = getattr(graph_module, operation_name)

    async def counted_operation(proposal, runtime):
        operation_calls.append(proposal)
        return await original_operation(proposal, runtime)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"mode": "complete", "notes": None, "data": _analysis_payload()},
            )
        return httpx.Response(
            status_code,
            content=b"{malformed-json",
            headers={"content-type": "application/json"},
        )

    monkeypatch.setattr(graph_module, operation_name, counted_operation)

    async def scenario():
        runtime = _runtime(handler, case)
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-malformed-{case.slug}-{status_code}.sqlite3"
            ) as saver:
                return await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_malformed_{case.slug}_{status_code}",
                    request_id=f"req_malformed_{case.slug}_{status_code}",
                    execution_id=f"exec_malformed_{case.slug}_{status_code}",
                    proposal=case.proposal,
                    original_approval=case.approval,
                )
        finally:
            await runtime.client.aclose()

    state = asyncio.run(scenario())
    intent = state.intents[0]
    write_requests = _write_requests(requests, case)
    expected_gets = 1 if case.slug == "specialist" else 0

    assert intent.status is expected_status
    assert intent.attempts == 1
    assert intent.error is not None
    assert intent.error.category is ApiErrorCategory.INVALID_RESPONSE
    assert intent.error.status_code == status_code
    assert len(operation_calls) == 1
    assert len(write_requests) == 1
    assert len(requests) == expected_gets + 1
    assert sum(request.method == "GET" for request in requests) == expected_gets
    assert "idempotency-key" not in write_requests[0].headers
    assert (
        state.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    ) is (expected_status is IntentStatus.UNCERTAIN)


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_prepared_checkpoint_is_observable_without_key_before_operation(
    tmp_path: Path,
    case: ActionCase,
    monkeypatch: pytest.MonkeyPatch,
):
    observed: list[AgentState] = []

    async def scenario():
        runtime = _runtime(
            lambda request: pytest.fail("operação falsa não deve alcançar HTTP"),
            case,
        )
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)

                async def inspect_prepared(proposal, operation_runtime):
                    snapshot = await graph.aget_state(
                        {
                            "configurable": {
                                "thread_id": f"thread_prepared_{case.slug}"
                            }
                        }
                    )
                    observed.append(AgentState.model_validate(snapshot.values))
                    return ActionReceipt(
                        accepted=True,
                        action_id=f"act_prepared_{case.slug}",
                        message="Ação aceita.",
                    )

                monkeypatch.setattr(
                    graph_module,
                    _OPERATION_NAMES[case.slug],
                    inspect_prepared,
                )
                return await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_prepared_{case.slug}",
                    request_id=f"req_prepared_{case.slug}",
                    execution_id=f"exec_prepared_{case.slug}",
                    proposal=case.proposal,
                    original_approval=case.approval,
                )
        finally:
            await runtime.client.aclose()

    completed = asyncio.run(scenario())

    assert len(observed) == 1
    prepared = observed[0].intents[0]
    assert observed[0].step_count == 4
    assert prepared.status is IntentStatus.PREPARED
    assert prepared.attempts == 0
    assert prepared.prepared_execution_id == f"exec_prepared_{case.slug}"
    assert prepared.idempotency_key is None
    assert prepared.expires_at is None
    assert completed.intents[0].status is IntentStatus.COMPLETED


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_non_idempotent_prepare_does_not_read_clock_or_generate_key(
    tmp_path: Path,
    case: ActionCase,
    monkeypatch: pytest.MonkeyPatch,
):
    requests: list[httpx.Request] = []

    def forbidden_clock():
        raise AssertionError("ação não idempotente não deve consultar relógio")

    async def scenario():
        runtime = _runtime(_success_handler(requests), case)
        try:
            monkeypatch.setattr(graph_module, "_utc_now", forbidden_clock)
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                return await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_no_clock_{case.slug}",
                    request_id=f"req_no_clock_{case.slug}",
                    execution_id=f"exec_no_clock_{case.slug}",
                    proposal=case.proposal,
                    original_approval=case.approval,
                )
        finally:
            await runtime.client.aclose()

    state = asyncio.run(scenario())

    assert state.intents[0].status is IntentStatus.COMPLETED
    assert state.intents[0].idempotency_key is None
    assert state.intents[0].expires_at is None


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
@pytest.mark.parametrize("drift", ["permission", "approval", "scope", "hash"])
def test_same_execution_revalidation_fails_before_operation_with_real_checkpoint(
    tmp_path: Path,
    case: ActionCase,
    drift: str,
    monkeypatch: pytest.MonkeyPatch,
):
    requests: list[httpx.Request] = []
    original_prepare = graph_module._prepare_intent

    def prepare_with_drift(state: AgentState) -> dict[str, object]:
        prepared = AgentState.model_validate(original_prepare(state))
        intent = prepared.intents[-1]
        intent_data = intent.model_dump(mode="python")
        state_data = prepared.model_dump(mode="python")
        if drift == "permission":
            state_data["permissions"] = frozenset({"read"})
        elif drift == "approval":
            state_data["approval"] = None
        elif drift == "hash":
            intent_data["payload_hash"] = "sha256:v1:" + "f" * 64
        else:
            scope_data = intent.scope.model_dump(mode="python")
            if case.slug == "specialist":
                scope_data["analysis_id"] = "an_9902"
            elif case.slug == "criticality":
                scope_data["asset_id"] = "asset_other"
            elif case.slug == "retraining":
                scope_data["model_id"] = "mdl_other"
            else:
                scope_data["justification"] = JUSTIFICATION + " divergente"
            intent_data["scope"] = case.scope_type.model_validate(scope_data)
        if drift in {"scope", "hash"}:
            replacement = WriteIntent.model_validate(intent_data)
            prepared = _replace_current_intent_for_test(prepared, replacement)
            state_data = prepared.model_dump(mode="python")
        return AgentState.model_validate(state_data).model_dump(mode="json")

    async def scenario():
        runtime = _runtime(lambda request: requests.append(request), case)
        try:
            monkeypatch.setattr(graph_module, "_prepare_intent", prepare_with_drift)
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}-{drift}.sqlite3"
            ) as saver:
                return await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_same_{case.slug}_{drift}",
                    request_id=f"req_same_{case.slug}_{drift}",
                    execution_id=f"exec_same_{case.slug}_{drift}",
                    proposal=case.proposal,
                    original_approval=case.approval,
                )
        finally:
            await runtime.client.aclose()

    state = asyncio.run(scenario())
    intent = state.intents[0]

    assert intent.status is IntentStatus.FAILED
    assert intent.attempts == 0
    assert intent.error is not None
    assert intent.error.code == {
        "permission": "AUTHORIZATION_CHANGED_BEFORE_DISPATCH",
        "approval": "AUTHORIZATION_CHANGED_BEFORE_DISPATCH",
        "scope": "INTENT_SCOPE_MISMATCH",
        "hash": "PAYLOAD_HASH_MISMATCH",
    }[drift]
    assert state.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert requests == []


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_same_execution_trusted_target_drift_fails_before_operation(
    tmp_path: Path,
    case: ActionCase,
    monkeypatch: pytest.MonkeyPatch,
):
    requests: list[httpx.Request] = []
    original_prepare = graph_module._prepare_intent
    runtime: WriteToolRuntime

    def prepare_then_change_trusted_target(state: AgentState) -> dict[str, object]:
        prepared = original_prepare(state)
        if case.slug in {"specialist", "criticality"}:
            object.__setattr__(runtime, "central_asset_id", "asset_other")
        elif case.slug == "retraining":
            object.__setattr__(runtime, "configured_model_id", "mdl_other")
        else:
            object.__setattr__(runtime, "current_case_id", "case_other")
        return prepared

    async def scenario():
        nonlocal runtime
        runtime = _runtime(lambda request: requests.append(request), case)
        try:
            monkeypatch.setattr(
                graph_module,
                "_prepare_intent",
                prepare_then_change_trusted_target,
            )
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                return await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_target_drift_{case.slug}",
                    request_id=f"req_target_drift_{case.slug}",
                    execution_id=f"exec_target_drift_{case.slug}",
                    proposal=case.proposal,
                    original_approval=case.approval,
                )
        finally:
            await runtime.client.aclose()

    state = asyncio.run(scenario())
    intent = state.intents[0]

    assert intent.status is IntentStatus.FAILED
    assert intent.attempts == 0
    assert intent.error is not None
    assert intent.error.code == "INTENT_SCOPE_MISMATCH"
    assert state.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert requests == []


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
@pytest.mark.parametrize("authorization_changed", [False, True])
def test_restart_guard_precedes_scope_hash_policy_preflight_and_operation(
    tmp_path: Path,
    case: ActionCase,
    authorization_changed: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_path = tmp_path / f"checkpoints-{case.slug}.sqlite3"
    operation_calls = 0

    async def crash_after_prepare(proposal, runtime):
        nonlocal operation_calls
        operation_calls += 1
        raise RuntimeError("queda depois do checkpoint prepared")

    async def forbidden_operation(*args, **kwargs):
        raise AssertionError("restart não pode repetir a operação")

    def forbidden_validation(*args, **kwargs):
        raise AssertionError("guard de execution_id deve ocorrer primeiro")

    async def scenario():
        initial_runtime = _runtime(
            lambda request: pytest.fail("operação falsa não deve alcançar HTTP"),
            case,
        )
        resumed_runtime = _runtime(
            lambda request: pytest.fail("restart não deve alcançar HTTP"),
            case,
            permissions=(
                frozenset({"read"})
                if authorization_changed
                else frozenset({"read", case.permission})
            ),
        )
        try:
            monkeypatch.setattr(
                graph_module,
                _OPERATION_NAMES[case.slug],
                crash_after_prepare,
            )
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                with pytest.raises(RuntimeError, match="checkpoint prepared"):
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=initial_runtime,
                        thread_id=f"thread_restart_{case.slug}",
                        request_id=f"req_restart_{case.slug}",
                        execution_id=f"exec_prepare_{case.slug}",
                        proposal=case.proposal,
                        original_approval=case.approval,
                    )
                prepared = AgentState.model_validate(
                    (
                        await graph.aget_state(
                            {
                                "configurable": {
                                    "thread_id": f"thread_restart_{case.slug}"
                                }
                            }
                        )
                    ).values
                )

            monkeypatch.setattr(
                graph_module,
                _OPERATION_NAMES[case.slug],
                forbidden_operation,
            )
            monkeypatch.setattr(
                graph_module,
                "_scope_from_proposal",
                forbidden_validation,
            )
            monkeypatch.setattr(
                graph_module,
                "_canonical_payload_hash",
                forbidden_validation,
            )
            monkeypatch.setattr(
                graph_module,
                "evaluate_write_policy",
                forbidden_validation,
            )
            async with open_checkpointer(checkpoint_path) as saver:
                uncertain = await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=resumed_runtime,
                    thread_id=f"thread_restart_{case.slug}",
                    request_id=f"req_restart_{case.slug}",
                    execution_id=f"exec_resume_{case.slug}",
                )
            return prepared, uncertain
        finally:
            await initial_runtime.client.aclose()
            await resumed_runtime.client.aclose()

    prepared, uncertain = asyncio.run(scenario())
    before = prepared.intents[0]
    after = uncertain.intents[0]

    assert before.status is IntentStatus.PREPARED
    assert before.prepared_execution_id == f"exec_prepare_{case.slug}"
    assert after.status is IntentStatus.UNCERTAIN
    assert after.attempts == 0
    assert after.error is not None
    assert after.error.code == "NON_IDEMPOTENT_OUTCOME_UNKNOWN_AFTER_RESUME"
    assert after.scope == before.scope
    assert after.idempotency_key is None
    assert after.expires_at is None
    assert uncertain.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert uncertain.review is None
    assert operation_calls == 1


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_failure_to_persist_prepared_checkpoint_prevents_operation_http(
    tmp_path: Path,
    case: ActionCase,
):
    requests: list[httpx.Request] = []
    failed = False

    async def scenario():
        nonlocal failed
        runtime = _runtime(_success_handler(requests), case)
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                original_put = saver.aput

                async def fail_prepared(config, checkpoint, metadata, new_versions):
                    nonlocal failed
                    if (
                        not failed
                        and IntentStatus.PREPARED.value
                        in _checkpoint_intent_statuses(checkpoint)
                    ):
                        failed = True
                        raise RuntimeError("falha injetada no checkpoint prepared")
                    return await original_put(
                        config,
                        checkpoint,
                        metadata,
                        new_versions,
                    )

                saver.aput = fail_prepared
                with pytest.raises(RuntimeError, match="checkpoint prepared"):
                    await invoke_agent(
                        build_agent_graph(saver),
                        request=_request(),
                        runtime=runtime,
                        thread_id=f"thread_checkpoint_{case.slug}",
                        request_id=f"req_checkpoint_{case.slug}",
                        execution_id=f"exec_checkpoint_{case.slug}",
                        proposal=case.proposal,
                        original_approval=case.approval,
                    )
        finally:
            await runtime.client.aclose()

    asyncio.run(scenario())

    assert failed is True
    assert requests == []


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_terminal_checkpoint_failure_restarts_uncertain_without_second_effect(
    tmp_path: Path,
    case: ActionCase,
):
    checkpoint_path = tmp_path / f"checkpoints-{case.slug}.sqlite3"
    requests: list[httpx.Request] = []
    terminal_failed = False

    async def scenario():
        nonlocal terminal_failed
        runtime = _runtime(_success_handler(requests), case)
        try:
            async with open_checkpointer(checkpoint_path) as saver:
                original_put = saver.aput
                original_put_writes = saver.aput_writes

                async def fail_terminal_checkpoint(
                    config,
                    checkpoint,
                    metadata,
                    new_versions,
                ):
                    nonlocal terminal_failed
                    if (
                        IntentStatus.COMPLETED.value
                        in _checkpoint_intent_statuses(checkpoint)
                    ):
                        terminal_failed = True
                        raise RuntimeError("falha injetada no checkpoint terminal")
                    return await original_put(
                        config,
                        checkpoint,
                        metadata,
                        new_versions,
                    )

                async def fail_terminal_writes(
                    config,
                    writes,
                    task_id,
                    task_path="",
                ):
                    nonlocal terminal_failed
                    if (
                        IntentStatus.COMPLETED.value
                        in _write_intent_statuses(writes)
                    ):
                        terminal_failed = True
                        raise RuntimeError("falha injetada no checkpoint terminal")
                    return await original_put_writes(
                        config,
                        writes,
                        task_id,
                        task_path,
                    )

                saver.aput = fail_terminal_checkpoint
                saver.aput_writes = fail_terminal_writes
                with pytest.raises(RuntimeError, match="checkpoint terminal"):
                    await invoke_agent(
                        build_agent_graph(saver),
                        request=_request(),
                        runtime=runtime,
                        thread_id=f"thread_terminal_{case.slug}",
                        request_id=f"req_terminal_{case.slug}",
                        execution_id=f"exec_terminal_{case.slug}",
                        proposal=case.proposal,
                        original_approval=case.approval,
                    )

            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                prepared = AgentState.model_validate(
                    (
                        await graph.aget_state(
                            {
                                "configurable": {
                                    "thread_id": f"thread_terminal_{case.slug}"
                                }
                            }
                        )
                    ).values
                )
                uncertain = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_terminal_{case.slug}",
                    request_id=f"req_terminal_{case.slug}",
                    execution_id=f"exec_terminal_resume_{case.slug}",
                )
            return prepared, uncertain
        finally:
            await runtime.client.aclose()

    prepared, uncertain = asyncio.run(scenario())

    assert terminal_failed is True
    assert prepared.intents[0].status is IntentStatus.PREPARED
    assert uncertain.intents[0].status is IntentStatus.UNCERTAIN
    assert uncertain.intents[0].attempts == 0
    assert uncertain.intents[0].error is not None
    assert (
        uncertain.intents[0].error.code
        == "NON_IDEMPOTENT_OUTCOME_UNKNOWN_AFTER_RESUME"
    )
    assert uncertain.review is None
    assert len(_write_requests(requests, case)) == 1
    if case.slug == "specialist":
        assert len([request for request in requests if request.method == "GET"]) == 1


@pytest.mark.parametrize(
    ("preflight_result", "expected_status"),
    [
        ("missing_read", IntentStatus.UNCERTAIN),
        ("http_404", IntentStatus.FAILED),
        ("http_500", IntentStatus.UNCERTAIN),
        ("invalid_response", IntentStatus.UNCERTAIN),
    ],
)
def test_specialist_preflight_error_counts_one_operation_and_zero_post(
    tmp_path: Path,
    preflight_result: str,
    expected_status: IntentStatus,
):
    case = ACTION_CASES[0]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if preflight_result == "http_404":
            return httpx.Response(
                404,
                json={"code": "NOT_FOUND", "message": "Análise ausente."},
            )
        if preflight_result == "http_500":
            return httpx.Response(
                500,
                json={"code": "INTERNAL_ERROR", "message": "Falha remota."},
            )
        return httpx.Response(200, json={"mode": "complete", "data": {}})

    async def scenario():
        runtime = _runtime(
            handler,
            case,
            permissions=(
                frozenset({"action_low"})
                if preflight_result == "missing_read"
                else frozenset({"read", "action_low"})
            ),
        )
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{preflight_result}.sqlite3"
            ) as saver:
                return await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_preflight_{preflight_result}",
                    request_id=f"req_preflight_{preflight_result}",
                    execution_id=f"exec_preflight_{preflight_result}",
                    proposal=case.proposal,
                    original_approval=case.approval,
                )
        finally:
            await runtime.client.aclose()

    state = asyncio.run(scenario())
    intent = state.intents[0]

    assert intent.status is expected_status
    assert intent.attempts == 1
    assert _write_requests(requests, case) == []
    assert len(requests) == (0 if preflight_result == "missing_read" else 1)


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_permission_revoked_while_waiting_does_not_prepare_or_dispatch(
    tmp_path: Path,
    case: ActionCase,
):
    requests: list[httpx.Request] = []

    async def scenario():
        initial_runtime = _runtime(lambda request: requests.append(request), case)
        revoked_runtime = _runtime(
            lambda request: requests.append(request),
            case,
            permissions=frozenset({"read"}),
        )
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                waiting = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=initial_runtime,
                    thread_id=f"thread_revoked_{case.slug}",
                    request_id=f"req_revoked_{case.slug}",
                    execution_id=f"exec_wait_{case.slug}",
                    proposal=case.proposal,
                )
                denied = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=revoked_runtime,
                    thread_id=f"thread_revoked_{case.slug}",
                    request_id=f"req_revoked_{case.slug}",
                    execution_id=f"exec_revoked_{case.slug}",
                    confirmation=ConfirmationReply(
                        intent_id=waiting.intents[0].intent_id,
                        decision="approve",
                    ),
                )
                return denied
        finally:
            await initial_runtime.client.aclose()
            await revoked_runtime.client.aclose()

    denied = asyncio.run(scenario())

    assert denied.step_count == 3
    assert denied.intents[0].status is IntentStatus.DENIED
    assert denied.intents[0].prepared_execution_id is None
    assert requests == []


def test_retraining_confirmation_keeps_persisted_model_when_runtime_changes(
    tmp_path: Path,
):
    case = ACTION_CASES[2]
    requests: list[httpx.Request] = []

    async def scenario():
        initial_runtime = _runtime(lambda request: requests.append(request), case)
        changed_runtime = _runtime(
            lambda request: requests.append(request),
            case,
            configured_model_id="mdl_other",
        )
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                waiting = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=initial_runtime,
                    thread_id="thread_model_changed",
                    request_id="req_model_changed",
                    execution_id="exec_model_waiting",
                    proposal=case.proposal,
                )
                denied = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=changed_runtime,
                    thread_id="thread_model_changed",
                    request_id="req_model_changed",
                    execution_id="exec_model_changed",
                    confirmation=ConfirmationReply(
                        intent_id=waiting.intents[0].intent_id,
                        decision="approve",
                    ),
                )
                return waiting, denied
        finally:
            await initial_runtime.client.aclose()
            await changed_runtime.client.aclose()

    waiting, denied = asyncio.run(scenario())

    assert waiting.intents[0].scope.model_id == "mdl_vib_v3"
    assert denied.approval is not None
    assert denied.approval.target_id == "mdl_vib_v3"
    assert denied.intents[0].status is IntentStatus.DENIED
    assert denied.intents[0].prepared_execution_id is None
    assert requests == []


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
@pytest.mark.parametrize(
    ("boundary", "expected_code"),
    [
        ("case", "WRITE_CASE_SCOPE_MISMATCH"),
        ("asset", "WRITE_ASSET_SCOPE_MISMATCH"),
    ],
)
def test_runtime_scope_drift_fails_at_boundary_before_checkpoint(
    tmp_path: Path,
    case: ActionCase,
    boundary: str,
    expected_code: str,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(
            lambda request: requests.append(request),
            case,
            current_case_id=(
                "case_other" if boundary == "case" else "case_tkt_exe_12"
            ),
            central_asset_id=(
                "asset_other" if boundary == "asset" else "asset_M101"
            ),
        )
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}-{boundary}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id=f"thread_boundary_{case.slug}_{boundary}",
                        request_id=f"req_boundary_{case.slug}_{boundary}",
                        execution_id=f"exec_boundary_{case.slug}_{boundary}",
                        proposal=case.proposal,
                        original_approval=case.approval,
                    )
                snapshot = await graph.aget_state(
                    {
                        "configurable": {
                            "thread_id": f"thread_boundary_{case.slug}_{boundary}"
                        }
                    }
                )
                return error.value, snapshot
        finally:
            await runtime.client.aclose()

    error, snapshot = asyncio.run(scenario())

    assert error.code == expected_code
    assert snapshot.values == {}
    assert requests == []


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_original_approval_confirmation_source_is_rejected_before_checkpoint(
    tmp_path: Path,
    case: ActionCase,
):
    requests: list[httpx.Request] = []
    wrong_source = TrustedActionApproval(
        action=case.approval.action,
        target_id=case.approval.target_id,
        material_parameters=case.approval.material_parameters,
        source=ApprovalSource.CONFIRMATION,
    )

    async def scenario():
        runtime = _runtime(lambda request: requests.append(request), case)
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id=f"thread_source_{case.slug}",
                        request_id=f"req_source_{case.slug}",
                        execution_id=f"exec_source_{case.slug}",
                        proposal=case.proposal,
                        original_approval=wrong_source,
                    )
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": f"thread_source_{case.slug}"}}
                )
                return error.value, snapshot
        finally:
            await runtime.client.aclose()

    error, snapshot = asyncio.run(scenario())

    assert error.code == "INVALID_ORIGINAL_APPROVAL_SOURCE"
    assert snapshot.values == {}
    assert requests == []


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_terminal_replay_is_immutable_and_rejects_drift_or_fake_confirmation(
    tmp_path: Path,
    case: ActionCase,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_success_handler(requests), case)
        config = {"configurable": {"thread_id": f"thread_replay_{case.slug}"}}
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                completed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_replay_{case.slug}",
                    request_id=f"req_replay_{case.slug}",
                    execution_id=f"exec_complete_{case.slug}",
                    proposal=case.proposal,
                    original_approval=case.approval,
                )
                before = await graph.aget_state(config)
                replayed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_replay_{case.slug}",
                    request_id=f"req_replay_{case.slug}",
                    execution_id=f"exec_replay_{case.slug}",
                    proposal=case.proposal,
                    original_approval=case.approval,
                )
                proposal_data = case.proposal.model_dump(mode="python")
                proposal_data["justification"] = JUSTIFICATION + " alterada"
                with pytest.raises(AgentInvocationProtocolError) as drift_error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id=f"thread_replay_{case.slug}",
                        request_id=f"req_replay_{case.slug}",
                        execution_id=f"exec_drift_{case.slug}",
                        proposal=type(case.proposal).model_validate(proposal_data),
                        original_approval=case.approval,
                    )
                with pytest.raises(
                    AgentInvocationProtocolError
                ) as approval_drift_error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id=f"thread_replay_{case.slug}",
                        request_id=f"req_replay_{case.slug}",
                        execution_id=f"exec_approval_drift_{case.slug}",
                        proposal=case.proposal,
                        original_approval=_conflicting_approval(case),
                    )
                with pytest.raises(AgentInvocationProtocolError) as stale_error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id=f"thread_replay_{case.slug}",
                        request_id=f"req_replay_{case.slug}",
                        execution_id=f"exec_stale_{case.slug}",
                        confirmation=ConfirmationReply(
                            intent_id=completed.intents[0].intent_id,
                            decision="approve",
                        ),
                    )
                after = await graph.aget_state(config)
                return (
                    completed,
                    replayed,
                    drift_error.value,
                    approval_drift_error.value,
                    stale_error.value,
                    before,
                    after,
                )
        finally:
            await runtime.client.aclose()

    (
        completed,
        replayed,
        drift_error,
        approval_drift_error,
        stale_error,
        before,
        after,
    ) = asyncio.run(scenario())

    assert replayed == completed
    assert drift_error.code == "PROPOSAL_DRIFT"
    assert approval_drift_error.code == "ORIGINAL_APPROVAL_DRIFT"
    assert stale_error.code == "STALE_CONFIRMATION"
    assert before.config == after.config
    assert before.values == after.values
    assert len(_write_requests(requests, case)) == 1


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_terminal_confirmation_replay_is_generic_and_decision_bound(
    tmp_path: Path,
    case: ActionCase,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_success_handler(requests), case)
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                waiting = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_confirm_replay_{case.slug}",
                    request_id=f"req_confirm_replay_{case.slug}",
                    execution_id=f"exec_wait_{case.slug}",
                    proposal=case.proposal,
                )
                approval = ConfirmationReply(
                    intent_id=waiting.intents[0].intent_id,
                    decision="approve",
                )
                completed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_confirm_replay_{case.slug}",
                    request_id=f"req_confirm_replay_{case.slug}",
                    execution_id=f"exec_approve_{case.slug}",
                    confirmation=approval,
                )
                replayed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_confirm_replay_{case.slug}",
                    request_id=f"req_confirm_replay_{case.slug}",
                    execution_id=f"exec_replay_{case.slug}",
                    confirmation=approval,
                )
                with pytest.raises(AgentInvocationProtocolError) as stale_error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id=f"thread_confirm_replay_{case.slug}",
                        request_id=f"req_confirm_replay_{case.slug}",
                        execution_id=f"exec_opposite_{case.slug}",
                        confirmation=ConfirmationReply(
                            intent_id=waiting.intents[0].intent_id,
                            decision="deny",
                        ),
                    )
                return completed, replayed, stale_error.value
        finally:
            await runtime.client.aclose()

    completed, replayed, stale_error = asyncio.run(scenario())

    assert replayed == completed
    assert stale_error.code == "STALE_CONFIRMATION"
    assert len(_write_requests(requests, case)) == 1


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_terminal_human_deny_replays_only_matching_deny(
    tmp_path: Path,
    case: ActionCase,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(lambda request: requests.append(request), case)
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                waiting = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_human_deny_{case.slug}",
                    request_id=f"req_human_deny_{case.slug}",
                    execution_id=f"exec_wait_{case.slug}",
                    proposal=case.proposal,
                )
                deny = ConfirmationReply(
                    intent_id=waiting.intents[0].intent_id,
                    decision="deny",
                )
                denied = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_human_deny_{case.slug}",
                    request_id=f"req_human_deny_{case.slug}",
                    execution_id=f"exec_deny_{case.slug}",
                    confirmation=deny,
                )
                replayed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_human_deny_{case.slug}",
                    request_id=f"req_human_deny_{case.slug}",
                    execution_id=f"exec_replay_{case.slug}",
                    confirmation=deny,
                )
                with pytest.raises(AgentInvocationProtocolError) as stale_error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id=f"thread_human_deny_{case.slug}",
                        request_id=f"req_human_deny_{case.slug}",
                        execution_id=f"exec_opposite_{case.slug}",
                        confirmation=ConfirmationReply(
                            intent_id=waiting.intents[0].intent_id,
                            decision="approve",
                        ),
                    )
                return denied, replayed, stale_error.value
        finally:
            await runtime.client.aclose()

    denied, replayed, stale_error = asyncio.run(scenario())

    assert replayed == denied
    assert denied.intents[0].status is IntentStatus.DENIED
    assert stale_error.code == "STALE_CONFIRMATION"
    assert requests == []


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_policy_deny_without_interrupt_is_not_confirmation_replay(
    tmp_path: Path,
    case: ActionCase,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(
            lambda request: requests.append(request),
            case,
            permissions=frozenset({"read"}),
        )
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                denied = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_policy_deny_{case.slug}",
                    request_id=f"req_policy_deny_{case.slug}",
                    execution_id=f"exec_policy_deny_{case.slug}",
                    proposal=case.proposal,
                )
                errors = []
                for decision in ("approve", "deny"):
                    with pytest.raises(AgentInvocationProtocolError) as error:
                        await invoke_agent(
                            graph,
                            request=_request(),
                            runtime=runtime,
                            thread_id=f"thread_policy_deny_{case.slug}",
                            request_id=f"req_policy_deny_{case.slug}",
                            execution_id=(
                                f"exec_policy_deny_{case.slug}_{decision}"
                            ),
                            confirmation=ConfirmationReply(
                                intent_id=denied.intents[0].intent_id,
                                decision=decision,
                            ),
                        )
                    errors.append(error.value)
                return denied, errors
        finally:
            await runtime.client.aclose()

    denied, errors = asyncio.run(scenario())

    assert denied.intents[0].status is IntentStatus.DENIED
    assert [error.code for error in errors] == [
        "STALE_CONFIRMATION",
        "STALE_CONFIRMATION",
    ]
    assert requests == []


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
@pytest.mark.parametrize("first_terminal", ["completed", "uncertain"])
def test_new_effect_after_terminal_requires_new_intent_and_confirmation(
    tmp_path: Path,
    case: ActionCase,
    first_terminal: str,
):
    requests: list[httpx.Request] = []
    writes = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal writes
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"mode": "complete", "notes": None, "data": _analysis_payload()},
            )
        writes += 1
        if first_terminal == "uncertain" and writes == 1:
            return httpx.Response(
                500,
                json={"code": "INTERNAL_ERROR", "message": "Falha remota."},
            )
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "action_id": f"act_new_{writes}",
                "message": "Ação aceita.",
            },
        )

    async def scenario():
        runtime = _runtime(handler, case)
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}-{first_terminal}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                first = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_new_{case.slug}_{first_terminal}",
                    request_id=f"req_first_{case.slug}_{first_terminal}",
                    execution_id=f"exec_first_{case.slug}_{first_terminal}",
                    proposal=case.proposal,
                    original_approval=case.approval,
                )
                waiting = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_new_{case.slug}_{first_terminal}",
                    request_id=f"req_second_{case.slug}_{first_terminal}",
                    execution_id=f"exec_wait_{case.slug}_{first_terminal}",
                    proposal=case.proposal,
                )
                second = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_new_{case.slug}_{first_terminal}",
                    request_id=f"req_second_{case.slug}_{first_terminal}",
                    execution_id=f"exec_second_{case.slug}_{first_terminal}",
                    confirmation=ConfirmationReply(
                        intent_id=waiting.intents[-1].intent_id,
                        decision="approve",
                    ),
                )
                return first, waiting, second
        finally:
            await runtime.client.aclose()

    first, waiting, second = asyncio.run(scenario())

    assert first.intents[0].status is (
        IntentStatus.COMPLETED
        if first_terminal == "completed"
        else IntentStatus.UNCERTAIN
    )
    assert waiting.intents[-1].status is IntentStatus.AWAITING_CONFIRMATION
    assert waiting.approval is None
    assert waiting.intents[0].intent_id != waiting.intents[1].intent_id
    assert second.intents[-1].status is IntentStatus.COMPLETED
    assert second.intents[-1].idempotency_key is None
    assert len(_write_requests(requests, case)) == 2


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_concurrent_same_request_on_one_thread_dispatches_once(
    tmp_path: Path,
    case: ActionCase,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_success_handler(requests), case)
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                calls = [
                    asyncio.create_task(
                        invoke_agent(
                            graph,
                            request=_request(),
                            runtime=runtime,
                            thread_id=f"thread_concurrent_{case.slug}",
                            request_id=f"req_concurrent_{case.slug}",
                            execution_id=f"exec_concurrent_{case.slug}_{index}",
                            proposal=case.proposal,
                            original_approval=case.approval,
                        )
                    )
                    for index in range(2)
                ]
                return await asyncio.gather(*calls)
        finally:
            await runtime.client.aclose()

    first, second = asyncio.run(scenario())

    assert first.intents == second.intents
    assert len(first.intents) == 1
    assert first.intents[0].status is IntentStatus.COMPLETED
    assert len(_write_requests(requests, case)) == 1


@pytest.mark.parametrize("case", ACTION_CASES, ids=lambda case: case.slug)
def test_different_threads_execute_independently(
    tmp_path: Path,
    case: ActionCase,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_success_handler(requests), case)
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{case.slug}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                calls = [
                    asyncio.create_task(
                        invoke_agent(
                            graph,
                            request=_request(),
                            runtime=runtime,
                            thread_id=f"thread_independent_{case.slug}_{index}",
                            request_id=f"req_independent_{case.slug}",
                            execution_id=f"exec_independent_{case.slug}_{index}",
                            proposal=case.proposal,
                            original_approval=case.approval,
                        )
                    )
                    for index in range(2)
                ]
                return await asyncio.gather(*calls)
        finally:
            await runtime.client.aclose()

    first, second = asyncio.run(scenario())

    assert first.thread_id != second.thread_id
    assert first.intents[0].intent_id != second.intents[0].intent_id
    assert len(_write_requests(requests, case)) == 2
