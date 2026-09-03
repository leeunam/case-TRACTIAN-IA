from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

import tractian_agent.graph as graph_module
from tractian_agent.checkpoint import open_checkpointer
from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import (
    ActionReceipt,
    ApiError,
    ApiErrorCategory,
    Identity,
    SupportRequest,
)
from tractian_agent.entrypoint import AgentInvocationProtocolError, invoke_agent
from tractian_agent.graph import build_agent_graph
from tractian_agent.observability import RecordingTelemetry, SpanName
from tractian_agent.state import AgentDecision, AgentState, ThreadScope
from tractian_agent.tools.runtime import ReadToolRuntime, WriteToolRuntime
from tractian_agent.write_contracts import (
    ConfirmationReply,
    IntentStatus,
    ReprocessIntentScope,
    WriteIntent,
)
from tractian_agent.write_policy import (
    ApprovalSource,
    PolicyDecision,
    PolicyReason,
    ReprocessProposal,
    RequestSpecialistAnalysisProposal,
    TrustedActionApproval,
    TrustedWriteContext,
    WritePolicyResult,
    canonical_write_payload_hash,
)


JUSTIFICATION = "O rolamento foi trocado e a análise precisa ser refeita."


def _request() -> SupportRequest:
    return SupportRequest(
        case_id="case_tkt_exe_12",
        ticket_id="TKT-EXE-12",
        asset_id="asset_M101",
        message="Reprocesse a análise depois da troca do rolamento.",
        identity=Identity(user_id="usr_ana", company_id="comp_forja_br"),
    )


def _proposal() -> ReprocessProposal:
    return ReprocessProposal(
        analysis_id="an_9901",
        justification=JUSTIFICATION,
    )


def _approval() -> TrustedActionApproval:
    return TrustedActionApproval(
        action="reprocess_analysis",
        target_id="an_9901",
        source=ApprovalSource.ORIGINAL_REQUEST,
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


def _success_handler(requests: list[httpx.Request]):
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
                "accepted": True,
                "action_id": "act_reprocess_01",
                "message": "Reprocesso aceito.",
            },
        )

    return handler


def _runtime(
    handler: Any,
    *,
    permissions: frozenset[str] = frozenset({"read", "action_low"}),
    central_asset_id: str = "asset_M101",
    current_case_id: str = "case_tkt_exe_12",
) -> WriteToolRuntime:
    return WriteToolRuntime.create(
        user_id="usr_ana",
        company_id="comp_forja_br",
        permissions=permissions,
        central_asset_id=central_asset_id,
        current_case_id=current_case_id,
        client=IndustrialApiClient(
            "https://simulator.test",
            transport=httpx.MockTransport(handler),
        ),
    )


def _expected_payload_hash() -> str:
    return canonical_write_payload_hash(
        _proposal(),
        trusted_context=_trusted_context(),
    )


def _trusted_context() -> TrustedWriteContext:
    return TrustedWriteContext(
        central_asset_id="asset_M101",
        current_case_id="case_tkt_exe_12",
        configured_model_id="mdl_vib_v3",
    )


def test_original_approval_completes_reprocess_in_five_steps(tmp_path: Path):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_success_handler(requests))
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                return await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_reprocess_direct",
                    request_id="req_reprocess_direct",
                    execution_id="exec_reprocess_direct",
                    proposal=_proposal(),
                    original_approval=_approval(),
                )
        finally:
            await runtime.client.aclose()

    state = asyncio.run(scenario())

    assert state.step_limit == 5
    assert state.step_count == 5
    assert state.pending_proposal == _proposal()
    assert len(state.intents) == 1
    intent = state.intents[0]
    assert intent.request_id == "req_reprocess_direct"
    assert intent.status is IntentStatus.COMPLETED
    assert intent.approval_source is ApprovalSource.ORIGINAL_REQUEST
    assert intent.payload_hash == _expected_payload_hash()
    assert intent.idempotency_key == f"tractian-agent:{intent.intent_id}"
    assert intent.attempts == 1
    assert intent.receipt is not None and intent.receipt.accepted is True
    action_evidence = [
        item for item in state.ledger.items if item.intent_id == intent.intent_id
    ]
    assert len(action_evidence) == 1
    assert action_evidence[0].action == "reprocess_analysis"
    assert action_evidence[0].tool is None
    assert action_evidence[0].resource == "/actions/act_reprocess_01"
    assert action_evidence[0].fact_path == "accepted"
    assert action_evidence[0].value.to_python() is True
    assert [(item.method, item.url.path) for item in requests] == [
        ("GET", "/analyses/an_9901"),
        ("POST", "/analyses/an_9901/reprocess"),
    ]
    assert requests[1].headers["idempotency-key"] == intent.idempotency_key


def test_policy_deny_finishes_in_two_steps_without_interrupt_key_or_http(
    tmp_path: Path,
):
    requests: list[httpx.Request] = []
    telemetry = RecordingTelemetry(pseudonym_key=b"p" * 32)

    async def scenario():
        runtime = _runtime(
            lambda request: requests.append(request),
            permissions=frozenset({"read"}),
        )
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver, telemetry=telemetry)
                state = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_reprocess_denied",
                    request_id="req_reprocess_denied",
                    execution_id="exec_reprocess_denied",
                    proposal=_proposal(),
                )
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_reprocess_denied"}}
                )
                return state, snapshot
        finally:
            await runtime.client.aclose()

    state, snapshot = asyncio.run(scenario())

    assert state.step_count == 2
    assert state.intents[0].status is IntentStatus.DENIED
    assert state.intents[0].idempotency_key is None
    assert snapshot.interrupts == ()
    assert requests == []
    policy_span = next(span for span in telemetry.spans if span.name is SpanName.POLICY)
    assert dict(policy_span.attributes)["outcome"] == "denied"
    assert dict(policy_span.attributes)["error_code"] == "policy_blocked"
    response_span = next(
        span for span in telemetry.spans if span.name is SpanName.RESPONSE
    )
    assert dict(response_span.attributes)["outcome"] == "denied"
    assert dict(response_span.attributes)["error_code"] == "policy_blocked"


def test_missing_approval_interrupts_with_persisted_intent_and_safe_prompt(
    tmp_path: Path,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(lambda request: requests.append(request))
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                state = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_reprocess_interrupt",
                    request_id="req_reprocess_interrupt",
                    execution_id="exec_reprocess_interrupt",
                    proposal=_proposal(),
                )
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_reprocess_interrupt"}}
                )
                return state, snapshot
        finally:
            await runtime.client.aclose()

    state, snapshot = asyncio.run(scenario())

    assert state.step_count == 2
    assert state.intents[0].status is IntentStatus.AWAITING_CONFIRMATION
    assert state.intents[0].idempotency_key is None
    assert snapshot.next == ("confirmation_gate",)
    assert len(snapshot.interrupts) == 1
    prompt = snapshot.interrupts[0].value
    assert prompt == {
        "intent_id": state.intents[0].intent_id,
        "action": "reprocess_analysis",
        "target_id": "an_9901",
        "justification": JUSTIFICATION,
        "payload_hash": _expected_payload_hash(),
    }
    assert not {
        "identity",
        "permissions",
        "idempotency_key",
        "runtime",
    } & set(prompt)
    assert requests == []


def test_confirmation_resume_approves_with_trusted_scope_and_completes(
    tmp_path: Path,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_success_handler(requests))
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                waiting = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_reprocess_confirmed",
                    request_id="req_reprocess_confirmed",
                    execution_id="exec_reprocess_waiting",
                    proposal=_proposal(),
                )

                async def forbidden_update(*args, **kwargs):
                    raise AssertionError(
                        "snapshot interrompido não pode usar aupdate_state"
                    )

                graph.aupdate_state = forbidden_update
                completed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_reprocess_confirmed",
                    request_id="req_reprocess_confirmed",
                    execution_id="exec_reprocess_confirmed",
                    confirmation=ConfirmationReply(
                        intent_id=waiting.intents[0].intent_id,
                        decision="approve",
                    ),
                )
                return waiting, completed
        finally:
            await runtime.client.aclose()

    waiting, completed = asyncio.run(scenario())

    assert waiting.step_count == 2
    assert completed.step_count == 5
    assert completed.execution_id == "exec_reprocess_confirmed"
    assert completed.approval is not None
    assert completed.approval.source is ApprovalSource.CONFIRMATION
    assert completed.approval.target_id == "an_9901"
    assert completed.intents[0].status is IntentStatus.COMPLETED
    assert completed.intents[0].approval_source is ApprovalSource.CONFIRMATION
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("boundary", "expected_code"),
    [
        ("runtime", "WRITE_RUNTIME_REQUIRED"),
        ("case", "WRITE_CASE_SCOPE_MISMATCH"),
        ("asset", "WRITE_ASSET_SCOPE_MISMATCH"),
        ("budget", "STEP_LIMIT_EXHAUSTED"),
    ],
)
def test_write_boundary_fails_before_first_checkpoint(
    tmp_path: Path,
    boundary: str,
    expected_code: str,
):
    requests: list[httpx.Request] = []

    async def scenario():
        write_runtime = _runtime(
            lambda request: requests.append(request),
            central_asset_id=("asset_other" if boundary == "asset" else "asset_M101"),
            current_case_id=("case_other" if boundary == "case" else "case_tkt_exe_12"),
        )
        runtime: ReadToolRuntime = write_runtime
        if boundary == "runtime":
            runtime = ReadToolRuntime.create(
                user_id="usr_ana",
                company_id="comp_forja_br",
                permissions=frozenset({"read", "action_low"}),
                central_asset_id="asset_M101",
                client=write_runtime.client,
            )
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{boundary}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id=f"thread_boundary_{boundary}",
                        request_id=f"req_boundary_{boundary}",
                        execution_id=f"exec_boundary_{boundary}",
                        step_limit=(4 if boundary == "budget" else None),
                        proposal=_proposal(),
                    )
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": f"thread_boundary_{boundary}"}}
                )
                return error.value, snapshot
        finally:
            await write_runtime.client.aclose()

    error, snapshot = asyncio.run(scenario())

    assert error.code == expected_code
    assert snapshot.values == {}
    assert requests == []


def test_original_approval_rejects_confirmation_source_before_checkpoint(
    tmp_path: Path,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(lambda request: requests.append(request))
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_invalid_original_source",
                        request_id="req_invalid_original_source",
                        execution_id="exec_invalid_original_source",
                        proposal=_proposal(),
                        original_approval=TrustedActionApproval(
                            action="reprocess_analysis",
                            target_id="an_9901",
                            source=ApprovalSource.CONFIRMATION,
                        ),
                    )
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_invalid_original_source"}}
                )
                return error.value, snapshot
        finally:
            await runtime.client.aclose()

    error, snapshot = asyncio.run(scenario())

    assert error.code == "INVALID_ORIGINAL_APPROVAL_SOURCE"
    assert snapshot.values == {}
    assert requests == []


@pytest.mark.parametrize(
    ("drift", "expected_code"),
    [
        ("proposal", "PROPOSAL_DRIFT"),
        ("approval", "ORIGINAL_APPROVAL_DRIFT"),
        ("new_request", "ACTIVE_INTENT_BLOCKS_NEW_REQUEST"),
    ],
)
def test_nonterminal_intent_rejects_drift_or_silent_replacement(
    tmp_path: Path,
    drift: str,
    expected_code: str,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(lambda request: requests.append(request))
        initial_approval = None
        if drift == "approval":
            initial_approval = TrustedActionApproval(
                action="reprocess_analysis",
                target_id="an_other",
                source=ApprovalSource.ORIGINAL_REQUEST,
            )
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{drift}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                waiting = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_drift_{drift}",
                    request_id=f"req_drift_{drift}",
                    execution_id="exec_drift_waiting",
                    proposal=_proposal(),
                    original_approval=initial_approval,
                )
                before = await graph.aget_state(
                    {"configurable": {"thread_id": f"thread_drift_{drift}"}}
                )
                next_proposal = _proposal()
                next_approval = None
                next_request_id = f"req_drift_{drift}"
                if drift == "proposal":
                    next_proposal = ReprocessProposal(
                        analysis_id="an_other",
                        justification=JUSTIFICATION,
                    )
                elif drift == "approval":
                    next_approval = _approval()
                else:
                    next_request_id = "req_replacement"
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id=f"thread_drift_{drift}",
                        request_id=next_request_id,
                        execution_id="exec_drift_invalid",
                        proposal=next_proposal,
                        original_approval=next_approval,
                    )
                after = await graph.aget_state(
                    {"configurable": {"thread_id": f"thread_drift_{drift}"}}
                )
                return waiting, error.value, before, after
        finally:
            await runtime.client.aclose()

    waiting, error, before, after = asyncio.run(scenario())

    assert waiting.intents[0].status is IntentStatus.AWAITING_CONFIRMATION
    assert error.code == expected_code
    assert before.config == after.config
    assert before.values == after.values
    assert requests == []


def test_confirmation_deny_finishes_in_three_steps_without_http(tmp_path: Path):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(lambda request: requests.append(request))
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                waiting = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_reprocess_human_denied",
                    request_id="req_reprocess_human_denied",
                    execution_id="exec_reprocess_waiting",
                    proposal=_proposal(),
                )
                return await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_reprocess_human_denied",
                    request_id="req_reprocess_human_denied",
                    execution_id="exec_reprocess_denied",
                    confirmation=ConfirmationReply(
                        intent_id=waiting.intents[0].intent_id,
                        decision="deny",
                    ),
                )
        finally:
            await runtime.client.aclose()

    denied = asyncio.run(scenario())

    assert denied.step_count == 3
    assert denied.intents[0].status is IntentStatus.DENIED
    assert denied.intents[0].decision.reason is PolicyReason.CONFIRMATION_REJECTED
    assert denied.intents[0].idempotency_key is None
    assert requests == []


def test_revoked_permission_is_rechecked_after_confirmation(tmp_path: Path):
    requests: list[httpx.Request] = []

    async def scenario():
        initial_runtime = _runtime(lambda request: requests.append(request))
        revoked_runtime = _runtime(
            lambda request: requests.append(request),
            permissions=frozenset({"read"}),
        )
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                waiting = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=initial_runtime,
                    thread_id="thread_reprocess_revoked",
                    request_id="req_reprocess_revoked",
                    execution_id="exec_reprocess_waiting",
                    proposal=_proposal(),
                )
                return await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=revoked_runtime,
                    thread_id="thread_reprocess_revoked",
                    request_id="req_reprocess_revoked",
                    execution_id="exec_reprocess_revoked",
                    confirmation=ConfirmationReply(
                        intent_id=waiting.intents[0].intent_id,
                        decision="approve",
                    ),
                )
        finally:
            await initial_runtime.client.aclose()
            await revoked_runtime.client.aclose()

    denied = asyncio.run(scenario())

    assert denied.step_count == 3
    assert denied.permissions == frozenset({"read"})
    assert denied.intents[0].status is IntentStatus.DENIED
    assert denied.intents[0].decision.reason is PolicyReason.MISSING_PERMISSION
    assert denied.intents[0].idempotency_key is None
    assert requests == []


@pytest.mark.parametrize(
    ("mode", "expected_code"),
    [
        ("missing", "CONFIRMATION_REQUIRED"),
        ("stale", "STALE_CONFIRMATION"),
        ("multiple", "AMBIGUOUS_CONFIRMATION"),
    ],
)
def test_invalid_confirmation_shapes_fail_before_mutation_or_http(
    tmp_path: Path,
    mode: str,
    expected_code: str,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(lambda request: requests.append(request))
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{mode}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                waiting = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_reprocess_{mode}",
                    request_id=f"req_reprocess_{mode}",
                    execution_id="exec_reprocess_waiting",
                    proposal=_proposal(),
                )
                config = {"configurable": {"thread_id": f"thread_reprocess_{mode}"}}
                before = await graph.aget_state(config)
                if mode == "multiple":
                    original_get_state = graph.aget_state
                    reads = 0

                    async def ambiguous_get_state(read_config):
                        nonlocal reads
                        snapshot = await original_get_state(read_config)
                        reads += 1
                        if reads == 1:
                            return SimpleNamespace(
                                values=snapshot.values,
                                config=snapshot.config,
                                next=snapshot.next,
                                interrupts=(
                                    snapshot.interrupts[0],
                                    snapshot.interrupts[0],
                                ),
                            )
                        return snapshot

                    graph.aget_state = ambiguous_get_state
                confirmation = None
                if mode != "missing":
                    confirmation = ConfirmationReply(
                        intent_id=(
                            "intent_stale"
                            if mode == "stale"
                            else waiting.intents[0].intent_id
                        ),
                        decision="approve",
                    )
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id=f"thread_reprocess_{mode}",
                        request_id=f"req_reprocess_{mode}",
                        execution_id="exec_invalid_confirmation",
                        confirmation=confirmation,
                    )
                if mode == "multiple":
                    graph.aget_state = original_get_state
                after = await graph.aget_state(config)
                return error.value, before, after
        finally:
            await runtime.client.aclose()

    error, before, after = asyncio.run(scenario())

    assert error.code == expected_code
    assert before.config == after.config
    assert before.values == after.values
    assert requests == []


def test_exact_terminal_confirmation_is_immutable_replay_and_stale_is_rejected(
    tmp_path: Path,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_success_handler(requests))
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                waiting = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_terminal_confirmation",
                    request_id="req_terminal_confirmation",
                    execution_id="exec_terminal_waiting",
                    proposal=_proposal(),
                )
                completed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_terminal_confirmation",
                    request_id="req_terminal_confirmation",
                    execution_id="exec_terminal_original",
                    confirmation=ConfirmationReply(
                        intent_id=waiting.intents[0].intent_id,
                        decision="approve",
                    ),
                )
                original_invoke = graph.ainvoke

                async def forbidden_invoke(*args, **kwargs):
                    raise AssertionError("replay terminal não pode executar o grafo")

                graph.ainvoke = forbidden_invoke
                replayed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_terminal_confirmation",
                    request_id="req_terminal_confirmation",
                    execution_id="exec_terminal_replay",
                    confirmation=ConfirmationReply(
                        intent_id=completed.intents[0].intent_id,
                        decision="approve",
                    ),
                )
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_terminal_confirmation",
                        request_id="req_terminal_confirmation",
                        execution_id="exec_terminal_stale",
                        confirmation=ConfirmationReply(
                            intent_id="intent_stale",
                            decision="approve",
                        ),
                    )
                graph.ainvoke = original_invoke
                return completed, replayed, error.value
        finally:
            await runtime.client.aclose()

    completed, replayed, stale_error = asyncio.run(scenario())

    assert replayed == completed
    assert replayed.execution_id == "exec_terminal_original"
    assert stale_error.code == "STALE_CONFIRMATION"
    assert len(requests) == 2


@pytest.mark.parametrize("drift", ["proposal", "approval"])
def test_terminal_replay_rejects_write_scope_drift_before_return(
    tmp_path: Path,
    drift: str,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_success_handler(requests))
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{drift}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                completed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_terminal_drift_{drift}",
                    request_id=f"req_terminal_drift_{drift}",
                    execution_id="exec_terminal_drift_original",
                    proposal=_proposal(),
                    original_approval=_approval(),
                )
                config = {
                    "configurable": {"thread_id": f"thread_terminal_drift_{drift}"}
                }
                before = await graph.aget_state(config)
                replay_proposal = _proposal()
                replay_approval = _approval()
                if drift == "proposal":
                    replay_proposal = ReprocessProposal(
                        analysis_id="an_other",
                        justification=JUSTIFICATION,
                    )
                else:
                    replay_approval = TrustedActionApproval(
                        action="reprocess_analysis",
                        target_id="an_other",
                        source=ApprovalSource.ORIGINAL_REQUEST,
                    )
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id=f"thread_terminal_drift_{drift}",
                        request_id=f"req_terminal_drift_{drift}",
                        execution_id="exec_terminal_drift_replay",
                        proposal=replay_proposal,
                        original_approval=replay_approval,
                    )
                after = await graph.aget_state(config)
                return completed, error.value, before, after
        finally:
            await runtime.client.aclose()

    completed, error, before, after = asyncio.run(scenario())

    assert completed.intents[0].status is IntentStatus.COMPLETED
    assert error.code == (
        "PROPOSAL_DRIFT" if drift == "proposal" else "ORIGINAL_APPROVAL_DRIFT"
    )
    assert before.config == after.config
    assert before.values == after.values
    assert len(requests) == 2


def test_terminal_confirmation_replay_rejects_opposite_consumed_decision(
    tmp_path: Path,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_success_handler(requests))
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                waiting = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_terminal_opposite",
                    request_id="req_terminal_opposite",
                    execution_id="exec_terminal_opposite_waiting",
                    proposal=_proposal(),
                )
                completed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_terminal_opposite",
                    request_id="req_terminal_opposite",
                    execution_id="exec_terminal_opposite_approved",
                    confirmation=ConfirmationReply(
                        intent_id=waiting.intents[0].intent_id,
                        decision="approve",
                    ),
                )
                config = {"configurable": {"thread_id": "thread_terminal_opposite"}}
                before = await graph.aget_state(config)
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_terminal_opposite",
                        request_id="req_terminal_opposite",
                        execution_id="exec_terminal_opposite_denied",
                        confirmation=ConfirmationReply(
                            intent_id=waiting.intents[0].intent_id,
                            decision="deny",
                        ),
                    )
                after = await graph.aget_state(config)
                return completed, error.value, before, after
        finally:
            await runtime.client.aclose()

    completed, error, before, after = asyncio.run(scenario())

    assert completed.approval is not None
    assert completed.approval.source is ApprovalSource.CONFIRMATION
    assert error.code == "STALE_CONFIRMATION"
    assert before.config == after.config
    assert before.values == after.values
    assert len(requests) == 2


def test_terminal_human_deny_accepts_only_matching_deny_replay(tmp_path: Path):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(lambda request: requests.append(request))
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                waiting = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_human_deny_replay",
                    request_id="req_human_deny_replay",
                    execution_id="exec_human_deny_waiting",
                    proposal=_proposal(),
                )
                denied = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_human_deny_replay",
                    request_id="req_human_deny_replay",
                    execution_id="exec_human_deny_original",
                    confirmation=ConfirmationReply(
                        intent_id=waiting.intents[0].intent_id,
                        decision="deny",
                    ),
                )
                replayed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_human_deny_replay",
                    request_id="req_human_deny_replay",
                    execution_id="exec_human_deny_replay",
                    confirmation=ConfirmationReply(
                        intent_id=waiting.intents[0].intent_id,
                        decision="deny",
                    ),
                )
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_human_deny_replay",
                        request_id="req_human_deny_replay",
                        execution_id="exec_human_deny_opposite",
                        confirmation=ConfirmationReply(
                            intent_id=waiting.intents[0].intent_id,
                            decision="approve",
                        ),
                    )
                return denied, replayed, error.value
        finally:
            await runtime.client.aclose()

    denied, replayed, error = asyncio.run(scenario())

    assert denied.intents[0].decision.reason is PolicyReason.CONFIRMATION_REJECTED
    assert denied.approval is None
    assert replayed == denied
    assert error.code == "STALE_CONFIRMATION"
    assert requests == []


@pytest.mark.parametrize("decision", ["approve", "deny"])
def test_policy_deny_without_interrupt_is_never_confirmation_replay(
    tmp_path: Path,
    decision: str,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(
            lambda request: requests.append(request),
            permissions=frozenset({"read"}),
        )
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-{decision}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                denied = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_policy_deny_replay_{decision}",
                    request_id=f"req_policy_deny_replay_{decision}",
                    execution_id="exec_policy_deny_original",
                    proposal=_proposal(),
                )
                config = {
                    "configurable": {
                        "thread_id": f"thread_policy_deny_replay_{decision}"
                    }
                }
                before = await graph.aget_state(config)
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id=f"thread_policy_deny_replay_{decision}",
                        request_id=f"req_policy_deny_replay_{decision}",
                        execution_id="exec_policy_deny_replay",
                        confirmation=ConfirmationReply(
                            intent_id=denied.intents[0].intent_id,
                            decision=decision,
                        ),
                    )
                after = await graph.aget_state(config)
                return denied, error.value, before, after
        finally:
            await runtime.client.aclose()

    denied, error, before, after = asyncio.run(scenario())

    assert denied.intents[0].decision.reason is PolicyReason.MISSING_PERMISSION
    assert error.code == "STALE_CONFIRMATION"
    assert before.config == after.config
    assert before.values == after.values
    assert requests == []


def test_other_write_action_uses_shared_policy_and_denies_without_permission(
    tmp_path: Path,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(
            lambda request: requests.append(request),
            permissions=frozenset({"read"}),
        )
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                state = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_specialist_denied",
                    request_id="req_specialist_denied",
                    execution_id="exec_specialist_denied",
                    proposal=RequestSpecialistAnalysisProposal(
                        analysis_id="an_9901",
                        justification=JUSTIFICATION,
                    ),
                )
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_specialist_denied"}}
                )
                return state, snapshot
        finally:
            await runtime.client.aclose()

    state, snapshot = asyncio.run(scenario())

    assert state.step_count == 2
    assert state.intents[0].status is IntentStatus.DENIED
    assert state.intents[0].decision.reason is PolicyReason.MISSING_PERMISSION
    assert snapshot.next == ()
    assert requests == []


def _legacy_intent(status: IntentStatus) -> WriteIntent:
    decision = {
        IntentStatus.AWAITING_CONFIRMATION: WritePolicyResult(
            decision=PolicyDecision.REQUIRE_CONFIRMATION,
            reason=PolicyReason.EXPLICIT_APPROVAL_REQUIRED,
        ),
        IntentStatus.DENIED: WritePolicyResult(
            decision=PolicyDecision.DENY,
            reason=PolicyReason.MISSING_PERMISSION,
        ),
    }[status]
    return WriteIntent(
        intent_id=f"legacy_{status.value}",
        scope=ReprocessIntentScope(
            action="reprocess_analysis",
            case_id="case_tkt_exe_12",
            company_id="comp_forja_br",
            user_id="usr_ana",
            analysis_id="an_9901",
            justification=JUSTIFICATION,
        ),
        payload_hash=_expected_payload_hash(),
        decision=decision,
        status=status,
    )


def test_active_legacy_intent_blocks_write_without_adoption_or_http(tmp_path: Path):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_success_handler(requests))
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                initial = AgentState(
                    request=_request(),
                    identity=runtime.identity,
                    permissions=runtime.permissions,
                    request_id="req_legacy_seed",
                    thread_id="thread_legacy_active",
                    execution_id="exec_legacy_seed",
                    thread_scope=ThreadScope(
                        thread_id="thread_legacy_active",
                        case_id="case_tkt_exe_12",
                        company_id="comp_forja_br",
                        user_id="usr_ana",
                    ),
                    trusted_write_context=_trusted_context(),
                    step_limit=3,
                    intents=(_legacy_intent(IntentStatus.AWAITING_CONFIRMATION),),
                )
                await graph.ainvoke(
                    initial.model_dump(mode="json"),
                    {"configurable": {"thread_id": "thread_legacy_active"}},
                    context=runtime,
                    durability="sync",
                )
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_legacy_active",
                        request_id="req_after_legacy",
                        execution_id="exec_after_legacy",
                        proposal=_proposal(),
                        original_approval=_approval(),
                    )
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_legacy_active"}}
                )
                return error.value, snapshot
        finally:
            await runtime.client.aclose()

    error, snapshot = asyncio.run(scenario())

    assert error.code == "LEGACY_INTENT_REQUIRES_REVIEW"
    restored = AgentState.model_validate(snapshot.values)
    assert restored.intents[0].request_id is None
    assert restored.intents[0].idempotency_key is None
    assert requests == []


def test_equal_proposals_in_distinct_requests_create_distinct_intents_and_keys(
    tmp_path: Path,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_success_handler(requests))
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                first = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_distinct_requests",
                    request_id="req_distinct_a",
                    execution_id="exec_distinct_a",
                    proposal=_proposal(),
                    original_approval=_approval(),
                )
                second = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_distinct_requests",
                    request_id="req_distinct_b",
                    execution_id="exec_distinct_b",
                    proposal=_proposal(),
                    original_approval=_approval(),
                )
                return first, second
        finally:
            await runtime.client.aclose()

    first, second = asyncio.run(scenario())

    assert first.request_id == "req_distinct_a"
    assert second.request_id == "req_distinct_b"
    assert len(first.intents) == 1
    assert len(second.intents) == 2
    assert second.intents[0].intent_id != second.intents[1].intent_id
    assert second.intents[0].idempotency_key != second.intents[1].idempotency_key
    assert len([request for request in requests if request.method == "POST"]) == 2


def test_historical_write_request_id_cannot_create_a_second_intent(
    tmp_path: Path,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_success_handler(requests))
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                for suffix in ("a", "b"):
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_historical_request",
                        request_id=f"req_historical_{suffix}",
                        execution_id=f"exec_historical_{suffix}",
                        proposal=_proposal(),
                        original_approval=_approval(),
                    )
                config = {"configurable": {"thread_id": "thread_historical_request"}}
                before = await graph.aget_state(config)
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_historical_request",
                        request_id="req_historical_a",
                        execution_id="exec_historical_reused",
                        proposal=_proposal(),
                        original_approval=_approval(),
                    )
                after = await graph.aget_state(config)
                return error.value, before, after
        finally:
            await runtime.client.aclose()

    error, before, after = asyncio.run(scenario())

    assert error.code == "REQUEST_ID_ALREADY_USED"
    assert before.config == after.config
    assert before.values == after.values
    assert len([request for request in requests if request.method == "POST"]) == 2


def test_new_read_after_completed_write_preserves_history_without_write_runtime(
    tmp_path: Path,
):
    requests: list[httpx.Request] = []

    async def scenario():
        write_runtime = _runtime(_success_handler(requests))
        read_runtime = ReadToolRuntime.create(
            user_id="usr_ana",
            company_id="comp_forja_br",
            permissions=frozenset({"read"}),
            central_asset_id="asset_M101",
            client=write_runtime.client,
        )
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                written = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=write_runtime,
                    thread_id="thread_write_then_read",
                    request_id="req_write_then_read_write",
                    execution_id="exec_write_then_read_write",
                    proposal=_proposal(),
                    original_approval=_approval(),
                )
                read = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=read_runtime,
                    thread_id="thread_write_then_read",
                    request_id="req_write_then_read_read",
                    execution_id="exec_write_then_read_read",
                )
                return written, read
        finally:
            await write_runtime.client.aclose()

    written, read = asyncio.run(scenario())

    assert written.intents[0].status is IntentStatus.COMPLETED
    assert read.request_id == "req_write_then_read_read"
    assert read.step_count == 3
    assert read.pending_proposal is None
    assert read.approval is None
    assert read.intents == written.intents
    assert len(requests) == 2


def test_concurrent_writes_return_each_callers_own_state_and_do_not_duplicate(
    tmp_path: Path,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_success_handler(requests))
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                first_task = asyncio.create_task(
                    invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_concurrent_writes",
                        request_id="req_concurrent_write_a",
                        execution_id="exec_concurrent_write_a",
                        proposal=_proposal(),
                        original_approval=_approval(),
                    )
                )
                second_task = asyncio.create_task(
                    invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_concurrent_writes",
                        request_id="req_concurrent_write_b",
                        execution_id="exec_concurrent_write_b",
                        proposal=_proposal(),
                        original_approval=_approval(),
                    )
                )
                return await asyncio.gather(first_task, second_task)
        finally:
            await runtime.client.aclose()

    first, second = asyncio.run(scenario())

    assert first.request_id == "req_concurrent_write_a"
    assert second.request_id == "req_concurrent_write_b"
    assert len(first.intents) == 1
    assert len(second.intents) == 2
    post_keys = [
        request.headers["idempotency-key"]
        for request in requests
        if request.method == "POST"
    ]
    assert len(post_keys) == 2
    assert len(set(post_keys)) == 2


def _receipt(*, accepted: bool = True) -> ActionReceipt:
    return ActionReceipt(
        accepted=accepted,
        action_id="act_matrix_01",
        message=(
            "Reprocesso aceito." if accepted else "Reprocesso recusado pela plataforma."
        ),
    )


def _api_error(
    category: ApiErrorCategory,
    code: str,
    *,
    status_code: int | None = None,
) -> ApiError:
    return ApiError(
        category=category,
        code=code,
        message="Falha normalizada na operação de reprocesso.",
        status_code=status_code,
    )


@pytest.mark.parametrize(
    ("results", "expected_status", "expected_attempts", "has_receipt"),
    [
        ([_receipt()], IntentStatus.COMPLETED, 1, True),
        ([_receipt(accepted=False)], IntentStatus.FAILED, 1, True),
        (
            [_api_error(ApiErrorCategory.API, "NOT_FOUND", status_code=404)],
            IntentStatus.FAILED,
            1,
            False,
        ),
        (
            [
                _api_error(
                    ApiErrorCategory.API,
                    "IDEMPOTENCY_PAYLOAD_CONFLICT",
                    status_code=409,
                )
            ],
            IntentStatus.FAILED,
            1,
            False,
        ),
        (
            [
                _api_error(
                    ApiErrorCategory.API,
                    "IDEMPOTENCY_IN_PROGRESS",
                    status_code=409,
                )
            ],
            IntentStatus.UNCERTAIN,
            1,
            False,
        ),
        (
            [
                _api_error(
                    ApiErrorCategory.API,
                    "IDEMPOTENCY_OUTCOME_UNKNOWN",
                    status_code=409,
                )
            ],
            IntentStatus.UNCERTAIN,
            1,
            False,
        ),
        (
            [
                _api_error(ApiErrorCategory.TIMEOUT, "REQUEST_TIMEOUT"),
                _receipt(),
            ],
            IntentStatus.COMPLETED,
            2,
            True,
        ),
        (
            [
                _api_error(ApiErrorCategory.TRANSPORT, "CONNECTION_LOST"),
                _receipt(accepted=False),
            ],
            IntentStatus.FAILED,
            2,
            True,
        ),
        (
            [
                _api_error(ApiErrorCategory.SERVER, "SERVER_ERROR", status_code=503),
                _api_error(ApiErrorCategory.TIMEOUT, "REQUEST_TIMEOUT"),
            ],
            IntentStatus.UNCERTAIN,
            2,
            False,
        ),
        (
            [
                _api_error(
                    ApiErrorCategory.INVALID_RESPONSE,
                    "INVALID_SCHEMA_RESPONSE",
                ),
                _api_error(ApiErrorCategory.API, "NOT_FOUND", status_code=404),
            ],
            IntentStatus.UNCERTAIN,
            2,
            False,
        ),
    ],
)
def test_reprocess_result_matrix_uses_at_most_one_same_key_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    results: list[ActionReceipt | ApiError],
    expected_status: IntentStatus,
    expected_attempts: int,
    has_receipt: bool,
):
    observed_keys: list[str] = []
    pending_results = list(results)

    async def fake_execute(proposal, runtime, *, idempotency_key):
        observed_keys.append(idempotency_key)
        assert proposal == _proposal()
        return pending_results.pop(0)

    monkeypatch.setattr(graph_module, "execute_reprocess_analysis", fake_execute)

    async def scenario():
        runtime = _runtime(
            lambda request: pytest.fail("operação falsa não deve alcançar HTTP")
        )
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                return await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_result_matrix",
                    request_id="req_result_matrix",
                    execution_id="exec_result_matrix",
                    proposal=_proposal(),
                    original_approval=_approval(),
                )
        finally:
            await runtime.client.aclose()

    state = asyncio.run(scenario())
    intent = state.intents[0]

    assert intent.status is expected_status
    assert intent.attempts == expected_attempts
    assert (intent.receipt is not None) is has_receipt
    assert len(observed_keys) == expected_attempts
    assert observed_keys == [intent.idempotency_key] * expected_attempts
    assert pending_results == []
    if intent.status is IntentStatus.COMPLETED:
        evidence = [
            item for item in state.ledger.items if item.intent_id == intent.intent_id
        ]
        assert len(evidence) == 1
        assert evidence[0].value.to_python() is True
    else:
        gaps = [gap for gap in state.ledger.gaps if gap.intent_id == intent.intent_id]
        assert len(gaps) == 1
        assert gaps[0].reason.value == (
            "unavailable" if intent.status is IntentStatus.UNCERTAIN else "error"
        )
    if (
        results
        and isinstance(results[0], ApiError)
        and results[0].code == "IDEMPOTENCY_PAYLOAD_CONFLICT"
    ):
        assert state.decision is AgentDecision.REQUIRE_HUMAN_REVIEW


@pytest.mark.parametrize(
    ("status_code", "expected_status", "expected_attempts"),
    [
        pytest.param(400, IntentStatus.FAILED, 1, id="400"),
        pytest.param(403, IntentStatus.FAILED, 1, id="403"),
        pytest.param(409, IntentStatus.FAILED, 1, id="409"),
        pytest.param(200, IntentStatus.UNCERTAIN, 2, id="2xx"),
    ],
)
def test_reprocess_malformed_http_response_uses_status_before_category(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    expected_status: IntentStatus,
    expected_attempts: int,
):
    requests: list[httpx.Request] = []
    operation_keys: list[str] = []
    original_operation = graph_module.execute_reprocess_analysis

    async def counted_operation(
        proposal,
        runtime,
        *,
        idempotency_key,
    ):
        operation_keys.append(idempotency_key)
        return await original_operation(
            proposal,
            runtime,
            idempotency_key=idempotency_key,
        )

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

    monkeypatch.setattr(
        graph_module,
        "execute_reprocess_analysis",
        counted_operation,
    )

    async def scenario():
        runtime = _runtime(handler)
        try:
            async with open_checkpointer(
                tmp_path / f"checkpoints-malformed-{status_code}.sqlite3"
            ) as saver:
                return await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_malformed_{status_code}",
                    request_id=f"req_malformed_{status_code}",
                    execution_id=f"exec_malformed_{status_code}",
                    proposal=_proposal(),
                    original_approval=_approval(),
                )
        finally:
            await runtime.client.aclose()

    state = asyncio.run(scenario())
    intent = state.intents[0]
    get_requests = [request for request in requests if request.method == "GET"]
    post_requests = [request for request in requests if request.method == "POST"]

    assert intent.status is expected_status
    assert intent.attempts == expected_attempts
    assert intent.error is not None
    assert intent.error.category is ApiErrorCategory.INVALID_RESPONSE
    assert intent.error.status_code == status_code
    assert len(operation_keys) == expected_attempts
    assert operation_keys == [intent.idempotency_key] * expected_attempts
    assert len(get_requests) == expected_attempts
    assert len(post_requests) == expected_attempts
    assert len(requests) == expected_attempts * 2
    assert {request.headers["idempotency-key"] for request in post_requests} == {
        intent.idempotency_key
    }


def test_prepared_checkpoint_is_observable_before_operation_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(_success_handler(requests))
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                original_execute = graph_module.execute_reprocess_analysis
                observed: list[AgentState] = []

                async def inspect_then_execute(
                    proposal,
                    operation_runtime,
                    *,
                    idempotency_key,
                ):
                    snapshot = await graph.aget_state(
                        {"configurable": {"thread_id": "thread_prepared_observable"}}
                    )
                    observed.append(AgentState.model_validate(snapshot.values))
                    return await original_execute(
                        proposal,
                        operation_runtime,
                        idempotency_key=idempotency_key,
                    )

                monkeypatch.setattr(
                    graph_module,
                    "execute_reprocess_analysis",
                    inspect_then_execute,
                )
                completed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_prepared_observable",
                    request_id="req_prepared_observable",
                    execution_id="exec_prepared_observable",
                    proposal=_proposal(),
                    original_approval=_approval(),
                )
                return observed, completed
        finally:
            await runtime.client.aclose()

    observed, completed = asyncio.run(scenario())

    assert len(observed) == 1
    prepared = observed[0].intents[0]
    assert observed[0].step_count == 4
    assert prepared.status is IntentStatus.PREPARED
    assert prepared.attempts == 0
    assert prepared.idempotency_key == completed.intents[0].idempotency_key
    assert len(requests) == 2


@pytest.mark.parametrize(
    ("mismatch", "expected_code"),
    [
        ("scope", "INTENT_SCOPE_MISMATCH"),
        ("hash", "PAYLOAD_HASH_MISMATCH"),
        ("key", "IDEMPOTENCY_KEY_INTENT_MISMATCH"),
    ],
)
def test_pre_dispatch_integrity_mismatch_fails_with_zero_http(
    mismatch: str,
    expected_code: str,
):
    requests: list[httpx.Request] = []
    runtime = _runtime(lambda request: requests.append(request))
    intent_id = "intent_integrity_01"
    scope = ReprocessIntentScope(
        action="reprocess_analysis",
        case_id="case_tkt_exe_12",
        company_id="comp_forja_br",
        user_id="usr_ana",
        analysis_id=("an_other" if mismatch == "scope" else "an_9901"),
        justification=JUSTIFICATION,
    )
    intent = WriteIntent(
        intent_id=intent_id,
        request_id="req_integrity",
        scope=scope,
        payload_hash=(
            "sha256:v1:" + "b" * 64 if mismatch == "hash" else _expected_payload_hash()
        ),
        decision=WritePolicyResult(
            decision=PolicyDecision.ALLOW,
            reason=PolicyReason.AUTHORIZED,
        ),
        status=IntentStatus.PREPARED,
        idempotency_key=(
            "tractian-agent:another-intent"
            if mismatch == "key"
            else f"tractian-agent:{intent_id}"
        ),
        expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
        prepared_execution_id="exec_integrity",
    )
    state_data = dict(
        request=_request(),
        identity=runtime.identity,
        permissions=runtime.permissions,
        request_id="req_integrity",
        thread_id="thread_integrity",
        execution_id="exec_integrity",
        thread_scope=ThreadScope(
            thread_id="thread_integrity",
            case_id="case_tkt_exe_12",
            company_id="comp_forja_br",
            user_id="usr_ana",
        ),
        step_count=4,
        step_limit=5,
        pending_proposal=_proposal(),
        approval=_approval(),
        intents=(intent,),
        trusted_write_context=_trusted_context(),
    )
    if mismatch in {"scope", "hash"}:
        with pytest.raises(ValidationError, match="ciclo de escrita"):
            AgentState(**state_data)
        asyncio.run(runtime.client.aclose())
        assert requests == []
        return
    state = AgentState(**state_data)

    async def scenario():
        try:
            result = await graph_module._execute_action(
                state,
                SimpleNamespace(context=runtime),
            )
            return AgentState.model_validate(result)
        finally:
            await runtime.client.aclose()

    failed = asyncio.run(scenario())

    assert failed.step_count == 5
    assert failed.intents[0].status is IntentStatus.FAILED
    assert failed.intents[0].attempts == 0
    assert failed.intents[0].error is not None
    assert failed.intents[0].error.code == expected_code
    assert failed.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert requests == []


@pytest.mark.parametrize("change", ["permission", "approval"])
def test_same_execution_authorization_change_fails_before_operation(
    change: str,
):
    requests: list[httpx.Request] = []
    runtime = _runtime(lambda request: requests.append(request))
    intent_id = "intent_authorization_same_execution"
    intent = WriteIntent(
        intent_id=intent_id,
        request_id="req_authorization_same_execution",
        scope=ReprocessIntentScope(
            action="reprocess_analysis",
            case_id="case_tkt_exe_12",
            company_id="comp_forja_br",
            user_id="usr_ana",
            analysis_id="an_9901",
            justification=JUSTIFICATION,
        ),
        payload_hash=_expected_payload_hash(),
        decision=WritePolicyResult(
            decision=PolicyDecision.ALLOW,
            reason=PolicyReason.AUTHORIZED,
        ),
        status=IntentStatus.PREPARED,
        idempotency_key=f"tractian-agent:{intent_id}",
        expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
        prepared_execution_id="exec_authorization_same_execution",
    )
    state = AgentState(
        request=_request(),
        identity=runtime.identity,
        permissions=(
            frozenset({"read"}) if change == "permission" else runtime.permissions
        ),
        request_id="req_authorization_same_execution",
        thread_id="thread_authorization_same_execution",
        execution_id="exec_authorization_same_execution",
        thread_scope=ThreadScope(
            thread_id="thread_authorization_same_execution",
            case_id="case_tkt_exe_12",
            company_id="comp_forja_br",
            user_id="usr_ana",
        ),
        step_count=4,
        step_limit=5,
        pending_proposal=_proposal(),
        approval=(None if change == "approval" else _approval()),
        intents=(intent,),
        trusted_write_context=_trusted_context(),
    )

    async def scenario():
        try:
            result = await graph_module._execute_action(
                state,
                SimpleNamespace(context=runtime),
            )
            return AgentState.model_validate(result)
        finally:
            await runtime.client.aclose()

    failed = asyncio.run(scenario())

    assert failed.intents[0].status is IntentStatus.FAILED
    assert failed.intents[0].attempts == 0
    assert failed.intents[0].error is not None
    assert failed.intents[0].error.code == "AUTHORIZATION_CHANGED_BEFORE_DISPATCH"
    assert failed.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert requests == []


def test_restart_with_revoked_permission_is_uncertain_without_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    requests: list[httpx.Request] = []
    original_execute = graph_module.execute_reprocess_analysis

    async def crash_after_prepared(*args, **kwargs):
        raise RuntimeError("queda depois do checkpoint prepared")

    async def scenario():
        initial_runtime = _runtime(lambda request: requests.append(request))
        revoked_runtime = _runtime(
            _success_handler(requests),
            permissions=frozenset({"read"}),
        )
        try:
            monkeypatch.setattr(
                graph_module,
                "execute_reprocess_analysis",
                crash_after_prepared,
            )
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                with pytest.raises(RuntimeError, match="checkpoint prepared"):
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=initial_runtime,
                        thread_id="thread_revoked_after_restart",
                        request_id="req_revoked_after_restart",
                        execution_id="exec_revoked_before_restart",
                        proposal=_proposal(),
                        original_approval=_approval(),
                    )
                prepared = AgentState.model_validate(
                    (
                        await graph.aget_state(
                            {
                                "configurable": {
                                    "thread_id": "thread_revoked_after_restart"
                                }
                            }
                        )
                    ).values
                )

            monkeypatch.setattr(
                graph_module,
                "execute_reprocess_analysis",
                original_execute,
            )
            async with open_checkpointer(checkpoint_path) as saver:
                uncertain = await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=revoked_runtime,
                    thread_id="thread_revoked_after_restart",
                    request_id="req_revoked_after_restart",
                    execution_id="exec_revoked_after_restart",
                )
            return prepared, uncertain
        finally:
            await initial_runtime.client.aclose()
            await revoked_runtime.client.aclose()

    prepared, uncertain = asyncio.run(scenario())

    assert prepared.intents[0].status is IntentStatus.PREPARED
    assert uncertain.intents[0].status is IntentStatus.UNCERTAIN
    assert uncertain.intents[0].attempts == 0
    assert uncertain.intents[0].error is not None
    assert uncertain.intents[0].error.code == "AUTHORIZATION_CHANGED_OUTCOME_UNKNOWN"
    assert uncertain.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert requests == []


def test_remote_commit_with_lost_response_retries_same_key_and_single_action(
    tmp_path: Path,
):
    posts: list[httpx.Request] = []
    committed: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"mode": "complete", "notes": None, "data": _analysis_payload()},
            )
        posts.append(request)
        key = request.headers["idempotency-key"]
        receipt = committed.setdefault(
            key,
            {
                "accepted": True,
                "action_id": "act_remote_commit_01",
                "message": "Reprocesso aceito.",
            },
        )
        if len(posts) == 1:
            raise httpx.ReadTimeout(
                "resposta perdida depois do commit", request=request
            )
        return httpx.Response(200, json=receipt)

    async def scenario():
        runtime = _runtime(handler)
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                return await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_lost_response",
                    request_id="req_lost_response",
                    execution_id="exec_lost_response",
                    proposal=_proposal(),
                    original_approval=_approval(),
                )
        finally:
            await runtime.client.aclose()

    completed = asyncio.run(scenario())
    intent = completed.intents[0]

    assert intent.status is IntentStatus.COMPLETED
    assert intent.attempts == 2
    assert len(posts) == 2
    assert len(committed) == 1
    assert posts[0].headers["idempotency-key"] == posts[1].headers["idempotency-key"]
    assert posts[0].content == posts[1].content


def _checkpoint_intent_statuses(checkpoint: dict[str, object]) -> set[str]:
    channel_values = checkpoint.get("channel_values", {})
    if not isinstance(channel_values, dict):
        return set()
    raw_intents = channel_values.get("intents", ())
    statuses: set[str] = set()
    for intent in raw_intents:
        if isinstance(intent, WriteIntent):
            statuses.add(intent.status.value)
        elif isinstance(intent, dict) and isinstance(intent.get("status"), str):
            statuses.add(intent["status"])
    return statuses


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


def test_failure_to_persist_prepared_checkpoint_prevents_all_http(
    tmp_path: Path,
):
    requests: list[httpx.Request] = []

    async def scenario():
        runtime = _runtime(lambda request: requests.append(request))
        failed = False
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
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
                with pytest.raises(
                    RuntimeError,
                    match="checkpoint prepared",
                ):
                    await invoke_agent(
                        build_agent_graph(saver),
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_prepared_checkpoint_failure",
                        request_id="req_prepared_checkpoint_failure",
                        execution_id="exec_prepared_checkpoint_failure",
                        proposal=_proposal(),
                        original_approval=_approval(),
                    )
            return failed
        finally:
            await runtime.client.aclose()

    failed = asyncio.run(scenario())

    assert failed is True
    assert requests == []


def test_restart_after_prepared_reuses_the_exact_persisted_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    observed_keys: list[str] = []

    async def crash_after_prepare(proposal, runtime, *, idempotency_key):
        observed_keys.append(idempotency_key)
        raise RuntimeError("queda inesperada antes de resultado terminal")

    async def complete_after_restart(proposal, runtime, *, idempotency_key):
        observed_keys.append(idempotency_key)
        return _receipt()

    async def scenario():
        runtime = _runtime(
            lambda request: pytest.fail("operação falsa não deve alcançar HTTP")
        )
        try:
            monkeypatch.setattr(
                graph_module,
                "execute_reprocess_analysis",
                crash_after_prepare,
            )
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                with pytest.raises(RuntimeError, match="queda inesperada"):
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_restart_prepared",
                        request_id="req_restart_prepared",
                        execution_id="exec_before_restart",
                        proposal=_proposal(),
                        original_approval=_approval(),
                    )
                before = AgentState.model_validate(
                    (
                        await graph.aget_state(
                            {"configurable": {"thread_id": "thread_restart_prepared"}}
                        )
                    ).values
                )

            monkeypatch.setattr(
                graph_module,
                "execute_reprocess_analysis",
                complete_after_restart,
            )
            async with open_checkpointer(checkpoint_path) as saver:
                completed = await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_restart_prepared",
                    request_id="req_restart_prepared",
                    execution_id="exec_after_restart",
                )
            return before, completed
        finally:
            await runtime.client.aclose()

    before, completed = asyncio.run(scenario())

    prepared = before.intents[0]
    terminal = completed.intents[0]
    assert prepared.status is IntentStatus.PREPARED
    assert prepared.prepared_execution_id == "exec_before_restart"
    assert completed.execution_id == "exec_after_restart"
    assert terminal.status is IntentStatus.COMPLETED
    assert terminal.idempotency_key == prepared.idempotency_key
    assert observed_keys == [prepared.idempotency_key, prepared.idempotency_key]


def test_prepared_resume_rejects_read_only_runtime_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    async def crash_after_prepare(*args, **kwargs):
        raise RuntimeError("queda com intenção prepared")

    monkeypatch.setattr(
        graph_module,
        "execute_reprocess_analysis",
        crash_after_prepare,
    )

    async def scenario():
        write_runtime = _runtime(
            lambda request: pytest.fail("operação falsa não deve alcançar HTTP")
        )
        read_runtime = ReadToolRuntime.create(
            user_id="usr_ana",
            company_id="comp_forja_br",
            permissions=frozenset({"read", "action_low"}),
            central_asset_id="asset_M101",
            client=write_runtime.client,
        )
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                with pytest.raises(RuntimeError, match="prepared"):
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=write_runtime,
                        thread_id="thread_prepared_read_runtime",
                        request_id="req_prepared_read_runtime",
                        execution_id="exec_prepared_write_runtime",
                        proposal=_proposal(),
                        original_approval=_approval(),
                    )
                config = {"configurable": {"thread_id": "thread_prepared_read_runtime"}}
                before = await graph.aget_state(config)
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=read_runtime,
                        thread_id="thread_prepared_read_runtime",
                        request_id="req_prepared_read_runtime",
                        execution_id="exec_prepared_read_only",
                    )
                after = await graph.aget_state(config)
                return error.value, before, after
        finally:
            await write_runtime.client.aclose()

    error, before, after = asyncio.run(scenario())

    assert error.code == "WRITE_RUNTIME_REQUIRED"
    assert before.config == after.config
    assert before.values == after.values


def test_expired_key_in_preparing_execution_fails_without_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    prepared_at = graph_module.datetime(2026, 8, 30, tzinfo=graph_module.timezone.utc)
    observed_times = iter([prepared_at, prepared_at + graph_module.timedelta(days=8)])
    operation_calls = 0

    def fake_now():
        return next(observed_times)

    async def forbidden_execute(*args, **kwargs):
        nonlocal operation_calls
        operation_calls += 1
        raise AssertionError("chave expirada não pode executar operação")

    monkeypatch.setattr(graph_module, "_utc_now", fake_now)
    monkeypatch.setattr(
        graph_module,
        "execute_reprocess_analysis",
        forbidden_execute,
    )

    async def scenario():
        runtime = _runtime(
            lambda request: pytest.fail("chave expirada não pode alcançar HTTP")
        )
        try:
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                return await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_expired_same_execution",
                    request_id="req_expired_same_execution",
                    execution_id="exec_expired_same_execution",
                    proposal=_proposal(),
                    original_approval=_approval(),
                )
        finally:
            await runtime.client.aclose()

    state = asyncio.run(scenario())
    intent = state.intents[0]

    assert intent.status is IntentStatus.FAILED
    assert intent.attempts == 0
    assert intent.error is not None
    assert intent.error.code == "IDEMPOTENCY_KEY_EXPIRED"
    assert operation_calls == 0


def test_expired_key_after_restart_is_uncertain_without_new_key_or_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    prepared_at = graph_module.datetime(2026, 8, 30, tzinfo=graph_module.timezone.utc)
    operation_calls = 0

    async def crash_after_prepare(*args, **kwargs):
        nonlocal operation_calls
        operation_calls += 1
        raise RuntimeError("queda antes do resultado checkpointado")

    async def forbidden_after_expiration(*args, **kwargs):
        nonlocal operation_calls
        operation_calls += 1
        raise AssertionError("retomada expirada não pode executar operação")

    async def scenario():
        runtime = _runtime(
            lambda request: pytest.fail("operação falsa não deve alcançar HTTP")
        )
        try:
            monkeypatch.setattr(graph_module, "_utc_now", lambda: prepared_at)
            monkeypatch.setattr(
                graph_module,
                "execute_reprocess_analysis",
                crash_after_prepare,
            )
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                with pytest.raises(RuntimeError, match="queda antes"):
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_expired_restart",
                        request_id="req_expired_restart",
                        execution_id="exec_expired_prepare",
                        proposal=_proposal(),
                        original_approval=_approval(),
                    )
                prepared = AgentState.model_validate(
                    (
                        await graph.aget_state(
                            {"configurable": {"thread_id": "thread_expired_restart"}}
                        )
                    ).values
                )

            monkeypatch.setattr(
                graph_module,
                "_utc_now",
                lambda: prepared_at + graph_module.timedelta(days=8),
            )
            monkeypatch.setattr(
                graph_module,
                "execute_reprocess_analysis",
                forbidden_after_expiration,
            )
            async with open_checkpointer(checkpoint_path) as saver:
                expired = await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_expired_restart",
                    request_id="req_expired_restart",
                    execution_id="exec_expired_resume",
                )
            return prepared, expired
        finally:
            await runtime.client.aclose()

    prepared, expired = asyncio.run(scenario())
    before = prepared.intents[0]
    after = expired.intents[0]

    assert before.status is IntentStatus.PREPARED
    assert after.status is IntentStatus.UNCERTAIN
    assert after.attempts == 0
    assert after.error is not None
    assert after.error.code == "IDEMPOTENCY_KEY_EXPIRED_OUTCOME_UNKNOWN"
    assert after.idempotency_key == before.idempotency_key
    assert operation_calls == 1


def test_terminal_checkpoint_failure_restarts_from_prepared_and_replays_receipt(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    posts: list[httpx.Request] = []
    committed: dict[str, dict[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"mode": "complete", "notes": None, "data": _analysis_payload()},
            )
        posts.append(request)
        receipt = committed.setdefault(
            request.headers["idempotency-key"],
            {
                "accepted": True,
                "action_id": "act_checkpoint_replay_01",
                "message": "Reprocesso aceito.",
            },
        )
        return httpx.Response(200, json=receipt)

    async def scenario():
        runtime = _runtime(handler)
        terminal_put_failed = False
        try:
            async with open_checkpointer(checkpoint_path) as saver:
                original_put_writes = saver.aput_writes
                original_put = saver.aput

                async def fail_terminal_checkpoint(
                    config,
                    checkpoint,
                    metadata,
                    new_versions,
                ):
                    nonlocal terminal_put_failed
                    if IntentStatus.COMPLETED.value in _checkpoint_intent_statuses(
                        checkpoint
                    ):
                        terminal_put_failed = True
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
                    nonlocal terminal_put_failed
                    if IntentStatus.COMPLETED.value in _write_intent_statuses(writes):
                        terminal_put_failed = True
                        raise RuntimeError("falha injetada no checkpoint terminal")
                    return await original_put_writes(
                        config,
                        writes,
                        task_id,
                        task_path,
                    )

                saver.aput_writes = fail_terminal_writes
                saver.aput = fail_terminal_checkpoint
                graph = build_agent_graph(saver)
                with pytest.raises(RuntimeError, match="checkpoint terminal"):
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_terminal_checkpoint_failure",
                        request_id="req_terminal_checkpoint_failure",
                        execution_id="exec_terminal_checkpoint_failure",
                        proposal=_proposal(),
                        original_approval=_approval(),
                    )
            async with open_checkpointer(checkpoint_path) as saver:
                reopened_graph = build_agent_graph(saver)
                prepared = AgentState.model_validate(
                    (
                        await reopened_graph.aget_state(
                            {
                                "configurable": {
                                    "thread_id": ("thread_terminal_checkpoint_failure")
                                }
                            }
                        )
                    ).values
                )
                completed = await invoke_agent(
                    reopened_graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_terminal_checkpoint_failure",
                    request_id="req_terminal_checkpoint_failure",
                    execution_id="exec_terminal_checkpoint_replay",
                )
            return terminal_put_failed, prepared, completed
        finally:
            await runtime.client.aclose()

    terminal_put_failed, prepared, completed = asyncio.run(scenario())

    assert terminal_put_failed is True
    assert prepared.intents[0].status is IntentStatus.PREPARED
    assert completed.intents[0].status is IntentStatus.COMPLETED
    assert completed.intents[0].attempts == 1
    assert completed.intents[0].idempotency_key == prepared.intents[0].idempotency_key
    assert len(posts) == 2
    assert len(committed) == 1
    assert posts[0].headers["idempotency-key"] == posts[1].headers["idempotency-key"]
