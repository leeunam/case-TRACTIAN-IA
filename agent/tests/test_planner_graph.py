from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr, ValidationError

import tractian_agent.graph as graph_module
import tractian_agent.entrypoint as entrypoint_module
from tractian_agent.checkpoint import open_checkpointer
from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import Identity, ResponseMode, SupportRequest
from tractian_agent.entrypoint import AgentInvocationProtocolError, invoke_agent
from tractian_agent.human_review import (
    ReviewApproveReply,
    ReviewEditReply,
    ReviewOperation,
    ReviewRejectReply,
    ReviewerIdentity,
    build_review_request,
)
from tractian_agent.evidence import compile_observations
from tractian_agent.graph import build_agent_graph
from tractian_agent.planner import (
    Planner,
    PlannerDecisionKind,
    PlannerStopReason,
    PlannerTerminalDecision,
)
from tractian_agent.state import (
    AgentDecision,
    AgentState,
    PersistedToolCall,
    PlannerUsage,
    ResumeAnchor,
    ThreadScope,
    ToolObservation,
    ReleaseGateOutcome,
    ReleaseGateReason,
    WriterDraft,
    WriterNextStep,
)
from tractian_agent.tools.analyses import (
    AnalysisListToolArtifact,
    AnalysisListToolOutcome,
    DegradedAnalysisListModelContent,
)
from tractian_agent.tools.assets import AssetToolArtifact, execute_get_asset
from tractian_agent.tools.knowledge import (
    ModelArtifact,
    ModelToolArtifact,
    ModelToolOutcome,
)
from tractian_agent.tools.observations import ToolSource
from tractian_agent.tools.runtime import ReadToolRuntime, WriteToolRuntime
from tractian_agent.write_contracts import (
    ConfirmationReply,
    IntentStatus,
    ReprocessIntentScope,
    WriteIntent,
)
from tractian_agent.write_policy import (
    ApprovalSource,
    EscalateCaseProposal,
    PolicyDecision,
    PolicyReason,
    ReprocessProposal,
    RequestModelRetrainingProposal,
    RequestSpecialistAnalysisProposal,
    TrustedWriteContext,
    WritePolicyResult,
    UpdateAssetCriticalityProposal,
    TrustedActionApproval,
    canonical_write_payload_hash,
)
from tractian_agent.writer import Writer


class _ScriptedPlannerModel(BaseChatModel):
    selector_responses: tuple[AIMessage, ...]
    terminal_responses: tuple[object, ...] = ()
    _selector_index: int = PrivateAttr(default=0)
    _terminal_index: int = PrivateAttr(default=0)
    _offered_tool_names: list[tuple[str, ...]] = PrivateAttr(default_factory=list)
    _events: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-planner-graph-model"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise AssertionError("o planner deve usar os wrappers públicos")

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> RunnableLambda:
        self._events.append("bind_tools")
        self._offered_tool_names.append(tuple(tool.name for tool in tools))

        async def select(_: list[BaseMessage]) -> AIMessage:
            self._events.append("selection_request")
            response = self.selector_responses[self._selector_index]
            self._selector_index += 1
            return response

        return RunnableLambda(select)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> RunnableLambda:
        assert schema is PlannerTerminalDecision
        assert include_raw is False
        self._events.append("with_structured_output")

        async def finalize(_: list[BaseMessage]) -> object:
            self._events.append("terminal_request")
            response = self.terminal_responses[self._terminal_index]
            self._terminal_index += 1
            return response

        return RunnableLambda(finalize)


class _EchoWriterModel(BaseChatModel):
    _events: list[str] = PrivateAttr(default_factory=list)
    _payloads: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "echo-writer-graph-model"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise AssertionError("o writer deve usar saída estruturada")

    def bind_tools(self, *args: Any, **kwargs: Any) -> RunnableLambda:
        raise AssertionError("o writer não pode receber tools")

    def with_structured_output(
        self,
        schema: object,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> RunnableLambda:
        assert schema is WriterDraft
        assert include_raw is False
        self._events.append("with_structured_output")

        async def write(messages: list[BaseMessage]) -> WriterDraft:
            self._events.append("writer_request")
            raw_payload = str(messages[-1].content)
            self._payloads.append(raw_payload)
            payload = json.loads(raw_payload)
            decision = AgentDecision(payload["decision"])
            next_step = {
                AgentDecision.GUIDE: WriterNextStep.MONITOR,
                AgentDecision.ACT: WriterNextStep.VERIFY_ACTION,
                AgentDecision.ESCALATE: WriterNextStep.AWAIT_ESCALATION,
                AgentDecision.REQUEST_INFORMATION: WriterNextStep.PROVIDE_INFORMATION,
                AgentDecision.REQUIRE_HUMAN_REVIEW: WriterNextStep.AWAIT_HUMAN_REVIEW,
            }[decision]
            return WriterDraft(
                decision=decision,
                evidence_ids=tuple(
                    sorted(fact["evidence_id"] for fact in payload["facts"])
                ),
                limitation_refs=tuple(
                    sorted(
                        limitation["limitation_ref"]
                        for limitation in payload["limitations"]
                    )
                ),
                next_step=next_step,
            )

        return RunnableLambda(write)


class _SequenceWriterGraphModel(BaseChatModel):
    responses: tuple[object, ...]
    _index: int = PrivateAttr(default=0)
    _payloads: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "sequence-writer-graph-model"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise AssertionError("o writer deve usar saída estruturada")

    def bind_tools(self, *args: Any, **kwargs: Any) -> RunnableLambda:
        raise AssertionError("o writer não pode receber tools")

    def with_structured_output(
        self,
        schema: object,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> RunnableLambda:
        assert schema is WriterDraft
        assert include_raw is False

        async def write(messages: list[BaseMessage]) -> object:
            self._payloads.append(str(messages[-1].content))
            response = self.responses[self._index]
            self._index += 1
            if isinstance(response, Exception):
                raise response
            return response

        return RunnableLambda(write)


class _HumanDispositionWriterModel(_EchoWriterModel):
    def with_structured_output(
        self,
        schema: object,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> RunnableLambda:
        assert schema is WriterDraft
        assert include_raw is False
        self._events.append("with_structured_output")

        async def write(messages: list[BaseMessage]) -> WriterDraft:
            self._events.append("writer_request")
            payload = json.loads(str(messages[-1].content))
            self._payloads.append(str(messages[-1].content))
            assert payload["decision"] == "guide"
            return WriterDraft(
                decision=AgentDecision.GUIDE,
                evidence_ids=tuple(
                    sorted(fact["evidence_id"] for fact in payload["facts"])
                ),
                limitation_refs=tuple(
                    sorted(
                        limitation["limitation_ref"]
                        for limitation in payload["limitations"]
                    )
                ),
                next_step=WriterNextStep.REQUEST_HUMAN_DISPOSITION,
            )

        return RunnableLambda(write)


class _ForbiddenWriterModel(BaseChatModel):
    _calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "forbidden-writer-graph-model"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise AssertionError("o writer não pode ser chamado nesta retomada")

    def bind_tools(self, *args: Any, **kwargs: Any) -> RunnableLambda:
        raise AssertionError("o writer não pode receber tools")

    def with_structured_output(
        self,
        schema: object,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> RunnableLambda:
        self._calls += 1
        raise AssertionError("o writer não pode ser chamado nesta retomada")


def _build_planner_graph(saver: object, model: BaseChatModel):
    return build_agent_graph(
        saver,
        planner=Planner(model),
        writer=Writer(_EchoWriterModel()),
    )


def _request(*, message: str = "Consulte o cadastro deste ativo.") -> SupportRequest:
    return SupportRequest(
        case_id="case_tkt_inv_04",
        ticket_id="TKT-INV-04",
        asset_id="asset_G501",
        message=message,
        identity=Identity(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
        ),
    )


def _asset_payload() -> dict[str, object]:
    return {
        "id": "asset_G501",
        "name": "Motor principal",
        "company_id": "comp_mineracao_andes",
        "criticality": "critical",
        "plant": "Planta 1",
        "line": "Britagem",
        "parent_asset_id": None,
        "machine_type": "motor_induction",
        "rotation_rpm": 1780.0,
        "bearing_pn": None,
        "bpfo_hz": None,
        "bpfi_hz": None,
        "bsf_hz": None,
        "ftf_hz": None,
        "line_frequency_hz": 60.0,
        "sensor_status": "online",
        "points": [
            {
                "id": "pt_G501_de",
                "asset_id": "asset_G501",
                "location": "DE",
                "sensor_status": "online",
            }
        ],
    }


async def _start_human_review(
    checkpoint_path,
    requests,
    *,
    writer_model=None,
    terminal_decision=None,
    step_limit=None,
    thread_id="thread_review_boundaries",
):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"mode": "complete", "notes": None, "data": _asset_payload()},
        )

    terminal_decision = terminal_decision or PlannerTerminalDecision(
        decision=PlannerDecisionKind.GUIDE,
        stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
    )
    planner_model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_asset",
                        "args": {"asset_id": "asset_G501"},
                        "id": "external_review_call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        ),
        terminal_responses=(terminal_decision,),
    )
    writer_model = writer_model or _HumanDispositionWriterModel()
    client = IndustrialApiClient(
        "https://industrial.test",
        transport=httpx.MockTransport(handler),
    )
    runtime = ReadToolRuntime.create(
        user_id="usr_pedro",
        company_id="comp_mineracao_andes",
        permissions=frozenset({"read"}),
        central_asset_id="asset_G501",
        client=client,
    )
    async with open_checkpointer(checkpoint_path) as saver:
        graph = build_agent_graph(
            saver,
            planner=Planner(planner_model),
            writer=Writer(writer_model),
        )
        waiting = await invoke_agent(
            graph,
            request=_request(),
            runtime=runtime,
            thread_id=thread_id,
            request_id="req_review_boundaries",
            execution_id="exec_review_boundaries",
            step_limit=step_limit,
        )
    return waiting, runtime, client, planner_model, writer_model


def _analysis_payload() -> dict[str, object]:
    return {
        "id": "an_9906",
        "asset_id": "asset_G501",
        "point_id": "pt_G501_de",
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


def _trusted_proposal_history(
    tool_name: str,
    request_id: str,
) -> tuple[tuple[PersistedToolCall, ...], tuple[ToolObservation, ...]]:
    if tool_name in {
        "propose_reprocess_analysis",
        "propose_request_specialist_analysis",
    }:
        call = PersistedToolCall(
            request_id=request_id,
            call_id="call_typed_analysis_authority",
            name="list_asset_analyses",
            arguments={"asset_id": "asset_G501"},
        )
        analyses = [{"id": "an_9906", "asset_id": "asset_G501"}]
        return (call,), (
            ToolObservation(
                request_id=request_id,
                call_id=call.call_id,
                content=DegradedAnalysisListModelContent(
                    mode=ResponseMode.PARTIAL,
                    notes="Descoberta tipada da solicitação.",
                    analyses=analyses,
                    total_analyses=1,
                    returned_analyses=1,
                    omitted_analyses=0,
                    truncated=False,
                    partial_data={},
                ).model_dump(mode="json"),
                artifact=AnalysisListToolArtifact(
                    tool_name=call.name,
                    arguments=call.arguments.to_python(),
                    source=ToolSource(
                        kind="industrial_api",
                        resource="/assets/asset_G501/analyses",
                    ),
                    outcome=AnalysisListToolOutcome(
                        mode=ResponseMode.PARTIAL,
                        notes="Descoberta tipada da solicitação.",
                        partial_data={},
                        analyses=analyses,
                        total_analyses=1,
                        returned_analyses=1,
                        omitted_analyses=0,
                    ),
                ),
            ),
        )
    if tool_name == "propose_request_model_retraining":
        call = PersistedToolCall(
            request_id=request_id,
            call_id="call_typed_model_authority",
            name="get_model",
            arguments={},
        )
        model = ModelArtifact(
            id="mdl_vib_v3",
            version="3.2.1",
            coverage=[],
            requirements={
                "min_completeness": 0.8,
                "min_snr_db": 12.0,
                "min_rotation_rpm": None,
            },
            processing_state="idle",
            last_run_at="2026-01-01T00:00:00Z",
        )
        return (call,), (
            ToolObservation(
                request_id=request_id,
                call_id=call.call_id,
                content=model.model_dump(mode="json"),
                artifact=ModelToolArtifact(
                    tool_name=call.name,
                    arguments={},
                    source=ToolSource(
                        kind="industrial_api",
                        resource="/models/mdl_vib_v3",
                    ),
                    outcome=ModelToolOutcome(
                        mode=ResponseMode.COMPLETE,
                        model=model,
                    ),
                ),
            ),
        )
    return (), ()


def test_planner_graph_executes_real_read_with_tool_node_and_finalizes(tmp_path):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"mode": "complete", "notes": None, "data": _asset_payload()},
        )

    model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_asset",
                        "args": {"asset_id": "asset_G501"},
                        "id": "call_get_asset_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="texto livre que não pode virar decisão"),
        ),
        terminal_responses=(
            PlannerTerminalDecision(
                decision=PlannerDecisionKind.GUIDE,
                stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
            ),
        ),
    )
    writer_model = _EchoWriterModel()
    writer = Writer(writer_model)

    async def scenario():
        checkpoint_path = tmp_path / "planner.sqlite3"
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        )
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=client,
        )
        try:
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(model),
                    writer=writer,
                )
                state = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_planner_read",
                    request_id="req_planner_read",
                    execution_id="exec_planner_read",
                )
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_planner_read"}}
                )
            async with open_checkpointer(checkpoint_path) as reopened_saver:
                reopened = AgentState.model_validate(
                    (
                        await build_agent_graph(
                            reopened_saver,
                            planner=Planner(model),
                            writer=writer,
                        ).aget_state(
                            {"configurable": {"thread_id": "thread_planner_read"}}
                        )
                    ).values
                )
            return state, snapshot, reopened
        finally:
            await client.aclose()

    state, snapshot, reopened = asyncio.run(scenario())

    assert len(requests) == 1
    assert requests[0].url.path == "/assets/asset_G501"
    assert state.step_limit == 24
    assert state.step_count == 7
    assert state.resume_anchor is ResumeAnchor.RELEASE_GATE
    assert state.decision is AgentDecision.GUIDE
    assert state.final_result is not None
    assert state.writer_draft is not None
    assert state.release_gate is not None
    assert state.release_gate.outcome is ReleaseGateOutcome.RELEASE
    assert len(state.tool_calls) == 1
    assert len(state.tool_observations) == 1
    observation = state.tool_observations[0]
    expected_id = hashlib.sha256(
        ("planner-v1\0req_planner_read\0" + "1").encode("utf-8")
    ).hexdigest()[:24]
    assert observation.call_id == f"call_planner_{expected_id}"
    assert type(observation.artifact.validated_read_artifact()) is AssetToolArtifact
    assert observation.content is not None
    assert observation.content.to_python()["id"] == "asset_G501"
    assert state.ledger.items
    assert state.ledger.gaps == ()
    assert reopened.ledger == state.ledger
    assert all(item.request_id == "req_planner_read" for item in state.ledger.items)
    assert all(item.call_id == observation.call_id for item in state.ledger.items)
    assert any(
        item.fact_path == "asset.criticality" and item.value.to_python() == "critical"
        for item in state.ledger.items
    )
    assert state.planner_terminal is not None
    assert state.planner_terminal.stop_reason == "sufficient_evidence"
    assert state.planner_failure is None
    assert snapshot.next == ()
    checkpoint_text = repr(snapshot.values)
    assert "AIMessage" not in checkpoint_text
    assert "ToolMessage" not in checkpoint_text
    assert "texto livre" not in checkpoint_text
    assert model._events == [
        "bind_tools",
        "selection_request",
        "bind_tools",
        "selection_request",
        "with_structured_output",
        "terminal_request",
    ]
    assert writer_model._events == ["with_structured_output", "writer_request"]


def test_human_approval_resumes_reopened_sqlite_without_repeating_producers(
    tmp_path,
    monkeypatch,
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"mode": "complete", "notes": None, "data": _asset_payload()},
        )

    planner_model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_asset",
                        "args": {"asset_id": "asset_G501"},
                        "id": "external_id_not_persisted",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        ),
        terminal_responses=(
            PlannerTerminalDecision(
                decision=PlannerDecisionKind.GUIDE,
                stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
            ),
        ),
    )
    writer_model = _HumanDispositionWriterModel()
    created_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(graph_module, "_utc_now", lambda: created_at)
    monkeypatch.setattr(
        entrypoint_module,
        "_utc_now",
        lambda: created_at.replace(minute=1),
    )

    async def scenario():
        checkpoint_path = tmp_path / "human-review.sqlite3"
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        )
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=client,
        )
        try:
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                waiting = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_human_review",
                    request_id="req_human_review",
                    execution_id="exec_before_review",
                )
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_human_review"}}
                )
            assert waiting.review_request is not None
            with pytest.raises(ValidationError):
                waiting.review_request.review_id = "changed"
            assert waiting.release_gate is not None
            forged_gate = waiting.release_gate.model_copy(
                update={"reason": ReleaseGateReason.NEXT_STEP_MISMATCH}
            )
            forged_request = build_review_request(
                request_id=waiting.request_id,
                request=waiting.request,
                thread_scope=waiting.thread_scope,
                permissions=waiting.permissions,
                gate=forged_gate,
                ledger=waiting.ledger,
                draft=waiting.writer_draft,
                created_at=waiting.review_request.created_at,
            )
            forged_state = waiting.model_dump(mode="json")
            forged_state["release_gate"] = None
            forged_state["review_request"] = forged_request.model_dump(mode="json")
            with pytest.raises(ValidationError, match="base do gate"):
                AgentState.model_validate(forged_state)
            async with open_checkpointer(checkpoint_path) as reopened_saver:
                reopened_graph = build_agent_graph(
                    reopened_saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                completed = await invoke_agent(
                    reopened_graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_human_review",
                    request_id="req_human_review",
                    execution_id="exec_after_review",
                    review_reply=ReviewApproveReply(
                        review_id=waiting.review_request.review_id,
                        operation="approve",
                    ),
                    reviewer=ReviewerIdentity(
                        reviewer_id="reviewer_01",
                        company_id="comp_mineracao_andes",
                        permission="review",
                    ),
                )
                replayed = await invoke_agent(
                    reopened_graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_human_review",
                    request_id="req_human_review",
                    execution_id="exec_replay_review",
                    review_reply=ReviewApproveReply(
                        review_id=waiting.review_request.review_id,
                        operation="approve",
                    ),
                    reviewer=ReviewerIdentity(
                        reviewer_id="reviewer_01",
                        company_id="comp_mineracao_andes",
                        permission="review",
                    ),
                )
                with pytest.raises(
                    AgentInvocationProtocolError,
                    match="diverge",
                ) as divergent:
                    await invoke_agent(
                        reopened_graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_human_review",
                        request_id="req_human_review",
                        execution_id="exec_divergent_review",
                        review_reply=ReviewApproveReply(
                            review_id=waiting.review_request.review_id,
                            operation="approve",
                        ),
                        reviewer=ReviewerIdentity(
                            reviewer_id="reviewer_02",
                            company_id="comp_mineracao_andes",
                            permission="review",
                        ),
                    )
            return waiting, snapshot, completed, replayed, divergent.value.code
        finally:
            await client.aclose()

    waiting, snapshot, completed, replayed, divergent_code = asyncio.run(scenario())

    assert waiting.final_result is None
    assert waiting.review_request is not None
    assert waiting.review_audit is None
    assert snapshot.next == ("await_human_review",)
    assert len(snapshot.interrupts) == 1
    assert requests and len(requests) == 1
    assert planner_model._selector_index == 2
    assert planner_model._terminal_index == 1
    assert writer_model._events == ["with_structured_output", "writer_request"]
    assert completed.final_result is not None
    assert completed.final_result.decision is AgentDecision.GUIDE
    assert completed.review_audit is not None
    assert completed.release_gate is not None
    assert completed.release_gate.outcome is ReleaseGateOutcome.RELEASE
    assert completed.release_gate.review_digest is not None
    assert completed.release_gate.review_audit_digest is not None
    forged_completed = completed.model_dump(mode="json")
    forged_completed["release_gate"] = None
    with pytest.raises(ValidationError):
        AgentState.model_validate(forged_completed)
    assert completed.step_count <= 24
    assert completed.approval == waiting.approval is None
    assert replayed == completed
    assert divergent_code == "DIVERGENT_REVIEW"


def test_eight_step_read_path_fails_closed_before_unfinishable_review(
    tmp_path,
    monkeypatch,
):
    created_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(graph_module, "_utc_now", lambda: created_at)
    requests: list[httpx.Request] = []

    async def scenario():
        state, runtime, client, planner_model, writer_model = (
            await _start_human_review(
                tmp_path / "review-budget-eight.sqlite3",
                requests,
                step_limit=8,
            )
        )
        await client.aclose()
        return state, planner_model, writer_model

    state, planner_model, writer_model = asyncio.run(scenario())
    assert state.step_count <= 8
    assert state.final_result is not None
    assert state.final_result.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert state.review_request is None
    assert state.planner_failure is not None
    assert state.planner_failure.code == "step_limit_exhausted"
    assert writer_model._events == []
    assert planner_model._terminal_index == 1
    assert len(requests) == 1


def test_legacy_review_checkpoint_exhausted_at_regate_uses_gate_terminal(
    tmp_path,
    monkeypatch,
):
    created_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(graph_module, "_utc_now", lambda: created_at)
    monkeypatch.setattr(
        entrypoint_module, "_utc_now", lambda: created_at + timedelta(minutes=1)
    )
    requests: list[httpx.Request] = []

    async def scenario():
        path = tmp_path / "review-regate-exhausted.sqlite3"
        waiting, runtime, client, planner_model, writer_model = (
            await _start_human_review(path, requests)
        )
        assert waiting.review_request is not None
        try:
            async with open_checkpointer(path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_review_boundaries"}}
                )
                limited = waiting.model_copy(
                    update={"step_limit": waiting.step_count + 1}
                )
                await graph.aupdate_state(
                    snapshot.config,
                    limited.model_dump(mode="json"),
                    as_node="release_gate",
                )
                await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_review_boundaries",
                    request_id="req_review_boundaries",
                    execution_id="exec_regate_exhausted_arm_interrupt",
                )
                return await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_review_boundaries",
                    request_id="req_review_boundaries",
                    execution_id="exec_regate_exhausted",
                    review_reply=ReviewApproveReply(
                        review_id=waiting.review_request.review_id,
                        operation="approve",
                    ),
                    reviewer=ReviewerIdentity(
                        reviewer_id="reviewer_budget",
                        company_id="comp_mineracao_andes",
                        permission="review",
                    ),
                )
        finally:
            await client.aclose()

    completed = asyncio.run(scenario())
    assert completed.step_count == completed.step_limit
    assert completed.planner_failure is None
    assert completed.release_gate is not None
    assert completed.release_gate.reason is ReleaseGateReason.STEP_BUDGET_EXHAUSTED
    assert completed.final_result is not None
    assert completed.final_result.evidence_ids == ()
    assert len(requests) == 1


@pytest.mark.parametrize("expired", [False, True])
def test_concurrent_review_replies_are_literal_idempotent_or_divergent(
    expired,
    tmp_path,
    monkeypatch,
):
    created_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(graph_module, "_utc_now", lambda: created_at)
    monkeypatch.setattr(
        entrypoint_module,
        "_utc_now",
        lambda: created_at
        + (timedelta(hours=24) if expired else timedelta(minutes=1)),
    )
    requests: list[httpx.Request] = []

    async def run_pair(name, *, divergent):
        path = tmp_path / f"review-concurrent-{name}-{expired}.sqlite3"
        waiting, runtime, client, planner_model, writer_model = (
            await _start_human_review(path, requests)
        )
        assert waiting.review_request is not None
        approve = ReviewApproveReply(
            review_id=waiting.review_request.review_id,
            operation="approve",
        )
        second_reply = (
            ReviewRejectReply(
                review_id=waiting.review_request.review_id,
                operation="reject",
            )
            if divergent
            else approve
        )
        reviewer = ReviewerIdentity(
            reviewer_id="reviewer_concurrent",
            company_id="comp_mineracao_andes",
            permission="review",
        )
        try:
            async with open_checkpointer(path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )

                async def submit(reply, suffix):
                    return await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_review_boundaries",
                        request_id="req_review_boundaries",
                        execution_id=f"exec_concurrent_{name}_{suffix}",
                        review_reply=reply,
                        reviewer=reviewer,
                    )

                return await asyncio.gather(
                    submit(approve, "first"),
                    submit(second_reply, "second"),
                    return_exceptions=True,
                )
        finally:
            await client.aclose()

    async def scenario():
        return await run_pair("same", divergent=False), await run_pair(
            "different", divergent=True
        )

    same, different = asyncio.run(scenario())
    assert all(isinstance(result, AgentState) for result in same)
    assert same[0] == same[1]
    completed = [result for result in different if isinstance(result, AgentState)]
    errors = [
        result
        for result in different
        if isinstance(result, AgentInvocationProtocolError)
    ]
    assert len(completed) == 1
    assert len(errors) == 1
    assert errors[0].code == "DIVERGENT_REVIEW"
    assert len(requests) == 2


def test_review_reply_cannot_cross_threads_with_same_request_and_clock(
    tmp_path,
    monkeypatch,
):
    created_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(graph_module, "_utc_now", lambda: created_at)
    monkeypatch.setattr(
        entrypoint_module, "_utc_now", lambda: created_at + timedelta(minutes=1)
    )
    requests: list[httpx.Request] = []

    async def scenario():
        first_path = tmp_path / "review-thread-a.sqlite3"
        second_path = tmp_path / "review-thread-b.sqlite3"
        first, runtime_a, client_a, _, _ = await _start_human_review(
            first_path, requests
        )
        second, runtime_b, client_b, planner_b, writer_b = (
            await _start_human_review(
                second_path, requests, thread_id="thread_review_other"
            )
        )
        assert first.review_request is not None
        assert second.review_request is not None
        try:
            async with open_checkpointer(second_path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_b),
                    writer=Writer(writer_b),
                )
                with pytest.raises(AgentInvocationProtocolError) as stale:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime_b,
                        thread_id="thread_review_other",
                        request_id="req_review_boundaries",
                        execution_id="exec_cross_thread_reply",
                        review_reply=ReviewRejectReply(
                            review_id=first.review_request.review_id,
                            operation="reject",
                        ),
                        reviewer=ReviewerIdentity(
                            reviewer_id="reviewer_cross_thread",
                            company_id="comp_mineracao_andes",
                            permission="review",
                        ),
                    )
            return (
                first.review_request.review_id,
                second.review_request.review_id,
                stale.value.code,
            )
        finally:
            await client_a.aclose()
            await client_b.aclose()

    first_id, second_id, code = asyncio.run(scenario())
    assert first_id != second_id
    assert code == "STALE_REVIEW"


def test_pending_review_rejects_new_request_stale_id_and_wrong_company_then_rejects(
    tmp_path,
    monkeypatch,
):
    created_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(graph_module, "_utc_now", lambda: created_at)
    monkeypatch.setattr(
        entrypoint_module, "_utc_now", lambda: created_at.replace(minute=1)
    )
    requests: list[httpx.Request] = []

    async def scenario():
        checkpoint_path = tmp_path / "review-reject.sqlite3"
        waiting, runtime, client, planner_model, writer_model = (
            await _start_human_review(checkpoint_path, requests)
        )
        assert waiting.review_request is not None
        review_id = waiting.review_request.review_id
        try:
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                with pytest.raises(AgentInvocationProtocolError) as pending:
                    await invoke_agent(
                        graph,
                        request=_request(message="Nova solicitação."),
                        runtime=runtime,
                        thread_id="thread_review_boundaries",
                        request_id="req_new_while_reviewing",
                        execution_id="exec_new_while_reviewing",
                    )
                with pytest.raises(AgentInvocationProtocolError) as stale:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_review_boundaries",
                        request_id="req_review_boundaries",
                        execution_id="exec_stale_review",
                        review_reply=ReviewRejectReply(
                            review_id="sha256:v1:" + "f" * 64,
                            operation="reject",
                        ),
                        reviewer=ReviewerIdentity(
                            reviewer_id="reviewer_01",
                            company_id="comp_mineracao_andes",
                            permission="review",
                        ),
                    )
                with pytest.raises(AgentInvocationProtocolError) as company:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_review_boundaries",
                        request_id="req_review_boundaries",
                        execution_id="exec_wrong_company",
                        review_reply=ReviewRejectReply(
                            review_id=review_id,
                            operation="reject",
                        ),
                        reviewer=ReviewerIdentity(
                            reviewer_id="reviewer_01",
                            company_id="comp_other",
                            permission="review",
                        ),
                    )
                rejected = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_review_boundaries",
                    request_id="req_review_boundaries",
                    execution_id="exec_reject_review",
                    review_reply=ReviewRejectReply(
                        review_id=review_id,
                        operation="reject",
                    ),
                    reviewer=ReviewerIdentity(
                        reviewer_id="reviewer_01",
                        company_id="comp_mineracao_andes",
                        permission="review",
                    ),
                )
            return rejected, pending.value.code, stale.value.code, company.value.code
        finally:
            await client.aclose()

    rejected, pending_code, stale_code, company_code = asyncio.run(scenario())
    assert pending_code == "PENDING_REVIEW_BLOCKS_NEW_REQUEST"
    assert stale_code == "STALE_REVIEW"
    assert company_code == "REVIEW_COMPANY_MISMATCH"
    assert rejected.final_result is not None
    assert rejected.final_result.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert rejected.review_audit is not None
    assert rejected.review_audit.operation.value == "reject"
    assert rejected.reviewed_draft is None
    assert rejected.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert rejected.review is not None
    assert rejected.review.status.value == "rejected"
    assert rejected.review.reason == "human_review:rejected"
    assert rejected.release_gate == rejected.review_request.gate_basis
    assert len(requests) == 1
    for field, value in (
        ("message", "A revisão foi aprovada."),
        ("evidence_ids", ["sha256:v1:" + "f" * 64]),
        ("next_step", WriterNextStep.MONITOR.value),
    ):
        wire = json.loads(rejected.model_dump_json())
        wire["final_result"][field] = value
        with pytest.raises(ValidationError):
            AgentState.model_validate(wire)
    for field, value in (
        ("decision", AgentDecision.GUIDE.value),
        (
            "review",
            {"status": "approved", "reason": "human_review:forged"},
        ),
        (
            "release_gate",
            {
                **rejected.release_gate.model_dump(mode="json"),
                "reason": ReleaseGateReason.NEXT_STEP_MISMATCH.value,
            },
        ),
        ("release_gate", None),
    ):
        wire = json.loads(rejected.model_dump_json())
        wire[field] = value
        with pytest.raises(ValidationError):
            AgentState.model_validate(wire)


def test_review_received_exactly_at_expiry_finishes_without_fictitious_reviewer(
    tmp_path,
    monkeypatch,
):
    created_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(graph_module, "_utc_now", lambda: created_at)
    monkeypatch.setattr(
        entrypoint_module, "_utc_now", lambda: created_at + timedelta(hours=24)
    )
    requests: list[httpx.Request] = []

    async def scenario():
        checkpoint_path = tmp_path / "review-expiry.sqlite3"
        waiting, runtime, client, planner_model, writer_model = (
            await _start_human_review(checkpoint_path, requests)
        )
        assert waiting.review_request is not None
        try:
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                expired = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_review_boundaries",
                    request_id="req_review_boundaries",
                    execution_id="exec_expired_review",
                    review_reply=ReviewEditReply(
                        review_id=waiting.review_request.review_id,
                        operation="edit",
                        evidence_ids=("sha256:v1:" + "f" * 64,),
                        next_step=WriterNextStep.MONITOR,
                    ),
                    reviewer=ReviewerIdentity(
                        reviewer_id="reviewer_not_recorded",
                        company_id="comp_mineracao_andes",
                        permission="review",
                    ),
                )
                replayed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_review_boundaries",
                    request_id="req_review_boundaries",
                    execution_id="exec_expired_review_replay",
                    review_reply=ReviewEditReply(
                        review_id=waiting.review_request.review_id,
                        operation="edit",
                        evidence_ids=("sha256:v1:" + "f" * 64,),
                        next_step=WriterNextStep.MONITOR,
                    ),
                    reviewer=ReviewerIdentity(
                        reviewer_id="reviewer_not_recorded",
                        company_id="comp_mineracao_andes",
                        permission="review",
                    ),
                )
                with pytest.raises(AgentInvocationProtocolError) as divergent:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=runtime,
                        thread_id="thread_review_boundaries",
                        request_id="req_review_boundaries",
                        execution_id="exec_expired_review_divergent",
                        review_reply=ReviewEditReply(
                            review_id=waiting.review_request.review_id,
                            operation="edit",
                            evidence_ids=("sha256:v1:" + "f" * 64,),
                            next_step=WriterNextStep.MONITOR,
                        ),
                        reviewer=ReviewerIdentity(
                            reviewer_id="reviewer_other",
                            company_id="comp_mineracao_andes",
                            permission="review",
                        ),
                    )
                return expired, replayed, divergent.value.code
        finally:
            await client.aclose()

    expired, replayed, divergent_code = asyncio.run(scenario())
    assert expired.final_result is not None
    assert expired.review_expiry is not None
    assert expired.review_expiry.expired_at == expired.review_request.expires_at
    assert expired.review_audit is None
    assert expired.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert expired.review is not None
    assert expired.review.status.value == "required"
    assert expired.review.reason == "human_review:expired"
    assert expired.release_gate == expired.review_request.gate_basis
    assert "reviewer_not_recorded" not in expired.model_dump_json()
    assert replayed == expired
    assert divergent_code == "DIVERGENT_REVIEW"
    assert len(requests) == 1
    for field, value in (
        ("message", "A revisão continua válida."),
        ("evidence_ids", ["sha256:v1:" + "f" * 64]),
        ("next_step", WriterNextStep.MONITOR.value),
    ):
        wire = json.loads(expired.model_dump_json())
        wire["final_result"][field] = value
        with pytest.raises(ValidationError):
            AgentState.model_validate(wire)
    for field, value in (
        ("decision", AgentDecision.GUIDE.value),
        (
            "review",
            {"status": "approved", "reason": "human_review:forged"},
        ),
        (
            "release_gate",
            {
                **expired.release_gate.model_dump(mode="json"),
                "reason": ReleaseGateReason.NEXT_STEP_MISMATCH.value,
            },
        ),
        ("release_gate", None),
    ):
        wire = json.loads(expired.model_dump_json())
        wire[field] = value
        with pytest.raises(ValidationError):
            AgentState.model_validate(wire)


def test_new_request_closes_expired_review_before_starting_fresh_work(
    tmp_path,
    monkeypatch,
):
    created_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(graph_module, "_utc_now", lambda: created_at)
    monkeypatch.setattr(entrypoint_module, "_utc_now", lambda: created_at)
    requests: list[httpx.Request] = []

    async def scenario():
        checkpoint_path = tmp_path / "review-auto-expiry.sqlite3"
        waiting, runtime, client, _, _ = await _start_human_review(
            checkpoint_path, requests
        )
        assert waiting.review_request is not None
        forged_missing_gate = waiting.model_dump(mode="json")
        forged_missing_gate["release_gate"] = None
        with pytest.raises(ValidationError, match="gate suspenso"):
            AgentState.model_validate(forged_missing_gate)
        drifted = waiting.continue_with(
            request=_request(),
            identity=runtime.identity,
            permissions=frozenset(),
            request_id=waiting.request_id,
            execution_id="exec_review_permission_drift",
            trusted_write_context=waiting.trusted_write_context,
        )
        assert drifted.release_gate is None
        assert AgentState.model_validate_json(drifted.model_dump_json()) == drifted
        for field, value in (
            ("permissions", ["read"]),
            ("resume_anchor", ResumeAnchor.AWAIT_HUMAN_REVIEW.value),
            ("resume_anchor", ResumeAnchor.WRITER.value),
            (
                "review",
                {"status": "approved", "reason": "human_review:approve"},
            ),
        ):
            forged_drift = drifted.model_dump(mode="json")
            forged_drift[field] = value
            with pytest.raises(ValidationError, match="gate suspenso"):
                AgentState.model_validate(forged_drift)
        old_review_id = waiting.review_request.review_id
        expired_at = waiting.review_request.expires_at
        monkeypatch.setattr(graph_module, "_utc_now", lambda: expired_at)
        monkeypatch.setattr(entrypoint_module, "_utc_now", lambda: expired_at)
        fresh_planner_model = _ScriptedPlannerModel(
            selector_responses=(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_asset",
                            "args": {"asset_id": "asset_G501"},
                            "id": "fresh_review_call",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ),
            terminal_responses=(
                PlannerTerminalDecision(
                    decision=PlannerDecisionKind.GUIDE,
                    stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
                ),
            ),
        )
        try:
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(fresh_planner_model),
                    writer=Writer(_HumanDispositionWriterModel()),
                )
                before = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_review_boundaries"}}
                )
                before_json = json.dumps(
                    before.values,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )

                async def assert_denied_without_mutation(
                    *,
                    denied_request,
                    denied_runtime,
                    denied_request_id,
                    denied_execution_id,
                    expected_code,
                ):
                    with pytest.raises(AgentInvocationProtocolError) as denied:
                        await invoke_agent(
                            graph,
                            request=denied_request,
                            runtime=denied_runtime,
                            thread_id="thread_review_boundaries",
                            request_id=denied_request_id,
                            execution_id=denied_execution_id,
                        )
                    assert denied.value.code == expected_code
                    after = await graph.aget_state(
                        {
                            "configurable": {
                                "thread_id": "thread_review_boundaries"
                            }
                        }
                    )
                    after_json = json.dumps(
                        after.values,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    assert after_json == before_json
                    assert after.next == before.next
                    assert after.interrupts == before.interrupts

                for suffix, identity in (
                    (
                        "tenant",
                        Identity(
                            user_id="usr_attacker",
                            company_id="comp_attacker",
                        ),
                    ),
                    (
                        "user",
                        Identity(
                            user_id="usr_attacker",
                            company_id="comp_mineracao_andes",
                        ),
                    ),
                ):
                    denied_runtime = ReadToolRuntime.create(
                        user_id=identity.user_id,
                        company_id=identity.company_id,
                        permissions=frozenset({"read"}),
                        central_asset_id="asset_G501",
                        client=client,
                    )
                    await assert_denied_without_mutation(
                        denied_request=_request().model_copy(
                            update={"identity": identity}
                        ),
                        denied_runtime=denied_runtime,
                        denied_request_id=f"req_attacker_{suffix}",
                        denied_execution_id=f"exec_attacker_{suffix}",
                        expected_code="THREAD_SCOPE_MISMATCH",
                    )
                for suffix, denied_request, denied_runtime, denied_request_id, denied_execution_id, expected_code in (
                    (
                        "case",
                        _request().model_copy(update={"case_id": "case_other"}),
                        runtime,
                        "req_other_case",
                        "exec_other_case",
                        "THREAD_SCOPE_MISMATCH",
                    ),
                    (
                        "request",
                        _request(message="Payload divergente na mesma request."),
                        runtime,
                        waiting.request_id,
                        "exec_request_replay",
                        "REQUEST_ID_PAYLOAD_MISMATCH",
                    ),
                    (
                        "execution",
                        _request(message="Nova request com execution repetida."),
                        runtime,
                        "req_execution_replay",
                        waiting.execution_id,
                        "EXECUTION_ID_ALREADY_USED",
                    ),
                    (
                        "asset",
                        _request().model_copy(update={"asset_id": "asset_other"}),
                        ReadToolRuntime.create(
                            user_id="usr_pedro",
                            company_id="comp_mineracao_andes",
                            permissions=frozenset({"read"}),
                            central_asset_id="asset_other",
                            client=client,
                        ),
                        "req_other_asset",
                        "exec_other_asset",
                        "THREAD_SCOPE_MISMATCH",
                    ),
                ):
                    await assert_denied_without_mutation(
                        denied_request=denied_request,
                        denied_runtime=denied_runtime,
                        denied_request_id=denied_request_id,
                        denied_execution_id=denied_execution_id,
                        expected_code=expected_code,
                    )
                no_read_runtime = ReadToolRuntime.create(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                    permissions=frozenset(),
                    central_asset_id="asset_G501",
                    client=client,
                )
                with pytest.raises(AgentInvocationProtocolError) as no_read:
                    await invoke_agent(
                        graph,
                        request=_request(message="Nova request sem leitura."),
                        runtime=no_read_runtime,
                        thread_id="thread_review_boundaries",
                        request_id="req_no_read_after_expiry",
                        execution_id="exec_no_read_after_expiry",
                    )
                assert no_read.value.code == "READ_PERMISSION_REQUIRED"
                expired_snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_review_boundaries"}}
                )
                expired = AgentState.model_validate(expired_snapshot.values)
                assert expired.review_expiry is not None
                assert expired.review_expiry.trigger == "new_request"
                assert expired.final_result is not None
                assert expired.final_result.evidence_ids == ()
                fresh = await invoke_agent(
                    graph,
                    request=_request(message="Nova solicitação após o vencimento."),
                    runtime=runtime,
                    thread_id="thread_review_boundaries",
                    request_id="req_after_expired_review",
                    execution_id="exec_after_expired_review",
                )
                return old_review_id, fresh
        finally:
            await client.aclose()

    old_review_id, fresh = asyncio.run(scenario())
    assert fresh.request_id == "req_after_expired_review"
    assert fresh.final_result is None
    assert fresh.review_request is not None
    assert fresh.review_request.review_id != old_review_id
    assert len(requests) == 2


def test_human_edit_preserves_selected_order_and_releases_through_gate(
    tmp_path,
    monkeypatch,
):
    created_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(graph_module, "_utc_now", lambda: created_at)
    monkeypatch.setattr(
        entrypoint_module, "_utc_now", lambda: created_at.replace(minute=1)
    )
    requests: list[httpx.Request] = []

    async def scenario():
        path = tmp_path / "review-edit.sqlite3"
        waiting, runtime, client, planner_model, writer_model = (
            await _start_human_review(path, requests)
        )
        assert waiting.review_request is not None
        selected = tuple(reversed(waiting.review_request.eligible_evidence_ids[:2]))
        try:
            async with open_checkpointer(path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                completed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_review_boundaries",
                    request_id="req_review_boundaries",
                    execution_id="exec_edit_review",
                    review_reply=ReviewEditReply(
                        review_id=waiting.review_request.review_id,
                        operation="edit",
                        evidence_ids=selected,
                        next_step=WriterNextStep.MONITOR,
                    ),
                    reviewer=ReviewerIdentity(
                        reviewer_id="reviewer_01",
                        company_id="comp_mineracao_andes",
                        permission="review",
                    ),
                )
                literal = ReviewEditReply(
                    review_id=waiting.review_request.review_id,
                    operation="edit",
                    evidence_ids=selected,
                    next_step=WriterNextStep.MONITOR,
                )
                replayed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_review_boundaries",
                    request_id="req_review_boundaries",
                    execution_id="exec_edit_review_replay",
                    review_reply=literal,
                    reviewer=ReviewerIdentity(
                        reviewer_id="reviewer_01",
                        company_id="comp_mineracao_andes",
                        permission="review",
                    ),
                )
                divergent_codes = []
                for suffix, divergent_reply in (
                    (
                        "order",
                        literal.model_copy(
                            update={"evidence_ids": tuple(reversed(selected))}
                        ),
                    ),
                    (
                        "next",
                        literal.model_copy(
                            update={
                                "next_step": WriterNextStep.REQUEST_HUMAN_DISPOSITION
                            }
                        ),
                    ),
                ):
                    with pytest.raises(AgentInvocationProtocolError) as divergent:
                        await invoke_agent(
                            graph,
                            request=_request(),
                            runtime=runtime,
                            thread_id="thread_review_boundaries",
                            request_id="req_review_boundaries",
                            execution_id=f"exec_edit_review_divergent_{suffix}",
                            review_reply=divergent_reply,
                            reviewer=ReviewerIdentity(
                                reviewer_id="reviewer_01",
                                company_id="comp_mineracao_andes",
                                permission="review",
                            ),
                        )
                    divergent_codes.append(divergent.value.code)
            return completed, selected, replayed, divergent_codes
        finally:
            await client.aclose()

    completed, selected, replayed, divergent_codes = asyncio.run(scenario())
    assert completed.reviewed_draft is not None
    assert completed.reviewed_draft.evidence_ids == selected
    assert completed.final_result is not None
    assert completed.final_result.evidence_ids == selected
    assert completed.release_gate.outcome is ReleaseGateOutcome.RELEASE
    assert replayed == completed
    assert divergent_codes == ["DIVERGENT_REVIEW", "DIVERGENT_REVIEW"]
    assert len(requests) == 1


def test_permission_revoked_before_regate_blocks_release_without_second_review(
    tmp_path,
    monkeypatch,
):
    created_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(graph_module, "_utc_now", lambda: created_at)
    monkeypatch.setattr(
        entrypoint_module, "_utc_now", lambda: created_at.replace(minute=1)
    )
    requests: list[httpx.Request] = []

    async def scenario():
        path = tmp_path / "review-revoked.sqlite3"
        waiting, runtime, client, planner_model, writer_model = (
            await _start_human_review(path, requests)
        )
        assert waiting.review_request is not None
        revoked = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset(),
            central_asset_id="asset_G501",
            client=client,
        )
        try:
            async with open_checkpointer(path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                with pytest.raises(AgentInvocationProtocolError) as denied:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=revoked,
                        thread_id="thread_review_boundaries",
                        request_id="req_review_boundaries",
                        execution_id="exec_revoked_review",
                        review_reply=ReviewApproveReply(
                            review_id=waiting.review_request.review_id,
                            operation="approve",
                        ),
                        reviewer=ReviewerIdentity(
                            reviewer_id="reviewer_01",
                            company_id="comp_mineracao_andes",
                            permission="review",
                        ),
                    )
                with pytest.raises(AgentInvocationProtocolError) as replay_denied:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=revoked,
                        thread_id="thread_review_boundaries",
                        request_id="req_review_boundaries",
                        execution_id="exec_revoked_review_replay",
                        review_reply=ReviewApproveReply(
                            review_id=waiting.review_request.review_id,
                            operation="approve",
                        ),
                        reviewer=ReviewerIdentity(
                            reviewer_id="reviewer_01",
                            company_id="comp_mineracao_andes",
                            permission="review",
                        ),
                    )
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_review_boundaries"}}
                )
            return (
                denied.value.code,
                replay_denied.value.code,
                AgentState.model_validate(snapshot.values),
                snapshot,
            )
        finally:
            await runtime.client.aclose()

    denial_code, replay_code, internal, snapshot = asyncio.run(scenario())
    assert denial_code == "READ_PERMISSION_REQUIRED"
    assert replay_code == "READ_PERMISSION_REQUIRED"
    assert internal.release_gate.reason.value == "permission_incompatible"
    assert internal.release_gate.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert internal.final_result is not None
    assert internal.final_result.evidence_ids == ()
    assert snapshot.next == ()
    assert snapshot.interrupts == ()
    assert len(requests) == 1


@pytest.mark.parametrize("operation", ["edit", "reject", "expiry"])
def test_review_boundary_without_read_never_returns_technical_state(
    operation,
    tmp_path,
    monkeypatch,
):
    created_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(graph_module, "_utc_now", lambda: created_at)
    requests: list[httpx.Request] = []

    async def scenario():
        path = tmp_path / f"review-no-read-{operation}.sqlite3"
        waiting, runtime, client, planner_model, writer_model = (
            await _start_human_review(path, requests)
        )
        assert waiting.review_request is not None
        received_at = (
            waiting.review_request.expires_at
            if operation == "expiry"
            else created_at + timedelta(minutes=1)
        )
        monkeypatch.setattr(entrypoint_module, "_utc_now", lambda: received_at)
        revoked = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset(),
            central_asset_id="asset_G501",
            client=client,
        )
        reply = (
            ReviewEditReply(
                review_id=waiting.review_request.review_id,
                operation="edit",
                evidence_ids=waiting.review_request.eligible_evidence_ids,
                next_step=WriterNextStep.MONITOR,
            )
            if operation == "edit"
            else ReviewRejectReply(
                review_id=waiting.review_request.review_id,
                operation="reject",
            )
        )
        reviewer = ReviewerIdentity(
            reviewer_id="reviewer_no_read",
            company_id="comp_mineracao_andes",
            permission="review",
        )
        try:
            async with open_checkpointer(path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                codes = []
                for suffix in ("first", "replay"):
                    with pytest.raises(AgentInvocationProtocolError) as denied:
                        await invoke_agent(
                            graph,
                            request=_request(),
                            runtime=revoked,
                            thread_id="thread_review_boundaries",
                            request_id="req_review_boundaries",
                            execution_id=f"exec_no_read_{operation}_{suffix}",
                            review_reply=reply,
                            reviewer=reviewer,
                        )
                    codes.append(denied.value.code)
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_review_boundaries"}}
                )
            return codes, AgentState.model_validate(snapshot.values), snapshot
        finally:
            await runtime.client.aclose()

    codes, internal, snapshot = asyncio.run(scenario())
    assert codes == ["READ_PERMISSION_REQUIRED", "READ_PERMISSION_REQUIRED"]
    assert internal.final_result is not None
    assert internal.final_result.evidence_ids == ()
    assert snapshot.next == ()
    assert snapshot.interrupts == ()
    assert len(requests) == 1


def test_two_writer_failures_can_be_edited_without_a_third_model_call(
    tmp_path,
    monkeypatch,
):
    created_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(graph_module, "_utc_now", lambda: created_at)
    monkeypatch.setattr(
        entrypoint_module, "_utc_now", lambda: created_at.replace(minute=1)
    )
    requests: list[httpx.Request] = []
    failing_writer = _SequenceWriterGraphModel(
        responses=({"invalid": True}, {"invalid": True})
    )

    async def scenario():
        path = tmp_path / "review-writer-failure.sqlite3"
        waiting, runtime, client, planner_model, _ = await _start_human_review(
            path,
            requests,
            writer_model=failing_writer,
        )
        assert waiting.review_request is not None
        try:
            async with open_checkpointer(path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(failing_writer),
                )
                return await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_review_boundaries",
                    request_id="req_review_boundaries",
                    execution_id="exec_writer_failure_edit",
                    review_reply=ReviewEditReply(
                        review_id=waiting.review_request.review_id,
                        operation="edit",
                        evidence_ids=waiting.review_request.eligible_evidence_ids,
                        next_step=WriterNextStep.MONITOR,
                    ),
                    reviewer=ReviewerIdentity(
                        reviewer_id="reviewer_01",
                        company_id="comp_mineracao_andes",
                        permission="review",
                    ),
                )
        finally:
            await client.aclose()

    completed = asyncio.run(scenario())
    assert failing_writer._index == 2
    assert completed.writer_attempts == 2
    assert completed.writer_failure is not None
    assert completed.reviewed_draft is not None
    assert completed.release_gate.outcome is ReleaseGateOutcome.RELEASE


@pytest.mark.parametrize("expired", [False, True])
def test_planner_explicit_review_is_not_converted_to_guide_and_has_no_second_loop(
    expired,
    tmp_path,
    monkeypatch,
):
    created_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(graph_module, "_utc_now", lambda: created_at)
    monkeypatch.setattr(
        entrypoint_module,
        "_utc_now",
        lambda: created_at
        + (timedelta(hours=24) if expired else timedelta(minutes=1)),
    )
    requests: list[httpx.Request] = []
    explicit_review = PlannerTerminalDecision(
        decision=PlannerDecisionKind.REQUIRE_HUMAN_REVIEW,
        stop_reason=PlannerStopReason.HUMAN_REVIEW_REQUIRED,
    )

    async def scenario():
        path = tmp_path / f"planner-explicit-review-{expired}.sqlite3"
        waiting, runtime, client, planner_model, writer_model = (
            await _start_human_review(
                path,
                requests,
                writer_model=_EchoWriterModel(),
                terminal_decision=explicit_review,
            )
        )
        assert waiting.review_request is not None
        assert waiting.review_request.allowed_operations == (
            ReviewOperation.REJECT,
        )
        try:
            async with open_checkpointer(path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                completed = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_review_boundaries",
                    request_id="req_review_boundaries",
                    execution_id=f"exec_explicit_review_{expired}",
                    review_reply=(
                        ReviewEditReply(
                            review_id=waiting.review_request.review_id,
                            operation="edit",
                            evidence_ids=("sha256:v1:" + "f" * 64,),
                            next_step=WriterNextStep.MONITOR,
                        )
                        if expired
                        else ReviewRejectReply(
                            review_id=waiting.review_request.review_id,
                            operation="reject",
                        )
                    ),
                    reviewer=ReviewerIdentity(
                        reviewer_id="reviewer_01",
                        company_id="comp_mineracao_andes",
                        permission="review",
                    ),
                )
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_review_boundaries"}}
                )
            return completed, snapshot
        finally:
            await client.aclose()

    completed, snapshot = asyncio.run(scenario())
    assert completed.planner_terminal.decision == "require_human_review"
    assert completed.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert completed.release_gate.reason.value == "human_review_requested"
    assert completed.final_result is not None
    assert (completed.review_expiry is not None) is expired
    assert snapshot.next == ()
    assert snapshot.interrupts == ()


def test_crash_after_review_audit_with_revoked_read_regates_without_duplicate(
    tmp_path,
    monkeypatch,
):
    created_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(graph_module, "_utc_now", lambda: created_at)
    monkeypatch.setattr(
        entrypoint_module, "_utc_now", lambda: created_at.replace(minute=1)
    )
    requests: list[httpx.Request] = []

    async def scenario():
        path = tmp_path / "review-audit-crash.sqlite3"
        waiting, runtime, client, planner_model, writer_model = (
            await _start_human_review(path, requests)
        )
        assert waiting.review_request is not None
        reply = ReviewApproveReply(
            review_id=waiting.review_request.review_id,
            operation="approve",
        )
        reviewer = ReviewerIdentity(
            reviewer_id="reviewer_01",
            company_id="comp_mineracao_andes",
            permission="review",
        )
        try:
            async with open_checkpointer(path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                uninterrupted = graph.ainvoke

                async def crash_after_audit(input, config, **kwargs):
                    return await uninterrupted(
                        input,
                        config,
                        interrupt_after=["await_human_review"],
                        **kwargs,
                    )

                graph.ainvoke = crash_after_audit
                audited = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_review_boundaries",
                    request_id="req_review_boundaries",
                    execution_id="exec_audit_before_crash",
                    review_reply=reply,
                    reviewer=reviewer,
                )
                assert audited.review_audit is not None
                assert audited.permissions == frozenset({"read"})
                forged_audited = audited.model_dump(mode="json")
                forged_audited["release_gate"] = None
                with pytest.raises(ValidationError, match="gate suspenso"):
                    AgentState.model_validate(forged_audited)
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_review_boundaries"}}
                )
            async with open_checkpointer(path) as saver:
                resumed_graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                revoked = ReadToolRuntime.create(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                    permissions=frozenset(),
                    central_asset_id="asset_G501",
                    client=client,
                )
                with pytest.raises(AgentInvocationProtocolError) as denied:
                    await invoke_agent(
                        resumed_graph,
                        request=_request(),
                        runtime=revoked,
                        thread_id="thread_review_boundaries",
                        request_id="req_review_boundaries",
                        execution_id="exec_replay_after_crash",
                        review_reply=reply,
                        reviewer=reviewer,
                    )
                completed_snapshot = await resumed_graph.aget_state(
                    {"configurable": {"thread_id": "thread_review_boundaries"}}
                )
            return (
                audited,
                snapshot,
                AgentState.model_validate(completed_snapshot.values),
                denied.value.code,
            )
        finally:
            await client.aclose()

    audited, snapshot, completed, denial_code = asyncio.run(scenario())
    assert audited.final_result is None
    assert audited.review_audit is not None
    assert snapshot.next == ("release_gate",)
    assert completed.final_result is not None
    assert completed.review_audit == audited.review_audit
    assert completed.release_gate.reason is ReleaseGateReason.PERMISSION_INCOMPATIBLE
    assert denial_code == "READ_PERMISSION_REQUIRED"
    assert len(requests) == 1


def test_planner_rejects_coerced_rotation_in_raw_read_artifact(
    tmp_path,
    monkeypatch,
):
    model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_asset",
                        "args": {"asset_id": "asset_G501"},
                        "id": "call_coerced_rotation",
                        "type": "tool_call",
                    }
                ],
            ),
        )
    )

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={
                        "mode": "complete",
                        "notes": None,
                        "data": _asset_payload(),
                    },
                )
            ),
        )
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=client,
        )
        result = await execute_get_asset("asset_G501", runtime)
        raw_artifact = result.artifact.model_dump(mode="json")
        raw_artifact["outcome"]["asset"]["technical_configuration"][
            "rotation_rpm"
        ] = "1780.0"

        class CoercedReadToolNode:
            def __init__(self, *args, **kwargs):
                pass

            async def ainvoke(self, *args, **kwargs):
                return {
                    "messages": [
                        ToolMessage(
                            content=json.dumps(
                                result.content.model_dump(mode="json")
                            ),
                            artifact=raw_artifact,
                            name="get_asset",
                            tool_call_id="call_coerced_rotation",
                            status="success",
                        )
                    ]
                }

        monkeypatch.setattr(
            "tractian_agent.graph.ToolNode",
            CoercedReadToolNode,
        )
        request = _request()
        state = AgentState(
            request=request,
            identity=runtime.identity,
            permissions=runtime.permissions,
            request_id="req_coerced_rotation",
            thread_id="thread_coerced_rotation",
            execution_id="exec_coerced_rotation",
            thread_scope=ThreadScope(
                thread_id="thread_coerced_rotation",
                case_id=request.case_id,
                company_id=runtime.identity.company_id,
                user_id=runtime.identity.user_id,
            ),
            step_limit=20,
        )
        config = {"configurable": {"thread_id": state.thread_id}}
        try:
            async with open_checkpointer(
                tmp_path / "coerced-rotation.sqlite3"
            ) as saver:
                graph = _build_planner_graph(saver, model)
                await graph.ainvoke(
                    state.model_dump(mode="json"),
                    config,
                    context=runtime,
                    durability="sync",
                )
                return AgentState.model_validate(
                    (await graph.aget_state(config)).values
                )
        finally:
            await client.aclose()

    state = asyncio.run(scenario())

    assert state.tool_observations == ()
    assert state.planner_failure is not None
    assert state.planner_failure.code == "invalid_tool_result"


@pytest.mark.parametrize(
    ("tool_name", "arguments", "permission", "message", "expected"),
    [
        (
            "propose_reprocess_analysis",
            {
                "analysis_id": "an_9906",
                "justification": "Há dados novos para repetir a análise.",
            },
            "action_low",
            "Reprocesse a análise an_9906.",
            ReprocessProposal(
                analysis_id="an_9906",
                justification="Há dados novos para repetir a análise.",
            ),
        ),
        (
            "propose_request_specialist_analysis",
            {
                "analysis_id": "an_9906",
                "justification": "A limitação exige análise especializada.",
            },
            "action_low",
            "Solicite especialista para an_9906.",
            RequestSpecialistAnalysisProposal(
                analysis_id="an_9906",
                justification="A limitação exige análise especializada.",
            ),
        ),
        (
            "propose_update_asset_criticality",
            {
                "criticality": "critical",
                "justification": "O impacto operacional exige prioridade máxima.",
            },
            "action_high",
            "Atualize a criticidade do ativo central.",
            UpdateAssetCriticalityProposal(
                criticality="critical",
                justification="O impacto operacional exige prioridade máxima.",
            ),
        ),
        (
            "propose_request_model_retraining",
            {
                "justification": "Erros sistemáticos sustentam novo treinamento.",
            },
            "action_high",
            "Solicite retreinamento do modelo mdl_vib_v3.",
            RequestModelRetrainingProposal(
                justification="Erros sistemáticos sustentam novo treinamento.",
            ),
        ),
        (
            "propose_escalate_case",
            {
                "justification": "O caso ultrapassa o atendimento remoto.",
            },
            "escalate",
            "Escale este caso para atendimento humano.",
            EscalateCaseProposal(
                justification="O caso ultrapassa o atendimento remoto.",
            ),
        ),
    ],
)
def test_each_planner_proposal_runs_through_tool_node_without_effect(
    tmp_path,
    tool_name,
    arguments,
    permission,
    message,
    expected,
):
    def forbidden_http(_: httpx.Request) -> httpx.Response:
        raise AssertionError("proposal tool não pode executar efeito HTTP")

    model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": arguments,
                        "id": f"call_{tool_name}",
                        "type": "tool_call",
                    }
                ],
            ),
        )
    )
    request = _request(message=message)
    request_id = f"req_{tool_name}"
    trusted_calls, trusted_observations = _trusted_proposal_history(
        tool_name,
        request_id,
    )

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(forbidden_http),
        )
        runtime = WriteToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({permission}),
            central_asset_id="asset_G501",
            current_case_id="case_tkt_inv_04",
            configured_model_id="mdl_vib_v3",
            client=client,
        )
        state = AgentState(
            request=request,
            identity=runtime.identity,
            permissions=runtime.permissions,
            request_id=request_id,
            thread_id=f"thread_{tool_name}",
            execution_id=f"exec_{tool_name}",
            thread_scope=ThreadScope(
                thread_id=f"thread_{tool_name}",
                case_id=request.case_id,
                company_id=runtime.identity.company_id,
                user_id=runtime.identity.user_id,
            ),
                tool_calls=trusted_calls,
                tool_observations=trusted_observations,
                ledger=compile_observations(
                    trusted_observations,
                    recorded_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
                ),
                planner_usage=PlannerUsage(
                request_id=request_id,
                selection_count=len(trusted_calls),
            ),
            step_limit=20,
        )
        config = {"configurable": {"thread_id": state.thread_id}}
        try:
            async with open_checkpointer(
                tmp_path / f"{tool_name}.sqlite3"
            ) as saver:
                graph = _build_planner_graph(saver, model)
                await graph.ainvoke(
                    state.model_dump(mode="json"),
                    config,
                    context=runtime,
                    durability="sync",
                    interrupt_after=["planner_tool"],
                )
                return await graph.aget_state(config)
        finally:
            await client.aclose()

    snapshot = asyncio.run(scenario())
    persisted = AgentState.model_validate(snapshot.values)

    assert snapshot.next == ("write_policy",)
    assert persisted.resume_anchor is ResumeAnchor.PLANNER_TOOL
    assert persisted.step_count == 3
    assert persisted.pending_proposal == expected
    assert persisted.trusted_write_context == TrustedWriteContext(
        central_asset_id="asset_G501",
        current_case_id="case_tkt_inv_04",
        configured_model_id="mdl_vib_v3",
    )
    assert persisted.tool_calls[-1].name == tool_name
    assert persisted.tool_observations == trusted_observations
    checkpoint_text = repr(snapshot.values)
    assert "ToolMessage" not in checkpoint_text
    assert "effect_executed" not in checkpoint_text


@pytest.mark.parametrize(
    "malformation",
    [
        "missing-content-status",
        "missing-artifact-kind",
        "missing-effect-executed",
        "coerced-effect-executed",
    ],
)
def test_planner_rejects_noncanonical_raw_proposal_wire(
    tmp_path,
    monkeypatch,
    malformation,
):
    arguments = {
        "criticality": "critical",
        "justification": "O impacto operacional exige prioridade máxima.",
    }
    proposal = {
        "action": "update_asset_criticality",
        **arguments,
    }
    content = {"status": "proposed", "proposal": proposal}
    artifact = {
        "kind": "write_proposal",
        "tool_name": "propose_update_asset_criticality",
        "proposal": proposal,
        "effect_executed": False,
    }
    if malformation == "missing-content-status":
        content.pop("status")
    elif malformation == "missing-artifact-kind":
        artifact.pop("kind")
    elif malformation == "missing-effect-executed":
        artifact.pop("effect_executed")
    else:
        artifact["effect_executed"] = 0

    class MalformedToolNode:
        def __init__(self, *args, **kwargs):
            pass

        async def ainvoke(self, *args, **kwargs):
            return {
                "messages": [
                    ToolMessage(
                        content=json.dumps(content),
                        artifact=artifact,
                        name="propose_update_asset_criticality",
                        tool_call_id="call_noncanonical_proposal",
                        status="success",
                    )
                ]
            }

    monkeypatch.setattr("tractian_agent.graph.ToolNode", MalformedToolNode)
    model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_update_asset_criticality",
                        "args": arguments,
                        "id": "call_noncanonical_proposal",
                        "type": "tool_call",
                    }
                ],
            ),
        )
    )

    async def scenario():
        client = IndustrialApiClient("https://industrial.test")
        runtime = WriteToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"action_high"}),
            central_asset_id="asset_G501",
            current_case_id="case_tkt_inv_04",
            client=client,
        )
        request = _request(message="Atualize a criticidade do ativo central.")
        state = AgentState(
            request=request,
            identity=runtime.identity,
            permissions=runtime.permissions,
            request_id=f"req_noncanonical_{malformation}",
            thread_id=f"thread_noncanonical_{malformation}",
            execution_id="exec_noncanonical_proposal",
            thread_scope=ThreadScope(
                thread_id=f"thread_noncanonical_{malformation}",
                case_id=request.case_id,
                company_id=runtime.identity.company_id,
                user_id=runtime.identity.user_id,
            ),
            step_limit=20,
        )
        config = {"configurable": {"thread_id": state.thread_id}}
        try:
            async with open_checkpointer(
                tmp_path / f"noncanonical-{malformation}.sqlite3"
            ) as saver:
                graph = _build_planner_graph(saver, model)
                await graph.ainvoke(
                    state.model_dump(mode="json"),
                    config,
                    context=runtime,
                    durability="sync",
                )
                return AgentState.model_validate(
                    (await graph.aget_state(config)).values
                )
        finally:
            await client.aclose()

    state = asyncio.run(scenario())

    assert state.pending_proposal is None
    assert state.planner_failure is not None
    assert state.planner_failure.stage == "planner_tool"
    assert state.planner_failure.code == "invalid_tool_result"


@pytest.mark.parametrize(
    ("interrupt_after", "expected_next", "expected_anchor", "expected_steps"),
    [
        ("ingest", "planner_select", ResumeAnchor.INGEST, 1),
        ("planner_select", "planner_tool", ResumeAnchor.PLANNER_SELECT, 2),
        ("planner_tool", "planner_select", ResumeAnchor.PLANNER_TOOL, 3),
    ],
)
def test_planner_cycle_resumes_from_each_new_nonterminal_node(
    tmp_path,
    interrupt_after,
    expected_next,
    expected_anchor,
    expected_steps,
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"mode": "complete", "notes": None, "data": _asset_payload()},
        )

    model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_asset",
                        "args": {"asset_id": "asset_G501"},
                        "id": "call_resume_asset",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content=""),
        ),
        terminal_responses=(
            PlannerTerminalDecision(
                decision=PlannerDecisionKind.GUIDE,
                stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
            ),
        ),
    )

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        )
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=client,
        )
        request = _request()
        state = AgentState(
            request=request,
            identity=runtime.identity,
            permissions=runtime.permissions,
            request_id="req_resume_planner",
            thread_id=f"thread_resume_{interrupt_after}",
            execution_id="exec_before_resume",
            thread_scope=ThreadScope(
                thread_id=f"thread_resume_{interrupt_after}",
                case_id=request.case_id,
                company_id=runtime.identity.company_id,
                user_id=runtime.identity.user_id,
            ),
            step_limit=20,
        )
        config = {"configurable": {"thread_id": state.thread_id}}
        try:
            async with open_checkpointer(
                tmp_path / f"resume-{interrupt_after}.sqlite3"
            ) as saver:
                graph = _build_planner_graph(saver, model)
                await graph.ainvoke(
                    state.model_dump(mode="json"),
                    config,
                    context=runtime,
                    durability="sync",
                    interrupt_after=[interrupt_after],
                )
                partial = await graph.aget_state(config)
                resumed = await invoke_agent(
                    graph,
                    request=request,
                    runtime=runtime,
                    thread_id=state.thread_id,
                    request_id=state.request_id,
                    execution_id="exec_after_resume",
                )
            return partial, resumed
        finally:
            await client.aclose()

    partial, resumed = asyncio.run(scenario())

    assert partial.next == (expected_next,)
    assert partial.values["resume_anchor"] == expected_anchor.value
    assert partial.values["step_count"] == expected_steps
    assert resumed.execution_id == "exec_after_resume"
    assert resumed.resume_anchor is ResumeAnchor.RELEASE_GATE
    assert resumed.final_result is not None
    assert resumed.planner_failure is None
    assert len(requests) == 1


def test_planner_generated_proposal_uses_policy_and_executes_once(tmp_path):
    requests: list[httpx.Request] = []
    justification = "O impacto operacional exige criticidade máxima."

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "PATCH"
        assert request.url.path == "/assets/asset_G501"
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "action_id": "act_planner_criticality",
                "message": "Criticidade atualizada.",
            },
        )

    model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_update_asset_criticality",
                        "args": {
                            "criticality": "critical",
                            "justification": justification,
                        },
                        "id": "call_propose_criticality",
                        "type": "tool_call",
                    }
                ],
            ),
        )
    )
    writer_model = _EchoWriterModel()
    writer = Writer(writer_model)
    approval = TrustedActionApproval(
        action="update_asset_criticality",
        target_id="asset_G501",
        material_parameters={"criticality": "critical"},
        source=ApprovalSource.ORIGINAL_REQUEST,
    )

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        )
        runtime = WriteToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"action_high"}),
            central_asset_id="asset_G501",
            current_case_id="case_tkt_inv_04",
            client=client,
        )
        request = _request(message="Atualize a criticidade do ativo central.")
        try:
            async with open_checkpointer(tmp_path / "planner-write.sqlite3") as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(model),
                    writer=writer,
                )
                completed = await invoke_agent(
                    graph,
                    request=request,
                    runtime=runtime,
                    thread_id="thread_planner_write",
                    request_id="req_planner_write",
                    execution_id="exec_planner_write",
                    original_approval=approval,
                )
                replayed = await invoke_agent(
                    graph,
                    request=request,
                    runtime=runtime,
                    thread_id="thread_planner_write",
                    request_id="req_planner_write",
                    execution_id="exec_delivery_retry",
                    original_approval=approval,
                )
            return completed, replayed
        finally:
            await client.aclose()

    completed, replayed = asyncio.run(scenario())

    assert completed.step_count == 9
    assert completed.step_limit == 24
    assert completed.resume_anchor is ResumeAnchor.RELEASE_GATE
    assert completed.decision is AgentDecision.ACT
    assert completed.pending_proposal == UpdateAssetCriticalityProposal(
        criticality="critical",
        justification=justification,
    )
    assert completed.intents[0].status.value == "completed"
    assert completed.release_gate is not None
    assert completed.release_gate.outcome is ReleaseGateOutcome.RELEASE
    assert completed.final_result is not None
    assert completed.final_result.evidence_ids == tuple(
        item.evidence_id for item in completed.ledger.items
    )
    assert "asset_G501" not in completed.final_result.message
    assert writer_model._events == ["with_structured_output", "writer_request"]
    assert replayed == completed
    assert len(requests) == 1


@pytest.mark.parametrize(
    (
        "slug",
        "proposal",
        "approval",
        "permission",
        "write_method",
        "write_path",
        "decision",
        "requires_preflight",
    ),
    [
        (
            "reprocess",
            ReprocessProposal(
                analysis_id="an_9906",
                justification="Há dados novos para repetir a análise.",
            ),
            TrustedActionApproval(
                action="reprocess_analysis",
                target_id="an_9906",
                source=ApprovalSource.ORIGINAL_REQUEST,
            ),
            "action_low",
            "POST",
            "/analyses/an_9906/reprocess",
            AgentDecision.ACT,
            True,
        ),
        (
            "specialist",
            RequestSpecialistAnalysisProposal(
                analysis_id="an_9906",
                justification="A limitação exige análise especializada.",
            ),
            TrustedActionApproval(
                action="request_specialist_analysis",
                target_id="an_9906",
                source=ApprovalSource.ORIGINAL_REQUEST,
            ),
            "action_low",
            "POST",
            "/analyses/an_9906/request-specialist",
            AgentDecision.ACT,
            True,
        ),
        (
            "criticality",
            UpdateAssetCriticalityProposal(
                criticality="critical",
                justification="O impacto operacional exige prioridade máxima.",
            ),
            TrustedActionApproval(
                action="update_asset_criticality",
                target_id="asset_G501",
                material_parameters={"criticality": "critical"},
                source=ApprovalSource.ORIGINAL_REQUEST,
            ),
            "action_high",
            "PATCH",
            "/assets/asset_G501",
            AgentDecision.ACT,
            False,
        ),
        (
            "retraining",
            RequestModelRetrainingProposal(
                justification="Erros sistemáticos sustentam novo treinamento.",
            ),
            TrustedActionApproval(
                action="request_model_retraining",
                target_id="mdl_vib_v3",
                source=ApprovalSource.ORIGINAL_REQUEST,
            ),
            "action_high",
            "POST",
            "/models/mdl_vib_v3/request-retraining",
            AgentDecision.ACT,
            False,
        ),
        (
            "escalation",
            EscalateCaseProposal(
                justification="O caso ultrapassa o atendimento remoto.",
            ),
            TrustedActionApproval(
                action="escalate_case",
                target_id="case_tkt_inv_04",
                source=ApprovalSource.ORIGINAL_REQUEST,
            ),
            "escalate",
            "POST",
            "/cases/case_tkt_inv_04/escalate",
            AgentDecision.ESCALATE,
            False,
        ),
    ],
)
def test_all_write_flows_use_public_planner_writer_gate_and_replay_once(
    tmp_path,
    slug,
    proposal,
    approval,
    permission,
    write_method,
    write_path,
    decision,
    requires_preflight,
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            assert requires_preflight
            assert request.url.path == "/analyses/an_9906"
            return httpx.Response(
                200,
                json={"mode": "complete", "notes": None, "data": _analysis_payload()},
            )
        assert request.method == write_method
        assert request.url.path == write_path
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "action_id": f"act_{slug}_writer_gate",
                "message": "Ação aceita pela plataforma.",
            },
        )

    writer_model = _EchoWriterModel()
    planner_model = _ScriptedPlannerModel(selector_responses=())

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        )
        runtime = WriteToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read", permission}),
            central_asset_id="asset_G501",
            current_case_id="case_tkt_inv_04",
            configured_model_id="mdl_vib_v3",
            client=client,
        )
        request = _request(message="Execute a ação industrial solicitada.")
        checkpoint_path = tmp_path / f"flow-{slug}.sqlite3"
        try:
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                uninterrupted_ainvoke = graph.ainvoke

                async def interrupt_after_effect(input, config, **kwargs):
                    return await uninterrupted_ainvoke(
                        input,
                        config,
                        interrupt_after=["execute_action"],
                        **kwargs,
                    )

                graph.ainvoke = interrupt_after_effect
                partial = await invoke_agent(
                    graph,
                    request=request,
                    runtime=runtime,
                    thread_id=f"thread_flow_{slug}",
                    request_id=f"req_flow_{slug}",
                    execution_id=f"exec_flow_{slug}",
                    proposal=proposal,
                    original_approval=approval,
                )
            async with open_checkpointer(checkpoint_path) as saver:
                resumed_graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                completed = await invoke_agent(
                    resumed_graph,
                    request=request,
                    runtime=runtime,
                    thread_id=f"thread_flow_{slug}",
                    request_id=f"req_flow_{slug}",
                    execution_id=f"exec_flow_{slug}_resume",
                    proposal=proposal,
                    original_approval=approval,
                )
                replayed = await invoke_agent(
                    resumed_graph,
                    request=request,
                    runtime=runtime,
                    thread_id=f"thread_flow_{slug}",
                    request_id=f"req_flow_{slug}",
                    execution_id=f"exec_flow_{slug}_delivery_retry",
                    proposal=proposal,
                    original_approval=approval,
                )
            return partial, completed, replayed
        finally:
            await client.aclose()

    partial, completed, replayed = asyncio.run(scenario())

    assert partial.step_count == 5
    assert partial.resume_anchor is ResumeAnchor.EXECUTE_ACTION
    assert partial.final_result is None
    assert completed.step_count == 7
    assert completed.step_limit == 24
    assert completed.decision is decision
    assert completed.resume_anchor is ResumeAnchor.RELEASE_GATE
    assert completed.release_gate is not None
    assert completed.release_gate.outcome is ReleaseGateOutcome.RELEASE
    assert completed.final_result is not None
    assert completed.final_result.decision is decision
    assert approval.target_id not in completed.final_result.message
    assert writer_model._events == ["with_structured_output", "writer_request"]
    assert planner_model._events == []
    assert replayed == completed
    assert len(requests) == (2 if requires_preflight else 1)
    if slug == "reprocess":
        write_request = requests[-1]
        assert write_request.headers["idempotency-key"] == (
            completed.intents[0].idempotency_key
        )


@pytest.mark.parametrize(
    ("slug", "proposal", "approval", "permission", "operation_name"),
    [
        (
            "specialist",
            RequestSpecialistAnalysisProposal(
                analysis_id="an_9906",
                justification="A limitação exige análise especializada.",
            ),
            TrustedActionApproval(
                action="request_specialist_analysis",
                target_id="an_9906",
                source=ApprovalSource.ORIGINAL_REQUEST,
            ),
            "action_low",
            "execute_request_specialist_analysis",
        ),
        (
            "criticality",
            UpdateAssetCriticalityProposal(
                criticality="critical",
                justification="O impacto operacional exige prioridade máxima.",
            ),
            TrustedActionApproval(
                action="update_asset_criticality",
                target_id="asset_G501",
                material_parameters={"criticality": "critical"},
                source=ApprovalSource.ORIGINAL_REQUEST,
            ),
            "action_high",
            "execute_update_asset_criticality",
        ),
        (
            "retraining",
            RequestModelRetrainingProposal(
                justification="Erros sistemáticos sustentam novo treinamento.",
            ),
            TrustedActionApproval(
                action="request_model_retraining",
                target_id="mdl_vib_v3",
                source=ApprovalSource.ORIGINAL_REQUEST,
            ),
            "action_high",
            "execute_request_model_retraining",
        ),
        (
            "escalation",
            EscalateCaseProposal(
                justification="O caso ultrapassa o atendimento remoto.",
            ),
            TrustedActionApproval(
                action="escalate_case",
                target_id="case_tkt_inv_04",
                source=ApprovalSource.ORIGINAL_REQUEST,
            ),
            "escalate",
            "execute_escalate_case",
        ),
    ],
)
@pytest.mark.parametrize("runtime_drift", [False, True], ids=["same-scope", "drift"])
def test_planner_resume_of_cross_execution_prepared_action_skips_writer_and_gate(
    tmp_path,
    monkeypatch,
    slug,
    proposal,
    approval,
    permission,
    operation_name,
    runtime_drift,
):
    operation_calls = 0
    http_calls: list[httpx.Request] = []
    graph_calls = 0
    planner_model = _ScriptedPlannerModel(selector_responses=())
    writer_model = _ForbiddenWriterModel()

    async def crash_after_prepared(*args: Any, **kwargs: Any) -> None:
        nonlocal operation_calls
        operation_calls += 1
        raise RuntimeError("queda depois do checkpoint prepared")

    async def forbidden_graph_call(*args: Any, **kwargs: Any) -> None:
        nonlocal graph_calls
        graph_calls += 1
        raise AssertionError("replay terminal não pode executar o grafo")

    def forbidden_http(request: httpx.Request) -> httpx.Response:
        http_calls.append(request)
        raise AssertionError("retomada conservadora não pode alcançar HTTP")

    async def scenario() -> tuple[AgentState, AgentState, AgentState]:
        checkpoint_path = tmp_path / (
            f"planner-cross-execution-{slug}-{runtime_drift}.sqlite3"
        )
        thread_id = f"thread_planner_cross_execution_{slug}_{runtime_drift}"
        request_id = f"req_planner_cross_execution_{slug}_{runtime_drift}"
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(forbidden_http),
        )
        initial_runtime = WriteToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read", permission}),
            central_asset_id="asset_G501",
            current_case_id="case_tkt_inv_04",
            configured_model_id="mdl_vib_v3",
            client=client,
        )
        drift = (
            {
                "specialist": {"central_asset_id": "asset_other"},
                "criticality": {"central_asset_id": "asset_other"},
                "retraining": {"configured_model_id": "mdl_other"},
                "escalation": {"current_case_id": "case_other"},
            }[slug]
            if runtime_drift
            else {}
        )
        resumed_runtime = WriteToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read", permission}),
            central_asset_id=drift.get("central_asset_id", "asset_G501"),
            current_case_id=drift.get("current_case_id", "case_tkt_inv_04"),
            configured_model_id=drift.get("configured_model_id", "mdl_vib_v3"),
            client=client,
        )
        request = _request(message="Atualize a criticidade do ativo central.")
        monkeypatch.setattr(
            graph_module,
            operation_name,
            crash_after_prepared,
        )
        try:
            async with open_checkpointer(checkpoint_path) as saver:
                with pytest.raises(RuntimeError, match="checkpoint prepared"):
                    await invoke_agent(
                        build_agent_graph(
                            saver,
                            planner=Planner(planner_model),
                            writer=Writer(writer_model),
                        ),
                        request=request,
                        runtime=initial_runtime,
                        thread_id=thread_id,
                        request_id=request_id,
                        execution_id="exec_prepare",
                        proposal=proposal,
                        original_approval=approval,
                    )

            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                prepared = AgentState.model_validate(
                    (
                        await graph.aget_state(
                            {"configurable": {"thread_id": thread_id}}
                        )
                    ).values
                )
                uncertain = await invoke_agent(
                    graph,
                    request=request,
                    runtime=resumed_runtime,
                    thread_id=thread_id,
                    request_id=request_id,
                    execution_id="exec_resume",
                )

            async with open_checkpointer(checkpoint_path) as saver:
                replay_graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                terminal_snapshot = await replay_graph.aget_state(
                    {"configurable": {"thread_id": thread_id}}
                )
                assert terminal_snapshot.next == ()
                replay_graph.ainvoke = forbidden_graph_call
                replayed = await invoke_agent(
                    replay_graph,
                    request=request,
                    runtime=initial_runtime,
                    thread_id=thread_id,
                    request_id=request_id,
                    execution_id="exec_replay",
                )
            return prepared, uncertain, replayed
        finally:
            await client.aclose()

    prepared, uncertain, replayed = asyncio.run(scenario())

    intent = uncertain.intents[0]
    assert prepared.resume_anchor is ResumeAnchor.PREPARE_INTENT
    assert prepared.step_count == 4
    assert uncertain.resume_anchor is ResumeAnchor.EXECUTE_ACTION
    assert uncertain.step_count == 5
    assert uncertain.step_limit == 24
    assert intent.status is IntentStatus.UNCERTAIN
    assert intent.attempts == 0
    assert intent.error is not None
    assert intent.error.code == "NON_IDEMPOTENT_OUTCOME_UNKNOWN_AFTER_RESUME"
    assert intent.error.message == (
        "A execução preparadora terminou sem resultado terminal observável."
    )
    assert uncertain.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert uncertain.release_gate is None
    assert uncertain.writer_attempts == 0
    assert uncertain.writer_draft is None
    assert uncertain.writer_failure is None
    assert uncertain.final_result is not None
    assert uncertain.final_result.message == (
        "O resultado remoto da ação é desconhecido e ela não será "
        "reenviada automaticamente."
    )
    assert uncertain.final_result.evidence_ids == ()
    assert uncertain.final_result.limitation_refs == ()
    assert uncertain.final_result.next_step is None
    assert approval.target_id not in uncertain.final_result.message
    assert proposal.justification not in uncertain.final_result.message
    assert approval.target_id not in intent.error.message
    assert proposal.justification not in intent.error.message
    assert uncertain.has_coherent_terminal_result()
    assert replayed == uncertain
    assert planner_model._events == []
    assert writer_model._calls == 0
    assert operation_calls == 1
    assert http_calls == []
    assert graph_calls == 0


@pytest.mark.parametrize(
    ("status_code", "expected_status"),
    [(400, IntentStatus.FAILED), (500, IntentStatus.UNCERTAIN)],
)
def test_other_action_failures_still_use_writer_and_release_gate(
    tmp_path,
    status_code,
    expected_status,
):
    proposal = UpdateAssetCriticalityProposal(
        criticality="critical",
        justification="O impacto operacional exige prioridade máxima.",
    )
    approval = TrustedActionApproval(
        action="update_asset_criticality",
        target_id="asset_G501",
        material_parameters={"criticality": "critical"},
        source=ApprovalSource.ORIGINAL_REQUEST,
    )
    planner_model = _ScriptedPlannerModel(selector_responses=())
    writer_model = _EchoWriterModel()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={
                "code": (
                    "VALIDATION_ERROR"
                    if status_code == 400
                    else "INTERNAL_ERROR"
                ),
                "message": "Falha remota sanitizada.",
            },
        )

    async def scenario() -> AgentState:
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        )
        runtime = WriteToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read", "action_high"}),
            central_asset_id="asset_G501",
            current_case_id="case_tkt_inv_04",
            configured_model_id="mdl_vib_v3",
            client=client,
        )
        try:
            async with open_checkpointer(
                tmp_path / f"planner-other-action-failure-{status_code}.sqlite3"
            ) as saver:
                return await invoke_agent(
                    build_agent_graph(
                        saver,
                        planner=Planner(planner_model),
                        writer=Writer(writer_model),
                    ),
                    request=_request(
                        message="Atualize a criticidade do ativo central."
                    ),
                    runtime=runtime,
                    thread_id=f"thread_planner_other_failure_{status_code}",
                    request_id=f"req_planner_other_failure_{status_code}",
                    execution_id=f"exec_planner_other_failure_{status_code}",
                    proposal=proposal,
                    original_approval=approval,
                )
        finally:
            await client.aclose()

    state = asyncio.run(scenario())

    assert state.intents[0].status is expected_status
    assert state.intents[0].attempts == 1
    assert state.resume_anchor is ResumeAnchor.RELEASE_GATE
    assert state.writer_attempts == 1
    assert state.writer_draft is not None
    assert state.release_gate is not None
    assert state.final_result is None
    assert state.review_request is not None
    assert planner_model._events == []
    assert writer_model._events == ["with_structured_output", "writer_request"]


def test_writer_graph_projection_excludes_every_non_allowlisted_state_sentinel(
    tmp_path,
):
    identity = Identity(
        user_id="usr_identity_sentinel",
        company_id="comp_identity_sentinel",
    )
    first_request = SupportRequest(
        case_id="case_tkt_inv_04",
        ticket_id="TKT-INV-04",
        asset_id="asset_G501",
        message="FIRST_REQUEST_SENTINEL",
        identity=identity,
    )
    second_request = SupportRequest(
        case_id="case_tkt_inv_04",
        ticket_id="TKT-INV-04",
        asset_id="asset_G501",
        message="REQUEST_SENTINEL",
        identity=identity,
    )
    proposal = UpdateAssetCriticalityProposal(
        criticality="critical",
        justification=(
            "PROPOSAL_SENTINEL: o impacto operacional exige prioridade máxima."
        ),
    )
    approval = TrustedActionApproval(
        action=proposal.action,
        target_id="asset_G501",
        material_parameters={"criticality": "critical"},
        source=ApprovalSource.ORIGINAL_REQUEST,
    )
    planner_model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_asset",
                        "args": {"asset_id": "asset_G501"},
                        "id": "call_history_sentinel",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content=""),
        ),
        terminal_responses=(
            PlannerTerminalDecision(
                decision=PlannerDecisionKind.GUIDE,
                stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
            ),
        ),
    )
    writer_model = _EchoWriterModel()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            payload = _asset_payload()
            payload["name"] = "HISTORY_ARTIFACT_SENTINEL"
            payload["company_id"] = identity.company_id
            return httpx.Response(
                200,
                json={"mode": "complete", "notes": None, "data": payload},
            )
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "action_id": "act_allowlisted_sentinel",
                "message": "RECEIPT_SENTINEL",
            },
        )

    async def scenario():
        client = IndustrialApiClient(
            "https://runtime-sentinel.invalid",
            transport=httpx.MockTransport(handler),
        )
        runtime = WriteToolRuntime.create(
            user_id=identity.user_id,
            company_id=identity.company_id,
            permissions=frozenset({"read", "action_high"}),
            central_asset_id="asset_G501",
            current_case_id="case_tkt_inv_04",
            client=client,
        )
        try:
            async with open_checkpointer(tmp_path / "writer-allowlist.sqlite3") as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                first = await invoke_agent(
                    graph,
                    request=first_request,
                    runtime=runtime,
                    thread_id="thread_writer_allowlist",
                    request_id="req_writer_allowlist_history",
                    execution_id="exec_writer_allowlist_history",
                )
                second = await invoke_agent(
                    graph,
                    request=second_request,
                    runtime=runtime,
                    thread_id="thread_writer_allowlist",
                    request_id="req_writer_allowlist_current",
                    execution_id="exec_writer_allowlist_current",
                    proposal=proposal,
                    original_approval=approval,
                )
            return first, second
        finally:
            await client.aclose()

    first, second = asyncio.run(scenario())

    assert first.final_result is not None
    assert len(second.ledger_history) == 1
    assert "HISTORY_ARTIFACT_SENTINEL" in first.final_result.message
    current_writer_payload = writer_model._payloads[-1]
    for forbidden in (
        "FIRST_REQUEST_SENTINEL",
        "REQUEST_SENTINEL",
        "usr_identity_sentinel",
        "comp_identity_sentinel",
        "action_high",
        "PROPOSAL_SENTINEL",
        "RECEIPT_SENTINEL",
        "act_allowlisted_sentinel",
        "HISTORY_ARTIFACT_SENTINEL",
        "runtime-sentinel.invalid",
        "asset_G501",
    ):
        assert forbidden not in current_writer_payload


def test_writer_format_repair_survives_checkpoint_without_repeating_planner(
    tmp_path,
):
    planner_model = _ScriptedPlannerModel(
        selector_responses=(AIMessage(content=""),),
        terminal_responses=(
            PlannerTerminalDecision(
                decision=PlannerDecisionKind.REQUEST_INFORMATION,
                stop_reason=PlannerStopReason.MISSING_INFORMATION,
                missing_information="Informe o ponto de medição.",
            ),
        ),
    )
    writer_model = _SequenceWriterGraphModel(
        responses=(
            {"decision": "request_information", "texto": "inválido"},
            WriterDraft(
                decision=AgentDecision.REQUEST_INFORMATION,
                evidence_ids=(),
                limitation_refs=(),
                next_step=WriterNextStep.PROVIDE_INFORMATION,
            ),
        )
    )
    writer = Writer(writer_model)

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(
                lambda request: pytest.fail(f"HTTP inesperado: {request.url}")
            ),
        )
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=client,
        )
        request = _request(message="Investigue, mas falta identificar o ponto.")
        state = AgentState(
            request=request,
            identity=runtime.identity,
            permissions=runtime.permissions,
            request_id="req_writer_repair",
            thread_id="thread_writer_repair",
            execution_id="exec_writer_repair_1",
            thread_scope=ThreadScope(
                thread_id="thread_writer_repair",
                case_id=request.case_id,
                company_id=runtime.identity.company_id,
                user_id=runtime.identity.user_id,
            ),
            step_limit=24,
        )
        config = {"configurable": {"thread_id": state.thread_id}}
        checkpoint_path = tmp_path / "writer-repair.sqlite3"
        try:
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=writer,
                )
                await graph.ainvoke(
                    state.model_dump(mode="json"),
                    config,
                    context=runtime,
                    durability="sync",
                    interrupt_after=["writer"],
                )
                partial = await graph.aget_state(config)
            async with open_checkpointer(checkpoint_path) as saver:
                resumed = await invoke_agent(
                    build_agent_graph(
                        saver,
                        planner=Planner(planner_model),
                        writer=writer,
                    ),
                    request=request,
                    runtime=runtime,
                    thread_id=state.thread_id,
                    request_id=state.request_id,
                    execution_id="exec_writer_repair_2",
                )
            return partial, resumed
        finally:
            await client.aclose()

    partial, resumed = asyncio.run(scenario())

    assert partial.next == ("writer",)
    assert partial.values["writer_attempts"] == 1
    assert partial.values["writer_failure"]["code"] == "invalid_structured_output"
    assert resumed.writer_attempts == 2
    assert resumed.writer_failure is None
    assert resumed.decision is AgentDecision.REQUEST_INFORMATION
    assert resumed.resume_anchor is ResumeAnchor.RELEASE_GATE
    assert resumed.final_result is not None
    assert resumed.final_result.message == (
        "Para continuar, informe: Informe o ponto de medição."
    )
    assert writer_model._payloads[0] == writer_model._payloads[1]
    assert planner_model._events == [
        "bind_tools",
        "selection_request",
        "with_structured_output",
        "terminal_request",
    ]


def test_valid_writer_checkpoint_resumes_only_the_release_gate(tmp_path):
    planner_model = _ScriptedPlannerModel(
        selector_responses=(AIMessage(content=""),),
        terminal_responses=(
            PlannerTerminalDecision(
                decision=PlannerDecisionKind.REQUEST_INFORMATION,
                stop_reason=PlannerStopReason.MISSING_INFORMATION,
                missing_information="Informe o ponto de medição.",
            ),
        ),
    )
    writer_model = _EchoWriterModel()
    checkpoint_path = tmp_path / "writer-valid-checkpoint.sqlite3"

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(
                lambda request: pytest.fail(f"HTTP inesperado: {request.url}")
            ),
        )
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=client,
        )
        request = _request(message="Investigue, mas falta identificar o ponto.")
        try:
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                uninterrupted_ainvoke = graph.ainvoke

                async def interrupt_after_writer(input, config, **kwargs):
                    return await uninterrupted_ainvoke(
                        input,
                        config,
                        interrupt_after=["writer"],
                        **kwargs,
                    )

                graph.ainvoke = interrupt_after_writer
                partial = await invoke_agent(
                    graph,
                    request=request,
                    runtime=runtime,
                    thread_id="thread_writer_valid_checkpoint",
                    request_id="req_writer_valid_checkpoint",
                    execution_id="exec_writer_valid_checkpoint_1",
                )
            async with open_checkpointer(checkpoint_path) as saver:
                completed = await invoke_agent(
                    build_agent_graph(
                        saver,
                        planner=Planner(planner_model),
                        writer=Writer(writer_model),
                    ),
                    request=request,
                    runtime=runtime,
                    thread_id="thread_writer_valid_checkpoint",
                    request_id="req_writer_valid_checkpoint",
                    execution_id="exec_writer_valid_checkpoint_2",
                )
            return partial, completed
        finally:
            await client.aclose()

    partial, completed = asyncio.run(scenario())

    assert partial.resume_anchor is ResumeAnchor.WRITER
    assert partial.writer_attempts == 1
    assert partial.writer_draft is not None
    assert partial.final_result is None
    assert completed.resume_anchor is ResumeAnchor.RELEASE_GATE
    assert completed.writer_attempts == 1
    assert completed.final_result is not None
    assert completed.final_result.decision is AgentDecision.REQUEST_INFORMATION
    assert writer_model._events == ["with_structured_output", "writer_request"]
    assert planner_model._events == [
        "bind_tools",
        "selection_request",
        "with_structured_output",
        "terminal_request",
    ]


@pytest.mark.parametrize(
    ("responses", "expected_attempts", "expected_code"),
    [
        (
            (
                {"decision": "guide", "technical_text": "primeiro inválido"},
                {"decision": "guide", "technical_text": "segundo inválido"},
            ),
            2,
            "invalid_structured_output",
        ),
        ((RuntimeError("RAW_SECRET_OUTPUT"),), 1, "model_failure"),
    ],
)
def test_writer_failure_stops_safely_without_hidden_retry(
    tmp_path,
    responses,
    expected_attempts,
    expected_code,
):
    planner_model = _ScriptedPlannerModel(
        selector_responses=(AIMessage(content=""),),
        terminal_responses=(
            PlannerTerminalDecision(
                decision=PlannerDecisionKind.REQUIRE_HUMAN_REVIEW,
                stop_reason=PlannerStopReason.HUMAN_REVIEW_REQUIRED,
            ),
        ),
    )
    writer_model = _SequenceWriterGraphModel(responses=responses)

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(
                lambda request: pytest.fail(f"HTTP inesperado: {request.url}")
            ),
        )
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=client,
        )
        try:
            async with open_checkpointer(
                tmp_path / f"writer-failure-{expected_code}.sqlite3"
            ) as saver:
                return await invoke_agent(
                    build_agent_graph(
                        saver,
                        planner=Planner(planner_model),
                        writer=Writer(writer_model),
                    ),
                    request=_request(),
                    runtime=runtime,
                    thread_id=f"thread_writer_failure_{expected_code}",
                    request_id=f"req_writer_failure_{expected_code}",
                    execution_id=f"exec_writer_failure_{expected_code}",
                )
        finally:
            await client.aclose()

    state = asyncio.run(scenario())

    assert state.writer_attempts == expected_attempts
    assert state.writer_failure is not None
    assert state.writer_failure.code.value == expected_code
    assert state.release_gate is not None
    assert state.release_gate.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert state.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert state.final_result is None
    assert state.review_request is not None
    state_wire = state.model_dump_json()
    assert "inválido" not in state_wire
    assert "RAW_SECRET_OUTPUT" not in state_wire
    assert len(writer_model._payloads) == expected_attempts


def test_write_policy_never_creates_a_second_intent_for_request_id(tmp_path):
    request = _request(message="Atualize a criticidade do ativo central.")
    existing_proposal = ReprocessProposal(
        analysis_id="an_historical",
        justification="Intenção terminal já registrada para esta solicitação.",
    )
    existing_intent = WriteIntent(
        intent_id="intent_existing_request",
        request_id="req_duplicate_intent",
        scope=ReprocessIntentScope(
            action="reprocess_analysis",
            case_id=request.case_id,
            company_id=request.identity.company_id,
            user_id=request.identity.user_id,
            analysis_id="an_historical",
            justification="Intenção terminal já registrada para esta solicitação.",
        ),
        payload_hash=canonical_write_payload_hash(existing_proposal),
        decision=WritePolicyResult(
            decision=PolicyDecision.DENY,
            reason=PolicyReason.MISSING_PERMISSION,
        ),
        status=IntentStatus.DENIED,
    )

    def forbidden_http(_: httpx.Request) -> httpx.Response:
        raise AssertionError("request_id duplicada não pode alcançar HTTP")

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(forbidden_http),
        )
        runtime = WriteToolRuntime.create(
            user_id=request.identity.user_id,
            company_id=request.identity.company_id,
            permissions=frozenset({"read"}),
            central_asset_id=request.asset_id,
            current_case_id=request.case_id,
            client=client,
        )
        state = AgentState(
            request=request,
            identity=runtime.identity,
            permissions=runtime.permissions,
            request_id="req_duplicate_intent",
            thread_id="thread_duplicate_intent",
            execution_id="exec_duplicate_intent",
            thread_scope=ThreadScope(
                thread_id="thread_duplicate_intent",
                case_id=request.case_id,
                company_id=request.identity.company_id,
                user_id=request.identity.user_id,
            ),
            step_limit=5,
            trusted_write_context=TrustedWriteContext(
                central_asset_id=request.asset_id,
                current_case_id=request.case_id,
                configured_model_id=runtime.configured_model_id,
            ),
            pending_proposal=existing_proposal,
            intents=(existing_intent,),
        )
        config = {"configurable": {"thread_id": state.thread_id}}
        try:
            async with open_checkpointer(
                tmp_path / "duplicate-intent.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver)
                with pytest.raises(ValueError, match="request_id já possui intenção"):
                    await graph.ainvoke(
                        state.model_dump(mode="json"),
                        config,
                        context=runtime,
                        durability="sync",
                    )
                return await graph.aget_state(config)
        finally:
            await client.aclose()

    snapshot = asyncio.run(scenario())
    restored = AgentState.model_validate(snapshot.values)

    assert len(restored.intents) == 1
    assert restored.intents[0] == existing_intent


def test_policy_requires_confirmation_when_proposal_exceeds_original_approval(
    tmp_path,
):
    justification = "O impacto operacional exige criticidade máxima."
    model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_update_asset_criticality",
                        "args": {
                            "criticality": "critical",
                            "justification": justification,
                        },
                        "id": "call_unapproved_criticality",
                        "type": "tool_call",
                    }
                ],
            ),
        )
    )
    narrower_approval = TrustedActionApproval(
        action="update_asset_criticality",
        target_id="asset_G501",
        material_parameters={"criticality": "low"},
        source=ApprovalSource.ORIGINAL_REQUEST,
    )

    def forbidden_http(_: httpx.Request) -> httpx.Response:
        raise AssertionError("policy deny não pode executar efeito HTTP")

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(forbidden_http),
        )
        runtime = WriteToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"action_high"}),
            central_asset_id="asset_G501",
            current_case_id="case_tkt_inv_04",
            client=client,
        )
        try:
            async with open_checkpointer(tmp_path / "planner-deny.sqlite3") as saver:
                return await invoke_agent(
                    _build_planner_graph(saver, model),
                    request=_request(message="Atualize a criticidade do ativo central."),
                    runtime=runtime,
                    thread_id="thread_planner_deny",
                    request_id="req_planner_deny",
                    execution_id="exec_planner_deny",
                    original_approval=narrower_approval,
                )
        finally:
            await client.aclose()

    state = asyncio.run(scenario())

    assert state.step_count == 4
    assert state.resume_anchor is ResumeAnchor.WRITE_POLICY
    assert state.pending_proposal == UpdateAssetCriticalityProposal(
        criticality="critical",
        justification=justification,
    )
    assert state.intents[0].status.value == "awaiting_confirmation"
    assert state.decision is AgentDecision.REQUEST_CONFIRMATION
    assert state.final_result is None


def test_planner_confirmation_resume_reaches_writer_gate_without_repeating_effect(
    tmp_path,
):
    requests: list[httpx.Request] = []
    justification = "O impacto operacional exige criticidade máxima."
    planner_model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_update_asset_criticality",
                        "args": {
                            "criticality": "critical",
                            "justification": justification,
                        },
                        "id": "call_confirmed_criticality",
                        "type": "tool_call",
                    }
                ],
            ),
        )
    )
    writer_model = _EchoWriterModel()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "PATCH"
        assert request.url.path == "/assets/asset_G501"
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "action_id": "act_confirmed_writer_gate",
                "message": "Ação aceita pela plataforma.",
            },
        )

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        )
        runtime = WriteToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read", "action_high"}),
            central_asset_id="asset_G501",
            current_case_id="case_tkt_inv_04",
            client=client,
        )
        request = _request(message="Atualize a criticidade do ativo central.")
        try:
            async with open_checkpointer(tmp_path / "confirmed-writer.sqlite3") as saver:
                graph = build_agent_graph(
                    saver,
                    planner=Planner(planner_model),
                    writer=Writer(writer_model),
                )
                partial = await invoke_agent(
                    graph,
                    request=request,
                    runtime=runtime,
                    thread_id="thread_confirmed_writer",
                    request_id="req_confirmed_writer",
                    execution_id="exec_confirmed_writer_1",
                )
                confirmation = ConfirmationReply(
                    intent_id=partial.intents[0].intent_id,
                    decision="approve",
                )
                completed = await invoke_agent(
                    graph,
                    request=request,
                    runtime=runtime,
                    thread_id="thread_confirmed_writer",
                    request_id="req_confirmed_writer",
                    execution_id="exec_confirmed_writer_2",
                    confirmation=confirmation,
                )
                replayed = await invoke_agent(
                    graph,
                    request=request,
                    runtime=runtime,
                    thread_id="thread_confirmed_writer",
                    request_id="req_confirmed_writer",
                    execution_id="exec_confirmed_writer_3",
                    confirmation=confirmation,
                )
            return partial, completed, replayed
        finally:
            await client.aclose()

    partial, completed, replayed = asyncio.run(scenario())

    assert partial.resume_anchor is ResumeAnchor.WRITE_POLICY
    assert partial.final_result is None
    assert partial.intents[0].status is IntentStatus.AWAITING_CONFIRMATION
    assert completed.resume_anchor is ResumeAnchor.RELEASE_GATE
    assert completed.release_gate is not None
    assert completed.release_gate.outcome is ReleaseGateOutcome.RELEASE
    assert completed.intents[0].status is IntentStatus.COMPLETED
    assert completed.approval is not None
    assert completed.approval.source is ApprovalSource.CONFIRMATION
    assert replayed == completed
    assert len(requests) == 1
    assert writer_model._events == ["with_structured_output", "writer_request"]


def test_resume_preserves_write_runtime_boundary_for_original_approval(tmp_path):
    approval = TrustedActionApproval(
        action="update_asset_criticality",
        target_id="asset_G501",
        material_parameters={"criticality": "critical"},
        source=ApprovalSource.ORIGINAL_REQUEST,
    )
    model = _ScriptedPlannerModel(selector_responses=(AIMessage(content=""),))

    async def scenario():
        client = IndustrialApiClient("https://industrial.test")
        write_runtime = WriteToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"action_high"}),
            central_asset_id="asset_G501",
            current_case_id="case_tkt_inv_04",
            client=client,
        )
        read_runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"action_high"}),
            central_asset_id="asset_G501",
            client=client,
        )
        request = _request(message="Atualize a criticidade do ativo central.")
        state = AgentState(
            request=request,
            identity=write_runtime.identity,
            permissions=write_runtime.permissions,
            request_id="req_resume_original_approval",
            thread_id="thread_resume_original_approval",
            execution_id="exec_before_resume",
            thread_scope=ThreadScope(
                thread_id="thread_resume_original_approval",
                case_id=request.case_id,
                company_id=write_runtime.identity.company_id,
                user_id=write_runtime.identity.user_id,
            ),
            trusted_write_context=TrustedWriteContext(
                central_asset_id=write_runtime.central_asset_id,
                current_case_id=write_runtime.current_case_id,
                configured_model_id=write_runtime.configured_model_id,
            ),
            step_limit=20,
            approval=approval,
        )
        config = {"configurable": {"thread_id": state.thread_id}}
        try:
            async with open_checkpointer(tmp_path / "resume-boundary.sqlite3") as saver:
                graph = _build_planner_graph(saver, model)
                await graph.ainvoke(
                    state.model_dump(mode="json"),
                    config,
                    context=write_runtime,
                    durability="sync",
                    interrupt_after=["ingest"],
                )
                before = await graph.aget_state(config)
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=request,
                        runtime=read_runtime,
                        thread_id=state.thread_id,
                        request_id=state.request_id,
                        execution_id="exec_after_resume",
                    )
                after = await graph.aget_state(config)
            return error.value, before, after
        finally:
            await client.aclose()

    error, before, after = asyncio.run(scenario())

    assert error.code == "WRITE_RUNTIME_REQUIRED"
    assert before.config == after.config
    assert before.values == after.values


def test_repeated_planner_call_terminates_safely_without_second_http(tmp_path):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"mode": "complete", "notes": None, "data": _asset_payload()},
        )

    def asset_call(call_id: str) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_asset",
                    "args": {"asset_id": "asset_G501"},
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        )

    model = _ScriptedPlannerModel(
        selector_responses=(asset_call("call_asset_1"), asset_call("call_asset_2"))
    )

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        )
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=client,
        )
        try:
            async with open_checkpointer(tmp_path / "repeated.sqlite3") as saver:
                return await invoke_agent(
                    _build_planner_graph(saver, model),
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_repeated_tool",
                    request_id="req_repeated_tool",
                    execution_id="exec_repeated_tool",
                )
        finally:
            await client.aclose()

    state = asyncio.run(scenario())

    assert len(requests) == 1
    assert len(state.tool_calls) == 1
    assert len(state.tool_observations) == 1
    assert state.step_count == 4
    assert state.final_result is not None
    assert state.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert state.planner_failure is not None
    assert state.planner_failure.stage == "planner_select"
    assert state.planner_failure.code == "repeated_tool_call"
    assert state.planner_usage.selection_count == 2


def test_read_transport_error_is_observed_and_never_retried(tmp_path):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("simulated transport failure", request=request)

    model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_asset",
                        "args": {"asset_id": "asset_G501"},
                        "id": "call_transport_error",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content=""),
        ),
        terminal_responses=(
            PlannerTerminalDecision(
                decision=PlannerDecisionKind.REQUIRE_HUMAN_REVIEW,
                stop_reason=PlannerStopReason.HUMAN_REVIEW_REQUIRED,
            ),
        ),
    )

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        )
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=client,
        )
        try:
            async with open_checkpointer(tmp_path / "transport.sqlite3") as saver:
                return await invoke_agent(
                    _build_planner_graph(saver, model),
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_transport_error",
                    request_id="req_transport_error",
                    execution_id="exec_transport_error",
                )
        finally:
            await client.aclose()

    state = asyncio.run(scenario())

    assert attempts == 1
    assert len(state.tool_observations) == 1
    observation = state.tool_observations[0]
    assert observation.artifact.outcome.error is not None
    assert observation.artifact.outcome.error.category.value == "transport"
    assert state.ledger.items == ()
    assert {gap.reason.value for gap in state.ledger.gaps} == {"error"}
    assert state.ledger.gaps[0].call_id == observation.call_id
    assert state.planner_failure is None
    assert state.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert state.resume_anchor is ResumeAnchor.RELEASE_GATE


def test_degraded_read_is_persisted_and_reaches_structured_terminal(tmp_path):
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            json={
                "mode": "partial",
                "notes": "Pontos temporariamente indisponíveis.",
                "data": {
                    "id": "asset_G501",
                    "sensor_status": "offline",
                },
            },
        )

    model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_asset",
                        "args": {"asset_id": "asset_G501"},
                        "id": "call_degraded_asset",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content=""),
        ),
        terminal_responses=(
            PlannerTerminalDecision(
                decision=PlannerDecisionKind.REQUEST_INFORMATION,
                stop_reason=PlannerStopReason.MISSING_INFORMATION,
                missing_information="Aguardar a telemetria dos pontos do ativo.",
            ),
        ),
    )

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        )
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=client,
        )
        try:
            async with open_checkpointer(tmp_path / "degraded.sqlite3") as saver:
                return await invoke_agent(
                    _build_planner_graph(saver, model),
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_degraded_asset",
                    request_id="req_degraded_asset",
                    execution_id="exec_degraded_asset",
                )
        finally:
            await client.aclose()

    state = asyncio.run(scenario())

    assert attempts == 1
    assert state.planner_failure is None
    assert state.decision is AgentDecision.REQUEST_INFORMATION
    assert state.tool_observations[0].content is not None
    assert state.tool_observations[0].content.to_python() == {
        "mode": "partial",
        "notes": "Pontos temporariamente indisponíveis.",
        "partial_data": {
            "id": "asset_G501",
            "sensor_status": "offline",
        },
    }
    artifact = state.tool_observations[0].artifact.validated_read_artifact()
    assert isinstance(artifact, AssetToolArtifact)
    assert artifact.outcome.mode is not None
    assert artifact.outcome.mode.value == "partial"
    assert state.ledger.items
    assert all(item.quality.value == "partial" for item in state.ledger.items)
    assert {gap.reason.value for gap in state.ledger.gaps} == {"partial"}


def test_public_reads_preserve_conflict_through_sqlite_reopen(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        point_id = request.url.params.get("point_id")
        return httpx.Response(
            200,
            json={
                "mode": "complete",
                "notes": None,
                "data": {
                    "asset_id": "asset_G501",
                    "point_id": point_id or "pt_G501_de",
                    "completeness": 0.61 if point_id else 0.98,
                    "freshness_minutes": 2,
                    "snr_db": 24.5,
                    "staleness_flag": False,
                },
            },
        )

    def quality_call(call_id: str, point_id: str | None) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_data_quality",
                    "args": (
                        {"asset_id": "asset_G501"}
                        if point_id is None
                        else {"asset_id": "asset_G501", "point_id": point_id}
                    ),
                    "id": call_id,
                    "type": "tool_call",
                }
            ],
        )

    model = _ScriptedPlannerModel(
        selector_responses=(
            quality_call("call_quality_asset", None),
            quality_call("call_quality_point", "pt_G501_de"),
            AIMessage(content=""),
        ),
        terminal_responses=(
            PlannerTerminalDecision(
                decision=PlannerDecisionKind.REQUIRE_HUMAN_REVIEW,
                stop_reason=PlannerStopReason.HUMAN_REVIEW_REQUIRED,
            ),
        ),
    )

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        )
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=client,
        )
        config = {"configurable": {"thread_id": "thread_quality_conflict"}}
        try:
            async with open_checkpointer(tmp_path / "conflict.sqlite3") as saver:
                graph = _build_planner_graph(saver, model)
                state = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_quality_conflict",
                    request_id="req_quality_conflict",
                    execution_id="exec_quality_conflict",
                )
            async with open_checkpointer(tmp_path / "conflict.sqlite3") as saver:
                reopened = AgentState.model_validate(
                    (await _build_planner_graph(saver, model).aget_state(config)).values
                )
            return state, reopened
        finally:
            await client.aclose()

    state, reopened = asyncio.run(scenario())

    assert len(state.tool_observations) == 2
    assert state.ledger.items
    assert state.ledger.gaps == ()
    assert state.ledger.conflicts
    assert state.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert reopened.ledger == state.ledger


def test_public_obsolete_read_preserves_gap_through_sqlite_reopen(tmp_path):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mode": "complete",
                "notes": None,
                "data": {
                    "asset_id": "asset_G501",
                    "point_id": "pt_G501_de",
                    "completeness": 0.98,
                    "freshness_minutes": 2,
                    "snr_db": 24.5,
                    "staleness_flag": True,
                },
            },
        )

    model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_data_quality",
                        "args": {"asset_id": "asset_G501"},
                        "id": "call_quality_obsolete",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content=""),
        ),
        terminal_responses=(
            PlannerTerminalDecision(
                decision=PlannerDecisionKind.REQUEST_INFORMATION,
                stop_reason=PlannerStopReason.MISSING_INFORMATION,
                missing_information="Aguardar dados atuais de qualidade.",
            ),
        ),
    )

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        )
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=client,
        )
        config = {"configurable": {"thread_id": "thread_quality_obsolete"}}
        try:
            async with open_checkpointer(tmp_path / "obsolete.sqlite3") as saver:
                graph = _build_planner_graph(saver, model)
                state = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_quality_obsolete",
                    request_id="req_quality_obsolete",
                    execution_id="exec_quality_obsolete",
                )
            async with open_checkpointer(tmp_path / "obsolete.sqlite3") as saver:
                reopened = AgentState.model_validate(
                    (await _build_planner_graph(saver, model).aget_state(config)).values
                )
            return state, reopened
        finally:
            await client.aclose()

    state, reopened = asyncio.run(scenario())

    assert state.ledger.items
    assert all(item.quality.value == "obsolete" for item in state.ledger.items)
    assert {gap.reason.value for gap in state.ledger.gaps} == {"obsolete"}
    assert state.ledger.conflicts == ()
    assert state.decision is AgentDecision.REQUEST_INFORMATION
    assert reopened.ledger == state.ledger


def test_public_new_request_archives_ledger_through_sqlite_reopen(tmp_path):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mode": "complete",
                "notes": None,
                "data": {
                    "asset_id": "asset_G501",
                    "point_id": "pt_G501_de",
                    "completeness": 0.98,
                    "freshness_minutes": 2,
                    "snr_db": 24.5,
                    "staleness_flag": False,
                },
            },
        )

    model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_data_quality",
                        "args": {"asset_id": "asset_G501"},
                        "id": "call_history_quality",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content=""),
            AIMessage(content=""),
        ),
        terminal_responses=(
            PlannerTerminalDecision(
                decision=PlannerDecisionKind.REQUEST_INFORMATION,
                stop_reason=PlannerStopReason.MISSING_INFORMATION,
                missing_information="Aguardar a próxima solicitação.",
            ),
            PlannerTerminalDecision(
                decision=PlannerDecisionKind.REQUEST_INFORMATION,
                stop_reason=PlannerStopReason.MISSING_INFORMATION,
                missing_information="Sem leitura nova para esta solicitação.",
            ),
        ),
    )

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        )
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=client,
        )
        checkpoint_path = tmp_path / "history.sqlite3"
        config = {"configurable": {"thread_id": "thread_ledger_history"}}
        try:
            async with open_checkpointer(checkpoint_path) as saver:
                first = await invoke_agent(
                    _build_planner_graph(saver, model),
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_ledger_history",
                    request_id="req_ledger_history_one",
                    execution_id="exec_ledger_history_one",
                )
            async with open_checkpointer(checkpoint_path) as saver:
                second = await invoke_agent(
                    _build_planner_graph(saver, model),
                    request=_request(message="Nova solicitação independente."),
                    runtime=runtime,
                    thread_id="thread_ledger_history",
                    request_id="req_ledger_history_two",
                    execution_id="exec_ledger_history_two",
                )
            async with open_checkpointer(checkpoint_path) as saver:
                restored = AgentState.model_validate(
                    (await _build_planner_graph(saver, model).aget_state(config)).values
                )
            return first, second, restored
        finally:
            await client.aclose()

    first, second, restored = asyncio.run(scenario())

    assert first.ledger.items
    assert second.ledger.items == ()
    assert second.ledger_history == (first.ledger,)
    assert restored.ledger_history == (first.ledger,)
    assert all(item.request_id == "req_ledger_history_one" for item in restored.ledger_history[0].items)


def test_unexpected_tool_failure_terminates_safely_without_retry(tmp_path):
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("sensitive adapter failure")

    model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_asset",
                        "args": {"asset_id": "asset_G501"},
                        "id": "call_unexpected_failure",
                        "type": "tool_call",
                    }
                ],
            ),
        )
    )

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        )
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=client,
        )
        try:
            async with open_checkpointer(tmp_path / "tool-failure.sqlite3") as saver:
                graph = _build_planner_graph(saver, model)
                state = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_tool_failure",
                    request_id="req_tool_failure",
                    execution_id="exec_tool_failure",
                )
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_tool_failure"}}
                )
            return state, snapshot
        finally:
            await client.aclose()

    state, snapshot = asyncio.run(scenario())

    assert attempts == 1
    assert state.step_count == 3
    assert len(state.tool_calls) == 1
    assert state.tool_observations == ()
    assert state.planner_failure is not None
    assert state.planner_failure.stage == "planner_tool"
    assert state.planner_failure.code == "tool_execution_failed"
    assert state.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert "sensitive adapter failure" not in repr(snapshot.values)


def test_planner_reserves_fixed_write_steps_before_proposal_tool(tmp_path):
    model = _ScriptedPlannerModel(
        selector_responses=(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_update_asset_criticality",
                        "args": {
                            "criticality": "critical",
                            "justification": (
                                "O impacto operacional exige criticidade máxima."
                            ),
                        },
                        "id": "call_budgeted_proposal",
                        "type": "tool_call",
                    }
                ],
            ),
        )
    )

    def forbidden_http(_: httpx.Request) -> httpx.Response:
        raise AssertionError("orçamento insuficiente não pode tocar HTTP")

    async def scenario():
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(forbidden_http),
        )
        runtime = WriteToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"action_high"}),
            central_asset_id="asset_G501",
            current_case_id="case_tkt_inv_04",
            client=client,
        )
        try:
            async with open_checkpointer(tmp_path / "budget.sqlite3") as saver:
                return await invoke_agent(
                    _build_planner_graph(saver, model),
                    request=_request(message="Atualize a criticidade do ativo central."),
                    runtime=runtime,
                    thread_id="thread_budgeted_proposal",
                    request_id="req_budgeted_proposal",
                    execution_id="exec_budgeted_proposal",
                    step_limit=6,
                )
        finally:
            await client.aclose()

    state = asyncio.run(scenario())

    assert state.step_count == 2
    assert state.pending_proposal is None
    assert state.tool_calls == ()
    assert state.planner_failure is not None
    assert state.planner_failure.code == "step_limit_exhausted"
    assert state.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
