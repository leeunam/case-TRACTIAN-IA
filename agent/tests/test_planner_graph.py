from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
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
from pydantic import PrivateAttr

from tractian_agent.checkpoint import open_checkpointer
from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import Identity, ResponseMode, SupportRequest
from tractian_agent.entrypoint import AgentInvocationProtocolError, invoke_agent
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
    WritePolicyResult,
    UpdateAssetCriticalityProposal,
    TrustedActionApproval,
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
    assert state.final_result is not None
    assert state.final_result.evidence_ids == ()
    state_wire = state.model_dump_json()
    assert "inválido" not in state_wire
    assert "RAW_SECRET_OUTPUT" not in state_wire
    assert len(writer_model._payloads) == expected_attempts


def test_write_policy_never_creates_a_second_intent_for_request_id(tmp_path):
    request = _request(message="Atualize a criticidade do ativo central.")
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
        payload_hash="sha256:v1:" + "b" * 64,
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
            pending_proposal=UpdateAssetCriticalityProposal(
                criticality="critical",
                justification="O impacto operacional exige criticidade máxima.",
            ),
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
