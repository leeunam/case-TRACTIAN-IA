from __future__ import annotations

import asyncio
from collections.abc import Sequence
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
from tractian_agent.contracts import Identity, SupportRequest
from tractian_agent.entrypoint import AgentInvocationProtocolError, invoke_agent
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
    ResumeAnchor,
    ThreadScope,
)
from tractian_agent.tools.assets import AssetToolArtifact, execute_get_asset
from tractian_agent.tools.runtime import ReadToolRuntime, WriteToolRuntime
from tractian_agent.write_contracts import (
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
            async with open_checkpointer(tmp_path / "planner.sqlite3") as saver:
                graph = build_agent_graph(saver, planner=Planner(model))
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
            return state, snapshot
        finally:
            await client.aclose()

    state, snapshot = asyncio.run(scenario())

    assert len(requests) == 1
    assert requests[0].url.path == "/assets/asset_G501"
    assert state.step_limit == 20
    assert state.step_count == 5
    assert state.resume_anchor is ResumeAnchor.PLANNER_FINALIZE
    assert state.decision is AgentDecision.GUIDE
    assert state.final_result is not None
    assert len(state.tool_calls) == 1
    assert len(state.tool_observations) == 1
    observation = state.tool_observations[0]
    assert observation.call_id == "call_get_asset_1"
    assert type(observation.artifact.validated_read_artifact()) is AssetToolArtifact
    assert observation.content is not None
    assert observation.content.to_python()["id"] == "asset_G501"
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
                graph = build_agent_graph(saver, planner=Planner(model))
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
            request_id=f"req_{tool_name}",
            thread_id=f"thread_{tool_name}",
            execution_id=f"exec_{tool_name}",
            thread_scope=ThreadScope(
                thread_id=f"thread_{tool_name}",
                case_id=request.case_id,
                company_id=runtime.identity.company_id,
                user_id=runtime.identity.user_id,
            ),
            step_limit=20,
        )
        config = {"configurable": {"thread_id": state.thread_id}}
        try:
            async with open_checkpointer(
                tmp_path / f"{tool_name}.sqlite3"
            ) as saver:
                graph = build_agent_graph(saver, planner=Planner(model))
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
    assert persisted.tool_observations == ()
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
                graph = build_agent_graph(saver, planner=Planner(model))
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
                graph = build_agent_graph(saver, planner=Planner(model))
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
    assert resumed.resume_anchor is ResumeAnchor.PLANNER_FINALIZE
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
                graph = build_agent_graph(saver, planner=Planner(model))
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

    assert completed.step_count == 7
    assert completed.step_limit == 20
    assert completed.resume_anchor is ResumeAnchor.EXECUTE_ACTION
    assert completed.decision is AgentDecision.ACT
    assert completed.pending_proposal == UpdateAssetCriticalityProposal(
        criticality="critical",
        justification=justification,
    )
    assert completed.intents[0].status.value == "completed"
    assert replayed == completed
    assert len(requests) == 1


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
                    build_agent_graph(saver, planner=Planner(model)),
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
                graph = build_agent_graph(saver, planner=Planner(model))
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
                    build_agent_graph(saver, planner=Planner(model)),
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
                    build_agent_graph(saver, planner=Planner(model)),
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
    assert state.planner_failure is None
    assert state.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
    assert state.resume_anchor is ResumeAnchor.PLANNER_FINALIZE


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
                    build_agent_graph(saver, planner=Planner(model)),
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
                graph = build_agent_graph(saver, planner=Planner(model))
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
                    build_agent_graph(saver, planner=Planner(model)),
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
