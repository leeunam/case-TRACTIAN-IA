import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import ValidationError

from tractian_agent.checkpoint import LocalCheckpointOwner, open_checkpointer
from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import ActionReceipt, Identity, ResponseMode, SupportRequest
from tractian_agent.entrypoint import AgentInvocationProtocolError, invoke_agent
from tractian_agent.evidence import compile_action_intents
from tractian_agent.graph import CompiledAgentGraph, build_agent_graph
from tractian_agent.state import (
    AgentDecision,
    AgentState,
    FinalResult,
    MessageRole,
    PlannerFailureRecord,
    PlannerTerminalRecord,
    PersistedToolArtifact,
    PersistedToolCall,
    ResumeAnchor,
    ReviewRecord,
    ReviewStatus,
    ThreadScope,
    ToolObservation,
)
from tractian_agent.tools.runtime import (
    ReadToolRuntime,
    TrustedIdentity,
    WriteToolRuntime,
)
from tractian_agent.write_contracts import (
    IntentStatus,
    ReprocessIntentScope,
    WriteIntent,
)
from tractian_agent.write_policy import (
    ApprovalSource,
    PolicyDecision,
    PolicyReason,
    ReprocessProposal,
    TrustedActionApproval,
    WritePolicyResult,
    canonical_write_payload_hash,
)


def _request(
    *,
    message: str = "Consulte o estado deste ativo.",
    user_id: str = "usr_pedro",
    company_id: str = "comp_mineracao_andes",
) -> SupportRequest:
    return SupportRequest(
        case_id="case_tkt_inv_04",
        ticket_id="TKT-INV-04",
        asset_id="asset_G501",
        message=message,
        identity=Identity(
            user_id=user_id,
            company_id=company_id,
        ),
    )


def _runtime(
    client: IndustrialApiClient,
    *,
    user_id: str = "usr_pedro",
    company_id: str = "comp_mineracao_andes",
    permissions: frozenset[str] = frozenset({"read"}),
    central_asset_id: str = "asset_G501",
) -> ReadToolRuntime:
    return ReadToolRuntime.create(
        user_id=user_id,
        company_id=company_id,
        permissions=permissions,
        central_asset_id=central_asset_id,
        client=client,
    )


def _initial_state(
    *,
    thread_id: str = "thread_case_tkt_inv_04",
    step_limit: int = 3,
    request: SupportRequest | None = None,
) -> AgentState:
    identity = TrustedIdentity(
        user_id="usr_pedro",
        company_id="comp_mineracao_andes",
    )
    return AgentState(
        request=_request() if request is None else request,
        identity=identity,
        permissions=frozenset({"read"}),
        request_id="req_01",
        thread_id=thread_id,
        execution_id="exec_01",
        thread_scope=ThreadScope(
            thread_id=thread_id,
            case_id="case_tkt_inv_04",
            company_id="comp_mineracao_andes",
            user_id="usr_pedro",
        ),
        step_limit=step_limit,
    )


def _denied_historical_intent(request_id: str) -> WriteIntent:
    return WriteIntent(
        intent_id=f"intent_{request_id}",
        request_id=request_id,
        scope=ReprocessIntentScope(
            action="reprocess_analysis",
            case_id="case_tkt_inv_04",
            company_id="comp_mineracao_andes",
            user_id="usr_pedro",
            analysis_id="an_historical",
            justification="Histórico terminal usado apenas para proveniência.",
        ),
        payload_hash="sha256:v1:" + "a" * 64,
        decision=WritePolicyResult(
            decision=PolicyDecision.DENY,
            reason=PolicyReason.MISSING_PERMISSION,
        ),
        status=IntentStatus.DENIED,
    )


def _historical_tool_artifact() -> PersistedToolArtifact:
    return PersistedToolArtifact(
        tool_name="historical_lookup",
        arguments={},
        source={"kind": "industrial_api", "resource": "/historical"},
        outcome={"mode": ResponseMode.COMPLETE},
    )


def _terminal_read_state() -> AgentState:
    state = _initial_state()
    values = state.model_dump(mode="python")
    values.update(
        decision=AgentDecision.GUIDE,
        final_result=FinalResult(
            decision=AgentDecision.GUIDE,
            message="Resultado de leitura já persistido.",
        ),
        resume_anchor=ResumeAnchor.FINISH,
        tool_observations=(
            ToolObservation(
                request_id=state.request_id,
                call_id="call_terminal_read",
                content={"status": "ok"},
                artifact=_historical_tool_artifact(),
            ),
        ),
    )
    return AgentState.model_validate(values)


def _terminal_planner_state() -> AgentState:
    state = _initial_state(step_limit=20)
    values = state.model_dump(mode="python")
    values.update(
        decision=AgentDecision.GUIDE,
        final_result=FinalResult(
            decision=AgentDecision.GUIDE,
            message="Resultado terminal validado do planner.",
        ),
        resume_anchor=ResumeAnchor.PLANNER_FINALIZE,
        planner_terminal=PlannerTerminalRecord(
            decision="guide",
            stop_reason="sufficient_evidence",
        ),
    )
    return AgentState.model_validate(values)


def _terminal_denial_state() -> AgentState:
    state = _initial_state(step_limit=5)
    values = state.model_dump(mode="python")
    values.update(
        pending_proposal=ReprocessProposal(
            analysis_id="an_historical",
            justification="A política deve conservar a negação persistida.",
        ),
        intents=(_denied_historical_intent(state.request_id),),
        decision=AgentDecision.GUIDE,
        final_result=FinalResult(
            decision=AgentDecision.GUIDE,
            message="A política recusou a ação.",
        ),
        resume_anchor=ResumeAnchor.WRITE_POLICY,
    )
    return AgentState.model_validate(values)


def _terminal_execution_state() -> AgentState:
    state = _initial_state(step_limit=5)
    proposal = ReprocessProposal(
        analysis_id="an_historical",
        justification="O reprocesso foi autorizado e concluído.",
    )
    completed_data = _denied_historical_intent(state.request_id).model_dump(
        mode="python"
    )
    completed_data.update(
        decision=WritePolicyResult(
            decision=PolicyDecision.ALLOW,
            reason=PolicyReason.AUTHORIZED,
        ),
        status=IntentStatus.COMPLETED,
        payload_hash=canonical_write_payload_hash(proposal),
        idempotency_key="tractian-agent:intent_req_01",
        expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        prepared_execution_id=state.execution_id,
        attempts=1,
        receipt=ActionReceipt(
            accepted=True,
            action_id="act_terminal_execution",
            message="Reprocesso concluído.",
        ),
    )
    completed = WriteIntent.model_validate(completed_data)
    values = state.model_dump(mode="python")
    values.update(
        pending_proposal=proposal,
        intents=(completed,),
        ledger=compile_action_intents(
            (completed,),
            recorded_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
        ),
        decision=AgentDecision.ACT,
        final_result=FinalResult(
            decision=AgentDecision.ACT,
            message="Reprocesso concluído.",
        ),
        resume_anchor=ResumeAnchor.EXECUTE_ACTION,
    )
    return AgentState.model_validate(values)


class _RecordingGraph:
    def __init__(
        self,
        values=None,
        *,
        next_nodes=(),
        interrupts=(),
        planner_enabled=False,
    ):
        self.values = values or {}
        self.next_nodes = next_nodes
        self.interrupts = interrupts
        self.planner_enabled = planner_enabled
        self.state_config = None
        self.invoke_config = None
        self.durability = None
        self.context = None
        self.as_node = None

    @asynccontextmanager
    async def thread_lock(self, thread_id):
        yield

    async def aget_state(self, config):
        self.state_config = config
        return SimpleNamespace(
            values=self.values,
            config=config,
            next=self.next_nodes,
            interrupts=self.interrupts,
        )

    async def ainvoke(self, input, config, *, context=None, durability):
        self.invoke_config = config
        self.durability = durability
        self.context = context
        if input is not None:
            self.values = input
        return self.values

    async def aupdate_state(self, config, values, *, as_node=None):
        self.values = values
        self.as_node = as_node
        return config


def test_entrypoint_requires_thread_configuration_and_sync_durability():
    async def scenario():
        graph = _RecordingGraph()
        async with IndustrialApiClient("https://industrial.test") as client:
            state = await invoke_agent(
                graph,
                request=_request(),
                runtime=_runtime(client),
                thread_id="thread_case_tkt_inv_04",
                request_id="req_01",
                execution_id="exec_01",
            )
        return graph, state

    graph, state = asyncio.run(scenario())

    expected_config = {
        "configurable": {"thread_id": "thread_case_tkt_inv_04"}
    }
    assert graph.state_config == expected_config
    assert graph.invoke_config == expected_config
    assert graph.durability == "sync"
    assert graph.context is not None
    assert graph.context.identity == state.identity
    assert isinstance(graph.context.client, IndustrialApiClient)
    assert state.thread_id == "thread_case_tkt_inv_04"


def test_continuation_also_uses_sync_durability_and_continue_with():
    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            first = await invoke_agent(
                _RecordingGraph(),
                request=_request(message="Primeiro pedido."),
                runtime=_runtime(client),
                thread_id="thread_case_tkt_inv_04",
                request_id="req_01",
                execution_id="exec_01",
            )
            continuation_graph = _RecordingGraph(first.model_dump(mode="json"))
            continued = await invoke_agent(
                continuation_graph,
                request=_request(message="Segundo pedido."),
                runtime=_runtime(client),
                thread_id="thread_case_tkt_inv_04",
                request_id="req_02",
                execution_id="exec_02",
            )
        return continuation_graph, continued

    graph, continued = asyncio.run(scenario())

    assert graph.durability == "sync"
    assert continued.request_id == "req_02"
    assert continued.execution_id == "exec_02"
    assert continued.step_count == 0


@pytest.mark.parametrize(
    "provenance",
    ["intent", "tool_call", "tool_observation"],
)
def test_new_request_rejects_any_historical_request_id_provenance(
    provenance: str,
):
    reused_request_id = "req_historical_reused"
    state = _initial_state()
    if provenance == "intent":
        state = state.model_copy(
            update={"intents": (_denied_historical_intent(reused_request_id),)}
        )
    elif provenance == "tool_call":
        state = state.model_copy(
            update={
                "tool_calls": (
                    PersistedToolCall(
                        request_id=reused_request_id,
                        call_id="call_historical",
                        name="historical_lookup",
                        arguments={},
                    ),
                )
            }
        )
    else:
        state = state.model_copy(
            update={
                "tool_observations": (
                    ToolObservation(
                        request_id=reused_request_id,
                        call_id="call_historical",
                        content={"status": "ok"},
                        artifact=_historical_tool_artifact(),
                    ),
                )
            }
        )
    persisted_values = state.model_dump(mode="json")
    graph = _RecordingGraph(persisted_values)

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            with pytest.raises(AgentInvocationProtocolError) as error:
                await invoke_agent(
                    graph,
                    request=_request(message="Nova leitura com ID antigo."),
                    runtime=_runtime(client),
                    thread_id=state.thread_id,
                    request_id=reused_request_id,
                    execution_id="exec_reused_history",
                )
        return error.value

    error = asyncio.run(scenario())

    assert error.code == "REQUEST_ID_ALREADY_USED"
    assert graph.values == persisted_values
    assert graph.as_node is None
    assert graph.invoke_config is None


@pytest.mark.parametrize(
    ("drift", "expected_code"),
    [
        ("identity", "RUNTIME_IDENTITY_SCOPE_MISMATCH"),
        ("asset_without_read", "RUNTIME_ASSET_SCOPE_MISMATCH"),
        ("permission", "READ_PERMISSION_REQUIRED"),
    ],
)
def test_terminal_read_replay_revalidates_current_runtime_before_return(
    drift: str,
    expected_code: str,
):
    state = _terminal_read_state()
    persisted_values = state.model_dump(mode="json")
    graph = _RecordingGraph(persisted_values)

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            runtime = _runtime(
                client,
                user_id=("usr_other" if drift == "identity" else "usr_pedro"),
                permissions=(
                    frozenset({"read"})
                    if drift == "identity"
                    else frozenset({"action_high"})
                ),
                central_asset_id=(
                    "asset_M101"
                    if drift == "asset_without_read"
                    else "asset_G501"
                ),
            )
            with pytest.raises(AgentInvocationProtocolError) as error:
                await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id=state.thread_id,
                    request_id=state.request_id,
                    execution_id="exec_terminal_scope_recheck",
                )
        return error.value

    error = asyncio.run(scenario())

    assert error.code == expected_code
    assert graph.values == persisted_values
    assert graph.as_node is None
    assert graph.invoke_config is None


def test_partial_read_scope_mismatch_fails_before_checkpoint_update():
    state = _initial_state().model_copy(
        update={"resume_anchor": ResumeAnchor.INGEST}
    )
    persisted_values = state.model_dump(mode="json")
    graph = _RecordingGraph(persisted_values, next_nodes=("route",))

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            with pytest.raises(AgentInvocationProtocolError) as error:
                await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=_runtime(
                        client,
                        permissions=frozenset({"action_high"}),
                        central_asset_id="asset_M101",
                    ),
                    thread_id=state.thread_id,
                    request_id=state.request_id,
                    execution_id="exec_partial_scope_recheck",
                )
        return error.value

    error = asyncio.run(scenario())

    assert error.code == "RUNTIME_ASSET_SCOPE_MISMATCH"
    assert graph.values == persisted_values
    assert graph.as_node is None
    assert graph.invoke_config is None


@pytest.mark.parametrize("thread_id", ["", "thread com espaço"])
def test_entrypoint_rejects_missing_or_non_opaque_thread_id(thread_id: str):
    async def scenario():
        graph = _RecordingGraph()
        async with IndustrialApiClient("https://industrial.test") as client:
            with pytest.raises(ValueError, match="thread_id é obrigatório"):
                await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=_runtime(client),
                    thread_id=thread_id,
                    request_id="req_01",
                    execution_id="exec_01",
                )
        return graph

    graph = asyncio.run(scenario())
    assert graph.state_config is None
    assert graph.invoke_config is None


def test_entrypoint_rejects_untrusted_context_before_checkpoint_access():
    async def scenario():
        graph = _RecordingGraph()
        with pytest.raises(TypeError, match="runtime autenticado é obrigatório"):
            await invoke_agent(
                graph,
                request=_request(),
                runtime=object(),
                thread_id="thread_case_tkt_inv_04",
                request_id="req_01",
                execution_id="exec_01",
            )
        return graph

    graph = asyncio.run(scenario())
    assert graph.state_config is None
    assert graph.invoke_config is None


@pytest.mark.parametrize(
    ("next_nodes", "expected_code"),
    [
        ((), "NON_TERMINAL_WITHOUT_PENDING_WORK"),
        (("route", "finish"), "MULTIPLE_PENDING_NODES"),
        (("unknown_node",), "UNKNOWN_PENDING_NODE"),
    ],
)
def test_nonterminal_resume_rejects_invalid_pending_work_shape(
    next_nodes: tuple[str, ...],
    expected_code: str,
):
    state = _initial_state()
    graph = _RecordingGraph(
        state.model_dump(mode="json"),
        next_nodes=next_nodes,
    )

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            with pytest.raises(AgentInvocationProtocolError) as error:
                await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=_runtime(client),
                    thread_id=state.thread_id,
                    request_id=state.request_id,
                    execution_id="exec_resume",
                )
        return error.value

    error = asyncio.run(scenario())

    assert error.code == expected_code
    assert graph.as_node is None
    assert graph.invoke_config is None


@pytest.mark.parametrize(
    ("anchor", "next_node"),
    [
        (ResumeAnchor.INGEST, "planner_select"),
        (ResumeAnchor.PLANNER_TOOL, "planner_select"),
    ],
)
def test_planner_select_resume_accepts_both_explicit_predecessors(
    anchor: ResumeAnchor,
    next_node: str,
):
    state = _initial_state(step_limit=20).model_copy(
        update={"resume_anchor": anchor}
    )
    graph = _RecordingGraph(
        state.model_dump(mode="json"),
        next_nodes=(next_node,),
        planner_enabled=True,
    )

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            return await invoke_agent(
                graph,
                request=_request(),
                runtime=_runtime(client),
                thread_id=state.thread_id,
                request_id=state.request_id,
                execution_id="exec_resume",
            )

    asyncio.run(scenario())

    assert graph.as_node == anchor.value


@pytest.mark.parametrize(
    ("raw_anchor", "next_node", "expected_code"),
    [
        (None, "route", "MISSING_RESUME_ANCHOR"),
        ("invented_node", "route", "UNKNOWN_RESUME_ANCHOR"),
        (ResumeAnchor.INGEST.value, "planner_tool", "RESUME_ANCHOR_MISMATCH"),
    ],
)
def test_resume_anchor_fails_closed_before_checkpoint_update(
    raw_anchor: str | None,
    next_node: str,
    expected_code: str,
):
    values = _initial_state(step_limit=20).model_dump(mode="json")
    if raw_anchor is None:
        values.pop("resume_anchor")
    else:
        values["resume_anchor"] = raw_anchor
    graph = _RecordingGraph(
        values,
        next_nodes=(next_node,),
        planner_enabled=True,
    )

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            with pytest.raises(AgentInvocationProtocolError) as error:
                await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=_runtime(client),
                    thread_id="thread_case_tkt_inv_04",
                    request_id="req_01",
                    execution_id="exec_resume",
                )
        return error.value

    error = asyncio.run(scenario())

    assert error.code == expected_code
    assert graph.as_node is None
    assert graph.invoke_config is None


@pytest.mark.parametrize("new_request", [False, True])
@pytest.mark.parametrize(
    ("raw_anchor", "expected_exception", "expected_code"),
    [
        (None, AgentInvocationProtocolError, "MISSING_RESUME_ANCHOR"),
        ("invented_node", AgentInvocationProtocolError, "UNKNOWN_RESUME_ANCHOR"),
        (ResumeAnchor.START.value, AgentInvocationProtocolError, "RESUME_ANCHOR_MISMATCH"),
        (ResumeAnchor.PLANNER_FINALIZE.value, ValidationError, None),
    ],
)
def test_terminal_resume_anchor_is_required_and_semantically_coherent(
    raw_anchor: str | None,
    expected_exception: type[Exception],
    expected_code: str | None,
    new_request: bool,
):
    state = _terminal_read_state()
    persisted_values = state.model_dump(mode="json")
    if raw_anchor is None:
        persisted_values.pop("resume_anchor")
    else:
        persisted_values["resume_anchor"] = raw_anchor
    graph = _RecordingGraph(persisted_values)

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            with pytest.raises(expected_exception) as error:
                await invoke_agent(
                    graph,
                    request=(
                        _request(message="Nova solicitação.")
                        if new_request
                        else _request()
                    ),
                    runtime=_runtime(client),
                    thread_id=state.thread_id,
                    request_id=("req_new_after_terminal" if new_request else state.request_id),
                    execution_id="exec_terminal_anchor_check",
                )
        return error.value

    error = asyncio.run(scenario())

    if expected_code is not None:
        assert isinstance(error, AgentInvocationProtocolError)
        assert error.code == expected_code
    assert graph.values == persisted_values
    assert graph.as_node is None
    assert graph.invoke_config is None


@pytest.mark.parametrize(
    ("planner_enabled", "anchor", "next_node"),
    [
        (False, ResumeAnchor.PLANNER_TOOL, "planner_select"),
        (True, ResumeAnchor.INGEST, "route"),
    ],
)
def test_partial_resume_rejects_checkpoint_from_other_graph_topology(
    planner_enabled: bool,
    anchor: ResumeAnchor,
    next_node: str,
):
    state = _initial_state(step_limit=20).model_copy(
        update={"resume_anchor": anchor}
    )
    persisted_values = state.model_dump(mode="json")
    graph = _RecordingGraph(
        persisted_values,
        next_nodes=(next_node,),
        planner_enabled=planner_enabled,
    )

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            with pytest.raises(AgentInvocationProtocolError) as error:
                await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=_runtime(client),
                    thread_id=state.thread_id,
                    request_id=state.request_id,
                    execution_id="exec_topology_mismatch",
                )
        return error.value

    error = asyncio.run(scenario())

    assert error.code == "RESUME_TOPOLOGY_MISMATCH"
    assert graph.as_node is None
    assert graph.invoke_config is None


def test_new_request_can_migrate_from_valid_terminal_fallback_to_planner():
    state = _terminal_read_state()
    graph = _RecordingGraph(
        state.model_dump(mode="json"),
        planner_enabled=True,
    )

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            return await invoke_agent(
                graph,
                request=_request(message="Novo ciclo no planner."),
                runtime=_runtime(client),
                thread_id=state.thread_id,
                request_id="req_migrated_to_planner",
                execution_id="exec_migrated_to_planner",
            )

    migrated = asyncio.run(scenario())

    assert graph.as_node == ResumeAnchor.START.value
    assert migrated.request_id == "req_migrated_to_planner"
    assert migrated.resume_anchor is ResumeAnchor.START
    assert migrated.step_limit == 20


def test_new_request_can_migrate_from_valid_terminal_planner_to_fallback():
    state = _terminal_planner_state()
    graph = _RecordingGraph(state.model_dump(mode="json"))

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            return await invoke_agent(
                graph,
                request=_request(message="Novo ciclo no fallback."),
                runtime=_runtime(client),
                thread_id=state.thread_id,
                request_id="req_migrated_to_fallback",
                execution_id="exec_migrated_to_fallback",
            )

    migrated = asyncio.run(scenario())

    assert graph.as_node == ResumeAnchor.START.value
    assert migrated.request_id == "req_migrated_to_fallback"
    assert migrated.resume_anchor is ResumeAnchor.START
    assert migrated.step_limit == 3


@pytest.mark.parametrize("mismatch", ["finish_with_intent", "failure_stage"])
def test_terminal_anchor_table_rejects_known_but_impossible_state(
    mismatch: str,
):
    if mismatch == "finish_with_intent":
        base = _terminal_read_state()
        values = base.model_dump(mode="python")
        values["intents"] = (_denied_historical_intent(base.request_id),)
        graph = _RecordingGraph(values, planner_enabled=True)

        async def scenario():
            async with IndustrialApiClient("https://industrial.test") as client:
                with pytest.raises(ValidationError, match="terminal diverge"):
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=_runtime(client),
                        thread_id=base.thread_id,
                        request_id=base.request_id,
                        execution_id="exec_impossible_terminal_anchor",
                    )

        asyncio.run(scenario())
    else:
        base = _initial_state(step_limit=20)
        values = base.model_dump(mode="python")
        values.update(
            decision=AgentDecision.REQUIRE_HUMAN_REVIEW,
            final_result=FinalResult(
                decision=AgentDecision.REQUIRE_HUMAN_REVIEW,
                message="Falha segura do planner.",
            ),
            review=ReviewRecord(
                status=ReviewStatus.REQUIRED,
                reason="planner:planner_tool:invalid_tool_result",
            ),
            planner_failure=PlannerFailureRecord(
                stage="planner_tool",
                code="invalid_tool_result",
            ),
            resume_anchor=ResumeAnchor.PLANNER_SELECT,
        )
        state = AgentState.model_validate(values)
        graph = _RecordingGraph(
            state.model_dump(mode="json"),
            planner_enabled=True,
        )

        async def scenario():
            async with IndustrialApiClient("https://industrial.test") as client:
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=_request(),
                        runtime=_runtime(client),
                        thread_id=state.thread_id,
                        request_id=state.request_id,
                        execution_id="exec_impossible_terminal_anchor",
                    )
            return error.value

        error = asyncio.run(scenario())
        assert error.code == "RESUME_ANCHOR_MISMATCH"

    assert graph.as_node is None
    assert graph.invoke_config is None


@pytest.mark.parametrize(
    "tamper",
    [
        "planner_guide_as_action",
        "fallback_finish_as_request_information",
        "denial_as_active_intent",
        "execution_as_guide",
        "execution_with_other_target",
        "execution_with_other_payload_hash",
    ],
)
def test_adulterated_terminal_checkpoint_fails_closed_before_return(
    tamper: str,
):
    if tamper == "planner_guide_as_action":
        persisted_values = _terminal_planner_state().model_dump(mode="json")
        persisted_values["decision"] = AgentDecision.ACT.value
        persisted_values["final_result"]["decision"] = AgentDecision.ACT.value
    elif tamper == "fallback_finish_as_request_information":
        persisted_values = _terminal_read_state().model_dump(mode="json")
        persisted_values["decision"] = AgentDecision.REQUEST_INFORMATION.value
        persisted_values["final_result"]["decision"] = (
            AgentDecision.REQUEST_INFORMATION.value
        )
    elif tamper == "denial_as_active_intent":
        persisted_values = _terminal_denial_state().model_dump(mode="json")
        persisted_values["intents"][0]["status"] = (
            IntentStatus.AWAITING_CONFIRMATION.value
        )
        persisted_values["intents"][0]["decision"] = {
            "decision": PolicyDecision.REQUIRE_CONFIRMATION.value,
            "reason": PolicyReason.EXPLICIT_APPROVAL_REQUIRED.value,
        }
    else:
        persisted_values = _terminal_execution_state().model_dump(mode="json")
        if tamper == "execution_as_guide":
            persisted_values["decision"] = AgentDecision.GUIDE.value
            persisted_values["final_result"]["decision"] = AgentDecision.GUIDE.value
        elif tamper == "execution_with_other_target":
            persisted_values["pending_proposal"]["analysis_id"] = "an_other"
        else:
            persisted_values["intents"][0]["payload_hash"] = (
                "sha256:v1:" + "b" * 64
            )
    graph = _RecordingGraph(persisted_values, planner_enabled=True)

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            with pytest.raises(ValidationError, match="terminal diverge"):
                await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=_runtime(client),
                    thread_id="thread_case_tkt_inv_04",
                    request_id="req_01",
                    execution_id="exec_adulterated_terminal",
                )

    asyncio.run(scenario())

    assert graph.as_node is None
    assert graph.invoke_config is None


def test_planner_step_limit_above_cap_only_blocks_same_request_resume():
    state = _terminal_read_state().model_copy(update={"step_limit": 21})

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            same_request_graph = _RecordingGraph(
                state.model_dump(mode="json"),
                planner_enabled=True,
            )
            with pytest.raises(AgentInvocationProtocolError) as error:
                await invoke_agent(
                    same_request_graph,
                    request=_request(),
                    runtime=_runtime(client),
                    thread_id=state.thread_id,
                    request_id=state.request_id,
                    execution_id="exec_same_over_cap",
                )
            new_request_graph = _RecordingGraph(
                state.model_dump(mode="json"),
                planner_enabled=True,
            )
            migrated = await invoke_agent(
                new_request_graph,
                request=_request(message="Novo ciclo com orçamento válido."),
                runtime=_runtime(client),
                thread_id=state.thread_id,
                request_id="req_after_over_cap",
                execution_id="exec_after_over_cap",
            )
        return error.value, same_request_graph, new_request_graph, migrated

    error, same_graph, new_graph, migrated = asyncio.run(scenario())

    assert error.code == "PLANNER_STEP_LIMIT_EXCEEDED"
    assert same_graph.as_node is None
    assert same_graph.invoke_config is None
    assert new_graph.as_node == ResumeAnchor.START.value
    assert migrated.step_limit == 20


def test_planner_builder_defaults_to_20_and_rejects_21_before_checkpoint():
    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            graph = _RecordingGraph(planner_enabled=True)
            state = await invoke_agent(
                graph,
                request=_request(),
                runtime=_runtime(client),
                thread_id="thread_planner_budget",
                request_id="req_planner_budget",
                execution_id="exec_planner_budget",
            )
            rejected = _RecordingGraph(planner_enabled=True)
            with pytest.raises(AgentInvocationProtocolError) as error:
                await invoke_agent(
                    rejected,
                    request=_request(),
                    runtime=_runtime(client),
                    thread_id="thread_planner_over_budget",
                    request_id="req_planner_over_budget",
                    execution_id="exec_planner_over_budget",
                    step_limit=21,
                )
        return state, rejected, error.value

    state, rejected, error = asyncio.run(scenario())

    assert state.step_limit == 20
    assert error.code == "PLANNER_STEP_LIMIT_EXCEEDED"
    assert rejected.state_config is None
    assert rejected.invoke_config is None


def test_original_approval_without_proposal_is_only_accepted_with_planner():
    approval = TrustedActionApproval(
        action="update_asset_criticality",
        target_id="asset_G501",
        material_parameters={"criticality": "critical"},
        source=ApprovalSource.ORIGINAL_REQUEST,
    )

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            runtime = WriteToolRuntime.create(
                user_id="usr_pedro",
                company_id="comp_mineracao_andes",
                permissions=frozenset({"action_high"}),
                central_asset_id="asset_G501",
                current_case_id="case_tkt_inv_04",
                client=client,
            )
            planner_graph = _RecordingGraph(planner_enabled=True)
            accepted = await invoke_agent(
                planner_graph,
                request=_request(),
                runtime=runtime,
                thread_id="thread_planner_approval",
                request_id="req_planner_approval",
                execution_id="exec_planner_approval",
                original_approval=approval,
            )
            fallback = _RecordingGraph()
            with pytest.raises(AgentInvocationProtocolError) as error:
                await invoke_agent(
                    fallback,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_fallback_approval",
                    request_id="req_fallback_approval",
                    execution_id="exec_fallback_approval",
                    original_approval=approval,
                )
        return accepted, planner_graph, fallback, error.value

    accepted, planner_graph, fallback, error = asyncio.run(scenario())

    assert accepted.approval == approval
    assert accepted.pending_proposal is None
    assert planner_graph.state_config is not None
    assert error.code == "ORIGINAL_APPROVAL_WITHOUT_PROPOSAL"
    assert fallback.state_config is None


def test_build_graph_rejects_checkpointer_outside_managed_lifecycle(
    tmp_path: Path,
):
    async def scenario():
        async with AsyncSqliteSaver.from_conn_string(
            str(tmp_path / "unmanaged.sqlite3")
        ) as saver:
            with pytest.raises(
                TypeError,
                match="construa-o com open_checkpointer",
            ):
                build_agent_graph(saver)

    asyncio.run(scenario())


def test_compiled_graph_constructor_rejects_owner_injection_before_access(
    tmp_path: Path,
):
    async def scenario():
        async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
            managed = build_agent_graph(saver)
            compiled = managed._graph
            accesses = []

            async def forbidden_get_state(*args, **kwargs):
                accesses.append("aget_state")
                raise AssertionError("construtor não pode ler checkpoint")

            async def forbidden_invoke(*args, **kwargs):
                accesses.append("ainvoke")
                raise AssertionError("construtor não pode invocar grafo")

            compiled.aget_state = forbidden_get_state
            compiled.ainvoke = forbidden_invoke
            with pytest.raises(TypeError):
                CompiledAgentGraph(compiled, LocalCheckpointOwner())
            assert accesses == []

            derived = CompiledAgentGraph(compiled)
            derived_entered = asyncio.Event()

            async def enter_derived_lock():
                async with derived.thread_lock("thread_constructor_probe"):
                    derived_entered.set()

            async with managed.thread_lock("thread_constructor_probe"):
                contender = asyncio.create_task(enter_derived_lock())
                await asyncio.sleep(0)
                assert not derived_entered.is_set()
            await contender
            assert derived_entered.is_set()
            assert accesses == []

    asyncio.run(scenario())


def test_ingest_node_requires_and_reads_trusted_runtime_context(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"

    async def scenario():
        async with IndustrialApiClient("https://context-marker.invalid") as client:
            runtime = _runtime(client)
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                missing_context_config = {
                    "configurable": {"thread_id": "thread_missing_context"}
                }
                with pytest.raises(
                    TypeError,
                    match="contexto confiável do grafo é obrigatório",
                ):
                    await graph.ainvoke(
                        _initial_state(
                            thread_id="thread_missing_context"
                        ).model_dump(mode="json"),
                        missing_context_config,
                        context=None,
                        durability="sync",
                    )

                valid_config = {
                    "configurable": {"thread_id": "thread_valid_context"}
                }
                result = await graph.ainvoke(
                    _initial_state(thread_id="thread_valid_context").model_dump(
                        mode="json"
                    ),
                    valid_config,
                    context=runtime,
                    durability="sync",
                )
                checkpoint = await saver.aget(valid_config)
        return result, checkpoint

    result, checkpoint = asyncio.run(scenario())
    assert result["final_result"]["decision"] == "guide"
    checkpoint_text = repr(checkpoint)
    assert "context-marker.invalid" not in checkpoint_text
    assert "IndustrialApiClient" not in checkpoint_text


def test_minimal_graph_finishes_a_simple_read_in_three_observable_steps(
    tmp_path: Path,
):
    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            runtime = _runtime(client)
            async with open_checkpointer(tmp_path / "checkpoints.sqlite3") as saver:
                graph = build_agent_graph(saver)
                state = await invoke_agent(
                    graph,
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_case_tkt_inv_04",
                    request_id="req_01",
                    execution_id="exec_01",
                    step_limit=3,
                )

                edges = {
                    (edge.source, edge.target)
                    for edge in graph.get_graph().edges
                    if edge.source != "__start__" and edge.target != "__end__"
                }

        assert {("ingest", "route"), ("route", "finish")} <= edges
        assert state.step_count == 3
        assert state.decision is AgentDecision.GUIDE
        assert state.final_result is not None
        assert state.final_result.decision is AgentDecision.GUIDE
        assert state.messages[-1].role is MessageRole.USER
        assert state.messages[-1].content == "Consulte o estado deste ativo."

    asyncio.run(scenario())


def test_reopened_saver_continues_state_without_checkpointing_runtime(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"

    async def scenario():
        async with IndustrialApiClient("https://runtime-one.invalid") as client:
            async with open_checkpointer(checkpoint_path) as saver:
                first = await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(message="Primeira leitura."),
                    runtime=_runtime(client),
                    thread_id="thread_case_tkt_inv_04",
                    request_id="req_01",
                    execution_id="exec_01",
                )

        async with IndustrialApiClient("https://runtime-two.invalid") as client:
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                continued = await invoke_agent(
                    graph,
                    request=_request(message="Segunda leitura."),
                    runtime=_runtime(client),
                    thread_id="thread_case_tkt_inv_04",
                    request_id="req_02",
                    execution_id="exec_02",
                )
                raw_checkpoint = await saver.aget(
                    {"configurable": {"thread_id": "thread_case_tkt_inv_04"}}
                )

        return first, continued, raw_checkpoint

    first, continued, raw_checkpoint = asyncio.run(scenario())

    assert first.execution_id == "exec_01"
    assert continued.execution_id == "exec_02"
    assert continued.request_id == "req_02"
    assert [message.content for message in continued.messages] == [
        "Primeira leitura.",
        "Segunda leitura.",
    ]
    checkpoint_text = repr(raw_checkpoint)
    assert "runtime-one.invalid" not in checkpoint_text
    assert "runtime-two.invalid" not in checkpoint_text
    assert "IndustrialApiClient" not in checkpoint_text


def test_continuation_fails_closed_when_authenticated_identity_changes(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            async with open_checkpointer(checkpoint_path) as saver:
                await invoke_agent(
                    build_agent_graph(saver),
                    request=_request(),
                    runtime=_runtime(client),
                    thread_id="thread_case_tkt_inv_04",
                    request_id="req_01",
                    execution_id="exec_01",
                )

        async with IndustrialApiClient("https://industrial.test") as client:
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                with pytest.raises(
                    ValidationError,
                    match="thread reutilizado fora do escopo confiável",
                ):
                    await invoke_agent(
                        graph,
                        request=_request(user_id="usr_intruso"),
                        runtime=_runtime(client, user_id="usr_intruso"),
                        thread_id="thread_case_tkt_inv_04",
                        request_id="req_02",
                        execution_id="exec_02",
                    )
                unchanged = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_case_tkt_inv_04"}}
                )

        return unchanged.values

    values = asyncio.run(scenario())
    assert values["execution_id"] == "exec_01"
    assert values["request_id"] == "req_01"


def test_threads_are_isolated_and_only_explicit_deletion_removes_one(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    config_a = {"configurable": {"thread_id": "thread_a"}}
    config_b = {"configurable": {"thread_id": "thread_b"}}

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                await invoke_agent(
                    graph,
                    request=_request(message="Leitura exclusiva A."),
                    runtime=_runtime(client),
                    thread_id="thread_a",
                    request_id="req_a",
                    execution_id="exec_a",
                )
                await invoke_agent(
                    graph,
                    request=_request(message="Leitura exclusiva B."),
                    runtime=_runtime(client),
                    thread_id="thread_b",
                    request_id="req_b",
                    execution_id="exec_b",
                )

        async with open_checkpointer(checkpoint_path) as saver:
            graph = build_agent_graph(saver)
            before_a = await graph.aget_state(config_a)
            before_b = await graph.aget_state(config_b)
            await saver.adelete_thread("thread_a")
            deleted_a = await graph.aget_state(config_a)
            preserved_b = await graph.aget_state(config_b)

        async with open_checkpointer(checkpoint_path) as saver:
            reopened_graph = build_agent_graph(saver)
            reopened_b = await reopened_graph.aget_state(config_b)

        return before_a, before_b, deleted_a, preserved_b, reopened_b

    before_a, before_b, deleted_a, preserved_b, reopened_b = asyncio.run(scenario())

    assert before_a.values["messages"][-1]["content"] == "Leitura exclusiva A."
    assert before_b.values["messages"][-1]["content"] == "Leitura exclusiva B."
    assert deleted_a.values == {}
    assert preserved_b.values["thread_id"] == "thread_b"
    assert reopened_b.values["thread_id"] == "thread_b"


def test_step_limit_is_not_exceeded_and_connection_closes_after_graph_error(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    config = {"configurable": {"thread_id": "thread_limited"}}

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            with pytest.raises(ValueError, match="orçamento de passos esgotado"):
                async with open_checkpointer(checkpoint_path) as saver:
                    await invoke_agent(
                        build_agent_graph(saver),
                        request=_request(),
                        runtime=_runtime(client),
                        thread_id="thread_limited",
                        request_id="req_limited",
                        execution_id="exec_limited",
                        step_limit=2,
                    )

            with pytest.raises(ValueError, match="no active connection"):
                await saver.conn.execute("SELECT 1")

        async with open_checkpointer(checkpoint_path) as reopened_saver:
            snapshot = await build_agent_graph(reopened_saver).aget_state(config)
        return snapshot.values

    values = asyncio.run(scenario())
    assert values["step_count"] == 2
    assert values["step_limit"] == 2
    assert values["final_result"] is None


def test_new_request_recovers_partial_thread_with_a_new_step_limit(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            runtime = _runtime(client)
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                with pytest.raises(ValueError, match="orçamento de passos esgotado"):
                    await invoke_agent(
                        graph,
                        request=_request(message="Pedido parcial."),
                        runtime=runtime,
                        thread_id="thread_recovered",
                        request_id="req_partial",
                        execution_id="exec_partial",
                        step_limit=2,
                    )

                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=_request(message="Pedido parcial."),
                        runtime=runtime,
                        thread_id="thread_recovered",
                        request_id="req_partial",
                        execution_id="exec_partial_retry",
                        step_limit=99,
                    )
                assert error.value.code == "STEP_LIMIT_EXHAUSTED"

                recovered = await invoke_agent(
                    graph,
                    request=_request(message="Novo pedido completo."),
                    runtime=runtime,
                    thread_id="thread_recovered",
                    request_id="req_recovered",
                    execution_id="exec_recovered",
                    step_limit=3,
                )
        return recovered

    recovered = asyncio.run(scenario())
    assert recovered.request_id == "req_recovered"
    assert recovered.execution_id == "exec_recovered"
    assert recovered.step_count == 3
    assert recovered.step_limit == 3
    assert [message.content for message in recovered.messages] == [
        "Pedido parcial.",
        "Novo pedido completo.",
    ]


def test_new_request_resumes_from_ingest_after_crash_between_update_and_invoke(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    config = {"configurable": {"thread_id": "thread_crash_after_start"}}
    request_a = _request(message="Ciclo anterior concluído.")
    request_b = _request(message="Novo ciclo recuperável.")

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            runtime = _runtime(client)
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                await invoke_agent(
                    graph,
                    request=request_a,
                    runtime=runtime,
                    thread_id="thread_crash_after_start",
                    request_id="req_before_crash",
                    execution_id="exec_before_crash",
                )
                original_get_state = graph.aget_state

                async def crash_before_graph_runs(*args, **kwargs):
                    raise RuntimeError("falha forçada após update START")

                graph.ainvoke = crash_before_graph_runs
                with pytest.raises(
                    RuntimeError,
                    match="falha forçada após update START",
                ):
                    await invoke_agent(
                        graph,
                        request=request_b,
                        runtime=runtime,
                        thread_id="thread_crash_after_start",
                        request_id="req_after_crash",
                        execution_id="exec_crashed",
                    )
                crashed = await original_get_state(config)

            async with open_checkpointer(checkpoint_path) as reopened_saver:
                reopened_graph = build_agent_graph(reopened_saver)
                before_resume = await reopened_graph.aget_state(config)
                recovered = await invoke_agent(
                    reopened_graph,
                    request=request_b,
                    runtime=runtime,
                    thread_id="thread_crash_after_start",
                    request_id="req_after_crash",
                    execution_id="exec_recovered",
                )
                terminal = await reopened_graph.aget_state(config)
        return crashed, before_resume, recovered, terminal

    crashed, before_resume, recovered, terminal = asyncio.run(scenario())

    assert crashed.next == ("ingest",)
    assert before_resume.next == ("ingest",)
    assert before_resume.values["request_id"] == "req_after_crash"
    assert before_resume.values["step_count"] == 0
    assert recovered.execution_id == "exec_recovered"
    assert recovered.step_count == 3
    assert recovered.final_result is not None
    assert [message.content for message in recovered.messages] == [
        "Ciclo anterior concluído.",
        "Novo ciclo recuperável.",
    ]
    assert terminal.next == ()
    assert terminal.values == recovered.model_dump(mode="json")


def test_same_terminal_request_replays_immutable_state_without_ainvoke(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    config = {"configurable": {"thread_id": "thread_terminal_replay"}}

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            runtime = _runtime(client)
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                completed = await invoke_agent(
                    graph,
                    request=_request(message="Pedido concluído."),
                    runtime=runtime,
                    thread_id="thread_terminal_replay",
                    request_id="req_terminal",
                    execution_id="exec_original",
                )
                original_get_state = graph.aget_state
                before = await original_get_state(config)

                async def forbidden_invoke(*args, **kwargs):
                    raise AssertionError("replay terminal não pode chamar ainvoke")

                graph.ainvoke = forbidden_invoke
                replayed = await invoke_agent(
                    graph,
                    request=_request(message="Pedido concluído."),
                    runtime=runtime,
                    thread_id="thread_terminal_replay",
                    request_id="req_terminal",
                    execution_id="exec_delivery_retry",
                    step_limit=99,
                )
                after = await original_get_state(config)
        return completed, replayed, before, after

    completed, replayed, before, after = asyncio.run(scenario())
    assert replayed == completed
    assert replayed.execution_id == "exec_original"
    assert before.config == after.config
    assert before.values == after.values


def test_same_partial_request_resumes_only_pending_nodes_and_preserves_budget(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    request = _request(message="Pedido parcialmente processado.")
    config = {"configurable": {"thread_id": "thread_partial_resume"}}

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            runtime = _runtime(client)
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                await graph.ainvoke(
                    _initial_state(
                        thread_id="thread_partial_resume",
                        request=request,
                    ).model_dump(mode="json"),
                    config,
                    context=runtime,
                    durability="sync",
                    interrupt_after=["ingest"],
                )
                partial = await graph.aget_state(config)
                resumed = await invoke_agent(
                    graph,
                    request=request,
                    runtime=runtime,
                    thread_id="thread_partial_resume",
                    request_id="req_01",
                    execution_id="exec_resumed",
                    step_limit=99,
                )
        return partial, resumed

    partial, resumed = asyncio.run(scenario())
    assert partial.next == ("route",)
    assert partial.values["step_count"] == 1
    assert partial.values["step_limit"] == 3
    assert resumed.execution_id == "exec_resumed"
    assert resumed.step_count == 3
    assert resumed.step_limit == 3
    assert [message.content for message in resumed.messages] == [
        "Pedido parcialmente processado."
    ]


def test_same_partial_request_after_route_resumes_only_finish(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    request = _request(message="Pedido roteado antes da retomada.")
    config = {"configurable": {"thread_id": "thread_resume_after_route"}}

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            runtime = _runtime(client)
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                await graph.ainvoke(
                    _initial_state(
                        thread_id="thread_resume_after_route",
                        request=request,
                    ).model_dump(mode="json"),
                    config,
                    context=runtime,
                    durability="sync",
                    interrupt_after=["route"],
                )
                partial = await graph.aget_state(config)
                resumed = await invoke_agent(
                    graph,
                    request=request,
                    runtime=runtime,
                    thread_id="thread_resume_after_route",
                    request_id="req_01",
                    execution_id="exec_after_route",
                )
        return partial, resumed

    partial, resumed = asyncio.run(scenario())

    assert partial.next == ("finish",)
    assert partial.values["step_count"] == 2
    assert resumed.step_count == 3
    assert resumed.final_result is not None
    assert [message.content for message in resumed.messages] == [
        "Pedido roteado antes da retomada."
    ]


def test_same_partial_request_with_insufficient_budget_fails_by_protocol(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    request = _request(message="Pedido sem orçamento suficiente.")
    config = {"configurable": {"thread_id": "thread_insufficient_budget"}}

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            runtime = _runtime(client)
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                await graph.ainvoke(
                    _initial_state(
                        thread_id="thread_insufficient_budget",
                        step_limit=2,
                        request=request,
                    ).model_dump(mode="json"),
                    config,
                    context=runtime,
                    durability="sync",
                    interrupt_after=["ingest"],
                )
                before = await graph.aget_state(config)
                with pytest.raises(AgentInvocationProtocolError) as error:
                    await invoke_agent(
                        graph,
                        request=request,
                        runtime=runtime,
                        thread_id="thread_insufficient_budget",
                        request_id="req_01",
                        execution_id="exec_retry",
                        step_limit=99,
                    )
                after = await graph.aget_state(config)
        return error.value, before, after

    error, before, after = asyncio.run(scenario())
    assert error.code == "STEP_LIMIT_EXHAUSTED"
    assert before.config == after.config
    assert before.values == after.values


def test_checkpoint_preserves_intent_expiration_and_never_creates_a_key(
    tmp_path: Path,
):
    expires_at = datetime(
        2026,
        9,
        6,
        9,
        30,
        17,
        123,
        tzinfo=timezone(timedelta(hours=-3)),
    )
    intent = WriteIntent(
        intent_id="intent_018f3a",
        scope=ReprocessIntentScope(
            action="reprocess_analysis",
            case_id="case_tkt_inv_04",
            company_id="comp_mineracao_andes",
            user_id="usr_pedro",
            analysis_id="an_9906",
            justification="Rolamento substituído; solicitar novo processamento.",
        ),
        payload_hash="sha256:v1:" + "a" * 64,
        decision=WritePolicyResult(
            decision=PolicyDecision.ALLOW,
            reason=PolicyReason.AUTHORIZED,
        ),
        status=IntentStatus.PREPARED,
        idempotency_key="tractian-agent:018f3a",
        expires_at=expires_at,
        prepared_execution_id="exec_01",
    )
    identity = TrustedIdentity(
        user_id="usr_pedro",
        company_id="comp_mineracao_andes",
    )
    initial = AgentState(
        request=_request(),
        identity=identity,
        permissions=frozenset({"read", "action_low"}),
        request_id="req_01",
        thread_id="thread_with_intent",
        execution_id="exec_01",
        thread_scope=ThreadScope(
            thread_id="thread_with_intent",
            case_id="case_tkt_inv_04",
            company_id="comp_mineracao_andes",
            user_id="usr_pedro",
        ),
        step_limit=3,
        intents=(intent,),
    )
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    config = {"configurable": {"thread_id": "thread_with_intent"}}

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            runtime = _runtime(
                client,
                permissions=frozenset({"read", "action_low"}),
            )
            async with open_checkpointer(checkpoint_path) as saver:
                await build_agent_graph(saver).ainvoke(
                    initial.model_dump(mode="json"),
                    config,
                    context=runtime,
                    durability="sync",
                )

        async with open_checkpointer(checkpoint_path) as saver:
            snapshot = await build_agent_graph(saver).aget_state(config)
        return AgentState.model_validate(snapshot.values)

    restored = asyncio.run(scenario())

    assert len(restored.intents) == 1
    restored_intent = restored.intents[0]
    assert restored_intent.intent_id == intent.intent_id
    assert restored_intent.idempotency_key == "tractian-agent:018f3a"
    assert restored_intent.expires_at is not None
    assert restored_intent.expires_at.isoformat() == expires_at.isoformat()


def test_concurrent_requests_on_the_same_thread_are_serialized_without_loss(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    config = {"configurable": {"thread_id": "thread_concurrent"}}

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                original_get_state = graph.aget_state
                first_snapshot_read = asyncio.Event()
                release_first = asyncio.Event()
                reads = 0

                async def delayed_get_state(read_config):
                    nonlocal reads
                    snapshot = await original_get_state(read_config)
                    reads += 1
                    if reads == 1:
                        first_snapshot_read.set()
                        await release_first.wait()
                    return snapshot

                graph.aget_state = delayed_get_state
                first_task = asyncio.create_task(
                    invoke_agent(
                        graph,
                        request=_request(message="Leitura concorrente A."),
                        runtime=_runtime(client),
                        thread_id="thread_concurrent",
                        request_id="req_concurrent_a",
                        execution_id="exec_concurrent_a",
                    )
                )
                await first_snapshot_read.wait()
                second_task = asyncio.create_task(
                    invoke_agent(
                        graph,
                        request=_request(message="Leitura concorrente B."),
                        runtime=_runtime(client),
                        thread_id="thread_concurrent",
                        request_id="req_concurrent_b",
                        execution_id="exec_concurrent_b",
                    )
                )
                await asyncio.sleep(0)
                release_first.set()
                results = await asyncio.gather(first_task, second_task)
                final_snapshot = await original_get_state(config)
        return results, final_snapshot.values

    results, values = asyncio.run(scenario())

    assert {result.request_id for result in results} == {
        "req_concurrent_a",
        "req_concurrent_b",
    }
    assert [message["content"] for message in values["messages"]] == [
        "Leitura concorrente A.",
        "Leitura concorrente B.",
    ]


def test_two_graph_wrappers_on_the_same_checkpoint_owner_serialize_the_thread(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    config = {"configurable": {"thread_id": "thread_shared_owner"}}

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            async with open_checkpointer(checkpoint_path) as saver:
                first_graph = build_agent_graph(saver)
                second_graph = build_agent_graph(saver)
                first_get_state = first_graph.aget_state
                second_get_state = second_graph.aget_state
                first_snapshot_read = asyncio.Event()
                second_snapshot_read = asyncio.Event()
                release_first = asyncio.Event()

                async def delayed_first_get_state(read_config):
                    snapshot = await first_get_state(read_config)
                    first_snapshot_read.set()
                    await release_first.wait()
                    return snapshot

                async def observed_second_get_state(read_config):
                    snapshot = await second_get_state(read_config)
                    second_snapshot_read.set()
                    return snapshot

                first_graph.aget_state = delayed_first_get_state
                second_graph.aget_state = observed_second_get_state
                first_task = asyncio.create_task(
                    invoke_agent(
                        first_graph,
                        request=_request(message="Owner compartilhado A."),
                        runtime=_runtime(client),
                        thread_id="thread_shared_owner",
                        request_id="req_shared_owner_a",
                        execution_id="exec_shared_owner_a",
                    )
                )
                await first_snapshot_read.wait()
                second_task = asyncio.create_task(
                    invoke_agent(
                        second_graph,
                        request=_request(message="Owner compartilhado B."),
                        runtime=_runtime(client),
                        thread_id="thread_shared_owner",
                        request_id="req_shared_owner_b",
                        execution_id="exec_shared_owner_b",
                    )
                )
                try:
                    await asyncio.wait_for(second_snapshot_read.wait(), timeout=0.1)
                except TimeoutError:
                    pass
                finally:
                    release_first.set()
                results = await asyncio.gather(first_task, second_task)
                final_snapshot = await first_get_state(config)
        return results, final_snapshot.values

    results, values = asyncio.run(scenario())

    assert {result.request_id for result in results} == {
        "req_shared_owner_a",
        "req_shared_owner_b",
    }
    assert [message["content"] for message in values["messages"]] == [
        "Owner compartilhado A.",
        "Owner compartilhado B.",
    ]


def test_concurrent_divergent_identity_fails_after_thread_is_bound(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                original_get_state = graph.aget_state
                first_snapshot_read = asyncio.Event()
                release_first = asyncio.Event()
                reads = 0

                async def delayed_get_state(read_config):
                    nonlocal reads
                    snapshot = await original_get_state(read_config)
                    reads += 1
                    if reads == 1:
                        first_snapshot_read.set()
                        await release_first.wait()
                    return snapshot

                graph.aget_state = delayed_get_state
                first_task = asyncio.create_task(
                    invoke_agent(
                        graph,
                        request=_request(),
                        runtime=_runtime(client),
                        thread_id="thread_identity_race",
                        request_id="req_owner",
                        execution_id="exec_owner",
                    )
                )
                await first_snapshot_read.wait()
                intruder_task = asyncio.create_task(
                    invoke_agent(
                        graph,
                        request=_request(user_id="usr_intruso"),
                        runtime=_runtime(client, user_id="usr_intruso"),
                        thread_id="thread_identity_race",
                        request_id="req_intruder",
                        execution_id="exec_intruder",
                    )
                )
                await asyncio.sleep(0)
                release_first.set()
                owner = await first_task
                with pytest.raises(
                    ValidationError,
                    match="thread reutilizado fora do escopo confiável",
                ):
                    await intruder_task
        return owner

    owner = asyncio.run(scenario())
    assert owner.identity.user_id == "usr_pedro"


def test_different_threads_do_not_share_the_same_local_lock(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"

    async def scenario():
        async with IndustrialApiClient("https://industrial.test") as client:
            async with open_checkpointer(checkpoint_path) as saver:
                graph = build_agent_graph(saver)
                original_invoke = graph.ainvoke
                first_invoke_started = asyncio.Event()
                release_first = asyncio.Event()

                async def delayed_invoke(input, config, **kwargs):
                    if config["configurable"]["thread_id"] == "thread_slow":
                        first_invoke_started.set()
                        await release_first.wait()
                    return await original_invoke(input, config, **kwargs)

                graph.ainvoke = delayed_invoke
                slow_task = asyncio.create_task(
                    invoke_agent(
                        graph,
                        request=_request(message="Thread lenta."),
                        runtime=_runtime(client),
                        thread_id="thread_slow",
                        request_id="req_slow",
                        execution_id="exec_slow",
                    )
                )
                await first_invoke_started.wait()
                fast_task = asyncio.create_task(
                    invoke_agent(
                        graph,
                        request=_request(message="Thread rápida."),
                        runtime=_runtime(client),
                        thread_id="thread_fast",
                        request_id="req_fast",
                        execution_id="exec_fast",
                    )
                )
                fast = await asyncio.wait_for(fast_task, timeout=1)
                release_first.set()
                slow = await slow_task
        return slow, fast

    slow, fast = asyncio.run(scenario())
    assert slow.thread_id == "thread_slow"
    assert fast.thread_id == "thread_fast"
