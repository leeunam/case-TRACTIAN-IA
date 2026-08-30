import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tractian_agent.checkpoint import open_checkpointer
from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import Identity, SupportRequest
from tractian_agent.entrypoint import AgentInvocationProtocolError, invoke_agent
from tractian_agent.graph import build_agent_graph
from tractian_agent.state import (
    AgentDecision,
    AgentState,
    MessageRole,
    ThreadScope,
)
from tractian_agent.tools.runtime import ReadToolRuntime, TrustedIdentity
from tractian_agent.write_contracts import (
    IntentStatus,
    ReprocessIntentScope,
    WriteIntent,
)
from tractian_agent.write_policy import (
    PolicyDecision,
    PolicyReason,
    WritePolicyResult,
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
) -> ReadToolRuntime:
    return ReadToolRuntime.create(
        user_id=user_id,
        company_id=company_id,
        permissions=permissions,
        central_asset_id="asset_G501",
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


class _RecordingGraph:
    def __init__(self, values=None):
        self.values = values or {}
        self.state_config = None
        self.invoke_config = None
        self.durability = None
        self.context = None

    @asynccontextmanager
    async def thread_lock(self, thread_id):
        yield

    async def aget_state(self, config):
        self.state_config = config
        return SimpleNamespace(values=self.values, config=config, next=())

    async def ainvoke(self, input, config, *, context=None, durability):
        self.invoke_config = config
        self.durability = durability
        self.context = context
        return self.values if input is None else input

    async def aupdate_state(self, config, values, *, as_node=None):
        self.values = values
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

        assert edges == {("ingest", "route"), ("route", "finish")}
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
