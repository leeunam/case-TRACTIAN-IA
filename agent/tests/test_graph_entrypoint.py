import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tractian_agent.checkpoint import open_checkpointer
from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import Identity, SupportRequest
from tractian_agent.entrypoint import invoke_agent
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
) -> ReadToolRuntime:
    return ReadToolRuntime.create(
        user_id=user_id,
        company_id=company_id,
        permissions=frozenset({"read"}),
        central_asset_id="asset_G501",
        client=client,
    )


class _RecordingGraph:
    def __init__(self, values=None):
        self.values = values or {}
        self.state_config = None
        self.invoke_config = None
        self.durability = None

    async def aget_state(self, config):
        self.state_config = config
        return SimpleNamespace(values=self.values)

    async def ainvoke(self, input, config, *, durability):
        self.invoke_config = config
        self.durability = durability
        return input


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
        async with open_checkpointer(checkpoint_path) as saver:
            await build_agent_graph(saver).ainvoke(
                initial.model_dump(mode="json"),
                config,
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
