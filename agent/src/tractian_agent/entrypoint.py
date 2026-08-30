"""Fronteira Python assíncrona do grafo determinístico."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Literal, Protocol

from langgraph.graph import START

from tractian_agent.contracts import SupportRequest
from tractian_agent.graph import MINIMAL_GRAPH_STEP_COUNT
from tractian_agent.state import AgentState, ThreadScope
from tractian_agent.tools.runtime import ReadToolRuntime


class AgentInvocationProtocolError(RuntimeError):
    """Checkpoint válido, mas incompatível com a operação solicitada."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class GraphStateSnapshot(Protocol):
    values: Mapping[str, object]
    config: dict[str, object]
    next: tuple[str, ...]


class AgentGraph(Protocol):
    def thread_lock(self, thread_id: str) -> AbstractAsyncContextManager[None]: ...

    async def aget_state(
        self,
        config: dict[str, object],
    ) -> GraphStateSnapshot: ...

    async def ainvoke(
        self,
        input: dict[str, object] | None,
        config: dict[str, object],
        *,
        context: ReadToolRuntime,
        durability: Literal["sync"],
    ) -> Mapping[str, object]: ...

    async def aupdate_state(
        self,
        config: dict[str, object],
        values: dict[str, object],
        *,
        as_node: str,
    ) -> dict[str, object]: ...


_PENDING_PREDECESSORS = {
    "ingest": START,
    "route": "ingest",
    "finish": "route",
}


def _require_opaque_id(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value or any(
        character.isspace() for character in value
    ):
        raise ValueError(f"{name} é obrigatório e não aceita espaços")
    return value


def _pending_predecessor(next_nodes: tuple[str, ...]) -> str:
    if not next_nodes:
        raise AgentInvocationProtocolError(
            "NON_TERMINAL_WITHOUT_PENDING_WORK",
            "checkpoint não terminal não possui trabalho pendente",
        )
    if len(next_nodes) != 1:
        raise AgentInvocationProtocolError(
            "MULTIPLE_PENDING_NODES",
            "checkpoint possui múltiplos nós pendentes",
        )
    pending_node = next_nodes[0]
    try:
        return _PENDING_PREDECESSORS[pending_node]
    except KeyError as error:
        raise AgentInvocationProtocolError(
            "UNKNOWN_PENDING_NODE",
            f"checkpoint possui nó pendente desconhecido: {pending_node}",
        ) from error


async def invoke_agent(
    graph: AgentGraph,
    *,
    request: SupportRequest,
    runtime: ReadToolRuntime,
    thread_id: str,
    request_id: str,
    execution_id: str,
    step_limit: int = 3,
) -> AgentState:
    """Cria ou continua estado confiável e executa com checkpoint síncrono."""
    if not isinstance(runtime, ReadToolRuntime):
        raise TypeError("runtime autenticado é obrigatório")
    thread_id = _require_opaque_id(thread_id, name="thread_id")
    request_id = _require_opaque_id(request_id, name="request_id")
    execution_id = _require_opaque_id(execution_id, name="execution_id")
    config: dict[str, object] = {"configurable": {"thread_id": thread_id}}

    async with graph.thread_lock(thread_id):
        snapshot = await graph.aget_state(config)
        persisted_values = snapshot.values
        if persisted_values:
            persisted = AgentState.model_validate(persisted_values)
            new_request = request_id != persisted.request_id
            state = persisted.continue_with(
                request=request,
                identity=runtime.identity,
                permissions=runtime.permissions,
                request_id=request_id,
                execution_id=execution_id,
                step_limit=step_limit,
            )
            if not new_request and persisted.final_result is not None:
                if snapshot.next:
                    raise AgentInvocationProtocolError(
                        "TERMINAL_WITH_PENDING_WORK",
                        "checkpoint terminal ainda possui trabalho pendente",
                    )
                return persisted
            if new_request:
                config = await graph.aupdate_state(
                    snapshot.config,
                    state.model_dump(mode="json"),
                    as_node=START,
                )
                invocation_input = None
            else:
                predecessor = _pending_predecessor(snapshot.next)
                if state.step_limit < MINIMAL_GRAPH_STEP_COUNT:
                    raise AgentInvocationProtocolError(
                        "STEP_LIMIT_EXHAUSTED",
                        "checkpoint parcial esgotou o orçamento de passos",
                    )
                config = await graph.aupdate_state(
                    snapshot.config,
                    state.model_dump(mode="json"),
                    as_node=predecessor,
                )
                invocation_input = None
        else:
            state = AgentState(
                request=request,
                identity=runtime.identity,
                permissions=runtime.permissions,
                request_id=request_id,
                thread_id=thread_id,
                execution_id=execution_id,
                thread_scope=ThreadScope(
                    thread_id=thread_id,
                    case_id=request.case_id,
                    company_id=runtime.identity.company_id,
                    user_id=runtime.identity.user_id,
                ),
                step_limit=step_limit,
            )
            invocation_input = state.model_dump(mode="json")

        result = await graph.ainvoke(
            invocation_input,
            config,
            context=runtime,
            durability="sync",
        )
    if not isinstance(result, Mapping):
        raise TypeError("o grafo devolveu estado inválido")
    return AgentState.model_validate(result)
