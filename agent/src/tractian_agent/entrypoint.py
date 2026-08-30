"""Fronteira Python assíncrona do grafo determinístico."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol

from tractian_agent.contracts import SupportRequest
from tractian_agent.state import AgentState, ThreadScope
from tractian_agent.tools.runtime import ReadToolRuntime


class GraphStateSnapshot(Protocol):
    values: Mapping[str, object]


class AgentGraph(Protocol):
    async def aget_state(
        self,
        config: dict[str, object],
    ) -> GraphStateSnapshot: ...

    async def ainvoke(
        self,
        input: dict[str, object],
        config: dict[str, object],
        *,
        durability: Literal["sync"],
    ) -> Mapping[str, object]: ...


def _require_opaque_id(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value or any(
        character.isspace() for character in value
    ):
        raise ValueError(f"{name} é obrigatório e não aceita espaços")
    return value


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

    snapshot = await graph.aget_state(config)
    persisted_values = snapshot.values
    if persisted_values:
        persisted = AgentState.model_validate(persisted_values)
        state = persisted.continue_with(
            request=request,
            identity=runtime.identity,
            permissions=runtime.permissions,
            request_id=request_id,
            execution_id=execution_id,
        )
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

    result = await graph.ainvoke(
        state.model_dump(mode="json"),
        config,
        durability="sync",
    )
    if not isinstance(result, Mapping):
        raise TypeError("o grafo devolveu estado inválido")
    return AgentState.model_validate(result)
