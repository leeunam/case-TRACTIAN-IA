"""Grafo determinístico mínimo da infraestrutura 5A."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import StateSnapshot

from tractian_agent.client import IndustrialApiClient
from tractian_agent.state import (
    AgentDecision,
    AgentState,
    FinalResult,
    MessageRole,
    PersistedMessage,
)
from tractian_agent.tools.runtime import ReadToolRuntime


MINIMAL_GRAPH_STEP_COUNT = 3


@dataclass
class _LockEntry:
    lock: asyncio.Lock
    users: int = 0


class _ThreadLockPool:
    """Locks locais efêmeros; não oferecem lease entre processos."""

    def __init__(self) -> None:
        self._entries: dict[str, _LockEntry] = {}
        self._registry_lock = asyncio.Lock()

    @asynccontextmanager
    async def hold(self, thread_id: str) -> AsyncIterator[None]:
        async with self._registry_lock:
            entry = self._entries.get(thread_id)
            if entry is None:
                entry = _LockEntry(lock=asyncio.Lock())
                self._entries[thread_id] = entry
            entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            async with self._registry_lock:
                entry.users -= 1
                if entry.users == 0:
                    del self._entries[thread_id]


class CompiledAgentGraph:
    """Grafo compilado e locks com o mesmo ciclo de vida local."""

    def __init__(self, graph: CompiledStateGraph) -> None:
        self._graph = graph
        self._thread_locks = _ThreadLockPool()

    def thread_lock(self, thread_id: str) -> AbstractAsyncContextManager[None]:
        return self._thread_locks.hold(thread_id)

    async def aget_state(self, config: RunnableConfig) -> StateSnapshot:
        return await self._graph.aget_state(config)

    async def ainvoke(
        self,
        input: object,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> object:
        return await self._graph.ainvoke(input, config, **kwargs)

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: object,
        **kwargs: Any,
    ) -> RunnableConfig:
        return await self._graph.aupdate_state(config, values, **kwargs)

    def get_graph(self) -> object:
        return self._graph.get_graph()


def _replace_state(state: AgentState, **changes: object) -> AgentState:
    data = {
        field_name: getattr(state, field_name)
        for field_name in type(state).model_fields
    }
    data.update(changes)
    return AgentState.model_validate(data)


def _checkpoint_update(state: AgentState) -> dict[str, object]:
    return state.model_dump(mode="json")


def _ingest(
    state: AgentState,
    runtime: Runtime[ReadToolRuntime],
) -> dict[str, object]:
    context = runtime.context
    if not isinstance(context, ReadToolRuntime) or not isinstance(
        context.client,
        IndustrialApiClient,
    ):
        raise TypeError("contexto confiável do grafo é obrigatório")
    if context.identity != state.identity or context.permissions != state.permissions:
        raise ValueError("contexto confiável diverge do estado persistido")
    if (
        state.request.asset_id is not None
        and context.central_asset_id != state.request.asset_id
    ):
        raise ValueError("ativo central do contexto diverge da solicitação")
    advanced = state.advance_step()
    updated = _replace_state(
        advanced,
        messages=(
            *advanced.messages,
            PersistedMessage(role=MessageRole.USER, content=advanced.request.message),
        ),
    )
    return _checkpoint_update(updated)


def _route(state: AgentState) -> dict[str, object]:
    advanced = state.advance_step()
    return _checkpoint_update(_replace_state(advanced, decision=AgentDecision.GUIDE))


def _finish(state: AgentState) -> dict[str, object]:
    advanced = state.advance_step()
    result = FinalResult(
        decision=AgentDecision.GUIDE,
        message="Fluxo determinístico de leitura concluído sem LLM.",
    )
    return _checkpoint_update(_replace_state(advanced, final_result=result))


def build_agent_graph(
    checkpointer: BaseCheckpointSaver[str],
) -> CompiledAgentGraph:
    """Compila o fluxo acíclico ``ingest → route → finish``."""
    builder = StateGraph(AgentState, context_schema=ReadToolRuntime)
    builder.add_node("ingest", _ingest)
    builder.add_node("route", _route)
    builder.add_node("finish", _finish)
    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "route")
    builder.add_edge("route", "finish")
    builder.add_edge("finish", END)
    return CompiledAgentGraph(builder.compile(checkpointer=checkpointer))
