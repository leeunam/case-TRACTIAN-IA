"""Grafo determinístico mínimo da infraestrutura 5A."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from tractian_agent.state import (
    AgentDecision,
    AgentState,
    FinalResult,
    MessageRole,
    PersistedMessage,
)


def _replace_state(state: AgentState, **changes: object) -> AgentState:
    data = {
        field_name: getattr(state, field_name)
        for field_name in type(state).model_fields
    }
    data.update(changes)
    return AgentState.model_validate(data)


def _checkpoint_update(state: AgentState) -> dict[str, object]:
    return state.model_dump(mode="json")


def _ingest(state: AgentState) -> dict[str, object]:
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
) -> CompiledStateGraph:
    """Compila o fluxo acíclico ``ingest → route → finish``."""
    builder = StateGraph(AgentState)
    builder.add_node("ingest", _ingest)
    builder.add_node("route", _route)
    builder.add_node("finish", _finish)
    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "route")
    builder.add_edge("route", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)
