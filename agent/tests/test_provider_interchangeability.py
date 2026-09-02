from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

from tractian_agent.checkpoint import open_checkpointer
from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import Identity, SupportRequest
from tractian_agent.entrypoint import invoke_agent
from tractian_agent.graph import build_agent_graph
from tractian_agent.model_provider import ModelConfig, ModelProvider
from tractian_agent.planner import (
    Planner,
    PlannerDecisionKind,
    PlannerStopReason,
    PlannerTerminalDecision,
)
from tractian_agent.state import AgentDecision, AgentState
from tractian_agent.tools.runtime import ReadToolRuntime, WriteToolRuntime
from tractian_agent.write_policy import PolicyDecision, PolicyReason


class _ProviderScriptedModel(BaseChatModel):
    external_call_id: str
    scenario: str
    _catalogs: list[tuple[str, ...]] = PrivateAttr(default_factory=list)
    _schemas: list[tuple[dict[str, object], ...]] = PrivateAttr(default_factory=list)
    _terminal_schemas: list[dict[str, object]] = PrivateAttr(default_factory=list)
    _selection_count: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "provider-interchangeability-test"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise AssertionError("o planner deve usar bind_tools")

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> RunnableLambda:
        self._catalogs.append(tuple(tool.name for tool in tools))
        self._schemas.append(
            tuple(tool.tool_call_schema.model_json_schema() for tool in tools)
        )

        async def select(_: list[BaseMessage]) -> AIMessage:
            self._selection_count += 1
            if self.scenario == "read" and self._selection_count > 1:
                return AIMessage(content="")
            if self.scenario == "read":
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_asset",
                            "args": {"asset_id": "asset_G501"},
                            "id": self.external_call_id,
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_update_asset_criticality",
                        "args": {
                            "criticality": "critical",
                            "justification": (
                                "O impacto operacional exige prioridade máxima."
                            ),
                        },
                        "id": self.external_call_id,
                        "type": "tool_call",
                    }
                ],
            )

        return RunnableLambda(select)

    def with_structured_output(self, schema: object, **kwargs: Any) -> RunnableLambda:
        assert schema is PlannerTerminalDecision
        assert kwargs == {"include_raw": False}
        self._terminal_schemas.append(schema.model_json_schema())

        async def finalize(_: list[BaseMessage]) -> PlannerTerminalDecision:
            return PlannerTerminalDecision(
                decision=PlannerDecisionKind.GUIDE,
                stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
            )

        return RunnableLambda(finalize)


class _FakeModelProvider:
    def __init__(self, external_call_id: str, *, scenario: str = "proposal") -> None:
        self._external_call_id = external_call_id
        self._scenario = scenario
        self.received_config: ModelConfig | None = None
        self.model: _ProviderScriptedModel | None = None

    def create_chat_model(self, config: ModelConfig) -> BaseChatModel:
        self.received_config = config
        self.model = _ProviderScriptedModel(
            external_call_id=self._external_call_id,
            scenario=self._scenario,
        )
        return self.model


def _request() -> SupportRequest:
    return SupportRequest(
        case_id="case_tkt_inv_04",
        ticket_id="TKT-INV-04",
        asset_id="asset_G501",
        message="Atualize a criticidade do ativo central.",
        identity=Identity(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
        ),
    )


def test_model_provider_swap_preserves_public_planner_contract_without_http(tmp_path):
    config = ModelConfig(
        model_id="fake/planner",
        temperature=0.0,
        timeout_seconds=1.0,
        max_output_tokens=32,
    )
    first: ModelProvider = _FakeModelProvider("provider-first-call")
    second: ModelProvider = _FakeModelProvider("provider-second-call")

    def forbidden_http(_: httpx.Request) -> httpx.Response:
        raise AssertionError("proposal sem confirmação não pode alcançar HTTP")

    async def run(provider: ModelProvider, database_name: str):
        model = provider.create_chat_model(config)
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
            async with open_checkpointer(tmp_path / database_name) as saver:
                state = await invoke_agent(
                    build_agent_graph(saver, planner=Planner(model)),
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_provider_swap",
                    request_id="req_provider_swap",
                    execution_id="exec_provider_swap",
                )
            assert isinstance(model, _ProviderScriptedModel)
            return state, model
        finally:
            await client.aclose()

    first_state, first_model = asyncio.run(run(first, "first.sqlite3"))
    second_state, second_model = asyncio.run(run(second, "second.sqlite3"))

    assert isinstance(first, _FakeModelProvider)
    assert isinstance(second, _FakeModelProvider)
    assert first.received_config is config
    assert second.received_config is config
    assert first_model._catalogs == second_model._catalogs
    assert first_model._schemas == second_model._schemas
    assert first_model._catalogs == [("propose_update_asset_criticality",)]
    for state in (first_state, second_state):
        assert state.decision is AgentDecision.REQUEST_CONFIRMATION
        assert state.final_result is None
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].name == "propose_update_asset_criticality"
        assert state.tool_calls[0].arguments.to_python() == {
            "criticality": "critical",
            "justification": "O impacto operacional exige prioridade máxima.",
        }
        assert len(state.intents) == 1
        assert state.intents[0].decision.decision is PolicyDecision.REQUIRE_CONFIRMATION
        assert state.intents[0].decision.reason is PolicyReason.EXPLICIT_APPROVAL_REQUIRED

    assert first_state.tool_calls == second_state.tool_calls
    assert first_state.intents[0].decision == second_state.intents[0].decision
    persisted = first_state.model_dump_json()
    assert "provider-first-call" not in persisted
    assert "provider-second-call" not in persisted


def test_provider_swap_preserves_read_catalog_and_pydantic_terminal_schema(tmp_path):
    config = ModelConfig(
        model_id="fake/planner",
        temperature=0.0,
        timeout_seconds=1.0,
        max_output_tokens=32,
    )
    first: ModelProvider = _FakeModelProvider("provider-first-read", scenario="read")
    second: ModelProvider = _FakeModelProvider("provider-second-read", scenario="read")

    def asset_response(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mode": "complete",
                "notes": None,
                "data": {
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
                    "points": [],
                },
            },
        )

    async def run(provider: ModelProvider, database_name: str):
        model = provider.create_chat_model(config)
        client = IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(asset_response),
        )
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=client,
        )
        try:
            async with open_checkpointer(tmp_path / database_name) as saver:
                state = await invoke_agent(
                    build_agent_graph(saver, planner=Planner(model)),
                    request=_request(),
                    runtime=runtime,
                    thread_id="thread_provider_read",
                    request_id="req_provider_read",
                    execution_id="exec_provider_read",
                )
            assert isinstance(model, _ProviderScriptedModel)
            return state, model
        finally:
            await client.aclose()

    first_state, first_model = asyncio.run(run(first, "first-read.sqlite3"))
    second_state, second_model = asyncio.run(run(second, "second-read.sqlite3"))

    assert first_model._catalogs == second_model._catalogs
    assert first_model._schemas == second_model._schemas
    assert first_model._catalogs[0][0] == "get_asset"
    assert first_model._terminal_schemas == second_model._terminal_schemas
    assert first_model._terminal_schemas == [PlannerTerminalDecision.model_json_schema()]
    assert first_state.decision is AgentDecision.GUIDE
    assert second_state.decision is AgentDecision.GUIDE
    assert first_state.tool_calls == second_state.tool_calls
    assert first_state.tool_calls[0].arguments.to_python() == {"asset_id": "asset_G501"}
    assert first_state.planner_terminal == second_state.planner_terminal
    assert "provider-first-read" not in first_state.model_dump_json()
    assert "provider-second-read" not in second_state.model_dump_json()
