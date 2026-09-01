import asyncio
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import LATEST_VERSION
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
import pytest
from pydantic import PrivateAttr, ValidationError

from tractian_agent.checkpoint import open_checkpointer
from tractian_agent.contracts import Identity, SupportRequest
from tractian_agent.planner import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT_VERSION,
    Planner,
    PlannerDecisionKind,
    PlannerDecisionTurn,
    PlannerErrorCode,
    PlannerProtocolError,
    PlannerStopReason,
    PlannerTerminalDecision,
    PlannerToolTurn,
)
from tractian_agent.state import (
    AgentState,
    PersistedToolCall,
    ThreadScope,
    ToolObservation,
)
from tractian_agent.tools.assets import get_asset
from tractian_agent.tools.observations import ToolArtifact, ToolOutcome, ToolSource
from tractian_agent.tools.runtime import TrustedIdentity


class _RecordingPlannerModel(BaseChatModel):
    selector_response: AIMessage
    terminal_response: object | None = None
    _events: list[str] = PrivateAttr(default_factory=list)
    _bound_tool_names: tuple[str, ...] = PrivateAttr(default=())
    _selection_messages: list[BaseMessage] = PrivateAttr(default_factory=list)
    _terminal_messages: list[BaseMessage] = PrivateAttr(default_factory=list)
    _structured_schema: type | dict[str, Any] | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "recording-planner-model"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise AssertionError("o planner deve usar somente os wrappers públicos")

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> RunnableLambda:
        self._events.append("bind_tools")
        self._bound_tool_names = tuple(tool.name for tool in tools)

        async def select(messages: list[BaseMessage]) -> AIMessage:
            self._events.append("selection_request")
            self._selection_messages = list(messages)
            return self.selector_response

        return RunnableLambda(select)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> RunnableLambda:
        self._events.append("with_structured_output")
        self._structured_schema = schema

        async def finalize(messages: list[BaseMessage]) -> object:
            self._events.append("terminal_request")
            self._terminal_messages = list(messages)
            return self.terminal_response

        return RunnableLambda(finalize)


def _request() -> SupportRequest:
    return SupportRequest(
        case_id="case_tkt_inv_04",
        ticket_id="TKT-INV-04",
        asset_id="asset_G501",
        message="Consulte o cadastro técnico deste ativo.",
        identity=Identity(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
        ),
    )


def test_planner_system_prompt_has_a_versioned_safe_role():
    normalized_prompt = PLANNER_SYSTEM_PROMPT.casefold()

    assert PLANNER_SYSTEM_PROMPT_VERSION == "planner-v1"
    assert PLANNER_SYSTEM_PROMPT_VERSION in PLANNER_SYSTEM_PROMPT
    assert "writer" in normalized_prompt
    assert "no máximo uma tool" in normalized_prompt
    assert "não invente evidência" in normalized_prompt
    assert "não executam efeito" in normalized_prompt
    assert "raciocínio interno" in normalized_prompt


def test_planner_selects_one_offered_tool_without_executing_it():
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_asset",
                    "args": {"asset_id": "asset_G501"},
                    "id": "call_get_asset_01",
                    "type": "tool_call",
                }
            ],
        )
    )

    result = asyncio.run(
        Planner(model).ainvoke(
            _request(),
            offered_tools=(get_asset,),
        )
    )

    assert isinstance(result, PlannerToolTurn)
    assert result.tool_call.call_id == "call_get_asset_01"
    assert result.tool_call.name == "get_asset"
    assert result.tool_call.arguments.to_python() == {"asset_id": "asset_G501"}
    assert PlannerToolTurn.model_validate_json(result.model_dump_json()) == result
    assert model._bound_tool_names == ("get_asset",)
    assert model._events == ["bind_tools", "selection_request"]
    assert isinstance(model._selection_messages[0], SystemMessage)
    assert model._selection_messages[0].content == PLANNER_SYSTEM_PROMPT


def test_planner_discards_selector_text_and_uses_a_separate_terminal_request():
    selector_decoy = "Resposta livre que não pode virar decisão nem resposta."
    terminal = PlannerTerminalDecision(
        decision=PlannerDecisionKind.GUIDE,
        stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
    )
    model = _RecordingPlannerModel(
        selector_response=AIMessage(content=selector_decoy),
        terminal_response=terminal,
    )

    result = asyncio.run(
        Planner(model).ainvoke(
            _request(),
            offered_tools=(get_asset,),
        )
    )

    assert isinstance(result, PlannerDecisionTurn)
    assert result.decision == terminal
    assert model._events == [
        "bind_tools",
        "selection_request",
        "with_structured_output",
        "terminal_request",
    ]
    assert model._structured_schema is PlannerTerminalDecision
    assert all(
        selector_decoy not in str(message.content)
        for message in model._terminal_messages
    )
    assert selector_decoy not in result.model_dump_json()
    assert PlannerDecisionTurn.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    ("decision", "stop_reason", "missing_information"),
    [
        (
            PlannerDecisionKind.GUIDE,
            PlannerStopReason.SUFFICIENT_EVIDENCE,
            None,
        ),
        (
            PlannerDecisionKind.REQUEST_INFORMATION,
            PlannerStopReason.MISSING_INFORMATION,
            "Informe o ponto de medição que deve ser investigado.",
        ),
        (
            PlannerDecisionKind.REQUIRE_HUMAN_REVIEW,
            PlannerStopReason.HUMAN_REVIEW_REQUIRED,
            None,
        ),
    ],
)
def test_terminal_decision_accepts_only_coherent_stop_contracts(
    decision,
    stop_reason,
    missing_information,
):
    terminal = PlannerTerminalDecision(
        decision=decision,
        stop_reason=stop_reason,
        missing_information=missing_information,
    )

    assert terminal.decision is decision
    assert terminal.stop_reason is stop_reason
    assert terminal.missing_information == missing_information


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {"decision": "act", "stop_reason": "sufficient_evidence"},
        {"decision": "escalate", "stop_reason": "human_review_required"},
        {
            "decision": "request_confirmation",
            "stop_reason": "missing_information",
            "missing_information": "Confirme a ação.",
        },
        {
            "decision": PlannerDecisionKind.GUIDE,
            "stop_reason": PlannerStopReason.MISSING_INFORMATION,
            "missing_information": "Informe o ativo.",
        },
        {
            "decision": PlannerDecisionKind.REQUEST_INFORMATION,
            "stop_reason": PlannerStopReason.MISSING_INFORMATION,
        },
        {
            "decision": PlannerDecisionKind.REQUIRE_HUMAN_REVIEW,
            "stop_reason": PlannerStopReason.HUMAN_REVIEW_REQUIRED,
            "missing_information": "Não deve acompanhar revisão humana.",
        },
        {
            "decision": PlannerDecisionKind.REQUEST_INFORMATION,
            "stop_reason": PlannerStopReason.MISSING_INFORMATION,
            "missing_information": "x" * 301,
        },
        {
            "decision": PlannerDecisionKind.GUIDE,
            "stop_reason": PlannerStopReason.SUFFICIENT_EVIDENCE,
            "response": "Texto do writer não pertence ao planner.",
        },
    ],
)
def test_terminal_decision_rejects_unsafe_or_incoherent_contracts(invalid_payload):
    with pytest.raises(ValidationError):
        PlannerTerminalDecision.model_validate(invalid_payload)


@pytest.mark.parametrize(
    ("selector_response", "expected_code"),
    [
        (
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "unknown_tool",
                        "args": {"asset_id": "asset_G501"},
                        "id": "call_unknown_01",
                        "type": "tool_call",
                    }
                ],
            ),
            PlannerErrorCode.UNKNOWN_TOOL,
        ),
        (
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_asset",
                        "args": {"asset_id": "asset_G501"},
                        "id": "call_first",
                        "type": "tool_call",
                    },
                    {
                        "name": "get_asset",
                        "args": {"asset_id": "asset_G502"},
                        "id": "call_second",
                        "type": "tool_call",
                    },
                ],
            ),
            PlannerErrorCode.MULTIPLE_TOOL_CALLS,
        ),
        (
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_asset",
                        "args": {"asset_id": "outside-public-schema"},
                        "id": "call_invalid_arguments",
                        "type": "tool_call",
                    }
                ],
            ),
            PlannerErrorCode.INVALID_TOOL_ARGUMENTS,
        ),
        (
            AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "name": "get_asset",
                        "args": "not-json",
                        "id": "call_malformed_arguments",
                        "error": "arguments are not valid JSON",
                        "type": "invalid_tool_call",
                    }
                ],
            ),
            PlannerErrorCode.INVALID_TOOL_ARGUMENTS,
        ),
    ],
)
def test_planner_fails_closed_for_unsafe_tool_selections(
    selector_response,
    expected_code,
):
    model = _RecordingPlannerModel(selector_response=selector_response)

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                offered_tools=(get_asset,),
            )
        )

    assert exc_info.value.code is expected_code
    assert model._events == ["bind_tools", "selection_request"]


def test_planner_fails_closed_for_invalid_terminal_output():
    model = _RecordingPlannerModel(
        selector_response=AIMessage(content="texto livre descartado"),
        terminal_response={
            "decision": "act",
            "stop_reason": "sufficient_evidence",
            "response": "não deve chegar ao cliente",
        },
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                offered_tools=(get_asset,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_TERMINAL_OUTPUT
    error_text = str(exc_info.value)
    assert "texto livre descartado" not in error_text
    assert "não deve chegar ao cliente" not in error_text


def test_planner_uses_only_persisted_next_turn_content_after_a_tool_call():
    call = PersistedToolCall(
        call_id="call_get_asset_01",
        name="get_asset",
        arguments={"asset_id": "asset_G501"},
    )
    observation = ToolObservation(
        call_id=call.call_id,
        content={"id": "asset_G501", "sensor_status": "online"},
        artifact=ToolArtifact(
            tool_name="get_asset",
            arguments={"asset_id": "asset_G501"},
            source=ToolSource(
                kind="industrial_api",
                resource="/assets/asset_G501",
            ),
            outcome=ToolOutcome(
                partial_data={"technical_detail": "artifact-only"}
            ),
        ),
    )
    terminal = PlannerTerminalDecision(
        decision=PlannerDecisionKind.GUIDE,
        stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
    )
    model = _RecordingPlannerModel(
        selector_response=AIMessage(content="não persistir"),
        terminal_response=terminal,
    )

    result = asyncio.run(
        Planner(model).ainvoke(
            _request(),
            offered_tools=(get_asset,),
            tool_calls=(call,),
            tool_observations=(observation,),
        )
    )

    assert isinstance(result, PlannerDecisionTurn)
    selection_tool_message = model._selection_messages[-1]
    assert isinstance(selection_tool_message, ToolMessage)
    assert selection_tool_message.tool_call_id == call.call_id
    assert selection_tool_message.content == (
        '{"id":"asset_G501","sensor_status":"online"}'
    )
    assert "artifact-only" not in str(model._selection_messages)
    assert model._terminal_messages == model._selection_messages


def test_planner_tool_cycle_survives_sqlite_close_and_reopen(tmp_path: Path):
    checkpoint_path = tmp_path / "planner-cycle.sqlite3"
    request = _request()
    call = PersistedToolCall(
        call_id="call_get_asset_01",
        name="get_asset",
        arguments={"asset_id": "asset_G501"},
    )
    observation = ToolObservation(
        call_id=call.call_id,
        content={"id": "asset_G501", "sensor_status": "online"},
        artifact=ToolArtifact(
            tool_name="get_asset",
            arguments={"asset_id": "asset_G501"},
            source=ToolSource(
                kind="industrial_api",
                resource="/assets/asset_G501",
            ),
            outcome=ToolOutcome(
                partial_data={"id": "asset_G501", "sensor_status": "online"}
            ),
        ),
    )
    state = AgentState(
        request=request,
        identity=TrustedIdentity(
            user_id=request.identity.user_id,
            company_id=request.identity.company_id,
        ),
        permissions=frozenset({"read"}),
        request_id="req_planner_01",
        thread_id="thread_planner_01",
        execution_id="exec_planner_01",
        thread_scope=ThreadScope(
            thread_id="thread_planner_01",
            case_id=request.case_id,
            company_id=request.identity.company_id,
            user_id=request.identity.user_id,
        ),
        tool_calls=(call,),
        tool_observations=(observation,),
        step_limit=3,
    )
    serialized_state = state.model_dump(mode="json")
    json.dumps(serialized_state, allow_nan=False)
    config = {
        "configurable": {
            "thread_id": state.thread_id,
            "checkpoint_ns": "",
        }
    }
    checkpoint = {
        "v": LATEST_VERSION,
        "id": "00000000-0000-6000-8000-000000000010",
        "ts": "2026-09-01T12:00:00+00:00",
        "channel_values": {"agent_state": serialized_state},
        "channel_versions": {"agent_state": "planner-v1"},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": ["agent_state"],
    }

    async def close_and_reopen():
        async with open_checkpointer(checkpoint_path) as saver:
            await saver.aput(
                config,
                checkpoint,
                {"source": "update", "step": 0, "parents": {}},
                {"agent_state": "planner-v1"},
            )
        async with open_checkpointer(checkpoint_path) as reopened_saver:
            return await reopened_saver.aget(config)

    restored_checkpoint = asyncio.run(close_and_reopen())

    assert restored_checkpoint is not None
    restored_state = AgentState.model_validate(
        restored_checkpoint["channel_values"]["agent_state"]
    )
    assert restored_state.tool_calls == (call,)
    assert restored_state.tool_observations == (observation,)
    assert restored_state.tool_observations[0].content is not None
    assert restored_state.tool_observations[0].content.to_python() == {
        "id": "asset_G501",
        "sensor_status": "online",
    }
