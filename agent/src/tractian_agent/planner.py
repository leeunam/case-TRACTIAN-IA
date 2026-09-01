"""Primeira fronteira isolada do planner, ainda fora do LangGraph."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
import json
from typing import Final, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import (
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from tractian_agent.contracts import StrictModel, SupportRequest
from tractian_agent.state import (
    PersistedSupportRequest,
    PersistedToolCall,
    ToolObservation,
)


PLANNER_SYSTEM_PROMPT_VERSION: Final = "planner-v1"
PLANNER_SYSTEM_PROMPT: Final = f"""\
prompt_version: {PLANNER_SYSTEM_PROMPT_VERSION}

Você é o planner do atendimento industrial, separado do writer. Sua função é
escolher a próxima tool oferecida ou encerrar com uma decisão estruturada; não
redija a resposta destinada ao cliente.

Use no máximo uma tool por turno e somente uma tool explicitamente oferecida.
Não invente evidência nem transforme hipótese em fato. Proposal tools apenas
registram propostas e não executam efeito industrial.

Não revele nem devolva raciocínio interno. Produza somente a chamada de tool
solicitada pela etapa de seleção ou os campos do schema da etapa terminal.
"""


class PlannerToolTurn(StrictModel):
    """Uma única chamada validada, pronta para entrar no estado."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["tool_call"] = "tool_call"
    tool_call: PersistedToolCall


class PlannerDecisionKind(str, Enum):
    GUIDE = "guide"
    REQUEST_INFORMATION = "request_information"
    REQUIRE_HUMAN_REVIEW = "require_human_review"


class PlannerStopReason(str, Enum):
    SUFFICIENT_EVIDENCE = "sufficient_evidence"
    MISSING_INFORMATION = "missing_information"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


class PlannerErrorCode(str, Enum):
    INVALID_SELECTION = "invalid_selection"
    INVALID_HISTORY = "invalid_history"
    DUPLICATE_TOOL_NAME = "duplicate_tool_name"
    UNKNOWN_TOOL = "unknown_tool"
    MULTIPLE_TOOL_CALLS = "multiple_tool_calls"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    INVALID_TERMINAL_OUTPUT = "invalid_terminal_output"


class PlannerProtocolError(RuntimeError):
    """Falha fechada que não preserva a saída livre ou inválida do modelo."""

    def __init__(self, code: PlannerErrorCode) -> None:
        self.code = code
        super().__init__(f"planner protocol error: {code.value}")


class PlannerTerminalDecision(StrictModel):
    """Decisão do planner; não contém texto final destinado ao cliente."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    decision: PlannerDecisionKind
    stop_reason: PlannerStopReason
    missing_information: str | None = Field(default=None, max_length=300)

    @field_validator("missing_information", mode="before")
    @classmethod
    def _normalize_missing_information(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("missing_information deve ser texto ou null")
        normalized = value.strip()
        if not normalized:
            raise ValueError("missing_information não pode ser vazio")
        return normalized

    @model_validator(mode="after")
    def _require_coherent_stop_contract(self) -> PlannerTerminalDecision:
        expected_reason = {
            PlannerDecisionKind.GUIDE: PlannerStopReason.SUFFICIENT_EVIDENCE,
            PlannerDecisionKind.REQUEST_INFORMATION: (
                PlannerStopReason.MISSING_INFORMATION
            ),
            PlannerDecisionKind.REQUIRE_HUMAN_REVIEW: (
                PlannerStopReason.HUMAN_REVIEW_REQUIRED
            ),
        }[self.decision]
        if self.stop_reason is not expected_reason:
            raise ValueError("stop_reason diverge da decisão terminal")
        requires_information = (
            self.decision is PlannerDecisionKind.REQUEST_INFORMATION
        )
        if requires_information != (self.missing_information is not None):
            raise ValueError("missing_information diverge da decisão terminal")
        return self


class PlannerDecisionTurn(StrictModel):
    """Encerramento validado da fatia do planner."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["decision"] = "decision"
    decision: PlannerTerminalDecision


class Planner:
    """Coordena as duas interfaces nativas do modelo sem executar tools."""

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    async def ainvoke(
        self,
        request: SupportRequest | PersistedSupportRequest,
        *,
        offered_tools: Sequence[BaseTool],
        tool_calls: Sequence[PersistedToolCall] = (),
        tool_observations: Sequence[ToolObservation] = (),
    ) -> PlannerToolTurn | PlannerDecisionTurn:
        tools = tuple(offered_tools)
        tool_names = tuple(tool.name for tool in tools)
        if len(tool_names) != len(set(tool_names)):
            raise PlannerProtocolError(PlannerErrorCode.DUPLICATE_TOOL_NAME)
        tools_by_name = {tool.name: tool for tool in tools}
        request_content = json.dumps(
            {
                "case_id": request.case_id,
                "ticket_id": request.ticket_id,
                "asset_id": request.asset_id,
                "message": request.message,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=request_content),
        ]
        calls = tuple(tool_calls)
        observations = tuple(tool_observations)
        if len(calls) != len(observations):
            raise PlannerProtocolError(PlannerErrorCode.INVALID_HISTORY)
        for call, observation in zip(calls, observations, strict=True):
            if observation.call_id != call.call_id or observation.content is None:
                raise PlannerProtocolError(PlannerErrorCode.INVALID_HISTORY)
            messages.extend(
                (
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": call.name,
                                "args": call.arguments.to_python(),
                                "id": call.call_id,
                                "type": "tool_call",
                            }
                        ],
                    ),
                    ToolMessage(
                        content=observation.content.encoded,
                        tool_call_id=call.call_id,
                        name=call.name,
                    ),
                )
            )
        selection = await self._model.bind_tools(tools).ainvoke(messages)
        if not isinstance(selection, AIMessage):
            raise PlannerProtocolError(PlannerErrorCode.INVALID_SELECTION)
        if selection.invalid_tool_calls:
            raise PlannerProtocolError(PlannerErrorCode.INVALID_TOOL_ARGUMENTS)
        if not selection.tool_calls:
            terminal_decision: PlannerTerminalDecision | None = None
            try:
                terminal_output = await self._model.with_structured_output(
                    PlannerTerminalDecision,
                    include_raw=False,
                ).ainvoke(messages)
                terminal_decision = PlannerTerminalDecision.model_validate(
                    terminal_output
                )
            except (TypeError, ValueError, ValidationError):
                pass
            if terminal_decision is None:
                raise PlannerProtocolError(
                    PlannerErrorCode.INVALID_TERMINAL_OUTPUT
                )
            return PlannerDecisionTurn(decision=terminal_decision)
        if len(selection.tool_calls) != 1:
            raise PlannerProtocolError(PlannerErrorCode.MULTIPLE_TOOL_CALLS)
        selected_call = selection.tool_calls[0]
        selected_tool = tools_by_name.get(selected_call["name"])
        if selected_tool is None:
            raise PlannerProtocolError(PlannerErrorCode.UNKNOWN_TOOL)
        tool_turn: PlannerToolTurn | None = None
        try:
            validated_arguments = selected_tool.tool_call_schema.model_validate(
                selected_call["args"]
            )
            tool_turn = PlannerToolTurn(
                tool_call=PersistedToolCall(
                    call_id=selected_call["id"],
                    name=selected_call["name"],
                    arguments=validated_arguments.model_dump(mode="json"),
                )
            )
        except (TypeError, ValueError, ValidationError):
            pass
        if tool_turn is None:
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_TOOL_ARGUMENTS
            )
        return tool_turn
