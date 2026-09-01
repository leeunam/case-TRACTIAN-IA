"""Primeira fronteira isolada do planner, ainda fora do LangGraph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
import json
import re
from typing import Final, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import (
    ConfigDict,
    Field,
    ValidationError,
    TypeAdapter,
    field_validator,
    model_validator,
)

from tractian_agent.contracts import StrictModel, SupportRequest
from tractian_agent.state import (
    AgentState,
    PlannerUsage,
    PersistedSupportRequest,
    PersistedToolCall,
    ToolObservation,
)
from tractian_agent.tools import READ_TOOLS, WRITE_PROPOSAL_TOOLS
from tractian_agent.tools.identifiers import (
    AnalysisId,
    KnowledgeDocumentId,
    ModelId,
)
from tractian_agent.tools.runtime import ReadToolRuntime, WriteToolRuntime


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


class PlannerLimits(StrictModel):
    """Orçamento fixo da fatia isolada do planner."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tool_calls: Literal[7] = 7
    selections: Literal[8] = 8
    finalizations: Literal[1] = 1
    context_characters: Literal[48_000] = 48_000


PLANNER_LIMITS: Final = PlannerLimits()

_ANALYSIS_ID_ADAPTER: Final = TypeAdapter(AnalysisId)
_KNOWLEDGE_ID_ADAPTER: Final = TypeAdapter(KnowledgeDocumentId)
_MODEL_ID_ADAPTER: Final = TypeAdapter(ModelId)
_ANALYSIS_ID_PATTERN: Final = re.compile(
    r"(?<![A-Za-z0-9_-])an_[A-Za-z0-9_-]{1,64}(?![A-Za-z0-9_-])"
)
_KNOWLEDGE_ID_PATTERN: Final = re.compile(
    r"(?<![A-Za-z0-9_-])kb_[A-Za-z0-9_-]{1,64}(?![A-Za-z0-9_-])"
)
_MODEL_ID_PATTERN: Final = re.compile(
    r"(?<![A-Za-z0-9_-])mdl_[A-Za-z0-9_-]{1,64}(?![A-Za-z0-9_-])"
)
_ASSET_ARGUMENT_TOOL_NAMES: Final = frozenset(
    {
        "get_asset",
        "list_asset_analyses",
        "get_baseline",
        "get_rms_series",
        "get_spectrum",
        "get_data_quality",
    }
)


def _contains_typed_id(
    texts: Sequence[str],
    *,
    pattern: re.Pattern[str],
    adapter: TypeAdapter[str],
) -> bool:
    for text in texts:
        for candidate in pattern.findall(text):
            try:
                adapter.validate_python(candidate, strict=True)
            except ValidationError:
                continue
            return True
    return False


def _typed_ids(
    texts: Sequence[str],
    *,
    pattern: re.Pattern[str],
    adapter: TypeAdapter[str],
) -> frozenset[str]:
    validated: set[str] = set()
    for text in texts:
        for candidate in pattern.findall(text):
            try:
                validated.add(adapter.validate_python(candidate, strict=True))
            except ValidationError:
                continue
    return frozenset(validated)


def _json_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            text
            for nested in value.values()
            for text in _json_strings(nested)
        )
    if isinstance(value, list):
        return tuple(text for nested in value for text in _json_strings(nested))
    return ()


def _current_request_observation_texts(state: AgentState) -> tuple[str, ...]:
    calls_by_id = {
        call.call_id: call
        for call in state.tool_calls
        if call.request_id == state.request_id
    }
    texts: list[str] = []
    for observation in state.tool_observations:
        call = calls_by_id.get(observation.call_id)
        if (
            call is None
            or observation.request_id != state.request_id
            or observation.content is None
            or observation.artifact.tool_name != call.name
            or observation.artifact.arguments != call.arguments
        ):
            continue
        texts.extend(_json_strings(observation.content.to_python()))
    return tuple(texts)


def select_planner_tools(
    state: AgentState,
    runtime: ReadToolRuntime,
) -> tuple[BaseTool, ...]:
    """Seleciona um subconjunto dos catálogos estáticos sem executar efeitos."""
    runtime_scope_matches = (
        runtime.identity == state.identity
        and runtime.permissions == state.permissions
        and (
            state.request.asset_id is None
            or runtime.central_asset_id == state.request.asset_id
        )
        and (
            not isinstance(runtime, WriteToolRuntime)
            or runtime.current_case_id == state.request.case_id
        )
    )
    if not runtime_scope_matches:
        raise PlannerProtocolError(PlannerErrorCode.RUNTIME_SCOPE_MISMATCH)
    observable_texts = (
        state.request.message,
        *_current_request_observation_texts(state),
    )
    has_analysis_id = _contains_typed_id(
        observable_texts,
        pattern=_ANALYSIS_ID_PATTERN,
        adapter=_ANALYSIS_ID_ADAPTER,
    )
    has_knowledge_id = _contains_typed_id(
        observable_texts,
        pattern=_KNOWLEDGE_ID_PATTERN,
        adapter=_KNOWLEDGE_ID_ADAPTER,
    )
    has_scoped_asset = (
        state.request.asset_id is not None
        and state.request.asset_id == runtime.central_asset_id
    )
    offered_reads: tuple[BaseTool, ...] = ()
    if "read" in state.permissions and "read" in runtime.permissions:
        offered_reads = tuple(
            tool
            for tool in READ_TOOLS
            if (tool.name not in _ASSET_ARGUMENT_TOOL_NAMES or has_scoped_asset)
            and (tool.name != "get_analysis" or has_analysis_id)
            and (tool.name != "get_knowledge_document" or has_knowledge_id)
        )
    if not isinstance(runtime, WriteToolRuntime):
        return offered_reads

    observed_model_ids = _typed_ids(
        observable_texts,
        pattern=_MODEL_ID_PATTERN,
        adapter=_MODEL_ID_ADAPTER,
    )
    has_scoped_case = state.request.case_id == runtime.current_case_id
    proposal_requirements = {
        "propose_reprocess_analysis": (
            "action_low",
            has_analysis_id,
        ),
        "propose_request_specialist_analysis": (
            "action_low",
            has_analysis_id,
        ),
        "propose_update_asset_criticality": (
            "action_high",
            has_scoped_asset,
        ),
        "propose_request_model_retraining": (
            "action_high",
            runtime.configured_model_id in observed_model_ids,
        ),
        "propose_escalate_case": (
            "escalate",
            has_scoped_case,
        ),
    }
    offered_proposals = tuple(
        tool
        for tool in WRITE_PROPOSAL_TOOLS
        if (
            (requirement := proposal_requirements[tool.name])[0]
            in state.permissions
            and requirement[0] in runtime.permissions
            and requirement[1]
        )
    )
    return (*offered_reads, *offered_proposals)


class PlannerContextStats(StrictModel):
    """Medição da representação realmente entregue à fronteira do modelo."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    characters: int = Field(ge=0)
    omitted_interactions: int = Field(ge=0)


class PlannerToolTurn(StrictModel):
    """Uma única chamada validada, pronta para entrar no estado."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["tool_call"] = "tool_call"
    tool_call: PersistedToolCall
    usage: PlannerUsage | None = None
    context: PlannerContextStats | None = None


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
    INVALID_USAGE = "invalid_usage"
    SELECTION_LIMIT_EXCEEDED = "selection_limit_exceeded"
    TOOL_CALL_LIMIT_EXCEEDED = "tool_call_limit_exceeded"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    FINALIZATION_LIMIT_EXCEEDED = "finalization_limit_exceeded"
    CONTEXT_LIMIT_EXCEEDED = "context_limit_exceeded"
    RUNTIME_SCOPE_MISMATCH = "runtime_scope_mismatch"


class PlannerProtocolError(RuntimeError):
    """Falha fechada que não preserva a saída livre ou inválida do modelo."""

    def __init__(
        self,
        code: PlannerErrorCode,
        *,
        usage: PlannerUsage | None = None,
    ) -> None:
        self.code = code
        self.usage = usage
        super().__init__(f"planner protocol error: {code.value}")


def _context_character_count(
    messages: Sequence[BaseMessage],
    tool_wire: Sequence[dict[str, object]],
) -> int:
    return len(
        json.dumps(
            {
                "messages": convert_to_openai_messages(messages),
                "tools": tool_wire,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _build_planner_context(
    request_content: str,
    interactions: Sequence[tuple[AIMessage, ToolMessage, bool]],
    tools: Sequence[BaseTool],
) -> tuple[list[BaseMessage], PlannerContextStats]:
    interaction_count = len(interactions)
    tool_wire = tuple(convert_to_openai_tool(tool) for tool in tools)
    for omitted in range(interaction_count + 1):
        if any(interaction[2] for interaction in interactions[:omitted]):
            break
        marker = json.dumps(
            {"omitted_interactions": omitted},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        messages: list[BaseMessage] = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            SystemMessage(content=marker),
            HumanMessage(content=request_content),
        ]
        for assistant_message, tool_message, _ in interactions[omitted:]:
            messages.extend((assistant_message, tool_message))
        characters = _context_character_count(messages, tool_wire)
        if characters <= PLANNER_LIMITS.context_characters:
            return messages, PlannerContextStats(
                characters=characters,
                omitted_interactions=omitted,
            )
    raise PlannerProtocolError(PlannerErrorCode.CONTEXT_LIMIT_EXCEEDED)


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
    usage: PlannerUsage | None = None
    context: PlannerContextStats | None = None


class Planner:
    """Coordena as duas interfaces nativas do modelo sem executar tools."""

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    async def ainvoke(
        self,
        request: SupportRequest | PersistedSupportRequest,
        *,
        offered_tools: Sequence[BaseTool],
        request_id: str | None = None,
        usage: PlannerUsage | None = None,
        tool_calls: Sequence[PersistedToolCall] = (),
        tool_observations: Sequence[ToolObservation] = (),
    ) -> PlannerToolTurn | PlannerDecisionTurn:
        active_usage = usage
        if active_usage is not None and active_usage.request_id != request_id:
            raise PlannerProtocolError(PlannerErrorCode.INVALID_USAGE)
        if (
            active_usage is not None
            and (
                active_usage.selection_count > PLANNER_LIMITS.selections
                or active_usage.finalization_count > PLANNER_LIMITS.finalizations
            )
        ):
            raise PlannerProtocolError(PlannerErrorCode.INVALID_USAGE)
        if (
            active_usage is not None
            and active_usage.finalization_count == PLANNER_LIMITS.finalizations
        ):
            raise PlannerProtocolError(PlannerErrorCode.FINALIZATION_LIMIT_EXCEEDED)
        if (
            active_usage is not None
            and active_usage.selection_count == PLANNER_LIMITS.selections
        ):
            raise PlannerProtocolError(PlannerErrorCode.SELECTION_LIMIT_EXCEEDED)
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
        calls = tuple(
            call for call in tool_calls if call.request_id == request_id
        )
        observations = tuple(
            observation
            for observation in tool_observations
            if observation.request_id == request_id
        )
        if len(calls) != len(observations):
            raise PlannerProtocolError(PlannerErrorCode.INVALID_HISTORY)
        call_ids = tuple(call.call_id for call in calls)
        observation_ids = tuple(observation.call_id for observation in observations)
        if (
            len(call_ids) != len(set(call_ids))
            or len(observation_ids) != len(set(observation_ids))
        ):
            raise PlannerProtocolError(PlannerErrorCode.INVALID_HISTORY)
        if len(calls) > PLANNER_LIMITS.tool_calls:
            raise PlannerProtocolError(PlannerErrorCode.INVALID_HISTORY)
        if len(calls) == PLANNER_LIMITS.tool_calls:
            raise PlannerProtocolError(PlannerErrorCode.TOOL_CALL_LIMIT_EXCEEDED)
        interactions: list[tuple[AIMessage, ToolMessage, bool]] = []
        for index, (call, observation) in enumerate(
            zip(calls, observations, strict=True)
        ):
            if (
                observation.call_id != call.call_id
                or observation.content is None
                or observation.artifact.tool_name != call.name
                or observation.artifact.arguments != call.arguments
            ):
                raise PlannerProtocolError(PlannerErrorCode.INVALID_HISTORY)
            interactions.append(
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
                    (
                        index == len(calls) - 1
                        or observation.artifact.outcome.error is not None
                        or (
                            observation.artifact.outcome.mode is not None
                            and observation.artifact.outcome.mode.value != "complete"
                        )
                    ),
                )
            )
        messages, context_stats = _build_planner_context(
            request_content,
            interactions,
            tools,
        )
        selection = await self._model.bind_tools(tools).ainvoke(messages)
        selection_usage = (
            None
            if active_usage is None
            else PlannerUsage(
                request_id=active_usage.request_id,
                selection_count=active_usage.selection_count + 1,
                finalization_count=active_usage.finalization_count,
            )
        )
        if not isinstance(selection, AIMessage):
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_SELECTION,
                usage=selection_usage,
            )
        if selection.invalid_tool_calls:
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_TOOL_ARGUMENTS,
                usage=selection_usage,
            )
        if not selection.tool_calls:
            final_usage = (
                None
                if selection_usage is None
                else PlannerUsage(
                    request_id=selection_usage.request_id,
                    selection_count=selection_usage.selection_count,
                    finalization_count=selection_usage.finalization_count + 1,
                )
            )
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
                    PlannerErrorCode.INVALID_TERMINAL_OUTPUT,
                    usage=final_usage,
                )
            return PlannerDecisionTurn(
                decision=terminal_decision,
                usage=final_usage,
                context=context_stats,
            )
        if len(selection.tool_calls) != 1:
            raise PlannerProtocolError(
                PlannerErrorCode.MULTIPLE_TOOL_CALLS,
                usage=selection_usage,
            )
        selected_call = selection.tool_calls[0]
        if selected_call["id"] in set(call_ids):
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_SELECTION,
                usage=selection_usage,
            )
        selected_tool = tools_by_name.get(selected_call["name"])
        if selected_tool is None:
            raise PlannerProtocolError(
                PlannerErrorCode.UNKNOWN_TOOL,
                usage=selection_usage,
            )
        tool_turn: PlannerToolTurn | None = None
        try:
            validated_arguments = selected_tool.tool_call_schema.model_validate(
                selected_call["args"]
            )
            tool_turn = PlannerToolTurn(
                tool_call=PersistedToolCall(
                    request_id=request_id,
                    call_id=selected_call["id"],
                    name=selected_call["name"],
                    arguments=validated_arguments.model_dump(mode="json"),
                ),
                usage=selection_usage,
                context=context_stats,
            )
        except (TypeError, ValueError, ValidationError):
            pass
        if tool_turn is None:
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_TOOL_ARGUMENTS,
                usage=selection_usage,
            )
        selected_fingerprint = _tool_call_fingerprint(tool_turn.tool_call)
        if any(
            _tool_call_fingerprint(prior_call) == selected_fingerprint
            for prior_call in calls
        ):
            raise PlannerProtocolError(
                PlannerErrorCode.REPEATED_TOOL_CALL,
                usage=selection_usage,
            )
        return tool_turn


def _tool_call_fingerprint(call: PersistedToolCall) -> str:
    """Identifica intenção da tool sem depender do ID atribuído pelo provider."""
    return json.dumps(
        {
            "arguments": call.arguments.to_python(),
            "tool": call.name,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
