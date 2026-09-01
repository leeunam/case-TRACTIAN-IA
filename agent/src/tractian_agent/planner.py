"""Primeira fronteira isolada do planner, ainda fora do LangGraph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
import re
from types import MappingProxyType
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
    PointId,
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
_POINT_ID_ADAPTER: Final = TypeAdapter(PointId)
_ANALYSIS_ID_PATTERN: Final = re.compile(
    r"(?<![A-Za-z0-9_-])an_[A-Za-z0-9_-]{1,64}(?![A-Za-z0-9_-])"
)
_KNOWLEDGE_ID_PATTERN: Final = re.compile(
    r"(?<![A-Za-z0-9_-])kb_[A-Za-z0-9_-]{1,64}(?![A-Za-z0-9_-])"
)
_MODEL_ID_PATTERN: Final = re.compile(
    r"(?<![A-Za-z0-9_-])mdl_[A-Za-z0-9_-]{1,64}(?![A-Za-z0-9_-])"
)
_POINT_ID_PATTERN: Final = re.compile(
    r"(?<![A-Za-z0-9_-])pt_[A-Za-z0-9_-]{1,64}(?![A-Za-z0-9_-])"
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
_POINT_ARGUMENT_TOOL_NAMES: Final = frozenset(
    {"get_baseline", "get_rms_series", "get_spectrum", "get_data_quality"}
)
_ANALYSIS_ARGUMENT_TOOL_NAMES: Final = frozenset(
    {
        "get_analysis",
        "propose_reprocess_analysis",
        "propose_request_specialist_analysis",
    }
)
_PLANNER_CATALOG: Final = (*READ_TOOLS, *WRITE_PROPOSAL_TOOLS)
_PLANNER_TOOLS_BY_NAME: Final[Mapping[str, BaseTool]] = MappingProxyType(
    {tool.name: tool for tool in _PLANNER_CATALOG}
)
if len(_PLANNER_TOOLS_BY_NAME) != len(_PLANNER_CATALOG):
    raise RuntimeError("Os catálogos do planner contêm nomes de tool duplicados.")


def _canonical_catalog_arguments(
    tool_name: str,
    arguments: object,
) -> dict[str, object] | None:
    tool = _PLANNER_TOOLS_BY_NAME.get(tool_name)
    if tool is None or not isinstance(arguments, Mapping):
        return None
    try:
        validated = tool.tool_call_schema.model_validate(arguments)
        explicit_wire = validated.model_dump(mode="json", exclude_unset=True)
        canonical_wire = validated.model_dump(mode="json")
        persisted_wire = json.dumps(
            arguments,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        validated_wire = json.dumps(
            explicit_wire,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, ValidationError):
        return None
    if persisted_wire != validated_wire:
        return None
    return canonical_wire


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


@dataclass(frozen=True)
class _PlannerAuthorizedTargets:
    analysis_ids: frozenset[str]
    knowledge_document_ids: frozenset[str]
    model_ids: frozenset[str]
    point_ids: frozenset[str]


def _validated_id(value: object, adapter: TypeAdapter[str]) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return adapter.validate_python(value, strict=True)
    except ValidationError:
        return None


def _ids_from_rows(
    value: object,
    field_name: str,
    adapter: TypeAdapter[str],
) -> set[str]:
    if not isinstance(value, list):
        return set()
    validated: set[str] = set()
    for row in value:
        if not isinstance(row, Mapping):
            continue
        candidate = _validated_id(row.get(field_name), adapter)
        if candidate is not None:
            validated.add(candidate)
    return validated


def _structured_observation_ids(
    call: PersistedToolCall,
    observation: ToolObservation,
) -> _PlannerAuthorizedTargets:
    content = (
        observation.content.to_python()
        if observation.content is not None
        else None
    )
    if not isinstance(content, Mapping):
        return _PlannerAuthorizedTargets(
            analysis_ids=frozenset(),
            knowledge_document_ids=frozenset(),
            model_ids=frozenset(),
            point_ids=frozenset(),
        )

    analysis_ids: set[str] = set()
    knowledge_document_ids: set[str] = set()
    model_ids: set[str] = set()
    point_ids: set[str] = set()
    if call.name == "get_asset":
        point_ids.update(
            _ids_from_rows(content.get("points"), "id", _POINT_ID_ADAPTER)
        )
    elif call.name == "list_asset_analyses":
        analyses = content.get("analyses")
        analysis_ids.update(_ids_from_rows(analyses, "id", _ANALYSIS_ID_ADAPTER))
        point_ids.update(_ids_from_rows(analyses, "point_id", _POINT_ID_ADAPTER))
    elif call.name == "get_analysis":
        analysis_id = _validated_id(content.get("id"), _ANALYSIS_ID_ADAPTER)
        point_id = _validated_id(content.get("point_id"), _POINT_ID_ADAPTER)
        if analysis_id is not None:
            analysis_ids.add(analysis_id)
        if point_id is not None:
            point_ids.add(point_id)
    elif call.name == "search_knowledge":
        knowledge_document_ids.update(
            _ids_from_rows(
                content.get("results"),
                "id",
                _KNOWLEDGE_ID_ADAPTER,
            )
        )
    elif call.name == "get_knowledge_document":
        document = content.get("document")
        source = document if isinstance(document, Mapping) else content
        document_id = _validated_id(source.get("id"), _KNOWLEDGE_ID_ADAPTER)
        if document_id is not None:
            knowledge_document_ids.add(document_id)
    elif call.name == "get_model":
        model = content.get("model")
        source = model if isinstance(model, Mapping) else content
        model_id = _validated_id(source.get("id"), _MODEL_ID_ADAPTER)
        if model_id is not None:
            model_ids.add(model_id)
    elif call.name in _ASSET_ARGUMENT_TOOL_NAMES:
        point_id = _validated_id(content.get("point_id"), _POINT_ID_ADAPTER)
        if point_id is None and isinstance(content.get("partial_data"), Mapping):
            point_id = _validated_id(
                content["partial_data"].get("point_id"),
                _POINT_ID_ADAPTER,
            )
        if point_id is not None:
            point_ids.add(point_id)
    return _PlannerAuthorizedTargets(
        analysis_ids=frozenset(analysis_ids),
        knowledge_document_ids=frozenset(knowledge_document_ids),
        model_ids=frozenset(model_ids),
        point_ids=frozenset(point_ids),
    )


def _authorized_targets(
    request_message: str,
    interactions: Sequence[tuple[PersistedToolCall, ToolObservation]],
) -> _PlannerAuthorizedTargets:
    analysis_ids = set(
        _typed_ids(
            (request_message,),
            pattern=_ANALYSIS_ID_PATTERN,
            adapter=_ANALYSIS_ID_ADAPTER,
        )
    )
    knowledge_document_ids = set(
        _typed_ids(
            (request_message,),
            pattern=_KNOWLEDGE_ID_PATTERN,
            adapter=_KNOWLEDGE_ID_ADAPTER,
        )
    )
    model_ids = set(
        _typed_ids(
            (request_message,),
            pattern=_MODEL_ID_PATTERN,
            adapter=_MODEL_ID_ADAPTER,
        )
    )
    point_ids = set(
        _typed_ids(
            (request_message,),
            pattern=_POINT_ID_PATTERN,
            adapter=_POINT_ID_ADAPTER,
        )
    )
    for call, observation in interactions:
        observed = _structured_observation_ids(call, observation)
        analysis_ids.update(observed.analysis_ids)
        knowledge_document_ids.update(observed.knowledge_document_ids)
        model_ids.update(observed.model_ids)
        point_ids.update(observed.point_ids)
    return _PlannerAuthorizedTargets(
        analysis_ids=frozenset(analysis_ids),
        knowledge_document_ids=frozenset(knowledge_document_ids),
        model_ids=frozenset(model_ids),
        point_ids=frozenset(point_ids),
    )


def _selected_targets_are_authorized(
    *,
    tool_name: str,
    arguments: Mapping[str, object],
    request: SupportRequest | PersistedSupportRequest,
    authorized: _PlannerAuthorizedTargets,
) -> bool:
    if "model_id" in arguments or "configured_model_id" in arguments:
        return False
    if tool_name in _ANALYSIS_ARGUMENT_TOOL_NAMES:
        if arguments.get("analysis_id") not in authorized.analysis_ids:
            return False
    if tool_name == "get_knowledge_document":
        if arguments.get("document_id") not in authorized.knowledge_document_ids:
            return False
    if tool_name in _ASSET_ARGUMENT_TOOL_NAMES:
        if (
            request.asset_id is None
            or arguments.get("asset_id") != request.asset_id
        ):
            return False
    if tool_name in _POINT_ARGUMENT_TOOL_NAMES:
        point_id = arguments.get("point_id")
        if point_id is not None and point_id not in authorized.point_ids:
            return False
    return True


def _current_request_interactions(
    state: AgentState,
) -> tuple[tuple[PersistedToolCall, ToolObservation], ...]:
    calls_by_id = {
        call.call_id: call
        for call in state.tool_calls
        if call.request_id == state.request_id
    }
    interactions: list[tuple[PersistedToolCall, ToolObservation]] = []
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
        authorized = _authorized_targets(state.request.message, interactions)
        if not _selected_targets_are_authorized(
            tool_name=call.name,
            arguments=call.arguments.to_python(),
            request=state.request,
            authorized=authorized,
        ):
            continue
        interactions.append((call, observation))
    return tuple(interactions)


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
    authorized = _authorized_targets(
        state.request.message,
        _current_request_interactions(state),
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
            and (tool.name != "get_analysis" or authorized.analysis_ids)
            and (
                tool.name != "get_knowledge_document"
                or authorized.knowledge_document_ids
            )
        )
    if not isinstance(runtime, WriteToolRuntime):
        return offered_reads

    has_scoped_case = state.request.case_id == runtime.current_case_id
    proposal_requirements = {
        "propose_reprocess_analysis": (
            "action_low",
            bool(authorized.analysis_ids),
        ),
        "propose_request_specialist_analysis": (
            "action_low",
            bool(authorized.analysis_ids),
        ),
        "propose_update_asset_criticality": (
            "action_high",
            has_scoped_asset,
        ),
        "propose_request_model_retraining": (
            "action_high",
            runtime.configured_model_id in authorized.model_ids,
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
    usage: PlannerUsage
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
    schemas: Sequence[BaseTool | type[StrictModel]],
    *,
    usage: PlannerUsage,
) -> tuple[list[BaseMessage], PlannerContextStats]:
    interaction_count = len(interactions)
    tool_wire = tuple(convert_to_openai_tool(schema) for schema in schemas)
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
    raise PlannerProtocolError(
        PlannerErrorCode.CONTEXT_LIMIT_EXCEEDED,
        usage=usage,
    )


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
    usage: PlannerUsage
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
        request_id: str,
        usage: PlannerUsage,
        tool_calls: Sequence[PersistedToolCall] = (),
        tool_observations: Sequence[ToolObservation] = (),
    ) -> PlannerToolTurn | PlannerDecisionTurn:
        active_usage = usage
        if (
            not isinstance(request_id, str)
            or not request_id
            or not isinstance(active_usage, PlannerUsage)
        ):
            raise PlannerProtocolError(PlannerErrorCode.INVALID_USAGE)
        if active_usage.request_id != request_id:
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_USAGE,
                usage=active_usage,
            )
        if (
            active_usage.selection_count > PLANNER_LIMITS.selections
            or active_usage.finalization_count > PLANNER_LIMITS.finalizations
        ):
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_USAGE,
                usage=active_usage,
            )
        if active_usage.finalization_count == PLANNER_LIMITS.finalizations:
            raise PlannerProtocolError(
                PlannerErrorCode.FINALIZATION_LIMIT_EXCEEDED,
                usage=active_usage,
            )
        if active_usage.selection_count == PLANNER_LIMITS.selections:
            raise PlannerProtocolError(
                PlannerErrorCode.SELECTION_LIMIT_EXCEEDED,
                usage=active_usage,
            )
        tools = tuple(offered_tools)
        tool_names = tuple(tool.name for tool in tools)
        if len(tool_names) != len(set(tool_names)):
            raise PlannerProtocolError(
                PlannerErrorCode.DUPLICATE_TOOL_NAME,
                usage=active_usage,
            )
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
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_HISTORY,
                usage=active_usage,
            )
        call_ids = tuple(call.call_id for call in calls)
        observation_ids = tuple(observation.call_id for observation in observations)
        if (
            len(call_ids) != len(set(call_ids))
            or len(observation_ids) != len(set(observation_ids))
        ):
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_HISTORY,
                usage=active_usage,
            )
        canonical_calls: list[PersistedToolCall] = []
        for call, observation in zip(calls, observations, strict=True):
            call_arguments = _canonical_catalog_arguments(
                call.name,
                call.arguments.to_python(),
            )
            artifact_arguments = _canonical_catalog_arguments(
                call.name,
                observation.artifact.arguments.to_python(),
            )
            if (
                observation.call_id != call.call_id
                or observation.content is None
                or observation.artifact.tool_name != call.name
                or call_arguments is None
                or artifact_arguments != call_arguments
            ):
                raise PlannerProtocolError(
                    PlannerErrorCode.INVALID_HISTORY,
                    usage=active_usage,
                )
            canonical_calls.append(
                PersistedToolCall(
                    request_id=call.request_id,
                    call_id=call.call_id,
                    name=call.name,
                    arguments=call_arguments,
                )
            )
        calls = tuple(canonical_calls)
        call_fingerprints = tuple(_tool_call_fingerprint(call) for call in calls)
        if len(call_fingerprints) != len(set(call_fingerprints)):
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_HISTORY,
                usage=active_usage,
            )
        if len(calls) > PLANNER_LIMITS.tool_calls:
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_HISTORY,
                usage=active_usage,
            )
        if len(calls) == PLANNER_LIMITS.tool_calls:
            raise PlannerProtocolError(
                PlannerErrorCode.TOOL_CALL_LIMIT_EXCEEDED,
                usage=active_usage,
            )
        interactions: list[tuple[AIMessage, ToolMessage, bool]] = []
        authorized_interactions: list[
            tuple[PersistedToolCall, ToolObservation]
        ] = []
        for index, (call, observation) in enumerate(
            zip(calls, observations, strict=True)
        ):
            historical_arguments = call.arguments.to_python()
            historical_authorized = _authorized_targets(
                request.message,
                authorized_interactions,
            )
            if not _selected_targets_are_authorized(
                tool_name=call.name,
                arguments=historical_arguments,
                request=request,
                authorized=historical_authorized,
            ):
                raise PlannerProtocolError(
                    PlannerErrorCode.INVALID_HISTORY,
                    usage=active_usage,
                )
            authorized_interactions.append((call, observation))
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
        authorized_targets = _authorized_targets(
            request.message,
            authorized_interactions,
        )
        messages, context_stats = _build_planner_context(
            request_content,
            interactions,
            tools,
            usage=active_usage,
        )
        selection = await self._model.bind_tools(tools).ainvoke(messages)
        selection_usage = PlannerUsage(
            request_id=active_usage.request_id,
            selection_count=active_usage.selection_count + 1,
            finalization_count=active_usage.finalization_count,
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
            terminal_messages, terminal_context_stats = _build_planner_context(
                request_content,
                interactions,
                (PlannerTerminalDecision,),
                usage=selection_usage,
            )
            final_usage = PlannerUsage(
                request_id=selection_usage.request_id,
                selection_count=selection_usage.selection_count,
                finalization_count=selection_usage.finalization_count + 1,
            )
            terminal_decision: PlannerTerminalDecision | None = None
            try:
                terminal_output = await self._model.with_structured_output(
                    PlannerTerminalDecision,
                    include_raw=False,
                ).ainvoke(terminal_messages)
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
                context=terminal_context_stats,
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
            arguments = validated_arguments.model_dump(mode="json")
            if _selected_targets_are_authorized(
                tool_name=selected_call["name"],
                arguments=arguments,
                request=request,
                authorized=authorized_targets,
            ):
                tool_turn = PlannerToolTurn(
                    tool_call=PersistedToolCall(
                        request_id=request_id,
                        call_id=selected_call["id"],
                        name=selected_call["name"],
                        arguments=arguments,
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
