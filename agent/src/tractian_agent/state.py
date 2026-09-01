"""Contratos JSON-safe do estado persistível do agente."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import json
import math
import re
from typing import Literal

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

from tractian_agent.contracts import ResponseMode, StrictModel, SupportRequest, ToolCall
from tractian_agent.tools.observations import ToolArtifact
from tractian_agent.tools.runtime import Permission, TrustedIdentity
from tractian_agent.write_contracts import IntentStatus, PersistedApiError, WriteIntent
from tractian_agent.write_policy import TrustedActionApproval, WriteProposal


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _key_segments(value: str) -> frozenset[str]:
    """Segmenta aliases snake/kebab/camel sem recorrer a substring livre."""
    separated_acronyms = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
    separated_camel = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        separated_acronyms,
    )
    return frozenset(
        segment.casefold()
        for segment in re.findall(r"[A-Za-z0-9]+", separated_camel)
    )


_TECHNICAL_FORBIDDEN_NAMES = frozenset(
    {
        "client",
        "transport",
        "runtime",
        "authorization",
        "apitoken",
        "token",
        "apikey",
        "credential",
        "credentials",
        "password",
        "secret",
        "cookie",
        "golden",
        "goldenset",
        "eval",
        "evaluation",
        "expectedpaths",
        "testscenarios",
        "evaluationseed",
        "seed",
        "rawhttpresponse",
        "reasoningtrace",
    }
)

_TECHNICAL_FORBIDDEN_SEGMENTS = frozenset(
    {
        "token",
        "password",
        "credential",
        "credentials",
        "authorization",
        "secret",
        "cookie",
        "evaluation",
        "eval",
    }
)

_TECHNICAL_FORBIDDEN_SEGMENT_PATTERNS = (
    frozenset({"access", "token"}),
    frozenset({"api", "token"}),
    frozenset({"api", "key"}),
    frozenset({"client", "secret"}),
    frozenset({"http", "response"}),
    frozenset({"response", "body"}),
    frozenset({"reasoning", "trace"}),
    frozenset({"golden", "set"}),
    frozenset({"expected", "path"}),
    frozenset({"expected", "paths"}),
    frozenset({"test", "scenario"}),
    frozenset({"test", "scenarios"}),
    frozenset({"evaluation", "seed"}),
)

_PUBLIC_ARGUMENT_FORBIDDEN_NAMES = _TECHNICAL_FORBIDDEN_NAMES | {
    "caseid",
    "companyid",
    "userid",
    "identity",
    "permissions",
    "approval",
    "threadid",
    "requestid",
    "executionid",
    "idempotencykey",
    "centralassetid",
    "configuredmodelid",
    "context",
    "url",
    "method",
    "header",
    "headers",
}

_PUBLIC_ARGUMENT_FORBIDDEN_SEGMENTS = _TECHNICAL_FORBIDDEN_SEGMENTS | {
    "identity",
    "permissions",
    "approval",
}

_PUBLIC_ARGUMENT_FORBIDDEN_SEGMENT_PATTERNS = (
    *_TECHNICAL_FORBIDDEN_SEGMENT_PATTERNS,
    frozenset({"trusted", "identity"}),
    frozenset({"action", "approval"}),
    frozenset({"agent", "thread", "id"}),
    frozenset({"thread", "id"}),
    frozenset({"request", "id"}),
    frozenset({"execution", "id"}),
    frozenset({"idempotency", "key"}),
    frozenset({"central", "asset", "id"}),
    frozenset({"configured", "model", "id"}),
    frozenset({"case", "id"}),
    frozenset({"company", "id"}),
    frozenset({"user", "id"}),
)


def _validate_json_boundary(
    value: JsonValue,
    *,
    forbidden_names: frozenset[str] | set[str],
    forbidden_segments: frozenset[str] | set[str] = frozenset(),
    forbidden_segment_patterns: tuple[frozenset[str], ...] = (),
) -> None:
    """Percorre um payload uma vez e aplica a política da sua fronteira."""
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            segments = _key_segments(key)
            if (
                _normalized_key(key) in forbidden_names
                or not segments.isdisjoint(forbidden_segments)
                or any(
                    pattern <= segments for pattern in forbidden_segment_patterns
                )
            ):
                raise ValueError("o estado contém um campo proibido")
            _validate_json_boundary(
                nested_value,
                forbidden_names=forbidden_names,
                forbidden_segments=forbidden_segments,
                forbidden_segment_patterns=forbidden_segment_patterns,
            )
    elif isinstance(value, list):
        for item in value:
            _validate_json_boundary(
                item,
                forbidden_names=forbidden_names,
                forbidden_segments=forbidden_segments,
                forbidden_segment_patterns=forbidden_segment_patterns,
            )
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("o estado contém número não finito")


_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)


class FrozenStateModel(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JsonSnapshot(FrozenStateModel):
    """JSON canônico imutável, desserializado sempre como uma nova cópia."""

    encoded: str

    @field_validator("encoded")
    @classmethod
    def _require_canonical_json(cls, value: str) -> str:
        try:
            validated = _JSON_VALUE_ADAPTER.validate_python(json.loads(value))
            return json.dumps(
                validated,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("encoded deve conter JSON canônico válido") from error

    @classmethod
    def capture(
        cls,
        value: object,
        *,
        forbidden_names: frozenset[str] | set[str],
        forbidden_segments: frozenset[str] | set[str] = frozenset(),
        forbidden_segment_patterns: tuple[frozenset[str], ...] = (),
    ) -> JsonSnapshot:
        if isinstance(value, cls):
            value = value.to_python()
        validated = _JSON_VALUE_ADAPTER.validate_python(value)
        _validate_json_boundary(
            validated,
            forbidden_names=forbidden_names,
            forbidden_segments=forbidden_segments,
            forbidden_segment_patterns=forbidden_segment_patterns,
        )
        return cls(
            encoded=json.dumps(
                validated,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )

    def to_python(self) -> JsonValue:
        return json.loads(self.encoded)

    def __getitem__(self, key: str | int) -> JsonValue:
        return self.to_python()[key]


def _snapshot_domain_value(value: object, validation_mode: str) -> object:
    if isinstance(value, JsonSnapshot):
        return value.to_python()
    if (
        isinstance(value, Mapping)
        and set(value) == {"encoded"}
        and isinstance(value["encoded"], str)
    ):
        return JsonSnapshot.model_validate(value).to_python()
    if validation_mode == "json":
        return JsonSnapshot.model_validate(value).to_python()
    return value


def _capture_public_argument_object(
    value: object,
    validation_mode: str,
) -> JsonSnapshot:
    domain_value = _snapshot_domain_value(value, validation_mode)
    if not isinstance(domain_value, Mapping) or any(
        not isinstance(key, str) for key in domain_value
    ):
        raise ValueError("arguments deve ser um objeto JSON com chaves string")
    return JsonSnapshot.capture(
        domain_value,
        forbidden_names=_PUBLIC_ARGUMENT_FORBIDDEN_NAMES,
        forbidden_segments=_PUBLIC_ARGUMENT_FORBIDDEN_SEGMENTS,
        forbidden_segment_patterns=_PUBLIC_ARGUMENT_FORBIDDEN_SEGMENT_PATTERNS,
    )


class PersistedRequestIdentity(FrozenStateModel):
    user_id: str = Field(min_length=1, pattern=r"^\S+$")
    company_id: str = Field(min_length=1, pattern=r"^\S+$")


class PersistedSupportRequest(FrozenStateModel):
    case_id: str = Field(min_length=1, pattern=r"^\S+$")
    ticket_id: str = Field(min_length=1, pattern=r"^\S+$")
    asset_id: str | None = Field(min_length=1, pattern=r"^\S+$")
    message: str = Field(min_length=1, pattern=r"\S")
    identity: PersistedRequestIdentity

    @model_validator(mode="before")
    @classmethod
    def _copy_shared_request(cls, value: object) -> object:
        if isinstance(value, SupportRequest):
            return value.model_dump(mode="python")
        return value


class PersistedToolCall(FrozenStateModel):
    request_id: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    call_id: str = Field(min_length=1, pattern=r"^\S+$")
    name: str = Field(min_length=1, pattern=r"^\S+$")
    arguments: JsonSnapshot

    @model_validator(mode="before")
    @classmethod
    def _copy_shared_call(cls, value: object) -> object:
        if isinstance(value, ToolCall):
            return value.model_dump(mode="python")
        return value

    @field_validator("arguments", mode="before")
    @classmethod
    def _snapshot_public_arguments(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> JsonSnapshot:
        return _capture_public_argument_object(value, info.mode)


class PersistedToolSource(FrozenStateModel):
    kind: Literal["industrial_api"]
    resource: str = Field(min_length=1, pattern=r"^/")


class PersistedToolOutcome(FrozenStateModel):
    mode: ResponseMode | None = None
    notes: str | None = None
    partial_data: JsonSnapshot | None = None
    error: PersistedApiError | None = None

    @field_validator("partial_data", mode="before")
    @classmethod
    def _snapshot_technical_result(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> JsonSnapshot | None:
        if value is None:
            return None
        return JsonSnapshot.capture(
            _snapshot_domain_value(value, info.mode),
            forbidden_names=_TECHNICAL_FORBIDDEN_NAMES,
            forbidden_segments=_TECHNICAL_FORBIDDEN_SEGMENTS,
            forbidden_segment_patterns=_TECHNICAL_FORBIDDEN_SEGMENT_PATTERNS,
        )


class PersistedToolArtifact(FrozenStateModel):
    tool_name: str = Field(min_length=1, pattern=r"^\S+$")
    arguments: JsonSnapshot
    source: PersistedToolSource
    outcome: PersistedToolOutcome
    truncated: bool = False
    omitted_items: int = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _copy_shared_artifact(cls, value: object) -> object:
        if isinstance(value, ToolArtifact):
            return value.model_dump(mode="python")
        return value

    @field_validator("arguments", mode="before")
    @classmethod
    def _snapshot_public_arguments(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> JsonSnapshot:
        return _capture_public_argument_object(value, info.mode)


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class AgentDecision(str, Enum):
    GUIDE = "guide"
    ACT = "act"
    ESCALATE = "escalate"
    REQUEST_INFORMATION = "request_information"
    REQUEST_CONFIRMATION = "request_confirmation"
    REQUIRE_HUMAN_REVIEW = "require_human_review"


class ReviewStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    APPROVED = "approved"
    REJECTED = "rejected"


class ThreadScope(FrozenStateModel):
    thread_id: str = Field(min_length=1, pattern=r"^\S+$")
    case_id: str = Field(min_length=1, pattern=r"^\S+$")
    company_id: str = Field(min_length=1, pattern=r"^\S+$")
    user_id: str = Field(min_length=1, pattern=r"^\S+$")


class PersistedMessage(FrozenStateModel):
    role: MessageRole
    content: str


class ToolObservation(FrozenStateModel):
    request_id: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    call_id: str = Field(min_length=1, pattern=r"^\S+$")
    content: JsonSnapshot | None = None
    artifact: PersistedToolArtifact

    @field_validator("content", mode="before")
    @classmethod
    def _snapshot_next_turn_content(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> JsonSnapshot | None:
        if value is None:
            return None
        return JsonSnapshot.capture(
            _snapshot_domain_value(value, info.mode),
            forbidden_names=_TECHNICAL_FORBIDDEN_NAMES,
            forbidden_segments=_TECHNICAL_FORBIDDEN_SEGMENTS,
            forbidden_segment_patterns=_TECHNICAL_FORBIDDEN_SEGMENT_PATTERNS,
        )


class StateEvidence(FrozenStateModel):
    evidence_id: str = Field(min_length=1, pattern=r"^\S+$")
    call_id: str = Field(min_length=1, pattern=r"^\S+$")
    value: JsonSnapshot

    @field_validator("value", mode="before")
    @classmethod
    def _snapshot_value(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> JsonSnapshot:
        return JsonSnapshot.capture(
            _snapshot_domain_value(value, info.mode),
            forbidden_names=_TECHNICAL_FORBIDDEN_NAMES,
            forbidden_segments=_TECHNICAL_FORBIDDEN_SEGMENTS,
            forbidden_segment_patterns=_TECHNICAL_FORBIDDEN_SEGMENT_PATTERNS,
        )


class FinalResult(FrozenStateModel):
    decision: AgentDecision
    message: str = Field(min_length=1, pattern=r"\S")


class ReviewRecord(FrozenStateModel):
    status: ReviewStatus
    reason: str | None = Field(default=None, min_length=1, pattern=r"\S")


class PlannerUsage(FrozenStateModel):
    """Contadores do planner vinculados a uma única solicitação."""

    request_id: str = Field(min_length=1, pattern=r"^\S+$")
    selection_count: int = Field(default=0, ge=0, le=8, strict=True)
    finalization_count: int = Field(default=0, ge=0, le=1, strict=True)


class AgentState(FrozenStateModel):
    request: PersistedSupportRequest
    identity: TrustedIdentity
    permissions: frozenset[Permission]
    request_id: str = Field(min_length=1, pattern=r"^\S+$")
    thread_id: str = Field(min_length=1, pattern=r"^\S+$")
    execution_id: str = Field(min_length=1, pattern=r"^\S+$")
    thread_scope: ThreadScope
    messages: tuple[PersistedMessage, ...] = ()
    tool_calls: tuple[PersistedToolCall, ...] = ()
    tool_observations: tuple[ToolObservation, ...] = ()
    evidence: tuple[StateEvidence, ...] = ()
    decision: AgentDecision | None = None
    step_count: int = Field(default=0, ge=0)
    step_limit: int = Field(gt=0)
    pending_proposal: WriteProposal | None = None
    approval: TrustedActionApproval | None = None
    intents: tuple[WriteIntent, ...] = ()
    final_result: FinalResult | None = None
    review: ReviewRecord | None = None
    planner_usage: PlannerUsage

    @model_validator(mode="before")
    @classmethod
    def _restore_request_bound_planner_state(cls, value: object) -> object:
        """Inicializa uso legado sem inventar proveniência para o histórico."""
        if isinstance(value, Mapping) and isinstance(value.get("request_id"), str):
            request_id = value["request_id"]
            restored = dict(value)
            if value.get("planner_usage") is None:
                restored["planner_usage"] = {
                    "request_id": request_id,
                    "selection_count": 0,
                    "finalization_count": 0,
                }
            return restored
        return value

    @field_validator("pending_proposal", mode="before")
    @classmethod
    def _restore_legacy_reprocess_proposal(cls, value: object) -> object:
        """Adiciona o discriminador somente ao shape inequívoco anterior."""
        if isinstance(value, Mapping) and set(value) == {
            "analysis_id",
            "justification",
        }:
            return {
                "action": "reprocess_analysis",
                "analysis_id": value["analysis_id"],
                "justification": value["justification"],
            }
        return value

    @model_validator(mode="after")
    def _validate_scope_and_budget(self) -> AgentState:
        expected_scope = (
            self.thread_id,
            self.request.case_id,
            self.identity.company_id,
            self.identity.user_id,
        )
        persisted_scope = (
            self.thread_scope.thread_id,
            self.thread_scope.case_id,
            self.thread_scope.company_id,
            self.thread_scope.user_id,
        )
        if expected_scope != persisted_scope:
            raise ValueError("thread reutilizado fora do escopo confiável")
        if (
            self.request.identity.company_id != self.identity.company_id
            or self.request.identity.user_id != self.identity.user_id
        ):
            raise ValueError("identidade da solicitação diverge da fronteira confiável")
        thread_intent_scope = (
            self.thread_scope.case_id,
            self.thread_scope.company_id,
            self.thread_scope.user_id,
        )
        if any(
            (
                intent.scope.case_id,
                intent.scope.company_id,
                intent.scope.user_id,
            )
            != thread_intent_scope
            for intent in self.intents
        ):
            raise ValueError("intenção persistida fora do escopo do thread")
        intent_ids = tuple(intent.intent_id for intent in self.intents)
        if len(intent_ids) != len(set(intent_ids)):
            raise ValueError("IDs de intenção devem ser únicos no estado")
        active_statuses = {
            IntentStatus.PROPOSED,
            IntentStatus.AWAITING_CONFIRMATION,
            IntentStatus.PREPARED,
        }
        active_request_ids = [
            intent.request_id
            for intent in self.intents
            if intent.request_id is not None
            and intent.status in active_statuses
        ]
        if len(active_request_ids) != len(set(active_request_ids)):
            raise ValueError("cada request_id aceita no máximo uma intenção ativa")
        if self.step_count > self.step_limit:
            raise ValueError("contador de passos excede o orçamento")
        if self.planner_usage.request_id != self.request_id:
            raise ValueError("uso do planner pertence a outra request_id")
        current_call_ids = tuple(
            call.call_id
            for call in self.tool_calls
            if call.request_id == self.request_id
        )
        current_observation_ids = tuple(
            observation.call_id
            for observation in self.tool_observations
            if observation.request_id == self.request_id
        )
        if len(current_call_ids) > 7:
            raise ValueError("histórico do planner excede sete tool calls")
        if (
            len(current_call_ids) != len(set(current_call_ids))
            or len(current_observation_ids) != len(set(current_observation_ids))
        ):
            raise ValueError("IDs de tool call devem ser únicos por request_id")
        return self

    def continue_with(
        self,
        *,
        request: SupportRequest,
        identity: TrustedIdentity,
        permissions: frozenset[Permission],
        request_id: str,
        execution_id: str,
        step_limit: int | None = None,
    ) -> AgentState:
        """Cria uma invocação no mesmo thread, validando novamente seu escopo."""
        if execution_id == self.execution_id:
            raise ValueError("cada continuação exige um novo execution_id")
        persisted_request = PersistedSupportRequest.model_validate(request)
        same_request = request_id == self.request_id
        if same_request and persisted_request != self.request:
            raise ValueError("retomada da mesma request_id exige solicitação idêntica")
        data = {
            field_name: getattr(self, field_name)
            for field_name in type(self).model_fields
        }
        data.update(
            request=persisted_request,
            identity=identity,
            permissions=permissions,
            request_id=request_id,
            execution_id=execution_id,
        )
        if not same_request:
            data.update(
                decision=None,
                step_count=0,
                step_limit=self.step_limit if step_limit is None else step_limit,
                pending_proposal=None,
                approval=None,
                final_result=None,
                review=None,
                planner_usage=PlannerUsage(request_id=request_id),
            )
        return type(self).model_validate(data)

    def advance_step(self) -> AgentState:
        """Avança uma unidade sem permitir ultrapassar o orçamento."""
        if self.step_count >= self.step_limit:
            raise ValueError("orçamento de passos esgotado")
        data = {
            field_name: getattr(self, field_name)
            for field_name in type(self).model_fields
        }
        data["step_count"] = self.step_count + 1
        return type(self).model_validate(data)
