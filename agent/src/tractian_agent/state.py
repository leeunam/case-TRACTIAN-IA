"""Contratos JSON-safe do estado persistível do agente."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import json
import math
from typing import Literal

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
    model_serializer,
    model_validator,
)

from tractian_agent.contracts import ResponseMode, StrictModel, SupportRequest, ToolCall
from tractian_agent.tools.observations import ToolArtifact
from tractian_agent.tools.runtime import Permission, TrustedIdentity
from tractian_agent.write_contracts import PersistedApiError, WriteIntent
from tractian_agent.write_policy import ReprocessProposal, TrustedActionApproval


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


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


def _validate_json_boundary(
    value: JsonValue,
    *,
    forbidden_names: frozenset[str] | set[str],
) -> None:
    """Percorre um payload uma vez e aplica a política da sua fronteira."""
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if _normalized_key(key) in forbidden_names:
                raise ValueError("o estado contém um campo proibido")
            _validate_json_boundary(nested_value, forbidden_names=forbidden_names)
    elif isinstance(value, list):
        for item in value:
            _validate_json_boundary(item, forbidden_names=forbidden_names)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("o estado contém número não finito")


_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)


class FrozenStateModel(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JsonSnapshot(FrozenStateModel):
    """JSON canônico imutável, desserializado sempre como uma nova cópia."""

    encoded: str

    @classmethod
    def capture(
        cls,
        value: object,
        *,
        forbidden_names: frozenset[str] | set[str],
    ) -> JsonSnapshot:
        if isinstance(value, cls):
            value = value.to_python()
        validated = _JSON_VALUE_ADAPTER.validate_python(value)
        _validate_json_boundary(validated, forbidden_names=forbidden_names)
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

    @model_serializer(mode="plain")
    def _serialize_as_json_value(self) -> JsonValue:
        return self.to_python()


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
    def _snapshot_public_arguments(cls, value: object) -> JsonSnapshot:
        return JsonSnapshot.capture(
            value,
            forbidden_names=_PUBLIC_ARGUMENT_FORBIDDEN_NAMES,
        )


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
    def _snapshot_technical_result(cls, value: object) -> JsonSnapshot | None:
        if value is None:
            return None
        return JsonSnapshot.capture(
            value,
            forbidden_names=_TECHNICAL_FORBIDDEN_NAMES,
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
    def _snapshot_public_arguments(cls, value: object) -> JsonSnapshot:
        return JsonSnapshot.capture(
            value,
            forbidden_names=_PUBLIC_ARGUMENT_FORBIDDEN_NAMES,
        )


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
    call_id: str = Field(min_length=1, pattern=r"^\S+$")
    artifact: PersistedToolArtifact


class StateEvidence(FrozenStateModel):
    evidence_id: str = Field(min_length=1, pattern=r"^\S+$")
    call_id: str = Field(min_length=1, pattern=r"^\S+$")
    value: JsonSnapshot

    @field_validator("value", mode="before")
    @classmethod
    def _snapshot_value(cls, value: object) -> JsonSnapshot:
        return JsonSnapshot.capture(
            value,
            forbidden_names=_TECHNICAL_FORBIDDEN_NAMES,
        )


class FinalResult(FrozenStateModel):
    decision: AgentDecision
    message: str = Field(min_length=1, pattern=r"\S")


class ReviewRecord(FrozenStateModel):
    status: ReviewStatus
    reason: str | None = Field(default=None, min_length=1, pattern=r"\S")


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
    pending_proposal: ReprocessProposal | None = None
    approval: TrustedActionApproval | None = None
    intents: tuple[WriteIntent, ...] = ()
    final_result: FinalResult | None = None
    review: ReviewRecord | None = None

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
        if self.step_count > self.step_limit:
            raise ValueError("contador de passos excede o orçamento")
        return self

    def continue_with(
        self,
        *,
        request: SupportRequest,
        identity: TrustedIdentity,
        permissions: frozenset[Permission],
        request_id: str,
        execution_id: str,
    ) -> AgentState:
        """Cria uma invocação no mesmo thread, validando novamente seu escopo."""
        if execution_id == self.execution_id:
            raise ValueError("cada continuação exige um novo execution_id")
        persisted_request = PersistedSupportRequest.model_validate(request)
        same_request = request_id == self.request_id
        if same_request and persisted_request != self.request:
            raise ValueError("retomada da mesma request_id exige solicitação idêntica")
        data = self.model_dump(mode="python")
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
                pending_proposal=None,
                approval=None,
                final_result=None,
                review=None,
            )
        return type(self).model_validate(data)

    def advance_step(self) -> AgentState:
        """Avança uma unidade sem permitir ultrapassar o orçamento."""
        if self.step_count >= self.step_limit:
            raise ValueError("orçamento de passos esgotado")
        data = self.model_dump(mode="python")
        data["step_count"] = self.step_count + 1
        return type(self).model_validate(data)
