"""Contratos JSON-safe do estado persistível do agente."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

from pydantic import ConfigDict, Field, JsonValue, field_validator, model_validator

from tractian_agent.contracts import StrictModel, SupportRequest, ToolCall
from tractian_agent.tools.observations import ToolArtifact, assert_safe_partial_json
from tractian_agent.tools.runtime import Permission, TrustedIdentity
from tractian_agent.write_contracts import WriteIntent
from tractian_agent.write_policy import ReprocessProposal, TrustedActionApproval


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _assert_safe_persisted_json(value: JsonValue) -> None:
    assert_safe_partial_json(value)
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if _normalized_key(key) == "transport":
                raise ValueError("o estado contém um campo proibido")
            _assert_safe_persisted_json(nested_value)
    elif isinstance(value, list):
        for item in value:
            _assert_safe_persisted_json(item)


class FrozenStateModel(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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
    artifact: ToolArtifact

    @model_validator(mode="after")
    def _validate_artifact_json(self) -> ToolObservation:
        _assert_safe_persisted_json(self.artifact.model_dump(mode="json"))
        return self


class StateEvidence(FrozenStateModel):
    evidence_id: str = Field(min_length=1, pattern=r"^\S+$")
    call_id: str = Field(min_length=1, pattern=r"^\S+$")
    value: JsonValue

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: JsonValue) -> JsonValue:
        _assert_safe_persisted_json(value)
        return value


class FinalResult(FrozenStateModel):
    decision: AgentDecision
    message: str = Field(min_length=1, pattern=r"\S")


class ReviewRecord(FrozenStateModel):
    status: ReviewStatus
    reason: str | None = Field(default=None, min_length=1, pattern=r"\S")


class AgentState(FrozenStateModel):
    request: SupportRequest
    identity: TrustedIdentity
    permissions: frozenset[Permission]
    request_id: str = Field(min_length=1, pattern=r"^\S+$")
    thread_id: str = Field(min_length=1, pattern=r"^\S+$")
    execution_id: str = Field(min_length=1, pattern=r"^\S+$")
    thread_scope: ThreadScope
    messages: tuple[PersistedMessage, ...] = ()
    tool_calls: tuple[ToolCall[dict[str, JsonValue]], ...] = ()
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

    @field_validator("tool_calls")
    @classmethod
    def _validate_tool_call_json(
        cls,
        calls: tuple[ToolCall[dict[str, JsonValue]], ...],
    ) -> tuple[ToolCall[dict[str, JsonValue]], ...]:
        for call in calls:
            _assert_safe_persisted_json(call.arguments)
        return calls

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
        data = self.model_dump(mode="python")
        data.update(
            request=request,
            identity=identity,
            permissions=permissions,
            request_id=request_id,
            execution_id=execution_id,
        )
        return type(self).model_validate(data)

    def advance_step(self) -> AgentState:
        """Avança uma unidade sem permitir ultrapassar o orçamento."""
        if self.step_count >= self.step_limit:
            raise ValueError("orçamento de passos esgotado")
        data = self.model_dump(mode="python")
        data["step_count"] = self.step_count + 1
        return type(self).model_validate(data)
