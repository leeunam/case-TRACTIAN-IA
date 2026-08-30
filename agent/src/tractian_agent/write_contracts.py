"""Contratos persistíveis de intenções de escrita."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StringConstraints, field_validator, model_validator

from tractian_agent.contracts import (
    ActionReceipt,
    ApiError,
    ApiErrorCategory,
    IdempotencyKey,
    StrictModel,
)
from tractian_agent.tools.identifiers import AnalysisId
from tractian_agent.write_policy import PolicyDecision, PolicyReason, WritePolicyResult


CanonicalHash = Annotated[
    str,
    StringConstraints(pattern=r"^sha256:v1:[0-9a-f]{64}$"),
]


class IntentStatus(str, Enum):
    PROPOSED = "proposed"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    PREPARED = "prepared"
    COMPLETED = "completed"
    DENIED = "denied"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class ReprocessIntentScope(StrictModel):
    """Escopo fechado da única ação suportada nesta fatia."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["reprocess_analysis"]
    case_id: str = Field(min_length=1, pattern=r"^\S+$")
    company_id: str = Field(min_length=1, pattern=r"^\S+$")
    user_id: str = Field(min_length=1, pattern=r"^\S+$")
    analysis_id: AnalysisId
    justification: str = Field(min_length=1, pattern=r"\S")


class PersistedActionReceipt(StrictModel):
    """Cópia imutável do recibo compartilhado para o checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    accepted: bool
    action_id: str = Field(min_length=1, pattern=r"^\S+$")
    message: str = Field(min_length=1, pattern=r"\S")

    @model_validator(mode="before")
    @classmethod
    def _copy_shared_receipt(cls, value: object) -> object:
        if isinstance(value, ActionReceipt):
            return value.model_dump(mode="python")
        return value


class PersistedApiError(StrictModel):
    """Cópia imutável do erro normalizado para o checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: Literal[False] = False
    category: ApiErrorCategory
    code: str = Field(min_length=1, pattern=r"^\S+$")
    message: str = Field(min_length=1, pattern=r"\S")
    status_code: int | None = Field(default=None, ge=100, le=599)

    @model_validator(mode="before")
    @classmethod
    def _copy_shared_error(cls, value: object) -> object:
        if isinstance(value, ApiError):
            return value.model_dump(mode="python")
        return value


class WriteIntent(StrictModel):
    """Registro observável de uma intenção no checkpointer do grafo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent_id: str = Field(min_length=1, pattern=r"^\S+$")
    scope: ReprocessIntentScope
    payload_hash: CanonicalHash
    decision: WritePolicyResult
    status: IntentStatus
    idempotency_key: IdempotencyKey | None = None
    expires_at: datetime | None = None
    prepared_execution_id: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^\S+$",
    )
    attempts: int = Field(default=0, ge=0)
    receipt: PersistedActionReceipt | None = None
    error: PersistedApiError | None = None

    @field_validator("expires_at")
    @classmethod
    def _require_aware_expiration(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("expires_at deve conter timezone")
        return value

    @model_validator(mode="after")
    def _validate_status_fields(self) -> WriteIntent:
        valid_reasons = {
            PolicyDecision.ALLOW: {PolicyReason.AUTHORIZED},
            PolicyDecision.REQUIRE_CONFIRMATION: {
                PolicyReason.EXPLICIT_APPROVAL_REQUIRED,
                PolicyReason.APPROVAL_SCOPE_MISMATCH,
            },
            PolicyDecision.DENY: {
                PolicyReason.MISSING_PERMISSION,
                PolicyReason.INVALID_JUSTIFICATION,
            },
        }
        if self.decision.reason not in valid_reasons[self.decision.decision]:
            raise ValueError("razão incompatível com a decisão da intenção")

        expected_decision = {
            IntentStatus.PROPOSED: PolicyDecision.ALLOW,
            IntentStatus.AWAITING_CONFIRMATION: PolicyDecision.REQUIRE_CONFIRMATION,
            IntentStatus.DENIED: PolicyDecision.DENY,
            IntentStatus.PREPARED: PolicyDecision.ALLOW,
            IntentStatus.COMPLETED: PolicyDecision.ALLOW,
            IntentStatus.FAILED: PolicyDecision.ALLOW,
            IntentStatus.UNCERTAIN: PolicyDecision.ALLOW,
        }[self.status]
        if self.decision.decision is not expected_decision:
            raise ValueError("decisão incompatível com o status da intenção")

        prepared_fields = (
            self.idempotency_key,
            self.expires_at,
            self.prepared_execution_id,
        )
        if self.status in {
            IntentStatus.PROPOSED,
            IntentStatus.AWAITING_CONFIRMATION,
            IntentStatus.DENIED,
        }:
            if (
                any(value is not None for value in prepared_fields)
                or self.attempts != 0
                or self.receipt is not None
                or self.error is not None
            ):
                raise ValueError("status anterior ao preparo contém dados de despacho")
            return self

        if any(value is None for value in prepared_fields):
            raise ValueError("intenção preparada exige chave, expiração e execução")
        if self.status is IntentStatus.PREPARED:
            if self.attempts != 0 or self.receipt is not None or self.error is not None:
                raise ValueError("status prepared não aceita tentativa ou resultado")
            return self

        if self.attempts < 1:
            raise ValueError("status terminal exige ao menos uma tentativa")
        if self.status is IntentStatus.COMPLETED:
            if self.receipt is None or self.error is not None:
                raise ValueError("status completed exige somente recibo")
        elif self.error is None or self.receipt is not None:
            raise ValueError("status failed/uncertain exige somente erro")
        return self
