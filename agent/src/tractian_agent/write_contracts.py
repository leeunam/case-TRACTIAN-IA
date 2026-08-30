"""Contratos persistíveis de intenções de escrita."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StringConstraints, field_validator, model_validator

from tractian_agent.contracts import ActionReceipt, ApiError, IdempotencyKey, StrictModel
from tractian_agent.tools.identifiers import AnalysisId
from tractian_agent.write_policy import WritePolicyResult


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
    receipt: ActionReceipt | None = None
    error: ApiError | None = None

    @field_validator("expires_at")
    @classmethod
    def _require_aware_expiration(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("expires_at deve conter timezone")
        return value

    @model_validator(mode="after")
    def _reject_receipt_and_error_together(self) -> WriteIntent:
        if self.receipt is not None and self.error is not None:
            raise ValueError("uma intenção não pode registrar recibo e erro juntos")
        return self
