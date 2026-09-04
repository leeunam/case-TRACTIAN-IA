from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DemoModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ExecutionStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_HUMAN = "waiting_human"
    COMPLETED = "completed"
    FAILED = "failed"


class DecisionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class DeliveryStatus(str, Enum):
    PENDING = "pending"
    DELIVERING = "delivering"
    DELIVERED = "delivered"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class DemoCase(DemoModel):
    id: str = Field(pattern=r"^case_[A-Za-z0-9_-]+$")
    ticket_id: str = Field(min_length=1)
    company_id: str = Field(min_length=1, pattern=r"^\S+$")
    requester_id: str = Field(min_length=1, pattern=r"^\S+$")
    asset_id: str = Field(min_length=1, pattern=r"^\S+$")
    initial_message: str = Field(min_length=1, pattern=r"\S")
    source_case_id: str | None = None
    immutable: bool
    simulation_mode: Literal["standard", "complete", "degraded", "custom_seed"] = (
        "standard"
    )
    seed: str | None = None
    created_at: datetime


class CreateCaseRequest(DemoModel):
    source_case_id: str | None = Field(default=None, pattern=r"^case_\S+$")
    company_id: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    requester_id: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    asset_id: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    message: str | None = Field(default=None, min_length=1, pattern=r"\S")
    simulation_mode: Literal["standard", "complete", "degraded", "custom_seed"] = (
        "standard"
    )
    seed: str | None = Field(
        default=None, min_length=1, max_length=100, pattern=r"^\S+$"
    )

    @model_validator(mode="after")
    def exactly_one_mode(self) -> "CreateCaseRequest":
        custom = (self.company_id, self.requester_id, self.asset_id, self.message)
        if self.simulation_mode == "custom_seed" and self.seed is None:
            raise ValueError("seed explícita é obrigatória no modo custom_seed")
        if self.simulation_mode != "custom_seed" and self.seed is not None:
            raise ValueError("seed explícita só é aceita no modo custom_seed")
        if self.source_case_id is not None:
            return self
        if any(value is None for value in custom):
            raise ValueError(
                "caso personalizado exige empresa, pessoa, ativo e mensagem"
            )
        return self


class CaseMessage(DemoModel):
    id: str = Field(pattern=r"^msg_\S+$")
    case_id: str = Field(pattern=r"^case_\S+$")
    persona_id: str = Field(min_length=1, pattern=r"^\S+$")
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, pattern=r"\S")
    created_at: datetime


class Execution(DemoModel):
    id: str = Field(pattern=r"^exec_\S+$")
    case_id: str = Field(pattern=r"^case_\S+$")
    message_id: str = Field(pattern=r"^msg_\S+$")
    status: ExecutionStatus
    provider: str | None = None
    fallback_reason: str | None = None
    trace_id: str | None = None
    error_code: str | None = None
    attempt: int = Field(default=0, ge=0)
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    resume_decision_id: str | None = None
    resume_kind: str | None = None
    created_at: datetime
    updated_at: datetime


class CaseEvent(DemoModel):
    id: int = Field(gt=0)
    case_id: str
    ordinal: int = Field(gt=0)
    kind: str = Field(min_length=1, pattern=r"^\S+$")
    payload: dict[str, object]
    created_at: datetime


class EnqueueMessageRequest(DemoModel):
    persona_id: str = Field(min_length=1, pattern=r"^\S+$")
    content: str = Field(min_length=1, max_length=12_000, pattern=r"\S")
    idempotency_key: str = Field(min_length=1, max_length=255, pattern=r"^\S+$")


class EnqueueMessageResponse(DemoModel):
    message: CaseMessage
    execution: Execution


class Persona(DemoModel):
    id: str = Field(min_length=1, pattern=r"^\S+$")
    name: str = Field(min_length=1, pattern=r"\S")
    profile: Literal["requester", "tractian", "authority"]
    company_id: str | None
    permissions: frozenset[str]


class CaseDetail(DemoModel):
    case: DemoCase
    messages: tuple[CaseMessage, ...]
    executions: tuple[Execution, ...]


class DemoConfig(DemoModel):
    mode: Literal["live"] = "live"
    warning: str = "Demonstração com dados e identidades simulados."
    industrial_api: Literal["configured"] = "configured"
    primary_provider: str
    fallback_provider: str
    slack_configured: bool


class AgentRunProjection(DemoModel):
    """Allowlist pública; nunca recebe o estado completo do LangGraph."""

    assistant_message: str = Field(min_length=1, pattern=r"\S")
    decision: Literal[
        "guide",
        "act",
        "escalate",
        "request_information",
        "request_confirmation",
        "require_human_review",
    ]
    trace_id: str = Field(min_length=1, pattern=r"^\S+$")
    provider: Literal["groq", "nvidia-nim"]
    fallback_reason: Literal["timeout", "rate_limit", "network", "server"] | None
    evidence_count: int = Field(ge=0)
    limitation_count: int = Field(ge=0)
    tool_names: tuple[str, ...]
    decision_candidate: "DecisionCandidate | None" = None


class DecisionCandidate(DemoModel):
    audience: Literal["requester", "tractian", "authority"]
    kind: Literal["action_confirmation", "action_authorization", "technical_review"]
    summary: str = Field(min_length=1, max_length=300, pattern=r"\S")
    scope: dict[str, object]
    required_permission: str | None = None
    resume_kind: Literal[
        "confirmation", "delegated_action", "technical_review", "acknowledgement"
    ]
    expires_at: datetime


class DecisionRequest(DemoModel):
    id: str = Field(pattern=r"^decision_\S+$")
    case_id: str
    execution_id: str
    company_id: str
    audience: Literal["requester", "tractian", "authority"]
    kind: str
    status: DecisionStatus
    subject_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    summary: str
    scope: dict[str, object]
    required_permission: str | None
    resume_kind: str
    allowed_operations: tuple[Literal["approve", "reject"], ...] = ("approve", "reject")
    expires_at: datetime
    created_at: datetime
    resolved_at: datetime | None = None
    resolved_by: str | None = None


class ResolveDecisionRequest(DemoModel):
    persona_id: str = Field(min_length=1, pattern=r"^\S+$")
    resolution: Literal["approve", "reject"]


class OutboxEvent(DemoModel):
    id: str = Field(pattern=r"^notification_\S+$")
    decision_id: str
    audience: Literal["tractian", "authority"]
    status: DeliveryStatus
    payload: dict[str, object]
    attempt: int = Field(ge=0)
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    external_id: str | None = None
    created_at: datetime
    updated_at: datetime


class RetryNotificationRequest(DemoModel):
    persona_id: str = Field(min_length=1, pattern=r"^\S+$")
