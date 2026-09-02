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
from tractian_agent.tools.identifiers import AnalysisId, AssetId, ModelId
from tractian_agent.write_policy import (
    AssetCriticality,
    PolicyDecision,
    PolicyReason,
    TrustedWriteContext,
    WriteMaterialParameters,
    WritePolicyResult,
    WriteProposal,
    canonical_write_payload_hash,
    resolve_action_scope,
)


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


class ConfirmationReply(StrictModel):
    """Resposta pública mínima para um interrupt de confirmação."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent_id: str = Field(min_length=1, pattern=r"^\S+$")
    decision: Literal["approve", "deny"]


class ReprocessIntentScope(StrictModel):
    """Escopo persistido do reprocessamento idempotente."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["reprocess_analysis"]
    case_id: str = Field(min_length=1, pattern=r"^\S+$")
    company_id: str = Field(min_length=1, pattern=r"^\S+$")
    user_id: str = Field(min_length=1, pattern=r"^\S+$")
    analysis_id: AnalysisId
    justification: str = Field(min_length=1, pattern=r"\S")


class RequestSpecialistAnalysisIntentScope(StrictModel):
    """Escopo persistido da solicitação de análise especializada."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["request_specialist_analysis"]
    case_id: str = Field(min_length=1, pattern=r"^\S+$")
    company_id: str = Field(min_length=1, pattern=r"^\S+$")
    user_id: str = Field(min_length=1, pattern=r"^\S+$")
    analysis_id: AnalysisId
    justification: str = Field(min_length=1, pattern=r"\S")


class UpdateAssetCriticalityIntentScope(StrictModel):
    """Escopo persistido da atualização de criticidade do ativo central."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["update_asset_criticality"]
    case_id: str = Field(min_length=1, pattern=r"^\S+$")
    company_id: str = Field(min_length=1, pattern=r"^\S+$")
    user_id: str = Field(min_length=1, pattern=r"^\S+$")
    asset_id: AssetId
    criticality: AssetCriticality
    justification: str = Field(min_length=1, pattern=r"\S")


class RequestModelRetrainingIntentScope(StrictModel):
    """Escopo persistido da solicitação de retreinamento do modelo atual."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["request_model_retraining"]
    case_id: str = Field(min_length=1, pattern=r"^\S+$")
    company_id: str = Field(min_length=1, pattern=r"^\S+$")
    user_id: str = Field(min_length=1, pattern=r"^\S+$")
    model_id: ModelId
    justification: str = Field(min_length=1, pattern=r"\S")


class EscalateCaseIntentScope(StrictModel):
    """Escopo persistido do escalonamento do caso atual."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["escalate_case"]
    case_id: str = Field(min_length=1, pattern=r"^\S+$")
    company_id: str = Field(min_length=1, pattern=r"^\S+$")
    user_id: str = Field(min_length=1, pattern=r"^\S+$")
    justification: str = Field(min_length=1, pattern=r"\S")


WriteIntentScope = Annotated[
    ReprocessIntentScope
    | RequestSpecialistAnalysisIntentScope
    | UpdateAssetCriticalityIntentScope
    | RequestModelRetrainingIntentScope
    | EscalateCaseIntentScope,
    Field(discriminator="action"),
]


def intent_scope_target_id(scope: WriteIntentScope) -> str:
    """Projeta o alvo já persistido sem consultar um runtime novo."""

    if isinstance(
        scope,
        (ReprocessIntentScope, RequestSpecialistAnalysisIntentScope),
    ):
        return scope.analysis_id
    if isinstance(scope, UpdateAssetCriticalityIntentScope):
        return scope.asset_id
    if isinstance(scope, RequestModelRetrainingIntentScope):
        return scope.model_id
    return scope.case_id


def intent_scope_material_parameters(
    scope: WriteIntentScope,
) -> WriteMaterialParameters:
    """Reconstrói somente parâmetros materiais já mostrados e persistidos."""

    if isinstance(scope, UpdateAssetCriticalityIntentScope):
        return WriteMaterialParameters(criticality=scope.criticality)
    return WriteMaterialParameters()


def proposal_matches_intent_scope(
    proposal: WriteProposal,
    scope: WriteIntentScope,
    *,
    payload_hash: str,
    trusted_context: TrustedWriteContext,
) -> bool:
    """Vincula proposal e intent ao alvo resolvido pela fronteira confiável."""
    if (
        proposal.action != scope.action
        or proposal.justification != scope.justification
        or canonical_write_payload_hash(proposal) != payload_hash
    ):
        return False
    expected_scope_type = {
        "reprocess_analysis": ReprocessIntentScope,
        "request_specialist_analysis": RequestSpecialistAnalysisIntentScope,
        "update_asset_criticality": UpdateAssetCriticalityIntentScope,
        "request_model_retraining": RequestModelRetrainingIntentScope,
        "escalate_case": EscalateCaseIntentScope,
    }[proposal.action]
    if not isinstance(scope, expected_scope_type):
        return False
    canonical = resolve_action_scope(
        proposal,
        trusted_context=trusted_context,
    )
    return (
        scope.case_id == trusted_context.current_case_id
        and canonical.target_id == intent_scope_target_id(scope)
        and canonical.material_parameters
        == intent_scope_material_parameters(scope)
    )


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
    request_id: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^\S+$",
    )
    scope: WriteIntentScope
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
    attempts: int = Field(default=0, ge=0, le=2)
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
                PolicyReason.CONFIRMATION_REJECTED,
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

        is_reprocess = isinstance(self.scope, ReprocessIntentScope)
        if not is_reprocess and self.attempts > 1:
            raise ValueError("ação não idempotente aceita no máximo uma tentativa")

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

        if self.prepared_execution_id is None:
            raise ValueError("intenção preparada exige execução")
        if is_reprocess:
            if self.idempotency_key is None or self.expires_at is None:
                raise ValueError(
                    "intenção de reprocesso preparada exige chave e expiração"
                )
        elif self.idempotency_key is not None or self.expires_at is not None:
            raise ValueError(
                "intenção não idempotente preparada proíbe chave e expiração"
            )
        if self.status is IntentStatus.PREPARED:
            if self.attempts != 0 or self.receipt is not None or self.error is not None:
                raise ValueError("status prepared não aceita tentativa ou resultado")
            return self

        if self.status is IntentStatus.COMPLETED:
            if self.attempts < 1:
                raise ValueError("status terminal exige ao menos uma tentativa")
            if (
                self.receipt is None
                or not self.receipt.accepted
                or self.error is not None
            ):
                raise ValueError("status completed exige somente recibo aceito")
            return self

        if self.receipt is not None:
            if (
                self.status is not IntentStatus.FAILED
                or self.receipt.accepted
                or self.error is not None
                or self.attempts < 1
            ):
                raise ValueError(
                    "status failed é o único que aceita recibo rejeitado"
                )
            return self

        if self.error is None:
            raise ValueError("status failed/uncertain exige erro ou recibo rejeitado")
        if self.attempts == 0:
            if is_reprocess:
                expected_codes = {
                    IntentStatus.FAILED: {
                        "AUTHORIZATION_CHANGED_BEFORE_DISPATCH",
                        "IDEMPOTENCY_KEY_EXPIRED",
                        "IDEMPOTENCY_KEY_INTENT_MISMATCH",
                        "PAYLOAD_HASH_MISMATCH",
                        "INTENT_SCOPE_MISMATCH",
                    },
                    IntentStatus.UNCERTAIN: {
                        "AUTHORIZATION_CHANGED_OUTCOME_UNKNOWN",
                        "IDEMPOTENCY_KEY_EXPIRED_OUTCOME_UNKNOWN",
                    },
                }[self.status]
            else:
                expected_codes = {
                    IntentStatus.FAILED: {
                        "AUTHORIZATION_CHANGED_BEFORE_DISPATCH",
                        "PAYLOAD_HASH_MISMATCH",
                        "INTENT_SCOPE_MISMATCH",
                    },
                    IntentStatus.UNCERTAIN: {
                        "NON_IDEMPOTENT_OUTCOME_UNKNOWN_AFTER_RESUME",
                    },
                }[self.status]
            if (
                self.error.code not in expected_codes
                or self.error.category is not ApiErrorCategory.API
            ):
                raise ValueError(
                    "zero tentativa só aceita falha local pré-despacho"
                )
        return self
