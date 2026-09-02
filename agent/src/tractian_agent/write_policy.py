"""Política determinística para propostas de escrita."""

from __future__ import annotations

from enum import Enum
import hashlib
import json
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, TypeAdapter, model_validator

from tractian_agent.contracts import StrictModel
from tractian_agent.tools.identifiers import AnalysisId, AssetId, CaseId, ModelId
from tractian_agent.tools.runtime import Permission


WriteAction = Literal[
    "reprocess_analysis",
    "request_specialist_analysis",
    "update_asset_criticality",
    "request_model_retraining",
    "escalate_case",
]
AssetCriticality = Literal["low", "medium", "high", "critical"]


class ApprovalSource(str, Enum):
    ORIGINAL_REQUEST = "original_request"
    CONFIRMATION = "confirmation"


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    DENY = "deny"


class PolicyReason(str, Enum):
    AUTHORIZED = "authorized"
    MISSING_PERMISSION = "missing_permission"
    EXPLICIT_APPROVAL_REQUIRED = "explicit_approval_required"
    APPROVAL_SCOPE_MISMATCH = "approval_scope_mismatch"
    INVALID_JUSTIFICATION = "invalid_justification"
    CONFIRMATION_REJECTED = "confirmation_rejected"


class ReprocessProposal(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["reprocess_analysis"] = "reprocess_analysis"
    analysis_id: AnalysisId
    justification: str


class RequestSpecialistAnalysisProposal(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["request_specialist_analysis"] = (
        "request_specialist_analysis"
    )
    analysis_id: AnalysisId
    justification: str


class UpdateAssetCriticalityProposal(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["update_asset_criticality"] = "update_asset_criticality"
    criticality: AssetCriticality
    justification: str


class RequestModelRetrainingProposal(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["request_model_retraining"] = "request_model_retraining"
    justification: str


class EscalateCaseProposal(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["escalate_case"] = "escalate_case"
    justification: str


WriteProposal = Annotated[
    ReprocessProposal
    | RequestSpecialistAnalysisProposal
    | UpdateAssetCriticalityProposal
    | RequestModelRetrainingProposal
    | EscalateCaseProposal,
    Field(discriminator="action"),
]


def canonical_write_payload_hash(proposal: WriteProposal) -> str:
    """Gera o hash do corpo que será enviado, sem incluir o alvo confiável."""
    body: dict[str, object]
    if isinstance(proposal, UpdateAssetCriticalityProposal):
        body = {
            "changes": {"criticality": proposal.criticality},
            "justification": proposal.justification,
        }
    else:
        body = {"justification": proposal.justification}
    encoded = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:v1:{hashlib.sha256(encoded).hexdigest()}"


class WriteMaterialParameters(StrictModel):
    """Parâmetros cujo valor faz parte do escopo aprovado."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    criticality: AssetCriticality | None = None


class TrustedWriteContext(StrictModel):
    """Alvos vindos da fronteira confiável, nunca dos argumentos do modelo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    central_asset_id: AssetId
    current_case_id: CaseId
    configured_model_id: ModelId


class TrustedActionApproval(StrictModel):
    """Aprovação criada pela fronteira confiável, nunca pelo modelo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: WriteAction
    target_id: str = Field(min_length=1, pattern=r"^\S+$")
    material_parameters: WriteMaterialParameters = Field(
        default_factory=WriteMaterialParameters
    )
    source: ApprovalSource

    @model_validator(mode="after")
    def _validate_material_parameters(self) -> TrustedActionApproval:
        has_criticality = self.material_parameters.criticality is not None
        if self.action == "update_asset_criticality" and not has_criticality:
            raise ValueError("aprovação de criticidade exige o novo valor")
        if self.action != "update_asset_criticality" and has_criticality:
            raise ValueError("a ação aprovada não aceita criticidade")
        if self.action in {"reprocess_analysis", "request_specialist_analysis"}:
            TypeAdapter(AnalysisId).validate_python(self.target_id)
        elif self.action == "update_asset_criticality":
            TypeAdapter(AssetId).validate_python(self.target_id)
        elif self.action == "request_model_retraining":
            TypeAdapter(ModelId).validate_python(self.target_id)
        else:
            TypeAdapter(CaseId).validate_python(self.target_id)
        return self


class WritePolicyResult(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: PolicyDecision
    reason: PolicyReason


def _result(decision: PolicyDecision, reason: PolicyReason) -> WritePolicyResult:
    return WritePolicyResult(decision=decision, reason=reason)


class CanonicalActionScope(StrictModel):
    """Escopo comparável entre a proposta e a aprovação confiável."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: WriteAction
    target_id: str = Field(min_length=1, pattern=r"^\S+$")
    material_parameters: WriteMaterialParameters = Field(
        default_factory=WriteMaterialParameters
    )


_REQUIRED_PERMISSION: dict[WriteAction, Permission] = {
    "reprocess_analysis": "action_low",
    "request_specialist_analysis": "action_low",
    "update_asset_criticality": "action_high",
    "request_model_retraining": "action_high",
    "escalate_case": "escalate",
}


def resolve_action_scope(
    proposal: WriteProposal,
    *,
    trusted_context: TrustedWriteContext | None = None,
) -> CanonicalActionScope:
    """Resolve alvos ocultos sem alterar a proposta recebida."""

    if isinstance(proposal, (ReprocessProposal, RequestSpecialistAnalysisProposal)):
        return CanonicalActionScope(
            action=proposal.action,
            target_id=proposal.analysis_id,
        )
    if trusted_context is None:
        raise ValueError("o contexto confiável é obrigatório para esta proposta")
    if isinstance(proposal, UpdateAssetCriticalityProposal):
        return CanonicalActionScope(
            action=proposal.action,
            target_id=trusted_context.central_asset_id,
            material_parameters=WriteMaterialParameters(
                criticality=proposal.criticality
            ),
        )
    if isinstance(proposal, RequestModelRetrainingProposal):
        return CanonicalActionScope(
            action=proposal.action,
            target_id=trusted_context.configured_model_id,
        )
    return CanonicalActionScope(
        action=proposal.action,
        target_id=trusted_context.current_case_id,
    )


def evaluate_write_policy(
    proposal: WriteProposal,
    *,
    permissions: frozenset[Permission],
    approval: TrustedActionApproval | None,
    trusted_context: TrustedWriteContext | None = None,
) -> WritePolicyResult:
    """Avalia uma proposta contra permissão, justificativa e escopo aprovado."""

    if _REQUIRED_PERMISSION[proposal.action] not in permissions:
        return _result(PolicyDecision.DENY, PolicyReason.MISSING_PERMISSION)
    if len(proposal.justification.strip()) < 20:
        return _result(PolicyDecision.DENY, PolicyReason.INVALID_JUSTIFICATION)
    if approval is None:
        return _result(
            PolicyDecision.REQUIRE_CONFIRMATION,
            PolicyReason.EXPLICIT_APPROVAL_REQUIRED,
        )
    expected_scope = resolve_action_scope(
        proposal,
        trusted_context=trusted_context,
    )
    approved_scope = CanonicalActionScope(
        action=approval.action,
        target_id=approval.target_id,
        material_parameters=approval.material_parameters,
    )
    if approved_scope != expected_scope:
        return _result(
            PolicyDecision.REQUIRE_CONFIRMATION,
            PolicyReason.APPROVAL_SCOPE_MISMATCH,
        )
    return _result(PolicyDecision.ALLOW, PolicyReason.AUTHORIZED)


def evaluate_reprocess_policy(
    proposal: ReprocessProposal,
    *,
    permissions: frozenset[Permission],
    approval: TrustedActionApproval | None,
) -> WritePolicyResult:
    """Decide se uma proposta de reprocesso pode avançar para execução."""

    return evaluate_write_policy(
        proposal,
        permissions=permissions,
        approval=approval,
    )
