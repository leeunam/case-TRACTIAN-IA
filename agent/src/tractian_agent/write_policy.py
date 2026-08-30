"""Política determinística para propostas de escrita."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import ConfigDict

from tractian_agent.contracts import StrictModel
from tractian_agent.tools.identifiers import AnalysisId
from tractian_agent.tools.runtime import Permission


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


class ReprocessProposal(StrictModel):
    analysis_id: AnalysisId
    justification: str


class TrustedActionApproval(StrictModel):
    """Aprovação criada pela fronteira confiável, nunca pelo modelo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["reprocess_analysis"]
    target_id: AnalysisId
    source: ApprovalSource


class WritePolicyResult(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: PolicyDecision
    reason: PolicyReason


def evaluate_reprocess_policy(
    proposal: ReprocessProposal,
    *,
    permissions: frozenset[Permission],
    approval: TrustedActionApproval | None,
) -> WritePolicyResult:
    """Decide se uma proposta de reprocesso pode avançar para execução."""

    if "action_low" not in permissions:
        return WritePolicyResult(
            decision=PolicyDecision.DENY,
            reason=PolicyReason.MISSING_PERMISSION,
        )
    if len(proposal.justification.strip()) < 20:
        return WritePolicyResult(
            decision=PolicyDecision.DENY,
            reason=PolicyReason.INVALID_JUSTIFICATION,
        )
    if approval is None:
        return WritePolicyResult(
            decision=PolicyDecision.REQUIRE_CONFIRMATION,
            reason=PolicyReason.EXPLICIT_APPROVAL_REQUIRED,
        )
    if approval.target_id != proposal.analysis_id:
        return WritePolicyResult(
            decision=PolicyDecision.REQUIRE_CONFIRMATION,
            reason=PolicyReason.APPROVAL_SCOPE_MISMATCH,
        )
    return WritePolicyResult(
        decision=PolicyDecision.ALLOW,
        reason=PolicyReason.AUTHORIZED,
    )
