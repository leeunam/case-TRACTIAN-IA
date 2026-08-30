import pytest
from pydantic import ValidationError

from tractian_agent.write_policy import (
    ApprovalSource,
    PolicyDecision,
    PolicyReason,
    ReprocessProposal,
    TrustedActionApproval,
    evaluate_reprocess_policy,
)


def test_allows_exact_explicit_reprocess_with_action_low():
    proposal = ReprocessProposal(
        analysis_id="an_9906",
        justification="Rolamento substituído; solicitar novo processamento.",
    )
    approval = TrustedActionApproval(
        action="reprocess_analysis",
        target_id="an_9906",
        source=ApprovalSource.ORIGINAL_REQUEST,
    )

    result = evaluate_reprocess_policy(
        proposal,
        permissions=frozenset({"read", "action_low"}),
        approval=approval,
    )

    assert result.decision is PolicyDecision.ALLOW
    assert result.reason is PolicyReason.AUTHORIZED


def test_denies_reprocess_without_action_low():
    proposal = ReprocessProposal(
        analysis_id="an_9906",
        justification="Rolamento substituído; solicitar novo processamento.",
    )
    approval = TrustedActionApproval(
        action="reprocess_analysis",
        target_id="an_9906",
        source=ApprovalSource.ORIGINAL_REQUEST,
    )

    result = evaluate_reprocess_policy(
        proposal,
        permissions=frozenset({"read"}),
        approval=approval,
    )

    assert result.decision is PolicyDecision.DENY
    assert result.reason is PolicyReason.MISSING_PERMISSION


def test_requires_confirmation_without_explicit_approval():
    proposal = ReprocessProposal(
        analysis_id="an_9906",
        justification="Rolamento substituído; solicitar novo processamento.",
    )

    result = evaluate_reprocess_policy(
        proposal,
        permissions=frozenset({"read", "action_low"}),
        approval=None,
    )

    assert result.decision is PolicyDecision.REQUIRE_CONFIRMATION
    assert result.reason is PolicyReason.EXPLICIT_APPROVAL_REQUIRED


def test_requires_confirmation_when_approved_target_differs():
    proposal = ReprocessProposal(
        analysis_id="an_9906",
        justification="Rolamento substituído; solicitar novo processamento.",
    )
    approval = TrustedActionApproval(
        action="reprocess_analysis",
        target_id="an_9907",
        source=ApprovalSource.CONFIRMATION,
    )

    result = evaluate_reprocess_policy(
        proposal,
        permissions=frozenset({"read", "action_low"}),
        approval=approval,
    )

    assert result.decision is PolicyDecision.REQUIRE_CONFIRMATION
    assert result.reason is PolicyReason.APPROVAL_SCOPE_MISMATCH


def test_denies_reprocess_with_short_justification():
    proposal = ReprocessProposal(
        analysis_id="an_9906",
        justification="Justificativa curta",
    )
    approval = TrustedActionApproval(
        action="reprocess_analysis",
        target_id="an_9906",
        source=ApprovalSource.ORIGINAL_REQUEST,
    )

    result = evaluate_reprocess_policy(
        proposal,
        permissions=frozenset({"read", "action_low"}),
        approval=approval,
    )

    assert result.decision is PolicyDecision.DENY
    assert result.reason is PolicyReason.INVALID_JUSTIFICATION


def test_policy_contract_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ReprocessProposal(
            analysis_id="an_9906",
            justification="Rolamento substituído; solicitar novo processamento.",
            expected_path=["POST /analyses/an_9906/reprocess"],
        )
