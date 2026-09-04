from datetime import datetime, timedelta, timezone

from tractian_agent.write_policy import (
    ApprovalSource,
    DelegatedApprovalAttestation,
    PolicyDecision,
    TrustedActionApproval,
    TrustedWriteContext,
    UpdateAssetCriticalityProposal,
    WriteMaterialParameters,
    delegated_subject_digest,
    evaluate_write_policy,
)


def test_exact_delegated_approval_supplies_required_permission() -> None:
    now = datetime.now(timezone.utc)
    scope = {
        "action": "update_asset_criticality",
        "target_id": "asset_M101",
        "material_parameters": {"criticality": "critical"},
        "company_id": "comp_forja_br",
    }
    approval = TrustedActionApproval(
        action="update_asset_criticality",
        target_id="asset_M101",
        material_parameters=WriteMaterialParameters(criticality="critical"),
        source=ApprovalSource.DELEGATED,
        delegation=DelegatedApprovalAttestation(
            decision_id="decision_1",
            approver_id="usr_ana",
            company_id="comp_forja_br",
            permission="action_high",
            subject_digest=delegated_subject_digest(scope),
            approved_at=now,
            expires_at=now + timedelta(hours=1),
        ),
    )
    result = evaluate_write_policy(
        UpdateAssetCriticalityProposal(
            criticality="critical",
            justification="Mudança aprovada pela autoridade responsável.",
        ),
        permissions=frozenset({"read"}),
        approval=approval,
        trusted_context=TrustedWriteContext(
            central_asset_id="asset_M101",
            current_case_id="case_tkt_ctx_01",
            configured_model_id="mdl_vib_v3",
        ),
    )
    assert result.decision is PolicyDecision.ALLOW


def test_delegated_approval_cannot_supply_wrong_permission() -> None:
    now = datetime.now(timezone.utc)
    approval = TrustedActionApproval(
        action="update_asset_criticality",
        target_id="asset_M101",
        material_parameters=WriteMaterialParameters(criticality="critical"),
        source=ApprovalSource.DELEGATED,
        delegation=DelegatedApprovalAttestation(
            decision_id="decision_1",
            approver_id="usr_lucas",
            company_id="comp_forja_br",
            permission="action_low",
            subject_digest=delegated_subject_digest(
                {
                    "action": "update_asset_criticality",
                    "target_id": "asset_M101",
                    "material_parameters": {"criticality": "critical"},
                    "company_id": "comp_forja_br",
                }
            ),
            approved_at=now,
            expires_at=now + timedelta(hours=1),
        ),
    )
    result = evaluate_write_policy(
        UpdateAssetCriticalityProposal(
            criticality="critical",
            justification="Mudança aprovada pela autoridade responsável.",
        ),
        permissions=frozenset({"read"}),
        approval=approval,
        trusted_context=TrustedWriteContext(
            central_asset_id="asset_M101",
            current_case_id="case_tkt_ctx_01",
            configured_model_id="mdl_vib_v3",
        ),
    )
    assert result.decision is PolicyDecision.DENY
