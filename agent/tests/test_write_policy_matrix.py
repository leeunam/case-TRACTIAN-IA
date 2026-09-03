import pytest
from pydantic import ValidationError

from tractian_agent.write_policy import (
    ApprovalSource,
    EscalateCaseProposal,
    PolicyDecision,
    PolicyReason,
    ReprocessProposal,
    RequestModelRetrainingProposal,
    RequestSpecialistAnalysisProposal,
    TrustedActionApproval,
    TrustedWriteContext,
    UpdateAssetCriticalityProposal,
    WriteMaterialParameters,
    canonical_write_payload_hash,
    evaluate_write_policy,
)


@pytest.mark.parametrize(
    ("proposal", "trusted_context", "changed_proposal", "changed_context"),
    [
        (
            ReprocessProposal(
                analysis_id="an_9906",
                justification="Há dados novos para reprocessar esta análise.",
            ),
            TrustedWriteContext(
                central_asset_id="asset_G501",
                current_case_id="case_tkt_inv_04",
                configured_model_id="mdl_vib_v3",
            ),
            ReprocessProposal(
                analysis_id="an_9910",
                justification="Há dados novos para reprocessar esta análise.",
            ),
            None,
        ),
        (
            RequestSpecialistAnalysisProposal(
                analysis_id="an_9906",
                justification="A limitação registrada exige análise especializada.",
            ),
            TrustedWriteContext(
                central_asset_id="asset_G501",
                current_case_id="case_tkt_inv_04",
                configured_model_id="mdl_vib_v3",
            ),
            RequestSpecialistAnalysisProposal(
                analysis_id="an_9910",
                justification="A limitação registrada exige análise especializada.",
            ),
            None,
        ),
        (
            UpdateAssetCriticalityProposal(
                criticality="high",
                justification="O impacto operacional exige criticidade mais alta.",
            ),
            TrustedWriteContext(
                central_asset_id="asset_G501",
                current_case_id="case_tkt_inv_04",
                configured_model_id="mdl_vib_v3",
            ),
            UpdateAssetCriticalityProposal(
                criticality="critical",
                justification="O impacto operacional exige criticidade mais alta.",
            ),
            TrustedWriteContext(
                central_asset_id="asset_G502",
                current_case_id="case_tkt_inv_04",
                configured_model_id="mdl_vib_v3",
            ),
        ),
        (
            RequestModelRetrainingProposal(
                justification="Erros sistemáticos sustentam solicitar novo treinamento.",
            ),
            TrustedWriteContext(
                central_asset_id="asset_G501",
                current_case_id="case_tkt_inv_04",
                configured_model_id="mdl_vib_v3",
            ),
            None,
            TrustedWriteContext(
                central_asset_id="asset_G501",
                current_case_id="case_tkt_inv_04",
                configured_model_id="mdl_vib_v4",
            ),
        ),
        (
            EscalateCaseProposal(
                justification="O caso ultrapassa o atendimento remoto disponível.",
            ),
            TrustedWriteContext(
                central_asset_id="asset_G501",
                current_case_id="case_tkt_inv_04",
                configured_model_id="mdl_vib_v3",
            ),
            None,
            TrustedWriteContext(
                central_asset_id="asset_G501",
                current_case_id="case_tkt_inv_05",
                configured_model_id="mdl_vib_v3",
            ),
        ),
    ],
)
def test_structural_hash_binds_each_action_target_and_material_body(
    proposal,
    trusted_context,
    changed_proposal,
    changed_context,
):
    original_hash = canonical_write_payload_hash(
        proposal,
        trusted_context=trusted_context,
    )

    changed_hash = canonical_write_payload_hash(
        proposal if changed_proposal is None else changed_proposal,
        trusted_context=(
            trusted_context if changed_context is None else changed_context
        ),
    )

    assert original_hash != changed_hash


def test_structural_hash_distinguishes_action_kind_for_the_same_target():
    context = TrustedWriteContext(
        central_asset_id="asset_G501",
        current_case_id="case_tkt_inv_04",
        configured_model_id="mdl_vib_v3",
    )
    justification = "A mesma análise exige uma ação industrial justificada."

    reprocess_hash = canonical_write_payload_hash(
        ReprocessProposal(
            analysis_id="an_9906",
            justification=justification,
        ),
        trusted_context=context,
    )
    specialist_hash = canonical_write_payload_hash(
        RequestSpecialistAnalysisProposal(
            analysis_id="an_9906",
            justification=justification,
        ),
        trusted_context=context,
    )

    assert reprocess_hash != specialist_hash


def test_allows_criticality_update_only_for_the_exact_trusted_scope():
    proposal = UpdateAssetCriticalityProposal(
        criticality="critical",
        justification="Ativo central sustenta uma elevação de criticidade.",
    )
    trusted_context = TrustedWriteContext(
        central_asset_id="asset_G501",
        current_case_id="case_tkt_inv_04",
        configured_model_id="mdl_vib_v3",
    )
    approval = TrustedActionApproval(
        action="update_asset_criticality",
        target_id="asset_G501",
        material_parameters=WriteMaterialParameters(criticality="critical"),
        source=ApprovalSource.CONFIRMATION,
    )

    result = evaluate_write_policy(
        proposal,
        permissions=frozenset({"read", "action_high"}),
        approval=approval,
        trusted_context=trusted_context,
    )

    assert result.decision is PolicyDecision.ALLOW
    assert result.reason is PolicyReason.AUTHORIZED


@pytest.mark.parametrize(
    ("proposal", "permission", "action", "target_id", "material_parameters"),
    [
        (
            ReprocessProposal(
                analysis_id="an_9906",
                justification="Há dados novos para reprocessar esta análise.",
            ),
            "action_low",
            "reprocess_analysis",
            "an_9906",
            WriteMaterialParameters(),
        ),
        (
            RequestSpecialistAnalysisProposal(
                analysis_id="an_9906",
                justification="A limitação registrada exige análise especializada.",
            ),
            "action_low",
            "request_specialist_analysis",
            "an_9906",
            WriteMaterialParameters(),
        ),
        (
            UpdateAssetCriticalityProposal(
                criticality="high",
                justification="O impacto operacional exige criticidade mais alta.",
            ),
            "action_high",
            "update_asset_criticality",
            "asset_G501",
            WriteMaterialParameters(criticality="high"),
        ),
        (
            RequestModelRetrainingProposal(
                justification="Erros sistemáticos sustentam solicitar novo treinamento.",
            ),
            "action_high",
            "request_model_retraining",
            "mdl_vib_v3",
            WriteMaterialParameters(),
        ),
        (
            EscalateCaseProposal(
                justification="O caso ultrapassa o atendimento remoto disponível.",
            ),
            "escalate",
            "escalate_case",
            "case_tkt_inv_04",
            WriteMaterialParameters(),
        ),
    ],
)
def test_permission_matrix_allows_each_proposal_for_its_exact_scope(
    proposal,
    permission,
    action,
    target_id,
    material_parameters,
):
    trusted_context = TrustedWriteContext(
        central_asset_id="asset_G501",
        current_case_id="case_tkt_inv_04",
        configured_model_id="mdl_vib_v3",
    )
    approval = TrustedActionApproval(
        action=action,
        target_id=target_id,
        material_parameters=material_parameters,
        source=ApprovalSource.ORIGINAL_REQUEST,
    )

    result = evaluate_write_policy(
        proposal,
        permissions=frozenset({"read", permission}),
        approval=approval,
        trusted_context=trusted_context,
    )

    assert result.decision is PolicyDecision.ALLOW
    assert result.reason is PolicyReason.AUTHORIZED


@pytest.mark.parametrize(
    ("action", "target_id", "material_parameters"),
    [
        ("reprocess_analysis", "mdl_vib_v3", WriteMaterialParameters()),
        (
            "request_specialist_analysis",
            "asset_G501",
            WriteMaterialParameters(),
        ),
        (
            "update_asset_criticality",
            "an_9906",
            WriteMaterialParameters(criticality="high"),
        ),
        ("request_model_retraining", "case_tkt_inv_04", WriteMaterialParameters()),
        ("escalate_case", "mdl_vib_v3", WriteMaterialParameters()),
    ],
)
def test_trusted_approval_rejects_target_id_from_another_action_kind(
    action,
    target_id,
    material_parameters,
):
    with pytest.raises(ValidationError):
        TrustedActionApproval(
            action=action,
            target_id=target_id,
            material_parameters=material_parameters,
            source=ApprovalSource.CONFIRMATION,
        )


@pytest.mark.parametrize(
    ("permissions", "justification", "approval", "expected_decision", "reason"),
    [
        (
            frozenset({"read"}),
            "curta",
            None,
            PolicyDecision.DENY,
            PolicyReason.MISSING_PERMISSION,
        ),
        (
            frozenset({"read", "escalate"}),
            "  aaaaaaaaaaaaaaaaaaa  ",
            None,
            PolicyDecision.DENY,
            PolicyReason.INVALID_JUSTIFICATION,
        ),
        (
            frozenset({"read", "escalate"}),
            "  aaaaaaaaaaaaaaaaaaaa  ",
            None,
            PolicyDecision.REQUIRE_CONFIRMATION,
            PolicyReason.EXPLICIT_APPROVAL_REQUIRED,
        ),
    ],
)
def test_policy_applies_permission_justification_and_approval_in_fixed_order(
    permissions,
    justification,
    approval,
    expected_decision,
    reason,
):
    result = evaluate_write_policy(
        EscalateCaseProposal(justification=justification),
        permissions=permissions,
        approval=approval,
        trusted_context=TrustedWriteContext(
            central_asset_id="asset_G501",
            current_case_id="case_tkt_inv_04",
            configured_model_id="mdl_vib_v3",
        ),
    )

    assert result.decision is expected_decision
    assert result.reason is reason


@pytest.mark.parametrize(
    ("proposal", "approval", "trusted_context"),
    [
        (
            ReprocessProposal(
                analysis_id="an_9906",
                justification="Há dados novos para reprocessar esta análise.",
            ),
            TrustedActionApproval(
                action="request_specialist_analysis",
                target_id="an_9906",
                source=ApprovalSource.CONFIRMATION,
            ),
            None,
        ),
        (
            ReprocessProposal(
                analysis_id="an_9907",
                justification="Há dados novos para reprocessar esta análise.",
            ),
            TrustedActionApproval(
                action="reprocess_analysis",
                target_id="an_9906",
                source=ApprovalSource.CONFIRMATION,
            ),
            None,
        ),
        (
            UpdateAssetCriticalityProposal(
                criticality="critical",
                justification="O impacto operacional exige criticidade mais alta.",
            ),
            TrustedActionApproval(
                action="update_asset_criticality",
                target_id="asset_G501",
                material_parameters=WriteMaterialParameters(criticality="high"),
                source=ApprovalSource.CONFIRMATION,
            ),
            TrustedWriteContext(
                central_asset_id="asset_G501",
                current_case_id="case_tkt_inv_04",
                configured_model_id="mdl_vib_v3",
            ),
        ),
        (
            UpdateAssetCriticalityProposal(
                criticality="high",
                justification="O impacto operacional exige criticidade mais alta.",
            ),
            TrustedActionApproval(
                action="update_asset_criticality",
                target_id="asset_G501",
                material_parameters=WriteMaterialParameters(criticality="high"),
                source=ApprovalSource.CONFIRMATION,
            ),
            TrustedWriteContext(
                central_asset_id="asset_G502",
                current_case_id="case_tkt_inv_04",
                configured_model_id="mdl_vib_v3",
            ),
        ),
        (
            RequestModelRetrainingProposal(
                justification="Erros sistemáticos sustentam solicitar novo treinamento.",
            ),
            TrustedActionApproval(
                action="request_model_retraining",
                target_id="mdl_vib_v3",
                source=ApprovalSource.CONFIRMATION,
            ),
            TrustedWriteContext(
                central_asset_id="asset_G501",
                current_case_id="case_tkt_inv_04",
                configured_model_id="mdl_vib_v4",
            ),
        ),
        (
            EscalateCaseProposal(
                justification="O caso ultrapassa o atendimento remoto disponível.",
            ),
            TrustedActionApproval(
                action="escalate_case",
                target_id="case_tkt_inv_04",
                source=ApprovalSource.CONFIRMATION,
            ),
            TrustedWriteContext(
                central_asset_id="asset_G501",
                current_case_id="case_tkt_inv_05",
                configured_model_id="mdl_vib_v3",
            ),
        ),
    ],
)
def test_policy_requires_confirmation_for_any_scope_divergence(
    proposal,
    approval,
    trusted_context,
):
    permission = {
        "reprocess_analysis": "action_low",
        "update_asset_criticality": "action_high",
        "request_model_retraining": "action_high",
        "escalate_case": "escalate",
    }[proposal.action]
    proposal_before = proposal.model_dump_json()
    approval_before = approval.model_dump_json()

    result = evaluate_write_policy(
        proposal,
        permissions=frozenset({"read", permission}),
        approval=approval,
        trusted_context=trusted_context,
    )

    assert result.decision is PolicyDecision.REQUIRE_CONFIRMATION
    assert result.reason is PolicyReason.APPROVAL_SCOPE_MISMATCH
    assert proposal.model_dump_json() == proposal_before
    assert approval.model_dump_json() == approval_before
