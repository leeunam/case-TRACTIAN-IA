from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tractian_agent.contracts import (
    ActionReceipt,
    ApiError,
    ApiErrorCategory,
    Identity,
    ResponseMode,
    SupportRequest,
)
from tractian_agent.evidence import canonical_evidence_id, compile_action_intents
from tractian_agent.release_gate import (
    ReleaseGateContext,
    evaluate_release,
    render_non_release_result,
    render_released_result,
)
from tractian_agent.state import (
    AgentDecision,
    AgentState,
    EvidenceItem,
    EvidenceConflict,
    EvidenceGap,
    EvidenceGapReason,
    EvidenceLedger,
    EvidenceQuality,
    EvidenceObsolescenceReason,
    EvidenceSourceKind,
    JsonSnapshot,
    PlannerTerminalRecord,
    ReleaseGateOutcome,
    ReleaseGateReason,
    ResumeAnchor,
    ThreadScope,
    WriterDraft,
    WriterNextStep,
)
from tractian_agent.tools.runtime import TrustedIdentity
from tractian_agent.write_contracts import (
    IntentStatus,
    ReprocessIntentScope,
    UpdateAssetCriticalityIntentScope,
    WriteIntent,
)
from tractian_agent.write_policy import (
    ApprovalSource,
    PolicyDecision,
    PolicyReason,
    ReprocessProposal,
    TrustedActionApproval,
    TrustedWriteContext,
    UpdateAssetCriticalityProposal,
    WritePolicyResult,
    canonical_write_payload_hash,
)
from tractian_agent.writer import build_writer_context


def _claimable_ledger() -> EvidenceLedger:
    item = EvidenceItem(
        evidence_id="sha256:v1:" + "b" * 64,
        request_id="req_gate_01",
        source_kind=EvidenceSourceKind.TOOL,
        call_id="call_gate_01",
        tool="get_asset",
        resource="/assets/asset_G501",
        fact_path="asset.criticality",
        value=JsonSnapshot.capture("high", forbidden_names=frozenset()),
        mode=ResponseMode.COMPLETE,
        source_at=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
        recorded_at=datetime(2026, 9, 2, 12, 1, tzinfo=timezone.utc),
        quality=EvidenceQuality.CLAIMABLE,
    )
    item = item.model_copy(update={"evidence_id": canonical_evidence_id(item)})
    return EvidenceLedger(
        request_id="req_gate_01",
        items=(item,),
    )


def _draft_for(
    decision: AgentDecision,
    ledger: EvidenceLedger,
    *,
    missing_information: str | None = None,
) -> WriterDraft:
    writer_context = build_writer_context(
        decision=decision,
        ledger=ledger,
        missing_information=missing_information,
    )
    return WriterDraft(
        decision=decision,
        evidence_ids=tuple(fact.evidence_id for fact in writer_context.facts),
        limitation_refs=tuple(
            limitation.limitation_ref for limitation in writer_context.limitations
        ),
        next_step={
            AgentDecision.GUIDE: WriterNextStep.MONITOR,
            AgentDecision.ACT: WriterNextStep.VERIFY_ACTION,
            AgentDecision.ESCALATE: WriterNextStep.AWAIT_ESCALATION,
            AgentDecision.REQUEST_INFORMATION: WriterNextStep.PROVIDE_INFORMATION,
            AgentDecision.REQUEST_CONFIRMATION: WriterNextStep.CONFIRM_ACTION,
            AgentDecision.REQUIRE_HUMAN_REVIEW: WriterNextStep.AWAIT_HUMAN_REVIEW,
        }[decision],
    )


def _trusted_write_context() -> TrustedWriteContext:
    return TrustedWriteContext(
        central_asset_id="asset_G501",
        current_case_id="case_tkt_inv_04",
        configured_model_id="mdl_vib_v3",
    )


def _completed_update_action() -> tuple[
    UpdateAssetCriticalityProposal,
    TrustedActionApproval,
    WriteIntent,
    EvidenceLedger,
]:
    proposal = UpdateAssetCriticalityProposal(
        criticality="critical",
        justification="O impacto operacional exige prioridade máxima.",
    )
    approval = TrustedActionApproval(
        action=proposal.action,
        target_id="asset_G501",
        material_parameters={"criticality": "critical"},
        source=ApprovalSource.ORIGINAL_REQUEST,
    )
    intent = WriteIntent(
        intent_id="intent_gate_action",
        request_id="req_gate_action",
        scope=UpdateAssetCriticalityIntentScope(
            action=proposal.action,
            case_id="case_tkt_inv_04",
            company_id="comp_mineracao_andes",
            user_id="usr_pedro",
            asset_id="asset_G501",
            criticality="critical",
            justification=proposal.justification,
        ),
        payload_hash=canonical_write_payload_hash(
            proposal,
            trusted_context=_trusted_write_context(),
        ),
        decision=WritePolicyResult(
            decision=PolicyDecision.ALLOW,
            reason=PolicyReason.AUTHORIZED,
        ),
        status=IntentStatus.COMPLETED,
        prepared_execution_id="exec_gate_action",
        attempts=1,
        receipt=ActionReceipt(
            accepted=True,
            action_id="act_gate_action",
            message="Mensagem externa que não vira fato.",
        ),
    )
    ledger = compile_action_intents(
        (intent,),
        recorded_at=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
    )
    return proposal, approval, intent, ledger


def _released_action_state() -> AgentState:
    proposal, approval, intent, ledger = _completed_update_action()
    draft = _draft_for(AgentDecision.ACT, ledger)
    gate_context = ReleaseGateContext(
        request_id="req_gate_action",
        decision=AgentDecision.ACT,
        ledger=ledger,
        draft=draft,
        permissions=frozenset({"action_high"}),
        trusted_write_context=_trusted_write_context(),
        intents=(intent,),
        proposal=proposal,
        approval=approval,
    )
    attestation = evaluate_release(gate_context)
    result = render_released_result(gate_context, attestation)
    request = SupportRequest(
        case_id="case_tkt_inv_04",
        ticket_id="TKT-INV-04",
        asset_id="asset_G501",
        message="Atualize a criticidade do ativo central.",
        identity=Identity(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
        ),
    )
    return AgentState(
        request=request,
        identity=TrustedIdentity.model_validate(request.identity.model_dump()),
        permissions=frozenset({"action_high"}),
        request_id="req_gate_action",
        thread_id="thread_gate_action",
        execution_id="exec_gate_action",
        thread_scope=ThreadScope(
            thread_id="thread_gate_action",
            case_id=request.case_id,
            company_id=request.identity.company_id,
            user_id=request.identity.user_id,
        ),
        trusted_write_context=_trusted_write_context(),
        ledger=ledger,
        decision=AgentDecision.ACT,
        step_count=9,
        step_limit=24,
        pending_proposal=proposal,
        approval=approval,
        intents=(intent,),
        final_result=result,
        writer_draft=draft,
        writer_attempts=1,
        release_gate=attestation,
        resume_anchor=ResumeAnchor.RELEASE_GATE,
    )


def test_gate_derives_act_from_the_completed_canonical_action() -> None:
    proposal, approval, intent, ledger = _completed_update_action()
    context = ReleaseGateContext(
        request_id="req_gate_action",
        decision=AgentDecision.ESCALATE,
        ledger=ledger,
        draft=_draft_for(AgentDecision.ESCALATE, ledger),
        permissions=frozenset({"action_high"}),
        trusted_write_context=_trusted_write_context(),
        intents=(intent,),
        proposal=proposal,
        approval=approval,
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.DECISION_MISMATCH


def test_action_with_cited_tool_evidence_requires_read_permission() -> None:
    proposal, approval, intent, action_ledger = _completed_update_action()
    technical = _claimable_ledger().items[0].model_copy(
        update={"request_id": "req_gate_action"}
    )
    technical = technical.model_copy(
        update={"evidence_id": canonical_evidence_id(technical)}
    )
    ledger = EvidenceLedger(
        request_id="req_gate_action",
        items=tuple(
            sorted(
                (*action_ledger.items, technical),
                key=lambda item: item.evidence_id,
            )
        ),
    )
    context = ReleaseGateContext(
        request_id="req_gate_action",
        decision=AgentDecision.ACT,
        ledger=ledger,
        draft=_draft_for(AgentDecision.ACT, ledger),
        permissions=frozenset({"action_high"}),
        trusted_write_context=_trusted_write_context(),
        intents=(intent,),
        proposal=proposal,
        approval=approval,
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.PERMISSION_INCOMPATIBLE


def test_receipt_only_action_does_not_require_read_permission() -> None:
    proposal, approval, intent, ledger = _completed_update_action()
    context = ReleaseGateContext(
        request_id="req_gate_action",
        decision=AgentDecision.ACT,
        ledger=ledger,
        draft=_draft_for(AgentDecision.ACT, ledger),
        permissions=frozenset({"action_high"}),
        trusted_write_context=_trusted_write_context(),
        intents=(intent,),
        proposal=proposal,
        approval=approval,
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.RELEASE


def test_read_permission_revoked_after_draft_blocks_cited_tool_evidence() -> None:
    proposal, approval, intent, action_ledger = _completed_update_action()
    technical = _claimable_ledger().items[0].model_copy(
        update={"request_id": "req_gate_action"}
    )
    technical = technical.model_copy(
        update={"evidence_id": canonical_evidence_id(technical)}
    )
    ledger = EvidenceLedger(
        request_id="req_gate_action",
        items=tuple(
            sorted(
                (*action_ledger.items, technical),
                key=lambda item: item.evidence_id,
            )
        ),
    )
    draft = _draft_for(AgentDecision.ACT, ledger)
    allowed = ReleaseGateContext(
        request_id="req_gate_action",
        decision=AgentDecision.ACT,
        ledger=ledger,
        draft=draft,
        permissions=frozenset({"read", "action_high"}),
        trusted_write_context=_trusted_write_context(),
        intents=(intent,),
        proposal=proposal,
        approval=approval,
    )
    revoked = ReleaseGateContext(
        request_id=allowed.request_id,
        decision=allowed.decision,
        ledger=allowed.ledger,
        draft=draft,
        permissions=frozenset({"action_high"}),
        trusted_write_context=allowed.trusted_write_context,
        intents=allowed.intents,
        proposal=allowed.proposal,
        approval=allowed.approval,
    )

    assert evaluate_release(allowed).outcome is ReleaseGateOutcome.RELEASE
    result = evaluate_release(revoked)
    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.PERMISSION_INCOMPATIBLE


def test_gate_does_not_hide_a_denied_write_intent_behind_read_evidence() -> None:
    proposal, _, completed, _ = _completed_update_action()
    denied = WriteIntent(
        intent_id="intent_gate_denied",
        request_id="req_gate_01",
        scope=completed.scope,
        payload_hash=completed.payload_hash,
        decision=WritePolicyResult(
            decision=PolicyDecision.DENY,
            reason=PolicyReason.MISSING_PERMISSION,
        ),
        status=IntentStatus.DENIED,
    )
    ledger = _claimable_ledger()
    context = ReleaseGateContext(
        request_id="req_gate_01",
        decision=AgentDecision.GUIDE,
        ledger=ledger,
        draft=_draft_for(AgentDecision.GUIDE, ledger),
        permissions=frozenset({"read"}),
        trusted_write_context=_trusted_write_context(),
        intents=(denied,),
        proposal=proposal,
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.INTENT_POLICY_MISMATCH


def test_gate_does_not_release_guide_for_an_orphan_write_proposal() -> None:
    proposal, _, _, _ = _completed_update_action()
    ledger = _claimable_ledger()
    context = ReleaseGateContext(
        request_id="req_gate_01",
        decision=AgentDecision.GUIDE,
        ledger=ledger,
        draft=_draft_for(AgentDecision.GUIDE, ledger),
        permissions=frozenset({"read"}),
        proposal=proposal,
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.INTENT_MISSING


def test_request_confirmation_requires_the_current_awaiting_intent() -> None:
    ledger = EvidenceLedger()
    context = ReleaseGateContext(
        request_id="req_gate_confirmation",
        decision=AgentDecision.REQUEST_CONFIRMATION,
        ledger=ledger,
        draft=_draft_for(AgentDecision.REQUEST_CONFIRMATION, ledger),
        permissions=frozenset({"action_high"}),
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.INTENT_MISSING


def test_request_confirmation_requires_the_action_permission() -> None:
    proposal, _, completed, _ = _completed_update_action()
    awaiting = WriteIntent(
        intent_id="intent_gate_confirmation",
        request_id="req_gate_confirmation",
        scope=completed.scope,
        payload_hash=completed.payload_hash,
        decision=WritePolicyResult(
            decision=PolicyDecision.REQUIRE_CONFIRMATION,
            reason=PolicyReason.EXPLICIT_APPROVAL_REQUIRED,
        ),
        status=IntentStatus.AWAITING_CONFIRMATION,
    )
    ledger = EvidenceLedger()
    context = ReleaseGateContext(
        request_id="req_gate_confirmation",
        decision=AgentDecision.REQUEST_CONFIRMATION,
        ledger=ledger,
        draft=_draft_for(AgentDecision.REQUEST_CONFIRMATION, ledger),
        permissions=frozenset(),
        trusted_write_context=_trusted_write_context(),
        intents=(awaiting,),
        proposal=proposal,
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.PERMISSION_INCOMPATIBLE


def test_request_confirmation_accepts_only_a_coherent_awaiting_intent() -> None:
    proposal, _, completed, _ = _completed_update_action()
    awaiting = WriteIntent(
        intent_id="intent_gate_confirmation",
        request_id="req_gate_confirmation",
        scope=completed.scope,
        payload_hash=completed.payload_hash,
        decision=WritePolicyResult(
            decision=PolicyDecision.REQUIRE_CONFIRMATION,
            reason=PolicyReason.EXPLICIT_APPROVAL_REQUIRED,
        ),
        status=IntentStatus.AWAITING_CONFIRMATION,
    )
    ledger = EvidenceLedger()
    context = ReleaseGateContext(
        request_id="req_gate_confirmation",
        decision=AgentDecision.REQUEST_CONFIRMATION,
        ledger=ledger,
        draft=_draft_for(AgentDecision.REQUEST_CONFIRMATION, ledger),
        permissions=frozenset({"action_high"}),
        trusted_write_context=_trusted_write_context(),
        intents=(awaiting,),
        proposal=proposal,
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUEST_CONFIRMATION
    assert result.reason is ReleaseGateReason.CONFIRMATION_REQUIRED


def test_request_confirmation_rejects_an_approval_incompatible_with_policy() -> None:
    proposal, approval, completed, _ = _completed_update_action()
    awaiting = WriteIntent(
        intent_id="intent_gate_confirmation",
        request_id="req_gate_confirmation",
        scope=completed.scope,
        payload_hash=completed.payload_hash,
        decision=WritePolicyResult(
            decision=PolicyDecision.REQUIRE_CONFIRMATION,
            reason=PolicyReason.EXPLICIT_APPROVAL_REQUIRED,
        ),
        status=IntentStatus.AWAITING_CONFIRMATION,
    )
    ledger = EvidenceLedger()
    context = ReleaseGateContext(
        request_id="req_gate_confirmation",
        decision=AgentDecision.REQUEST_CONFIRMATION,
        ledger=ledger,
        draft=_draft_for(AgentDecision.REQUEST_CONFIRMATION, ledger),
        permissions=frozenset({"action_high"}),
        trusted_write_context=_trusted_write_context(),
        intents=(awaiting,),
        proposal=proposal,
        approval=approval,
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.INTENT_POLICY_MISMATCH


def test_release_context_revalidates_the_status_policy_matrix() -> None:
    proposal, _, completed, _ = _completed_update_action()
    awaiting = WriteIntent(
        intent_id="intent_gate_confirmation",
        request_id="req_gate_confirmation",
        scope=completed.scope,
        payload_hash=completed.payload_hash,
        decision=WritePolicyResult(
            decision=PolicyDecision.REQUIRE_CONFIRMATION,
            reason=PolicyReason.EXPLICIT_APPROVAL_REQUIRED,
        ),
        status=IntentStatus.AWAITING_CONFIRMATION,
    ).model_copy(
        update={
            "decision": WritePolicyResult(
                decision=PolicyDecision.ALLOW,
                reason=PolicyReason.AUTHORIZED,
            )
        }
    )
    ledger = EvidenceLedger()

    with pytest.raises(ValidationError, match="status da intenção"):
        ReleaseGateContext(
            request_id="req_gate_confirmation",
            decision=AgentDecision.REQUEST_CONFIRMATION,
            ledger=ledger,
            draft=_draft_for(AgentDecision.REQUEST_CONFIRMATION, ledger),
            permissions=frozenset({"action_high"}),
            trusted_write_context=_trusted_write_context(),
            intents=(awaiting,),
            proposal=proposal,
        )


def test_completed_action_requires_the_exact_canonical_proposal() -> None:
    _, approval, intent, ledger = _completed_update_action()
    divergent = UpdateAssetCriticalityProposal(
        criticality="low",
        justification="A mesma ação com conteúdo material diferente.",
    )
    context = ReleaseGateContext(
        request_id="req_gate_action",
        decision=AgentDecision.ACT,
        ledger=ledger,
        draft=_draft_for(AgentDecision.ACT, ledger),
        permissions=frozenset({"action_high"}),
        trusted_write_context=_trusted_write_context(),
        intents=(intent,),
        proposal=divergent,
        approval=approval,
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.INTENT_POLICY_MISMATCH


def test_completed_action_binds_the_proposal_target_to_the_intent_scope() -> None:
    justification = "A análise precisa ser recalculada com dados corrigidos."
    canonical_proposal = ReprocessProposal(
        analysis_id="an_9906",
        justification=justification,
    )
    intent = WriteIntent(
        intent_id="intent_gate_reprocess",
        request_id="req_gate_reprocess",
        scope=ReprocessIntentScope(
            action="reprocess_analysis",
            case_id="case_tkt_inv_04",
            company_id="comp_mineracao_andes",
            user_id="usr_pedro",
            analysis_id="an_9906",
            justification=justification,
        ),
        payload_hash=canonical_write_payload_hash(canonical_proposal),
        decision=WritePolicyResult(
            decision=PolicyDecision.ALLOW,
            reason=PolicyReason.AUTHORIZED,
        ),
        status=IntentStatus.COMPLETED,
        idempotency_key="tractian-agent:intent_gate_reprocess",
        expires_at=datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
        prepared_execution_id="exec_gate_reprocess",
        attempts=1,
        receipt=ActionReceipt(
            accepted=True,
            action_id="act_gate_reprocess",
            message="Reprocesso aceito.",
        ),
    )
    ledger = compile_action_intents(
        (intent,),
        recorded_at=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
    )
    divergent_proposal = canonical_proposal.model_copy(
        update={"analysis_id": "an_9910"}
    )
    context = ReleaseGateContext(
        request_id="req_gate_reprocess",
        decision=AgentDecision.ACT,
        ledger=ledger,
        draft=_draft_for(AgentDecision.ACT, ledger),
        permissions=frozenset({"action_low"}),
        trusted_write_context=_trusted_write_context(),
        intents=(intent,),
        proposal=divergent_proposal,
        approval=TrustedActionApproval(
            action="reprocess_analysis",
            target_id="an_9906",
            source=ApprovalSource.ORIGINAL_REQUEST,
        ),
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.INTENT_POLICY_MISMATCH


def test_read_decision_must_match_the_canonical_planner_terminal() -> None:
    ledger = _claimable_ledger()
    context = ReleaseGateContext(
        request_id="req_gate_01",
        decision=AgentDecision.GUIDE,
        ledger=ledger,
        draft=_draft_for(AgentDecision.GUIDE, ledger),
        permissions=frozenset({"read"}),
        planner_terminal=PlannerTerminalRecord(
            decision="request_information",
            stop_reason="missing_information",
            missing_information="Informe o ponto de medição.",
        ),
        missing_information="Informe o ponto de medição.",
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.DECISION_MISMATCH


def test_gate_releases_only_the_canonical_facts_selected_by_id() -> None:
    ledger = _claimable_ledger()
    evidence_id = ledger.items[0].evidence_id
    draft = WriterDraft(
        decision=AgentDecision.GUIDE,
        evidence_ids=(evidence_id,),
        limitation_refs=(),
        next_step=WriterNextStep.MONITOR,
    )
    context = ReleaseGateContext(
        request_id="req_gate_01",
        decision=AgentDecision.GUIDE,
        ledger=ledger,
        draft=draft,
        permissions=frozenset({"read"}),
    )

    attestation = evaluate_release(context)
    result = render_released_result(context, attestation)

    assert attestation.outcome is ReleaseGateOutcome.RELEASE
    assert result.decision is AgentDecision.GUIDE
    assert result.evidence_ids == (evidence_id,)
    assert result.limitation_refs == ()
    assert result.next_step is WriterNextStep.MONITOR
    assert 'asset.criticality = "high"' in result.message
    assert "get_asset" in result.message
    assert evidence_id not in result.message


def test_projection_overflow_marker_blocks_release() -> None:
    base = _claimable_ledger()
    items = []
    for index in range(65):
        item = base.items[0].model_copy(
            update={"call_id": f"call_gate_overflow_{index:03d}"}
        )
        items.append(
            item.model_copy(update={"evidence_id": canonical_evidence_id(item)})
        )
    ledger = EvidenceLedger(
        request_id=base.request_id,
        items=tuple(sorted(items, key=lambda item: item.evidence_id)),
    )
    context = ReleaseGateContext(
        request_id="req_gate_01",
        decision=AgentDecision.GUIDE,
        ledger=ledger,
        draft=_draft_for(AgentDecision.GUIDE, ledger),
        permissions=frozenset({"read"}),
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.INSUFFICIENT_EVIDENCE


def test_request_information_keeps_structured_provenance_without_technical_text() -> None:
    ledger = _claimable_ledger()
    missing_information = "Informe o ponto de medição."
    draft = _draft_for(
        AgentDecision.REQUEST_INFORMATION,
        ledger,
        missing_information=missing_information,
    )
    context = ReleaseGateContext(
        request_id="req_gate_01",
        decision=AgentDecision.REQUEST_INFORMATION,
        ledger=ledger,
        draft=draft,
        permissions=frozenset({"read"}),
        missing_information=missing_information,
    )

    attestation = evaluate_release(context)
    result = render_non_release_result(context, attestation)

    assert attestation.outcome is ReleaseGateOutcome.REQUEST_INFORMATION
    assert result.evidence_ids == draft.evidence_ids
    assert result.limitation_refs == draft.limitation_refs
    assert result.next_step is WriterNextStep.PROVIDE_INFORMATION
    assert "high" not in result.message
    assert "get_asset" not in result.message


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("decision", ReleaseGateReason.DECISION_MISMATCH),
        ("unknown_id", ReleaseGateReason.EVIDENCE_REFERENCE_MISMATCH),
        ("omitted_limitation", ReleaseGateReason.LIMITATION_REFERENCE_MISMATCH),
        ("permission", ReleaseGateReason.PERMISSION_INCOMPATIBLE),
    ],
)
def test_gate_blocks_draft_and_permission_mismatches(
    mutation: str,
    expected_reason: ReleaseGateReason,
) -> None:
    ledger = _claimable_ledger()
    if mutation == "omitted_limitation":
        limited = ledger.items[0].model_copy(
            update={"limitations": ("janela reduzida",)}
        )
        limited = limited.model_copy(
            update={"evidence_id": canonical_evidence_id(limited)}
        )
        ledger = EvidenceLedger(
            request_id=ledger.request_id,
            items=(limited,),
        )
    draft = _draft_for(AgentDecision.GUIDE, ledger)
    permissions = frozenset({"read"})
    if mutation == "decision":
        draft = draft.model_copy(
            update={
                "decision": AgentDecision.ACT,
                "next_step": WriterNextStep.VERIFY_ACTION,
            }
        )
    elif mutation == "unknown_id":
        draft = draft.model_copy(
            update={"evidence_ids": ("sha256:v1:" + "f" * 64,)}
        )
    elif mutation == "omitted_limitation":
        draft = draft.model_copy(update={"limitation_refs": ()})
    else:
        permissions = frozenset()
    context = ReleaseGateContext(
        request_id="req_gate_01",
        decision=AgentDecision.GUIDE,
        ledger=ledger,
        draft=draft,
        permissions=permissions,
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is expected_reason
    with pytest.raises(ValueError, match="release"):
        render_released_result(context, result)


@pytest.mark.parametrize("degradation", ["gap", "partial", "obsolete", "conflict"])
def test_gate_blocks_every_current_global_evidence_degradation(
    degradation: str,
) -> None:
    base = _claimable_ledger()
    items = base.items
    gaps: tuple[EvidenceGap, ...] = ()
    conflicts: tuple[EvidenceConflict, ...] = ()
    if degradation == "gap":
        gaps = (
            EvidenceGap(
                reason=EvidenceGapReason.PARTIAL,
                request_id=base.request_id,
                call_id="call_gate_01",
            ),
        )
    elif degradation == "partial":
        items = (
            items[0].model_copy(update={"quality": EvidenceQuality.PARTIAL}),
        )
    elif degradation == "obsolete":
        items = (
            items[0].model_copy(
                update={
                    "quality": EvidenceQuality.OBSOLETE,
                    "obsolescence": (
                        EvidenceObsolescenceReason.DATA_QUALITY_STALE,
                    ),
                }
            ),
        )
    else:
        other = items[0].model_copy(
            update={
                "evidence_id": "sha256:v1:" + "c" * 64,
                "value": JsonSnapshot.capture(
                    "low",
                    forbidden_names=frozenset(),
                ),
            }
        )
        items = (*items, other)
        conflicts = (
            EvidenceConflict(
                canonical_key=items[0].canonical_key,
                evidence_ids=tuple(sorted(item.evidence_id for item in items)),
            ),
        )
    ledger = EvidenceLedger(
        request_id=base.request_id,
        items=items,
        gaps=gaps,
        conflicts=conflicts,
    )
    context = ReleaseGateContext(
        request_id="req_gate_01",
        decision=AgentDecision.GUIDE,
        ledger=ledger,
        draft=_draft_for(AgentDecision.GUIDE, ledger),
        permissions=frozenset({"read"}),
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.INSUFFICIENT_EVIDENCE


def test_gate_rejects_a_historical_item_inserted_in_the_current_ledger() -> None:
    base = _claimable_ledger()
    historical = base.items[0].model_copy(
        update={"request_id": "req_gate_historical"}
    )
    historical = historical.model_copy(
        update={"evidence_id": canonical_evidence_id(historical)}
    )
    ledger = EvidenceLedger(
        request_id="req_gate_01",
        items=(historical,),
    )
    context = ReleaseGateContext(
        request_id="req_gate_01",
        decision=AgentDecision.GUIDE,
        ledger=ledger,
        draft=_draft_for(AgentDecision.GUIDE, ledger),
        permissions=frozenset({"read"}),
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.REQUEST_MISMATCH


def test_gate_rejects_a_historical_gap_without_exposing_its_reference() -> None:
    base = _claimable_ledger()
    historical_gap = EvidenceGap(
        reason=EvidenceGapReason.PARTIAL,
        request_id="req_historical_secret",
        call_id="call_historical_secret",
        fact_path="asset.criticality",
    )
    ledger = EvidenceLedger(
        request_id=base.request_id,
        items=base.items,
        gaps=(historical_gap,),
    )
    context = ReleaseGateContext(
        request_id="req_gate_01",
        decision=AgentDecision.GUIDE,
        ledger=ledger,
        draft=_draft_for(AgentDecision.GUIDE, ledger),
        permissions=frozenset({"read"}),
    )

    attestation = evaluate_release(context)
    result = render_non_release_result(context, attestation)

    assert attestation.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert attestation.reason is ReleaseGateReason.REQUEST_MISMATCH
    assert result.limitation_refs == ()
    assert "historical" not in result.message


def test_gate_rejects_partial_evidence_marked_as_claimable() -> None:
    base = _claimable_ledger()
    incoherent = base.items[0].model_copy(
        update={"mode": ResponseMode.PARTIAL}
    )
    incoherent = incoherent.model_copy(
        update={"evidence_id": canonical_evidence_id(incoherent)}
    )
    ledger = EvidenceLedger(request_id=base.request_id, items=(incoherent,))
    context = ReleaseGateContext(
        request_id="req_gate_01",
        decision=AgentDecision.GUIDE,
        ledger=ledger,
        draft=_draft_for(AgentDecision.GUIDE, ledger),
        permissions=frozenset({"read"}),
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.INSUFFICIENT_EVIDENCE


def test_gate_recomputes_every_persisted_evidence_id() -> None:
    base = _claimable_ledger()
    tampered = base.items[0].model_copy(
        update={
            "value": JsonSnapshot.capture(
                "SENSITIVE_VALUE",
                forbidden_names=frozenset(),
            )
        }
    )
    ledger = EvidenceLedger(request_id=base.request_id, items=(tampered,))
    context = ReleaseGateContext(
        request_id="req_gate_01",
        decision=AgentDecision.GUIDE,
        ledger=ledger,
        draft=_draft_for(AgentDecision.GUIDE, ledger),
        permissions=frozenset({"read"}),
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.INSUFFICIENT_EVIDENCE


def test_gate_recompiles_action_evidence_and_rejects_the_wrong_resource() -> None:
    proposal, approval, intent, canonical = _completed_update_action()
    forged = canonical.items[0].model_copy(
        update={"resource": "/actions/action_FORGED"}
    )
    forged = forged.model_copy(
        update={"evidence_id": canonical_evidence_id(forged)}
    )
    ledger = EvidenceLedger(request_id=canonical.request_id, items=(forged,))
    context = ReleaseGateContext(
        request_id="req_gate_action",
        decision=AgentDecision.ACT,
        ledger=ledger,
        draft=_draft_for(AgentDecision.ACT, ledger),
        permissions=frozenset({"action_high"}),
        trusted_write_context=_trusted_write_context(),
        intents=(intent,),
        proposal=proposal,
        approval=approval,
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.ACTION_EVIDENCE_MISSING


def test_gate_recomputes_conflicts_instead_of_trusting_the_conflict_list() -> None:
    base = _claimable_ledger()
    divergent = base.items[0].model_copy(
        update={
            "call_id": "call_gate_02",
            "value": JsonSnapshot.capture("low", forbidden_names=frozenset()),
        }
    )
    divergent = divergent.model_copy(
        update={"evidence_id": canonical_evidence_id(divergent)}
    )
    ledger = EvidenceLedger(
        request_id=base.request_id,
        items=tuple(sorted((*base.items, divergent), key=lambda item: item.evidence_id)),
    )
    context = ReleaseGateContext(
        request_id="req_gate_01",
        decision=AgentDecision.GUIDE,
        ledger=ledger,
        draft=_draft_for(AgentDecision.GUIDE, ledger),
        permissions=frozenset({"read"}),
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.INSUFFICIENT_EVIDENCE


def test_gate_reports_uncertain_action_before_considering_its_evidence_gap() -> None:
    proposal = UpdateAssetCriticalityProposal(
        criticality="critical",
        justification="O impacto operacional exige prioridade máxima.",
    )
    approval = TrustedActionApproval(
        action=proposal.action,
        target_id="asset_G501",
        material_parameters={"criticality": "critical"},
        source=ApprovalSource.ORIGINAL_REQUEST,
    )
    intent = WriteIntent(
        intent_id="intent_gate_uncertain",
        request_id="req_gate_uncertain",
        scope=UpdateAssetCriticalityIntentScope(
            action=proposal.action,
            case_id="case_tkt_inv_04",
            company_id="comp_mineracao_andes",
            user_id="usr_pedro",
            asset_id="asset_G501",
            criticality="critical",
            justification=proposal.justification,
        ),
        payload_hash=canonical_write_payload_hash(
            proposal,
            trusted_context=_trusted_write_context(),
        ),
        decision=WritePolicyResult(
            decision=PolicyDecision.ALLOW,
            reason=PolicyReason.AUTHORIZED,
        ),
        status=IntentStatus.UNCERTAIN,
        prepared_execution_id="exec_gate_uncertain",
        attempts=1,
        error=ApiError(
            category=ApiErrorCategory.TIMEOUT,
            code="REMOTE_TIMEOUT",
            message="A resposta remota não chegou.",
        ),
    )
    ledger = compile_action_intents(
        (intent,),
        recorded_at=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
    )
    context = ReleaseGateContext(
        request_id="req_gate_uncertain",
        decision=AgentDecision.ACT,
        ledger=ledger,
        draft=_draft_for(AgentDecision.ACT, ledger),
        permissions=frozenset({"action_high"}),
        trusted_write_context=_trusted_write_context(),
        intents=(intent,),
        proposal=proposal,
        approval=approval,
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.INTENT_UNCERTAIN
    with pytest.raises(ValueError, match="release"):
        render_released_result(context, result)


def test_checkpoint_rejects_a_final_message_that_diverges_from_gate_renderer() -> None:
    state = _released_action_state()
    tampered = state.model_dump(mode="json")
    tampered["final_result"]["message"] = "Afirmação técnica inventada."

    with pytest.raises(ValidationError, match="renderer"):
        AgentState.model_validate(tampered)


def test_checkpoint_rejects_a_forged_decision_with_recomputed_digests() -> None:
    state = _released_action_state()
    assert state.writer_draft is not None
    forged_draft = state.writer_draft.model_copy(
        update={
            "decision": AgentDecision.ESCALATE,
            "next_step": WriterNextStep.AWAIT_ESCALATION,
        }
    )
    gate_context = ReleaseGateContext(
        request_id=state.request_id,
        decision=AgentDecision.ESCALATE,
        ledger=state.ledger,
        draft=forged_draft,
        permissions=state.permissions,
        trusted_write_context=state.trusted_write_context,
        intents=state.intents,
        proposal=state.pending_proposal,
        approval=state.approval,
    )
    recomputed = evaluate_release(gate_context)
    assert recomputed.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    forged_release = recomputed.model_copy(
        update={
            "outcome": ReleaseGateOutcome.RELEASE,
            "reason": ReleaseGateReason.PASSED,
        }
    )
    tampered = state.model_dump(mode="json")
    tampered["decision"] = AgentDecision.ESCALATE.value
    tampered["writer_draft"] = forged_draft.model_dump(mode="json")
    tampered["release_gate"] = forged_release.model_dump(mode="json")
    tampered["final_result"] = {
        "decision": AgentDecision.ESCALATE.value,
        "message": "O caso foi escalado pela plataforma.",
        "evidence_ids": list(forged_draft.evidence_ids),
        "limitation_refs": list(forged_draft.limitation_refs),
        "next_step": WriterNextStep.AWAIT_ESCALATION.value,
    }

    with pytest.raises(ValidationError, match="atestado"):
        AgentState.model_validate(tampered)


def test_checkpoint_binds_the_released_action_target_to_the_trusted_request() -> None:
    state = _released_action_state()
    intent = state.intents[0]
    assert isinstance(intent.scope, UpdateAssetCriticalityIntentScope)
    forged_trusted_context = _trusted_write_context().model_copy(
        update={"central_asset_id": "asset_G502"}
    )
    forged_intent = intent.model_copy(
        update={
            "scope": intent.scope.model_copy(update={"asset_id": "asset_G502"}),
            "payload_hash": canonical_write_payload_hash(
                state.pending_proposal,
                trusted_context=forged_trusted_context,
            ),
        }
    )
    forged_approval = state.approval.model_copy(update={"target_id": "asset_G502"})
    forged_ledger = compile_action_intents(
        (forged_intent,),
        recorded_at=state.ledger.items[0].recorded_at,
    )
    forged_draft = _draft_for(AgentDecision.ACT, forged_ledger)
    forged_context = ReleaseGateContext(
        request_id=state.request_id,
        decision=AgentDecision.ACT,
        ledger=forged_ledger,
        draft=forged_draft,
        permissions=state.permissions,
        trusted_write_context=forged_trusted_context,
        intents=(forged_intent,),
        proposal=state.pending_proposal,
        approval=forged_approval,
    )
    forged_gate = evaluate_release(forged_context)
    assert forged_gate.outcome is ReleaseGateOutcome.RELEASE
    forged_result = render_released_result(forged_context, forged_gate)
    tampered = state.model_dump(mode="json")
    tampered["intents"] = [forged_intent.model_dump(mode="json")]
    tampered["approval"] = forged_approval.model_dump(mode="json")
    tampered["trusted_write_context"] = forged_trusted_context.model_dump(
        mode="json"
    )
    tampered["ledger"] = forged_ledger.model_dump(mode="json")
    tampered["writer_draft"] = forged_draft.model_dump(mode="json")
    tampered["release_gate"] = forged_gate.model_dump(mode="json")
    tampered["final_result"] = forged_result.model_dump(mode="json")

    with pytest.raises(ValidationError, match="escopo confiável"):
        AgentState.model_validate(tampered)


def test_checkpoint_cannot_delete_the_trusted_action_scope() -> None:
    state = _released_action_state()
    tampered = state.model_dump(mode="json")
    tampered.pop("trusted_write_context")

    with pytest.raises(ValidationError, match="contexto confiável|atestado"):
        AgentState.model_validate(tampered)


def test_gate_attestation_binds_missing_information_even_if_message_is_recomputed() -> None:
    missing_information = "Informe o ponto de medição."
    planner_terminal = PlannerTerminalRecord(
        decision="request_information",
        stop_reason="missing_information",
        missing_information=missing_information,
    )
    ledger = EvidenceLedger()
    draft = _draft_for(
        AgentDecision.REQUEST_INFORMATION,
        ledger,
        missing_information=missing_information,
    )
    context = ReleaseGateContext(
        request_id="req_gate_missing_information",
        decision=AgentDecision.REQUEST_INFORMATION,
        ledger=ledger,
        draft=draft,
        permissions=frozenset({"read"}),
        trusted_write_context=_trusted_write_context(),
        planner_terminal=planner_terminal,
        missing_information=missing_information,
    )
    attestation = evaluate_release(context)
    result = render_non_release_result(context, attestation)
    request = SupportRequest(
        case_id="case_tkt_inv_04",
        ticket_id="TKT-INV-04",
        asset_id="asset_G501",
        message="Investigue, mas falta identificar o ponto.",
        identity=Identity(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
        ),
    )
    state = AgentState(
        request=request,
        identity=TrustedIdentity.model_validate(request.identity.model_dump()),
        permissions=frozenset({"read"}),
        request_id="req_gate_missing_information",
        thread_id="thread_gate_missing_information",
        execution_id="exec_gate_missing_information",
        thread_scope=ThreadScope(
            thread_id="thread_gate_missing_information",
            case_id=request.case_id,
            company_id=request.identity.company_id,
            user_id=request.identity.user_id,
        ),
        trusted_write_context=_trusted_write_context(),
        ledger=ledger,
        decision=AgentDecision.REQUEST_INFORMATION,
        step_count=5,
        step_limit=24,
        planner_terminal=planner_terminal,
        final_result=result,
        writer_draft=draft,
        writer_attempts=1,
        release_gate=attestation,
        resume_anchor=ResumeAnchor.RELEASE_GATE,
    )
    tampered = state.model_dump(mode="json")
    tampered["planner_terminal"]["missing_information"] = "Informe o sensor."
    tampered["final_result"]["message"] = "Para continuar, informe: Informe o sensor."

    with pytest.raises(ValidationError, match="atestado"):
        AgentState.model_validate(tampered)
