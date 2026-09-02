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
from tractian_agent.evidence import compile_action_intents
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
    UpdateAssetCriticalityIntentScope,
    WriteIntent,
)
from tractian_agent.write_policy import (
    ApprovalSource,
    PolicyDecision,
    PolicyReason,
    TrustedActionApproval,
    UpdateAssetCriticalityProposal,
    WritePolicyResult,
    canonical_write_payload_hash,
)
from tractian_agent.writer import build_writer_context


def _claimable_ledger() -> EvidenceLedger:
    return EvidenceLedger(
        request_id="req_gate_01",
        items=(
            EvidenceItem(
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
            ),
        ),
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
            AgentDecision.REQUIRE_HUMAN_REVIEW: WriterNextStep.AWAIT_HUMAN_REVIEW,
        }[decision],
    )


def test_gate_releases_only_the_canonical_facts_selected_by_id() -> None:
    evidence_id = "sha256:v1:" + "b" * 64
    ledger = _claimable_ledger()
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
        ledger = EvidenceLedger(
            request_id=ledger.request_id,
            items=(
                ledger.items[0].model_copy(
                    update={"limitations": ("janela reduzida",)}
                ),
            ),
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
        payload_hash=canonical_write_payload_hash(proposal),
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
        intents=(intent,),
        approval=approval,
    )

    result = evaluate_release(context)

    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.INTENT_UNCERTAIN
    with pytest.raises(ValueError, match="release"):
        render_released_result(context, result)


def test_checkpoint_rejects_a_final_message_that_diverges_from_gate_renderer() -> None:
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
        payload_hash=canonical_write_payload_hash(proposal),
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
    evidence_ids = tuple(sorted(item.evidence_id for item in ledger.items))
    draft = WriterDraft(
        decision=AgentDecision.ACT,
        evidence_ids=evidence_ids,
        limitation_refs=(),
        next_step=WriterNextStep.VERIFY_ACTION,
    )
    gate_context = ReleaseGateContext(
        request_id="req_gate_action",
        decision=AgentDecision.ACT,
        ledger=ledger,
        draft=draft,
        permissions=frozenset({"action_high"}),
        intents=(intent,),
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
    state = AgentState(
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
    tampered = state.model_dump(mode="json")
    tampered["final_result"]["message"] = "Afirmação técnica inventada."

    with pytest.raises(ValidationError, match="renderer"):
        AgentState.model_validate(tampered)


def test_gate_attestation_binds_missing_information_even_if_message_is_recomputed() -> None:
    missing_information = "Informe o ponto de medição."
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
        ledger=ledger,
        decision=AgentDecision.REQUEST_INFORMATION,
        step_count=5,
        step_limit=24,
        planner_terminal=PlannerTerminalRecord(
            decision="request_information",
            stop_reason="missing_information",
            missing_information=missing_information,
        ),
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
