"""Porta determinística que atesta o draft e renderiza somente fatos do ledger."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from pydantic import ConfigDict, Field, model_validator

from tractian_agent.contracts import ResponseMode, StrictModel
from tractian_agent.evidence import (
    assess_evidence,
    canonical_evidence_id,
    compile_action_intents,
    merge_ledgers,
)
from tractian_agent.human_review import review_audit_is_canonical
from tractian_agent.state import (
    AgentDecision,
    EvidenceLedger,
    EvidenceQuality,
    EvidenceSourceKind,
    EvidenceSufficiency,
    FinalResult,
    PlannerTerminalRecord,
    ReleaseGateOutcome,
    ReleaseGateReason,
    ReleaseGateRecord,
    ReviewAudit,
    ReviewRequest,
    ReviewResolution,
    ReviewedDraft,
    WriterDraft,
    WriterFailureRecord,
    WriterNextStep,
)
from tractian_agent.tools.runtime import Permission
from tractian_agent.write_contracts import (
    IntentStatus,
    WriteIntent,
    approval_matches_write_intent,
    proposal_matches_intent_scope,
)
from tractian_agent.write_policy import (
    PolicyDecision,
    PolicyReason,
    TrustedActionApproval,
    TrustedWriteContext,
    WriteProposal,
    evaluate_write_policy,
)
from tractian_agent.writer import (
    WriterContext,
    build_writer_context,
    limitation_descriptions,
)


_NEXT_STEP_BY_DECISION = {
    AgentDecision.GUIDE: WriterNextStep.MONITOR,
    AgentDecision.ACT: WriterNextStep.VERIFY_ACTION,
    AgentDecision.ESCALATE: WriterNextStep.AWAIT_ESCALATION,
    AgentDecision.REQUEST_INFORMATION: WriterNextStep.PROVIDE_INFORMATION,
    AgentDecision.REQUEST_CONFIRMATION: WriterNextStep.CONFIRM_ACTION,
    AgentDecision.REQUIRE_HUMAN_REVIEW: WriterNextStep.AWAIT_HUMAN_REVIEW,
}
_ACTION_PERMISSION = {
    "reprocess_analysis": "action_low",
    "request_specialist_analysis": "action_low",
    "update_asset_criticality": "action_high",
    "request_model_retraining": "action_high",
    "escalate_case": "escalate",
}


class ReleaseGateContext(StrictModel):
    """Allowlist pura construída do estado; não aceita runtime ou artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    request_id: str = Field(min_length=1, pattern=r"^\S+$")
    decision: AgentDecision
    ledger: EvidenceLedger
    draft: WriterDraft | ReviewedDraft | None = None
    permissions: frozenset[Permission]
    intents: tuple[WriteIntent, ...] = ()
    proposal: WriteProposal | None = None
    trusted_write_context: TrustedWriteContext | None = None
    planner_terminal: PlannerTerminalRecord | None = None
    approval: TrustedActionApproval | None = None
    missing_information: str | None = None
    writer_failure: WriterFailureRecord | None = None
    review_request: ReviewRequest | None = None
    review_audit: ReviewAudit | None = None
    review_resolution: ReviewResolution | None = None

    @model_validator(mode="after")
    def _require_current_intents(self) -> ReleaseGateContext:
        if any(intent.request_id != self.request_id for intent in self.intents):
            raise ValueError(
                "contexto do gate aceita somente intenções da request atual"
            )
        return self


def _digest(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:v1:{hashlib.sha256(encoded).hexdigest()}"


def _record(
    context: ReleaseGateContext,
    outcome: ReleaseGateOutcome,
    reason: ReleaseGateReason,
) -> ReleaseGateRecord:
    return ReleaseGateRecord(
        subject_decision=context.decision,
        outcome=outcome,
        reason=reason,
        draft_digest=_digest(
            context.draft.model_dump(mode="json") if context.draft is not None else None
        ),
        ledger_digest=_digest(context.ledger),
        context_digest=_digest(
            {
                "approval": (
                    context.approval.model_dump(mode="json")
                    if context.approval is not None
                    else None
                ),
                "decision": context.decision.value,
                "intents": [
                    intent.model_dump(mode="json")
                    for intent in sorted(
                        context.intents,
                        key=lambda candidate: candidate.intent_id,
                    )
                ],
                "missing_information": context.missing_information,
                "proposal": (
                    context.proposal.model_dump(mode="json")
                    if context.proposal is not None
                    else None
                ),
                "permissions": sorted(context.permissions),
                "planner_terminal": (
                    context.planner_terminal.model_dump(mode="json")
                    if context.planner_terminal is not None
                    else None
                ),
                "request_id": context.request_id,
                "trusted_write_context": (
                    context.trusted_write_context.model_dump(mode="json")
                    if context.trusted_write_context is not None
                    else None
                ),
                "writer_failure": (
                    context.writer_failure.model_dump(mode="json")
                    if context.writer_failure is not None
                    else None
                ),
            }
        ),
        review_digest=(
            _digest(context.review_request)
            if context.review_request is not None
            else None
        ),
        review_audit_digest=(
            _digest(context.review_audit) if context.review_audit is not None else None
        ),
    )


def _review(
    context: ReleaseGateContext,
    reason: ReleaseGateReason,
) -> ReleaseGateRecord:
    return _record(
        context,
        ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW,
        reason,
    )


def build_budget_exhausted_gate(
    context: ReleaseGateContext,
) -> ReleaseGateRecord:
    """Atestado terminal próprio quando nem o gate pode consumir outro passo."""

    return _review(context, ReleaseGateReason.STEP_BUDGET_EXHAUSTED)


def render_budget_exhausted_result(
    context: ReleaseGateContext,
    attestation: ReleaseGateRecord,
) -> FinalResult:
    if attestation != build_budget_exhausted_gate(context):
        raise ValueError("atestado não corresponde ao esgotamento do gate")
    return FinalResult(
        decision=AgentDecision.REQUIRE_HUMAN_REVIEW,
        message="O orçamento seguro do fluxo foi esgotado antes da liberação.",
        next_step=WriterNextStep.AWAIT_HUMAN_REVIEW,
    )


def _expected_context(context: ReleaseGateContext) -> WriterContext:
    return build_writer_context(
        decision=context.decision,
        ledger=context.ledger,
        missing_information=context.missing_information,
    )


def _current_intent(context: ReleaseGateContext) -> WriteIntent | None:
    return context.intents[0] if len(context.intents) == 1 else None


def _approval_matches_intent(
    context: ReleaseGateContext,
    intent: WriteIntent,
) -> bool:
    if context.proposal is None:
        return False
    return approval_matches_write_intent(
        context.proposal,
        intent,
        approval=context.approval,
        trusted_context=context.trusted_write_context,
    )


def _action_decision_is_canonical(
    context: ReleaseGateContext,
) -> ReleaseGateReason | None:
    intent = _current_intent(context)
    if intent is None:
        if (
            context.intents
            or context.proposal is not None
            or context.decision
            in {
                AgentDecision.ACT,
                AgentDecision.ESCALATE,
                AgentDecision.REQUEST_CONFIRMATION,
            }
        ):
            return ReleaseGateReason.INTENT_MISSING
        return None
    if context.trusted_write_context is None:
        return ReleaseGateReason.INTENT_POLICY_MISMATCH
    if intent.status is IntentStatus.DENIED:
        return ReleaseGateReason.INTENT_POLICY_MISMATCH
    if intent.status is IntentStatus.UNCERTAIN:
        return ReleaseGateReason.INTENT_UNCERTAIN
    if intent.status in {
        IntentStatus.PROPOSED,
        IntentStatus.PREPARED,
        IntentStatus.FAILED,
    }:
        return ReleaseGateReason.INTENT_NOT_COMPLETED
    if intent.status is IntentStatus.AWAITING_CONFIRMATION:
        if context.proposal is None or not proposal_matches_intent_scope(
            context.proposal,
            intent.scope,
            payload_hash=intent.payload_hash,
            trusted_context=context.trusted_write_context,
        ):
            return ReleaseGateReason.INTENT_POLICY_MISMATCH
        if context.decision is not AgentDecision.REQUEST_CONFIRMATION:
            return ReleaseGateReason.DECISION_MISMATCH
        if _ACTION_PERMISSION[intent.scope.action] not in context.permissions:
            return ReleaseGateReason.PERMISSION_INCOMPATIBLE
        current_policy = evaluate_write_policy(
            context.proposal,
            permissions=context.permissions,
            approval=context.approval,
            trusted_context=context.trusted_write_context,
        )
        if current_policy != intent.decision:
            return ReleaseGateReason.INTENT_POLICY_MISMATCH
        return None
    if context.proposal is None or not proposal_matches_intent_scope(
        context.proposal,
        intent.scope,
        payload_hash=intent.payload_hash,
        trusted_context=context.trusted_write_context,
    ):
        return ReleaseGateReason.INTENT_POLICY_MISMATCH
    expected = (
        AgentDecision.ESCALATE
        if intent.scope.action == "escalate_case"
        else AgentDecision.ACT
    )
    if context.decision is not expected:
        return ReleaseGateReason.DECISION_MISMATCH
    return None


def _planner_decision_is_canonical(
    context: ReleaseGateContext,
) -> ReleaseGateReason | None:
    if context.planner_terminal is None:
        return None
    if context.intents or context.proposal is not None:
        return ReleaseGateReason.INTENT_POLICY_MISMATCH
    if context.decision is not AgentDecision(context.planner_terminal.decision):
        return ReleaseGateReason.DECISION_MISMATCH
    if context.missing_information != context.planner_terminal.missing_information:
        return ReleaseGateReason.MISSING_INFORMATION_INVALID
    return None


def _ledger_integrity_reason(
    context: ReleaseGateContext,
) -> ReleaseGateReason | None:
    if any(item.request_id != context.request_id for item in context.ledger.items):
        return ReleaseGateReason.REQUEST_MISMATCH
    if any(gap.request_id != context.request_id for gap in context.ledger.gaps):
        return ReleaseGateReason.REQUEST_MISMATCH
    if any(
        item.evidence_id != canonical_evidence_id(item) for item in context.ledger.items
    ):
        return ReleaseGateReason.INSUFFICIENT_EVIDENCE
    if any(
        (
            item.quality is EvidenceQuality.CLAIMABLE
            and (item.mode is not ResponseMode.COMPLETE or bool(item.obsolescence))
        )
        or (item.quality is EvidenceQuality.OBSOLETE and not item.obsolescence)
        or (bool(item.obsolescence) and item.quality is not EvidenceQuality.OBSOLETE)
        for item in context.ledger.items
    ):
        return ReleaseGateReason.INSUFFICIENT_EVIDENCE
    canonical_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
    expected_actions = compile_action_intents(
        context.intents,
        recorded_at=canonical_time,
    )
    actual_action_items = tuple(
        sorted(
            (
                item.model_copy(update={"recorded_at": canonical_time})
                for item in context.ledger.items
                if item.source_kind is EvidenceSourceKind.ACTION
            ),
            key=lambda item: item.evidence_id,
        )
    )
    expected_action_items = tuple(
        sorted(expected_actions.items, key=lambda item: item.evidence_id)
    )
    actual_action_gaps = tuple(
        sorted(
            (gap for gap in context.ledger.gaps if gap.intent_id is not None),
            key=lambda gap: gap.model_dump_json(),
        )
    )
    expected_action_gaps = tuple(
        sorted(
            expected_actions.gaps,
            key=lambda gap: gap.model_dump_json(),
        )
    )
    if (
        actual_action_items != expected_action_items
        or actual_action_gaps != expected_action_gaps
    ):
        return ReleaseGateReason.ACTION_EVIDENCE_MISSING
    expected_conflicts = merge_ledgers(
        EvidenceLedger(
            request_id=context.ledger.request_id,
            items=context.ledger.items,
        )
    ).conflicts
    if context.ledger.conflicts != expected_conflicts:
        return ReleaseGateReason.INSUFFICIENT_EVIDENCE
    return None


def _action_is_trusted(context: ReleaseGateContext) -> ReleaseGateReason | None:
    intent = _current_intent(context)
    if intent is None:
        return ReleaseGateReason.INTENT_MISSING
    if intent.status is IntentStatus.UNCERTAIN:
        return ReleaseGateReason.INTENT_UNCERTAIN
    if intent.status is not IntentStatus.COMPLETED:
        return ReleaseGateReason.INTENT_NOT_COMPLETED
    if (
        intent.decision.decision is not PolicyDecision.ALLOW
        or intent.decision.reason is not PolicyReason.AUTHORIZED
    ):
        return ReleaseGateReason.INTENT_POLICY_MISMATCH
    required_permission = _ACTION_PERMISSION[intent.scope.action]
    if required_permission not in context.permissions:
        return ReleaseGateReason.PERMISSION_INCOMPATIBLE
    if not _approval_matches_intent(context, intent):
        return ReleaseGateReason.APPROVAL_MISMATCH
    if not any(
        item.source_kind is EvidenceSourceKind.ACTION
        and item.intent_id == intent.intent_id
        and item.action == intent.scope.action
        and item.fact_path == "accepted"
        and item.value.to_python() is True
        and item.claimable
        and item.evidence_id in context.draft.evidence_ids
        for item in context.ledger.items
    ):
        return ReleaseGateReason.ACTION_EVIDENCE_MISSING
    return None


def evaluate_release(context: ReleaseGateContext) -> ReleaseGateRecord:
    """Recalcula todos os invariantes; nenhum veredito vem do modelo."""
    reviewed = isinstance(context.draft, ReviewedDraft)
    if (context.writer_failure is not None and not reviewed) or context.draft is None:
        return _review(context, ReleaseGateReason.WRITER_FAILURE)
    if reviewed:
        request = context.review_request
        audit = context.review_audit
        resolution = context.review_resolution
        if request is None or audit is None or resolution is None:
            return _review(context, ReleaseGateReason.REQUEST_MISMATCH)
        if (
            request.request_id != context.request_id
            or request.subject_decision is not context.decision
            or audit.review_id != request.review_id
            or audit.review_digest != _digest(request)
            or audit.before_digest != _digest(request.draft)
            or audit.after_digest != _digest(context.draft)
            or audit.operation.value not in {"approve", "edit"}
            or not review_audit_is_canonical(
                request, audit, context.draft, context.ledger, resolution
            )
        ):
            return _review(context, ReleaseGateReason.REQUEST_MISMATCH)
    if context.ledger.request_id not in {context.request_id, None} or (
        context.ledger.request_id is None
        and bool(
            context.ledger.items or context.ledger.gaps or context.ledger.conflicts
        )
    ):
        return _review(context, ReleaseGateReason.REQUEST_MISMATCH)
    ledger_reason = _ledger_integrity_reason(context)
    if ledger_reason is not None:
        return _review(context, ledger_reason)
    action_decision_reason = _action_decision_is_canonical(context)
    if action_decision_reason is not None:
        return _review(context, action_decision_reason)
    planner_decision_reason = _planner_decision_is_canonical(context)
    if planner_decision_reason is not None:
        return _review(context, planner_decision_reason)
    if context.draft.decision is not context.decision:
        return _review(context, ReleaseGateReason.DECISION_MISMATCH)

    expected = _expected_context(context)
    evidence_ids = tuple(fact.evidence_id for fact in expected.facts)
    limitation_refs = tuple(
        limitation.limitation_ref for limitation in expected.limitations
    )
    reviewed_empty_is_safe = reviewed and (
        (context.decision, context.draft.next_step)
        in {
            (
                AgentDecision.REQUEST_INFORMATION,
                WriterNextStep.PROVIDE_INFORMATION,
            ),
            (
                AgentDecision.REQUEST_CONFIRMATION,
                WriterNextStep.CONFIRM_ACTION,
            ),
            (
                AgentDecision.REQUIRE_HUMAN_REVIEW,
                WriterNextStep.AWAIT_HUMAN_REVIEW,
            ),
        }
    )
    if (not reviewed and context.draft.evidence_ids != evidence_ids) or (
        reviewed
        and (
            (not context.draft.evidence_ids and not reviewed_empty_is_safe)
            or any(item not in evidence_ids for item in context.draft.evidence_ids)
        )
    ):
        return _review(context, ReleaseGateReason.EVIDENCE_REFERENCE_MISMATCH)
    evidence_by_id = {item.evidence_id: item for item in context.ledger.items}
    if "read" not in context.permissions and any(
        evidence_by_id[evidence_id].source_kind is EvidenceSourceKind.TOOL
        for evidence_id in context.draft.evidence_ids
    ):
        return _review(context, ReleaseGateReason.PERMISSION_INCOMPATIBLE)
    if context.draft.limitation_refs != limitation_refs:
        return _review(context, ReleaseGateReason.LIMITATION_REFERENCE_MISMATCH)
    human_disposition = (
        context.decision is AgentDecision.GUIDE
        and context.draft.next_step is WriterNextStep.REQUEST_HUMAN_DISPOSITION
    )
    if (
        context.draft.next_step is not _NEXT_STEP_BY_DECISION[context.decision]
        and not human_disposition
    ):
        return _review(context, ReleaseGateReason.NEXT_STEP_MISMATCH)
    if any(
        limitation.kind == "projection_overflow" for limitation in expected.limitations
    ):
        return _review(context, ReleaseGateReason.INSUFFICIENT_EVIDENCE)

    if context.decision is AgentDecision.REQUEST_INFORMATION:
        if (
            context.missing_information is None
            or not context.missing_information.strip()
        ):
            return _review(context, ReleaseGateReason.MISSING_INFORMATION_INVALID)
        return _record(
            context,
            ReleaseGateOutcome.REQUEST_INFORMATION,
            ReleaseGateReason.INFORMATION_REQUIRED,
        )
    if context.decision is AgentDecision.REQUEST_CONFIRMATION:
        return _record(
            context,
            ReleaseGateOutcome.REQUEST_CONFIRMATION,
            ReleaseGateReason.CONFIRMATION_REQUIRED,
        )
    if context.decision is AgentDecision.REQUIRE_HUMAN_REVIEW:
        return _review(context, ReleaseGateReason.HUMAN_REVIEW_REQUESTED)

    if context.decision is not AgentDecision.GUIDE:
        action_reason = _action_is_trusted(context)
        if action_reason is not None:
            return _review(context, action_reason)

    assessment = assess_evidence(context.ledger)
    has_degraded_item = any(
        item.quality is not EvidenceQuality.CLAIMABLE or bool(item.obsolescence)
        for item in context.ledger.items
    )
    if (
        assessment.status is not EvidenceSufficiency.SUFFICIENT
        or bool(context.ledger.gaps)
        or bool(context.ledger.conflicts)
        or has_degraded_item
    ):
        return _review(context, ReleaseGateReason.INSUFFICIENT_EVIDENCE)
    if context.decision is AgentDecision.GUIDE and "read" not in context.permissions:
        return _review(context, ReleaseGateReason.PERMISSION_INCOMPATIBLE)
    if human_disposition:
        return _review(context, ReleaseGateReason.HUMAN_DISPOSITION_REQUIRED)
    return _record(
        context,
        ReleaseGateOutcome.RELEASE,
        ReleaseGateReason.PASSED,
    )


def _canonical_value(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def render_released_result(
    context: ReleaseGateContext,
    attestation: ReleaseGateRecord,
) -> FinalResult:
    """Resolve valores somente no ledger após revalidar o atestado."""
    if attestation != evaluate_release(context) or (
        attestation.outcome is not ReleaseGateOutcome.RELEASE
    ):
        raise ValueError("somente um atestado release atual pode renderizar fatos")
    assert context.draft is not None
    items = {item.evidence_id: item for item in context.ledger.items}
    prefix = {
        AgentDecision.GUIDE: "Orientação fundamentada nas evidências registradas.",
        AgentDecision.ACT: "A ação foi concluída pela plataforma.",
        AgentDecision.ESCALATE: "O caso foi escalado pela plataforma.",
    }[context.decision]
    facts = []
    for evidence_id in context.draft.evidence_ids:
        item = items[evidence_id]
        source = item.tool if item.tool is not None else item.action
        assert source is not None
        facts.append(
            f"{item.fact_path} = {_canonical_value(item.value.to_python())} "
            f"(fonte {source}, recurso {item.resource})."
        )
    descriptions_by_ref = limitation_descriptions(context.ledger)
    limitation_messages = []
    for limitation_ref in context.draft.limitation_refs:
        description = descriptions_by_ref[limitation_ref]
        limitation_messages.append(f"Limitação considerada: {description}.")
    next_step = {
        WriterNextStep.MONITOR: "Próximo passo: monitore a condição.",
        WriterNextStep.VERIFY_ACTION: "Próximo passo: verifique o recibo da ação.",
        WriterNextStep.AWAIT_ESCALATION: "Próximo passo: aguarde o contato humano.",
    }[context.draft.next_step]
    message = " ".join((prefix, *facts, *limitation_messages, next_step))
    return FinalResult(
        decision=context.decision,
        message=message,
        evidence_ids=context.draft.evidence_ids,
        limitation_refs=context.draft.limitation_refs,
        next_step=context.draft.next_step,
    )


def render_non_release_result(
    context: ReleaseGateContext,
    attestation: ReleaseGateRecord,
) -> FinalResult:
    """Renderiza somente avisos seguros, sem fatos técnicos."""
    if attestation != evaluate_release(context) or (
        attestation.outcome is ReleaseGateOutcome.RELEASE
    ):
        raise ValueError("atestado não corresponde a uma saída segura")
    if attestation.outcome is ReleaseGateOutcome.REQUEST_INFORMATION:
        assert context.missing_information is not None
        assert context.draft is not None
        return FinalResult(
            decision=AgentDecision.REQUEST_INFORMATION,
            message=f"Para continuar, informe: {context.missing_information}",
            evidence_ids=context.draft.evidence_ids,
            limitation_refs=context.draft.limitation_refs,
            next_step=WriterNextStep.PROVIDE_INFORMATION,
        )
    if attestation.outcome is ReleaseGateOutcome.REQUEST_CONFIRMATION:
        return FinalResult(
            decision=AgentDecision.REQUEST_CONFIRMATION,
            message="A ação precisa de confirmação explícita antes de continuar.",
            next_step=WriterNextStep.CONFIRM_ACTION,
        )
    return FinalResult(
        decision=AgentDecision.REQUIRE_HUMAN_REVIEW,
        message="A resposta não foi liberada e exige revisão humana.",
        next_step=WriterNextStep.AWAIT_HUMAN_REVIEW,
    )


__all__ = [
    "ReleaseGateContext",
    "build_budget_exhausted_gate",
    "evaluate_release",
    "render_budget_exhausted_result",
    "render_non_release_result",
    "render_released_result",
]
