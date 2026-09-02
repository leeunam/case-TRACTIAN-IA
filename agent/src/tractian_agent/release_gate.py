"""Porta determinística que atesta o draft e renderiza somente fatos do ledger."""

from __future__ import annotations

import hashlib
import json

from pydantic import ConfigDict, Field, model_validator

from tractian_agent.contracts import StrictModel
from tractian_agent.evidence import assess_evidence
from tractian_agent.state import (
    AgentDecision,
    EvidenceLedger,
    EvidenceQuality,
    EvidenceSufficiency,
    FinalResult,
    ReleaseGateOutcome,
    ReleaseGateReason,
    ReleaseGateRecord,
    WriterDraft,
    WriterFailureRecord,
    WriterNextStep,
)
from tractian_agent.tools.runtime import Permission
from tractian_agent.write_contracts import (
    IntentStatus,
    WriteIntent,
    intent_scope_material_parameters,
    intent_scope_target_id,
)
from tractian_agent.write_policy import (
    ApprovalSource,
    PolicyDecision,
    PolicyReason,
    TrustedActionApproval,
)
from tractian_agent.writer import WriterContext, build_writer_context


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
    draft: WriterDraft | None = None
    permissions: frozenset[Permission]
    intents: tuple[WriteIntent, ...] = ()
    approval: TrustedActionApproval | None = None
    missing_information: str | None = None
    writer_failure: WriterFailureRecord | None = None

    @model_validator(mode="after")
    def _require_current_intents(self) -> ReleaseGateContext:
        if any(intent.request_id != self.request_id for intent in self.intents):
            raise ValueError("contexto do gate aceita somente intenções da request atual")
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
            context.draft.model_dump(mode="json")
            if context.draft is not None
            else None
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
                "permissions": sorted(context.permissions),
                "request_id": context.request_id,
                "writer_failure": (
                    context.writer_failure.model_dump(mode="json")
                    if context.writer_failure is not None
                    else None
                ),
            }
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


def _expected_context(context: ReleaseGateContext) -> WriterContext:
    return build_writer_context(
        decision=context.decision,
        ledger=context.ledger,
        missing_information=context.missing_information,
    )


def _current_intent(context: ReleaseGateContext) -> WriteIntent | None:
    return context.intents[0] if len(context.intents) == 1 else None


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
    expected_approval = TrustedActionApproval(
        action=intent.scope.action,
        target_id=intent_scope_target_id(intent.scope),
        material_parameters=intent_scope_material_parameters(intent.scope),
        source=(
            context.approval.source
            if context.approval is not None
            else ApprovalSource.CONFIRMATION
        ),
    )
    if context.approval != expected_approval:
        return ReleaseGateReason.APPROVAL_MISMATCH
    if not any(
        item.intent_id == intent.intent_id
        and item.action == intent.scope.action
        and item.fact_path == "accepted"
        and item.value.to_python() is True
        and item.claimable
        for item in context.ledger.items
    ):
        return ReleaseGateReason.ACTION_EVIDENCE_MISSING
    return None


def evaluate_release(context: ReleaseGateContext) -> ReleaseGateRecord:
    """Recalcula todos os invariantes; nenhum veredito vem do modelo."""
    if context.writer_failure is not None or context.draft is None:
        return _review(context, ReleaseGateReason.WRITER_FAILURE)
    if context.ledger.request_id not in {context.request_id, None} or (
        context.ledger.request_id is None
        and bool(
            context.ledger.items
            or context.ledger.gaps
            or context.ledger.conflicts
        )
    ):
        return _review(context, ReleaseGateReason.REQUEST_MISMATCH)
    if context.draft.decision is not context.decision:
        return _review(context, ReleaseGateReason.DECISION_MISMATCH)

    expected = _expected_context(context)
    evidence_ids = tuple(fact.evidence_id for fact in expected.facts)
    limitation_refs = tuple(
        limitation.limitation_ref for limitation in expected.limitations
    )
    if context.draft.evidence_ids != evidence_ids:
        return _review(context, ReleaseGateReason.EVIDENCE_REFERENCE_MISMATCH)
    if context.draft.limitation_refs != limitation_refs:
        return _review(context, ReleaseGateReason.LIMITATION_REFERENCE_MISMATCH)
    if context.draft.next_step is not _NEXT_STEP_BY_DECISION[context.decision]:
        return _review(context, ReleaseGateReason.NEXT_STEP_MISMATCH)

    if context.decision is AgentDecision.REQUEST_INFORMATION:
        if context.missing_information is None or not context.missing_information.strip():
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
    if context.decision is AgentDecision.GUIDE:
        if "read" not in context.permissions:
            return _review(context, ReleaseGateReason.PERMISSION_INCOMPATIBLE)
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
    limitations_by_ref = {
        limitation.limitation_ref: limitation
        for limitation in _expected_context(context).limitations
    }
    limitation_messages = []
    for limitation_ref in context.draft.limitation_refs:
        limitation = limitations_by_ref[limitation_ref]
        description = limitation.detail or limitation.reason or limitation.kind
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
    "evaluate_release",
    "render_non_release_result",
    "render_released_result",
]
