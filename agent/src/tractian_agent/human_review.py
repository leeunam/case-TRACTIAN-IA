"""Contratos públicos e funções puras da revisão humana retomável."""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json

from pydantic import ConfigDict, TypeAdapter, field_validator

from tractian_agent.contracts import StrictModel

from tractian_agent.state import (
    AgentDecision,
    EvidenceLedger,
    EvidenceQuality,
    ReleaseGateReason,
    ReleaseGateRecord,
    ReviewApproveReply,
    ReviewAudit,
    ReviewEditReply,
    ReviewExpiry,
    ReviewInterruptPayload,
    ReviewOperation,
    ReviewQuestion,
    ReviewRejectReply,
    ReviewReply,
    ReviewRequest,
    ReviewedDraft,
    ReviewerIdentity,
    WriterDraft,
    WriterNextStep,
)
from tractian_agent.writer import build_writer_context


_REVIEW_TTL = timedelta(hours=24)
_REPLY_ADAPTER = TypeAdapter(ReviewReply)


class ReviewResumeEnvelope(StrictModel):
    """Envelope interno criado somente pela fronteira autenticada."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    reply: ReviewReply
    reviewer: ReviewerIdentity
    received_at: datetime

    @field_validator("received_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("received_at exige UTC aware")
        return value


def canonical_digest(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"sha256:v1:{hashlib.sha256(encoded).hexdigest()}"


def _question(reason: ReleaseGateReason) -> ReviewQuestion:
    if reason is ReleaseGateReason.HUMAN_DISPOSITION_REQUIRED:
        return ReviewQuestion.CONFIRM_HUMAN_DISPOSITION
    if reason is ReleaseGateReason.WRITER_FAILURE:
        return ReviewQuestion.REBUILD_STRUCTURED_DRAFT
    return ReviewQuestion.ASSESS_BLOCKING_SAFETY


def _allowed_operations(
    reason: ReleaseGateReason,
    draft: WriterDraft | None,
) -> tuple[ReviewOperation, ...]:
    approve = (
        draft is not None
        and reason is ReleaseGateReason.HUMAN_DISPOSITION_REQUIRED
    )
    return (
        *((ReviewOperation.APPROVE,) if approve else ()),
        ReviewOperation.EDIT,
        ReviewOperation.REJECT,
    )


def build_review_request(
    *,
    request_id: str,
    request: object,
    gate: ReleaseGateRecord,
    ledger: EvidenceLedger,
    draft: WriterDraft | None,
    created_at: datetime,
) -> ReviewRequest:
    if gate.outcome.value != "require_human_review":
        raise ValueError("somente bloqueio do gate cria revisão")
    if created_at.tzinfo is None or created_at.utcoffset() != timedelta(0):
        raise ValueError("created_at exige UTC aware")
    eligible = tuple(
        sorted(
            item.evidence_id
            for item in ledger.items
            if item.request_id == request_id
            and item.quality is EvidenceQuality.CLAIMABLE
            and not item.obsolescence
        )
    )
    request_digest = canonical_digest(request)
    gate_digest = canonical_digest(gate)
    draft_digest = canonical_digest(draft)
    question = _question(gate.reason)
    operations = _allowed_operations(gate.reason, draft)
    expires_at = created_at + _REVIEW_TTL
    identity_fields = {
        "request_id": request_id,
        "request_digest": request_digest,
        "gate_digest": gate_digest,
        "draft_digest": draft_digest,
        "reason": gate.reason.value,
        "question": question.value,
        "subject_decision": gate.subject_decision.value,
        "eligible_evidence_ids": list(eligible),
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "allowed_operations": [item.value for item in operations],
    }
    return ReviewRequest(
        review_id=canonical_digest(
            {"review": identity_fields, "version": "human-review-v1"}
        ),
        request_id=request_id,
        request_digest=request_digest,
        gate_digest=gate_digest,
        gate_basis=gate,
        draft_digest=draft_digest,
        reason=gate.reason,
        question=question,
        subject_decision=gate.subject_decision,
        eligible_evidence_ids=eligible,
        draft=draft,
        created_at=created_at,
        expires_at=expires_at,
        allowed_operations=operations,
    )


def review_interrupt_payload(request: ReviewRequest) -> ReviewInterruptPayload:
    return ReviewInterruptPayload(
        review_id=request.review_id,
        reason=request.reason,
        question=request.question,
        eligible_evidence_ids=request.eligible_evidence_ids,
        draft_present=request.draft is not None,
        allowed_operations=request.allowed_operations,
        created_at=request.created_at,
        expires_at=request.expires_at,
    )


def build_reviewed_draft(
    request: ReviewRequest,
    reply: ReviewApproveReply | ReviewEditReply,
    ledger: EvidenceLedger,
) -> ReviewedDraft:
    if reply.review_id != request.review_id:
        raise ValueError("reply pertence a outra revisão")
    if ReviewOperation(reply.operation) not in request.allowed_operations:
        raise ValueError("operação não permitida para esta revisão")
    if isinstance(reply, ReviewApproveReply):
        if request.draft is None:
            raise ValueError("aprovação exige draft persistido")
        evidence_ids = request.draft.evidence_ids
    else:
        evidence_ids = reply.evidence_ids
    if not evidence_ids or any(
        item not in request.eligible_evidence_ids for item in evidence_ids
    ):
        raise ValueError("edição aceita somente evidências elegíveis atuais")
    limitations = build_writer_context(
        decision=request.subject_decision,
        ledger=ledger,
        missing_information=None,
    ).limitations
    next_step = (
        WriterNextStep.MONITOR
        if isinstance(reply, ReviewApproveReply)
        and request.subject_decision is AgentDecision.GUIDE
        else reply.next_step
        if isinstance(reply, ReviewEditReply)
        else request.draft.next_step
    )
    return ReviewedDraft(
        decision=request.subject_decision,
        evidence_ids=evidence_ids,
        limitation_refs=tuple(sorted(item.limitation_ref for item in limitations)),
        next_step=next_step,
    )


def build_review_audit(
    *,
    request: ReviewRequest,
    reply: ReviewReply,
    reviewer: ReviewerIdentity,
    received_at: datetime,
    reviewed_draft: ReviewedDraft | None,
) -> ReviewAudit:
    if reply.review_id != request.review_id:
        raise ValueError("reply pertence a outra revisão")
    if received_at.tzinfo is None or received_at.utcoffset() != timedelta(0):
        raise ValueError("received_at exige UTC aware")
    before = canonical_digest(request.draft)
    after = canonical_digest(reviewed_draft)
    return ReviewAudit(
        review_id=request.review_id,
        review_digest=canonical_digest(request),
        reviewer_id=reviewer.reviewer_id,
        company_id=reviewer.company_id,
        operation=ReviewOperation(reply.operation),
        received_at=received_at,
        reply_digest=canonical_digest(_REPLY_ADAPTER.dump_python(reply, mode="json")),
        before_digest=before,
        after_digest=after,
        structural_change=before != after,
    )


def review_is_valid(request: ReviewRequest, received_at: datetime) -> bool:
    if received_at.tzinfo is None or received_at.utcoffset() != timedelta(0):
        raise ValueError("received_at exige UTC aware")
    return received_at < request.expires_at


def review_audit_is_canonical(
    request: ReviewRequest,
    audit: ReviewAudit,
    reviewed_draft: ReviewedDraft | None,
) -> bool:
    if audit.operation is ReviewOperation.APPROVE:
        reply: ReviewReply = ReviewApproveReply(
            review_id=request.review_id,
            operation="approve",
        )
    elif audit.operation is ReviewOperation.EDIT:
        if reviewed_draft is None:
            return False
        reply = ReviewEditReply(
            review_id=request.review_id,
            operation="edit",
            evidence_ids=reviewed_draft.evidence_ids,
            next_step=reviewed_draft.next_step,
        )
    else:
        reply = ReviewRejectReply(
            review_id=request.review_id,
            operation="reject",
        )
    before = canonical_digest(request.draft)
    after = canonical_digest(reviewed_draft)
    return (
        audit.review_id == request.review_id
        and audit.review_digest == canonical_digest(request)
        and audit.operation in request.allowed_operations
        and audit.received_at < request.expires_at
        and audit.reply_digest == canonical_digest(reply)
        and audit.before_digest == before
        and audit.after_digest == after
        and audit.structural_change == (before != after)
        and (
            (audit.operation is ReviewOperation.REJECT)
            == (reviewed_draft is None)
        )
    )


__all__ = [
    "ReviewApproveReply", "ReviewAudit", "ReviewEditReply", "ReviewExpiry",
    "ReviewInterruptPayload", "ReviewOperation", "ReviewQuestion",
    "ReviewRejectReply", "ReviewReply", "ReviewRequest", "ReviewedDraft",
    "ReviewerIdentity", "ReviewResumeEnvelope", "build_review_audit", "build_review_request",
    "build_reviewed_draft", "canonical_digest", "review_interrupt_payload",
    "review_is_valid",
    "review_audit_is_canonical",
]
