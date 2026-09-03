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
    FinalResult,
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
    ReviewResolution,
    ReviewedDraft,
    ReviewerIdentity,
    ThreadScope,
    WriterDraft,
    WriterNextStep,
    allowed_review_operations,
)
from tractian_agent.tools.runtime import Permission
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
    return allowed_review_operations(
        reason,
        draft_present=draft is not None,
    )


def build_review_request(
    *,
    request_id: str,
    request: object,
    thread_scope: ThreadScope,
    permissions: frozenset[Permission],
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
        "thread_scope_digest": canonical_digest(thread_scope),
        "basis_permissions": sorted(permissions),
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
        thread_scope_digest=canonical_digest(thread_scope),
        basis_permissions=tuple(sorted(permissions)),
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
    empty_is_safe = (
        isinstance(reply, ReviewEditReply)
        and not evidence_ids
        and (
            (request.subject_decision, reply.next_step)
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
    )
    if (not evidence_ids and not empty_is_safe) or any(
        item not in request.eligible_evidence_ids for item in evidence_ids
    ):
        raise ValueError("edição aceita somente evidências elegíveis atuais")
    limitations = (
        ()
        if isinstance(reply, ReviewApproveReply)
        else build_writer_context(
            decision=request.subject_decision,
            ledger=ledger,
            missing_information=None,
        ).limitations
    )
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
        limitation_refs=(
            request.draft.limitation_refs
            if isinstance(reply, ReviewApproveReply)
            else tuple(sorted(item.limitation_ref for item in limitations))
        ),
        next_step=next_step,
    )


def build_review_audit(
    *,
    request: ReviewRequest,
    resolution: ReviewResolution,
    reviewed_draft: ReviewedDraft | None,
) -> ReviewAudit:
    reply = resolution.reply
    reviewer = resolution.reviewer
    received_at = resolution.received_at
    if reply.review_id != request.review_id:
        raise ValueError("reply pertence a outra revisão")
    if received_at.tzinfo is None or received_at.utcoffset() != timedelta(0):
        raise ValueError("received_at exige UTC aware")
    if not review_is_valid(request, received_at):
        raise ValueError("received_at está fora da janela da revisão")
    before = canonical_digest(request.draft)
    after = canonical_digest(reviewed_draft)
    return ReviewAudit(
        review_id=request.review_id,
        review_digest=canonical_digest(request),
        reviewer_id=reviewer.reviewer_id,
        company_id=reviewer.company_id,
        reviewer_permission=reviewer.permission,
        operation=ReviewOperation(reply.operation),
        received_at=received_at,
        reply_digest=canonical_digest(_REPLY_ADAPTER.dump_python(reply, mode="json")),
        resolution_digest=review_resolution_digest(resolution),
        before_digest=before,
        after_digest=after,
        structural_change=before != after,
    )


def review_is_valid(request: ReviewRequest, received_at: datetime) -> bool:
    if received_at.tzinfo is None or received_at.utcoffset() != timedelta(0):
        raise ValueError("received_at exige UTC aware")
    return request.created_at <= received_at < request.expires_at


def build_review_resolution(
    *,
    request: ReviewRequest,
    reply: ReviewReply,
    reviewer: ReviewerIdentity,
    received_at: datetime,
) -> ReviewResolution:
    if reply.review_id != request.review_id:
        raise ValueError("reply pertence a outra revisão")
    if received_at.tzinfo is None or received_at.utcoffset() != timedelta(0):
        raise ValueError("received_at exige UTC aware")
    if received_at < request.created_at:
        raise ValueError("received_at precede a criação da revisão")
    return ReviewResolution(
        review_id=request.review_id,
        reply=reply,
        reviewer=reviewer,
        received_at=received_at,
    )


def review_resolution_digest(resolution: ReviewResolution) -> str:
    return canonical_digest(resolution)


def review_resolution_subject_digest(
    reply: ReviewReply,
    reviewer: ReviewerIdentity,
) -> str:
    """Fingerprint estrutural do conteúdo confiável, sem o relógio do retry."""

    return canonical_digest(
        {
            "reply": _REPLY_ADAPTER.dump_python(reply, mode="json"),
            "reviewer": reviewer.model_dump(mode="json"),
        }
    )


def render_review_rejected_result() -> FinalResult:
    return FinalResult(
        decision=AgentDecision.REQUIRE_HUMAN_REVIEW,
        message="A revisão humana rejeitou a liberação da resposta.",
        next_step=WriterNextStep.AWAIT_HUMAN_REVIEW,
    )


def render_review_expired_result() -> FinalResult:
    return FinalResult(
        decision=AgentDecision.REQUIRE_HUMAN_REVIEW,
        message="A revisão expirou; uma nova solicitação é necessária.",
        next_step=WriterNextStep.AWAIT_HUMAN_REVIEW,
    )


def review_audit_is_canonical(
    request: ReviewRequest,
    audit: ReviewAudit,
    reviewed_draft: ReviewedDraft | None,
    ledger: EvidenceLedger,
    resolution: ReviewResolution,
) -> bool:
    reply = resolution.reply
    if audit.operation.value != reply.operation:
        return False
    try:
        expected_reviewed = (
            None
            if audit.operation is ReviewOperation.REJECT
            else build_reviewed_draft(request, reply, ledger)
        )
        if expected_reviewed != reviewed_draft:
            return False
        return audit == build_review_audit(
            request=request,
            resolution=resolution,
            reviewed_draft=expected_reviewed,
        )
    except (ValueError, TypeError):
        return False


__all__ = [
    "ReviewApproveReply", "ReviewAudit", "ReviewEditReply", "ReviewExpiry",
    "ReviewInterruptPayload", "ReviewOperation", "ReviewQuestion",
    "ReviewRejectReply", "ReviewReply", "ReviewRequest", "ReviewedDraft",
    "ReviewResolution", "ReviewerIdentity", "ReviewResumeEnvelope", "build_review_audit", "build_review_request",
    "build_review_resolution",
    "build_reviewed_draft", "canonical_digest", "review_interrupt_payload",
    "review_is_valid",
    "review_audit_is_canonical",
    "review_resolution_digest",
    "review_resolution_subject_digest",
    "render_review_expired_result",
    "render_review_rejected_result",
]
