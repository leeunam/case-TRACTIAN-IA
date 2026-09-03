from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from tractian_agent.contracts import ResponseMode
from tractian_agent.evidence import canonical_evidence_id
from tractian_agent.human_review import (
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
    build_review_audit,
    build_reviewed_draft,
    build_review_request,
    build_review_resolution,
    review_audit_is_canonical,
    review_is_valid,
    review_interrupt_payload,
)
from tractian_agent.release_gate import ReleaseGateContext, evaluate_release
from tractian_agent.state import (
    AgentDecision,
    EvidenceItem,
    EvidenceLedger,
    EvidenceQuality,
    EvidenceSourceKind,
    JsonSnapshot,
    ReleaseGateOutcome,
    ReleaseGateReason,
    ThreadScope,
    WriterDraft,
    WriterFailureCode,
    WriterFailureRecord,
    WriterNextStep,
)


NOW = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)


def _thread_scope(thread_id: str = "thread_review_01") -> ThreadScope:
    return ThreadScope(
        thread_id=thread_id,
        case_id="case_tkt_inv_04",
        company_id="comp_mineracao_andes",
        user_id="usr_pedro",
    )


def _reviewer() -> ReviewerIdentity:
    return ReviewerIdentity(
        reviewer_id="reviewer_01",
        company_id="comp_mineracao_andes",
        permission="review",
    )


def _resolution(request, reply, *, received_at=NOW + timedelta(minutes=1)):
    return build_review_resolution(
        request=request,
        reply=reply,
        reviewer=_reviewer(),
        received_at=received_at,
    )


def _ledger() -> EvidenceLedger:
    item = EvidenceItem(
        evidence_id="sha256:v1:" + "0" * 64,
        request_id="req_review_01",
        source_kind=EvidenceSourceKind.TOOL,
        call_id="call_review_01",
        tool="get_asset",
        resource="/assets/asset_G501",
        fact_path="asset.criticality",
        value=JsonSnapshot.capture("high", forbidden_names=frozenset()),
        mode=ResponseMode.COMPLETE,
        recorded_at=NOW,
        quality=EvidenceQuality.CLAIMABLE,
    )
    item = item.model_copy(update={"evidence_id": canonical_evidence_id(item)})
    return EvidenceLedger(request_id="req_review_01", items=(item,))


def _reviewable_context() -> ReleaseGateContext:
    ledger = _ledger()
    return ReleaseGateContext(
        request_id="req_review_01",
        decision=AgentDecision.GUIDE,
        ledger=ledger,
        draft=WriterDraft(
            decision=AgentDecision.GUIDE,
            evidence_ids=(ledger.items[0].evidence_id,),
            limitation_refs=(),
            next_step=WriterNextStep.REQUEST_HUMAN_DISPOSITION,
        ),
        permissions=frozenset({"read"}),
    )


def _request():
    context = _reviewable_context()
    gate = evaluate_release(context)
    return build_review_request(
        request_id=context.request_id,
        request={
            "case_id": "case_tkt_inv_04",
            "ticket_id": "TKT-INV-04",
            "asset_id": "asset_G501",
            "message": "Não exponha este texto no interrupt.",
            "identity": {
                "user_id": "usr_pedro",
                "company_id": "comp_mineracao_andes",
            },
        },
        thread_scope=_thread_scope(),
        permissions=context.permissions,
        gate=gate,
        ledger=context.ledger,
        draft=context.draft,
        created_at=NOW,
    )


def test_review_request_is_deterministic_strict_frozen_and_exactly_24_hours():
    first = _request()
    second = _request()

    assert first == second
    assert first.review_id.startswith("sha256:v1:")
    assert first.request_id == "req_review_01"
    assert first.reason is ReleaseGateReason.HUMAN_DISPOSITION_REQUIRED
    assert first.question is ReviewQuestion.CONFIRM_HUMAN_DISPOSITION
    assert first.subject_decision is AgentDecision.GUIDE
    assert first.expires_at == NOW + timedelta(hours=24)
    assert first.allowed_operations == (
        ReviewOperation.APPROVE,
        ReviewOperation.EDIT,
        ReviewOperation.REJECT,
    )
    assert type(first).model_validate_json(first.model_dump_json()) == first
    with pytest.raises(ValidationError):
        type(first).model_validate({**first.model_dump(), "extra": True})
    with pytest.raises(ValidationError):
        type(first).model_validate(
            {**first.model_dump(), "expires_at": NOW + timedelta(hours=23)}
        )
    with pytest.raises(ValidationError):
        type(first).model_validate(
            {**first.model_dump(), "created_at": NOW.replace(tzinfo=None)}
        )
    with pytest.raises(ValidationError):
        first.review_id = "changed"  # type: ignore[misc]


def test_review_contracts_reject_python_coercions_but_round_trip_json():
    request = _request()
    payload = review_interrupt_payload(request)
    edit = ReviewEditReply(
        review_id=request.review_id,
        operation="edit",
        evidence_ids=request.eligible_evidence_ids,
        next_step=WriterNextStep.MONITOR,
    )
    reviewed = build_reviewed_draft(request, edit, _ledger())
    audit = build_review_audit(
        request=request,
        resolution=_resolution(request, edit),
        reviewed_draft=reviewed,
    )
    expiry = ReviewExpiry(
        review_id=request.review_id,
        review_digest=audit.review_digest,
        resolution_digest=audit.resolution_digest,
        expired_at=request.expires_at,
    )
    resolution = _resolution(request, edit)
    models = (request, payload, edit, reviewed, audit, expiry, resolution)
    for model in models:
        assert type(model).model_validate_json(model.model_dump_json()) == model

    coercions = (
        (ReviewRequest, {**request.model_dump(), "eligible_evidence_ids": list(request.eligible_evidence_ids)}),
        (ReviewInterruptPayload, {**payload.model_dump(), "allowed_operations": list(payload.allowed_operations)}),
        (ReviewEditReply, {**edit.model_dump(), "evidence_ids": list(edit.evidence_ids)}),
        (ReviewedDraft, {**reviewed.model_dump(), "next_step": "monitor"}),
        (ReviewAudit, {**audit.model_dump(), "received_at": audit.received_at.isoformat()}),
        (ReviewExpiry, {**expiry.model_dump(), "expired_at": expiry.expired_at.isoformat()}),
        (ReviewResolution, {**resolution.model_dump(), "received_at": resolution.received_at.isoformat()}),
    )
    for model_type, value in coercions:
        with pytest.raises(ValidationError):
            model_type.model_validate(value)

    with pytest.raises(ValidationError):
        ReviewerIdentity.model_validate(
            {
                "reviewer_id": 1,
                "company_id": "comp_mineracao_andes",
                "permission": "review",
            }
        )
    with pytest.raises(ValidationError):
        ReviewApproveReply.model_validate(
            {
                "review_id": request.review_id,
                "operation": True,
            }
        )


def test_interrupt_payload_has_recursive_allowlist_and_no_sensitive_values():
    payload = review_interrupt_payload(_request())
    wire = payload.model_dump(mode="json")

    assert set(wire) == {
        "review_id",
        "reason",
        "question",
        "eligible_evidence_ids",
        "draft_present",
        "allowed_operations",
        "created_at",
        "expires_at",
    }
    forbidden_fragments = {
        "message",
        "identity",
        "permission",
        "value",
        "resource",
        "fact_path",
        "action",
        "target",
        "proposal",
        "intent",
        "receipt",
        "runtime",
        "rationale",
        "error",
        "secret",
        "asset_G501",
        "usr_pedro",
        "comp_mineracao_andes",
    }
    encoded = payload.model_dump_json().casefold()
    assert all(fragment not in encoded for fragment in forbidden_fragments)


def test_reviewer_identity_is_trusted_and_never_representable_in_reply():
    reviewer = ReviewerIdentity(
        reviewer_id="reviewer_01",
        company_id="comp_mineracao_andes",
        permission="review",
    )
    assert reviewer.permission == "review"
    with pytest.raises(ValidationError):
        ReviewerIdentity(
            reviewer_id="reviewer_01",
            company_id="comp_mineracao_andes",
            permission="read",  # type: ignore[arg-type]
        )

    review_id = _request().review_id
    reply_adapter = TypeAdapter(ReviewReply)
    replies = (
        ReviewApproveReply(review_id=review_id, operation="approve"),
        ReviewEditReply(
            review_id=review_id,
            operation="edit",
            evidence_ids=_request().eligible_evidence_ids,
            next_step=WriterNextStep.MONITOR,
        ),
        ReviewRejectReply(review_id=review_id, operation="reject"),
    )
    for reply in replies:
        assert reply_adapter.validate_json(reply.model_dump_json()) == reply
        wire = reply.model_dump(mode="json")
        assert not {
            "reviewer_id",
            "company_id",
            "permission",
            "author",
            "received_at",
            "rationale",
            "decision",
            "text",
            "limitations",
            "action",
        } & set(wire)
    with pytest.raises(ValidationError):
        ReviewApproveReply.model_validate(
            {
                "review_id": review_id,
                "operation": "approve",
                "reviewer_id": "forged",
            }
        )


def test_approve_is_not_offered_without_a_valid_draft():
    context = _reviewable_context().model_copy(
        update={"draft": None, "writer_failure": None}
    )
    gate = evaluate_release(context)
    request = build_review_request(
        request_id=context.request_id,
        request={"safe": True},
        thread_scope=_thread_scope(),
        permissions=context.permissions,
        gate=gate,
        ledger=context.ledger,
        draft=None,
        created_at=NOW,
    )

    assert gate.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert ReviewOperation.APPROVE not in request.allowed_operations
    assert request.allowed_operations == (
        ReviewOperation.EDIT,
        ReviewOperation.REJECT,
    )


def test_approve_clears_only_safe_human_disposition_and_releases():
    context = _reviewable_context()
    request = _request()
    reply = ReviewApproveReply(review_id=request.review_id, operation="approve")
    reviewed = build_reviewed_draft(request, reply, context.ledger)
    reviewer = ReviewerIdentity(
        reviewer_id="reviewer_01",
        company_id="comp_mineracao_andes",
        permission="review",
    )
    audit = build_review_audit(
        request=request,
        resolution=build_review_resolution(
            request=request,
            reply=reply,
            reviewer=reviewer,
            received_at=NOW + timedelta(minutes=1),
        ),
        reviewed_draft=reviewed,
    )
    resolution = build_review_resolution(
        request=request,
        reply=reply,
        reviewer=reviewer,
        received_at=NOW + timedelta(minutes=1),
    )
    regated = context.model_copy(
        update={
            "draft": reviewed,
            "review_request": request,
            "review_audit": audit,
            "review_resolution": resolution,
        }
    )

    assert reviewed.decision is AgentDecision.GUIDE
    assert reviewed.next_step is WriterNextStep.MONITOR
    assert evaluate_release(regated).outcome is ReleaseGateOutcome.RELEASE
    assert audit.reviewer_id == "reviewer_01"
    assert audit.structural_change is True
    assert "criticality" not in audit.model_dump_json()


def test_edit_preserves_human_order_but_derives_limitations_and_regates():
    context = _reviewable_context()
    request = _request()
    reply = ReviewEditReply(
        review_id=request.review_id,
        operation="edit",
        evidence_ids=request.eligible_evidence_ids,
        next_step=WriterNextStep.MONITOR,
    )
    reviewed = build_reviewed_draft(request, reply, context.ledger)
    audit = build_review_audit(
        request=request,
        resolution=_resolution(request, reply),
        reviewed_draft=reviewed,
    )

    assert reviewed.evidence_ids == reply.evidence_ids
    assert reviewed.limitation_refs == ()
    assert evaluate_release(
        context.model_copy(
            update={
                "draft": reviewed,
                "review_request": request,
                "review_audit": audit,
                "review_resolution": _resolution(request, reply),
            }
        )
    ).outcome is ReleaseGateOutcome.RELEASE

    with pytest.raises(ValueError, match="elegíveis"):
        build_reviewed_draft(
            request,
            reply.model_copy(update={"evidence_ids": ("sha256:v1:" + "f" * 64,)}),
            context.ledger,
        )
    with pytest.raises(ValidationError):
        ReviewEditReply(
            review_id=request.review_id,
            operation="edit",
            evidence_ids=(request.eligible_evidence_ids[0],) * 2,
            next_step=WriterNextStep.MONITOR,
        )


def test_hard_gate_reason_cannot_be_approved_or_overridden_by_edit():
    context = _reviewable_context().model_copy(update={"permissions": frozenset()})
    gate = evaluate_release(context)
    request = build_review_request(
        request_id=context.request_id,
        request={"safe": True},
        thread_scope=_thread_scope(),
        permissions=context.permissions,
        gate=gate,
        ledger=context.ledger,
        draft=context.draft,
        created_at=NOW,
    )

    assert gate.reason is ReleaseGateReason.PERMISSION_INCOMPATIBLE
    assert ReviewOperation.APPROVE not in request.allowed_operations
    with pytest.raises(ValueError, match="não permitida"):
        build_reviewed_draft(
            request,
            ReviewApproveReply(review_id=request.review_id, operation="approve"),
            context.ledger,
        )
    edited = build_reviewed_draft(
        request,
        ReviewEditReply(
            review_id=request.review_id,
            operation="edit",
            evidence_ids=request.eligible_evidence_ids,
            next_step=WriterNextStep.MONITOR,
        ),
        context.ledger,
    )
    edit_reply = ReviewEditReply(
        review_id=request.review_id,
        operation="edit",
        evidence_ids=request.eligible_evidence_ids,
        next_step=WriterNextStep.MONITOR,
    )
    audit = build_review_audit(
        request=request,
        resolution=_resolution(request, edit_reply),
        reviewed_draft=edited,
    )
    assert evaluate_release(
        context.model_copy(
            update={
                "draft": edited,
                "review_request": request,
                "review_audit": audit,
                "review_resolution": _resolution(request, edit_reply),
            }
        )
    ).reason is ReleaseGateReason.PERMISSION_INCOMPATIBLE


def test_writer_failure_can_be_structurally_edited_without_model_call():
    context = _reviewable_context().model_copy(
        update={
            "draft": None,
            "writer_failure": WriterFailureRecord(
                code=WriterFailureCode.INVALID_STRUCTURED_OUTPUT,
                attempts=2,
                repairable=True,
            ),
        }
    )
    gate = evaluate_release(context)
    request = build_review_request(
        request_id=context.request_id,
        request={"safe": True},
        thread_scope=_thread_scope(),
        permissions=context.permissions,
        gate=gate,
        ledger=context.ledger,
        draft=None,
        created_at=NOW,
    )
    reply = ReviewEditReply(
        review_id=request.review_id,
        operation="edit",
        evidence_ids=request.eligible_evidence_ids,
        next_step=WriterNextStep.MONITOR,
    )
    reviewed = build_reviewed_draft(request, reply, context.ledger)
    audit = build_review_audit(
        request=request,
        resolution=_resolution(request, reply),
        reviewed_draft=reviewed,
    )

    assert evaluate_release(
        context.model_copy(
            update={
                "draft": reviewed,
                "review_request": request,
                "review_audit": audit,
                "review_resolution": _resolution(request, reply),
            }
        )
    ).outcome is ReleaseGateOutcome.RELEASE


def test_reject_audit_contains_only_structural_digests_and_expiry_is_exclusive():
    request = _request()
    reply = ReviewRejectReply(review_id=request.review_id, operation="reject")
    reviewer = ReviewerIdentity(
        reviewer_id="reviewer_01",
        company_id="comp_mineracao_andes",
        permission="review",
    )
    audit = build_review_audit(
        request=request,
        resolution=build_review_resolution(
            request=request,
            reply=reply,
            reviewer=reviewer,
            received_at=request.expires_at - timedelta(microseconds=1),
        ),
        reviewed_draft=None,
    )

    assert audit.operation is ReviewOperation.REJECT
    assert audit.structural_change is True
    assert review_is_valid(request, request.expires_at - timedelta(microseconds=1))
    assert not review_is_valid(request, request.expires_at)
    assert not review_is_valid(request, request.expires_at + timedelta(microseconds=1))
    with pytest.raises(ValueError, match="UTC"):
        review_is_valid(request, request.expires_at.replace(tzinfo=None))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("request_digest", "sha256:v1:" + "f" * 64),
        ("gate_digest", "sha256:v1:" + "f" * 64),
        ("draft_digest", "sha256:v1:" + "f" * 64),
        ("eligible_evidence_ids", ("sha256:v1:" + "f" * 64,)),
    ],
)
def test_review_request_rejects_request_gate_draft_and_eligible_tamper(
    field,
    replacement,
):
    request = _request()
    with pytest.raises(ValidationError):
        type(request).model_validate(
            {**request.model_dump(), field: replacement}
        )


def test_regate_rejects_audit_tamper():
    context = _reviewable_context()
    request = _request()
    reply = ReviewApproveReply(review_id=request.review_id, operation="approve")
    reviewed = build_reviewed_draft(request, reply, context.ledger)
    audit = build_review_audit(
        request=request,
        resolution=_resolution(request, reply),
        reviewed_draft=reviewed,
    )
    tampered = audit.model_copy(
        update={"reply_digest": "sha256:v1:" + "f" * 64}
    )

    result = evaluate_release(
        context.model_copy(
            update={
                "draft": reviewed,
                "review_request": request,
                "review_audit": tampered,
                "review_resolution": _resolution(request, reply),
            }
        )
    )
    assert result.outcome is ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
    assert result.reason is ReleaseGateReason.REQUEST_MISMATCH


def test_review_id_is_bound_to_thread_scope():
    first = _request()
    context = _reviewable_context()
    second = build_review_request(
        request_id=context.request_id,
        request={
            "case_id": "case_tkt_inv_04",
            "ticket_id": "TKT-INV-04",
            "asset_id": "asset_G501",
            "message": "Não exponha este texto no interrupt.",
            "identity": {
                "user_id": "usr_pedro",
                "company_id": "comp_mineracao_andes",
            },
        },
        thread_scope=_thread_scope("thread_review_02"),
        permissions=context.permissions,
        gate=evaluate_release(context),
        ledger=context.ledger,
        draft=context.draft,
        created_at=NOW,
    )

    assert first.thread_scope_digest != second.thread_scope_digest
    assert first.review_id != second.review_id


def test_approve_audit_rejects_reviewed_draft_not_derived_from_original():
    context = _reviewable_context()
    request = _request()
    reply = ReviewApproveReply(review_id=request.review_id, operation="approve")
    resolution = _resolution(request, reply)
    forged = ReviewedDraft(
        decision=AgentDecision.GUIDE,
        evidence_ids=(),
        limitation_refs=(),
        next_step=WriterNextStep.MONITOR,
    )
    forged_audit = build_review_audit(
        request=request,
        resolution=resolution,
        reviewed_draft=forged,
    )

    assert not review_audit_is_canonical(
        request, forged_audit, forged, context.ledger, resolution
    )
    regated = evaluate_release(
        context.model_copy(
            update={
                "draft": forged,
                "review_request": request,
                "review_resolution": resolution,
                "review_audit": forged_audit,
            }
        )
    )
    assert regated.reason is ReleaseGateReason.REQUEST_MISMATCH


def test_audit_binds_trusted_author_and_boundary_time():
    request = _request()
    reply = ReviewRejectReply(review_id=request.review_id, operation="reject")
    resolution = _resolution(request, reply)
    audit = build_review_audit(
        request=request,
        resolution=resolution,
        reviewed_draft=None,
    )

    forged_author = audit.model_copy(update={"reviewer_id": "reviewer_forged"})
    assert not review_audit_is_canonical(
        request, forged_author, None, _ledger(), resolution
    )
    forged_time = audit.model_copy(
        update={"received_at": audit.received_at + timedelta(seconds=1)}
    )
    assert not review_audit_is_canonical(
        request, forged_time, None, _ledger(), resolution
    )
    with pytest.raises(ValueError, match="precede"):
        build_review_resolution(
            request=request,
            reply=reply,
            reviewer=_reviewer(),
            received_at=request.created_at - timedelta(microseconds=1),
        )


def test_request_information_writer_failure_accepts_fact_free_safe_edit():
    context = ReleaseGateContext(
        request_id="req_review_01",
        decision=AgentDecision.REQUEST_INFORMATION,
        ledger=EvidenceLedger(request_id="req_review_01"),
        draft=None,
        permissions=frozenset({"read"}),
        missing_information="qual é o identificador da análise?",
        writer_failure=WriterFailureRecord(
            code=WriterFailureCode.INVALID_STRUCTURED_OUTPUT,
            attempts=2,
            repairable=True,
        ),
    )
    request = build_review_request(
        request_id=context.request_id,
        request={"safe": True},
        thread_scope=_thread_scope(),
        permissions=context.permissions,
        gate=evaluate_release(context),
        ledger=context.ledger,
        draft=None,
        created_at=NOW,
    )
    reply = ReviewEditReply(
        review_id=request.review_id,
        operation="edit",
        evidence_ids=(),
        next_step=WriterNextStep.PROVIDE_INFORMATION,
    )

    reviewed = build_reviewed_draft(request, reply, context.ledger)
    assert reviewed.evidence_ids == ()
    assert reviewed.next_step is WriterNextStep.PROVIDE_INFORMATION
    resolution = _resolution(request, reply)
    audit = build_review_audit(
        request=request,
        resolution=resolution,
        reviewed_draft=reviewed,
    )
    regated = evaluate_release(
        context.model_copy(
            update={
                "draft": reviewed,
                "review_request": request,
                "review_resolution": resolution,
                "review_audit": audit,
            }
        )
    )
    assert regated.outcome is ReleaseGateOutcome.REQUEST_INFORMATION, regated.reason
