from datetime import datetime, timedelta, timezone
import json

import pytest
from pydantic import ValidationError

from tractian_agent.contracts import ActionReceipt, ApiError, ApiErrorCategory
from tractian_agent.write_contracts import (
    IntentStatus,
    PersistedActionReceipt,
    PersistedApiError,
    ReprocessIntentScope,
    WriteIntent,
)
from tractian_agent.write_policy import (
    PolicyDecision,
    PolicyReason,
    ReprocessProposal,
    WritePolicyResult,
)


def _scope(**changes: object) -> ReprocessIntentScope:
    data = {
        "action": "reprocess_analysis",
        "case_id": "case_tkt_inv_04",
        "company_id": "comp_mineracao_andes",
        "user_id": "usr_pedro",
        "analysis_id": "an_9906",
        "justification": "Rolamento substituído; solicitar novo processamento.",
    }
    data.update(changes)
    return ReprocessIntentScope(**data)


def _policy_result(
    decision: PolicyDecision = PolicyDecision.ALLOW,
    reason: PolicyReason | None = None,
) -> WritePolicyResult:
    resolved_reason = reason or {
        PolicyDecision.ALLOW: PolicyReason.AUTHORIZED,
        PolicyDecision.REQUIRE_CONFIRMATION: PolicyReason.EXPLICIT_APPROVAL_REQUIRED,
        PolicyDecision.DENY: PolicyReason.MISSING_PERMISSION,
    }[decision]
    return WritePolicyResult(decision=decision, reason=resolved_reason)


def _intent(**changes: object) -> WriteIntent:
    data = {
        "intent_id": "intent_018f3a",
        "scope": _scope(),
        "payload_hash": "sha256:v1:" + "a" * 64,
        "decision": _policy_result(),
        "status": IntentStatus.PREPARED,
        "idempotency_key": "tractian-agent:018f3a",
        "expires_at": datetime(
            2026,
            9,
            6,
            9,
            30,
            tzinfo=timezone(timedelta(hours=-3)),
        ),
        "prepared_execution_id": "exec_02",
        "attempts": 0,
        "receipt": None,
        "error": None,
    }
    data.update(changes)
    return WriteIntent(**data)


def test_write_intent_records_the_complete_persistable_contract():
    intent = _intent()

    assert intent.intent_id == "intent_018f3a"
    assert intent.scope.analysis_id == "an_9906"
    assert intent.payload_hash == "sha256:v1:" + "a" * 64
    assert intent.decision.decision is PolicyDecision.ALLOW
    assert intent.status is IntentStatus.PREPARED
    assert intent.idempotency_key == "tractian-agent:018f3a"
    assert intent.expires_at == datetime(
        2026,
        9,
        6,
        9,
        30,
        tzinfo=timezone(timedelta(hours=-3)),
    )
    assert intent.prepared_execution_id == "exec_02"
    assert intent.attempts == 0
    assert intent.receipt is None
    assert intent.error is None


def test_intent_status_is_the_closed_seven_value_enum():
    assert {status.value for status in IntentStatus} == {
        "proposed",
        "awaiting_confirmation",
        "prepared",
        "completed",
        "denied",
        "failed",
        "uncertain",
    }

    with pytest.raises(ValidationError):
        _intent(status="retrying")


def test_write_intent_accepts_typed_receipt_or_typed_error():
    completed = _intent(
        status=IntentStatus.COMPLETED,
        attempts=1,
        receipt=ActionReceipt(
            accepted=True,
            action_id="act_1234abcd",
            message="Reprocesso aceito.",
        ),
    )
    failed = _intent(
        status=IntentStatus.FAILED,
        attempts=1,
        error=ApiError(
            category=ApiErrorCategory.API,
            code="NOT_FOUND",
            message="Análise não encontrada.",
            status_code=404,
        ),
    )

    assert completed.receipt.action_id == "act_1234abcd"
    assert failed.error.code == "NOT_FOUND"


@pytest.mark.parametrize(
    ("status", "decision", "result_kind"),
    [
        (IntentStatus.PROPOSED, PolicyDecision.ALLOW, None),
        (
            IntentStatus.AWAITING_CONFIRMATION,
            PolicyDecision.REQUIRE_CONFIRMATION,
            None,
        ),
        (IntentStatus.DENIED, PolicyDecision.DENY, None),
        (IntentStatus.PREPARED, PolicyDecision.ALLOW, "prepared"),
        (IntentStatus.COMPLETED, PolicyDecision.ALLOW, "receipt"),
        (IntentStatus.FAILED, PolicyDecision.ALLOW, "error"),
        (IntentStatus.UNCERTAIN, PolicyDecision.ALLOW, "error"),
    ],
)
def test_write_intent_accepts_each_consistent_status_variant(
    status,
    decision,
    result_kind,
):
    changes = {"status": status, "decision": _policy_result(decision)}
    if result_kind is None:
        changes.update(
            idempotency_key=None,
            expires_at=None,
            prepared_execution_id=None,
        )
    elif result_kind == "receipt":
        changes.update(
            attempts=1,
            receipt=ActionReceipt(
                accepted=True,
                action_id="act_1234abcd",
                message="Reprocesso aceito.",
            ),
        )
    elif result_kind == "error":
        changes.update(
            attempts=1,
            error=ApiError(
                category=ApiErrorCategory.TRANSPORT,
                code="CONNECTION_LOST",
                message="Conexão encerrada sem resposta.",
            ),
        )

    intent = _intent(**changes)
    restored = WriteIntent.model_validate_json(intent.model_dump_json())

    assert intent.status is status
    assert intent.decision.decision is decision
    assert restored == intent
    if result_kind == "receipt":
        assert isinstance(restored.receipt, PersistedActionReceipt)
    elif result_kind == "error":
        assert isinstance(restored.error, PersistedApiError)


_VALID_REASONS = {
    PolicyDecision.ALLOW: {PolicyReason.AUTHORIZED},
    PolicyDecision.REQUIRE_CONFIRMATION: {
        PolicyReason.EXPLICIT_APPROVAL_REQUIRED,
        PolicyReason.APPROVAL_SCOPE_MISMATCH,
    },
    PolicyDecision.DENY: {
        PolicyReason.MISSING_PERMISSION,
        PolicyReason.INVALID_JUSTIFICATION,
    },
}


@pytest.mark.parametrize(
    ("decision", "reason"),
    [
        (decision, reason)
        for decision in PolicyDecision
        for reason in PolicyReason
    ],
)
def test_write_intent_enforces_reason_for_each_policy_decision(decision, reason):
    status = {
        PolicyDecision.ALLOW: IntentStatus.PROPOSED,
        PolicyDecision.REQUIRE_CONFIRMATION: IntentStatus.AWAITING_CONFIRMATION,
        PolicyDecision.DENY: IntentStatus.DENIED,
    }[decision]
    changes = {
        "status": status,
        "decision": _policy_result(decision, reason),
        "idempotency_key": None,
        "expires_at": None,
        "prepared_execution_id": None,
    }

    if reason in _VALID_REASONS[decision]:
        assert _intent(**changes).decision.reason is reason
    else:
        with pytest.raises(ValidationError, match="razão.*decisão"):
            _intent(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {
            "status": IntentStatus.PROPOSED,
            "decision": _policy_result(PolicyDecision.DENY),
            "idempotency_key": None,
            "expires_at": None,
            "prepared_execution_id": None,
        },
        {
            "status": IntentStatus.AWAITING_CONFIRMATION,
            "decision": _policy_result(PolicyDecision.ALLOW),
            "idempotency_key": None,
            "expires_at": None,
            "prepared_execution_id": None,
        },
        {
            "status": IntentStatus.PROPOSED,
            "idempotency_key": None,
            "expires_at": None,
            "prepared_execution_id": None,
            "attempts": 1,
        },
        {
            "status": IntentStatus.DENIED,
            "decision": _policy_result(PolicyDecision.DENY),
        },
        {"status": IntentStatus.PREPARED, "idempotency_key": None},
        {"status": IntentStatus.PREPARED, "expires_at": None},
        {"status": IntentStatus.PREPARED, "prepared_execution_id": None},
        {"status": IntentStatus.PREPARED, "attempts": 1},
        {
            "status": IntentStatus.PREPARED,
            "receipt": ActionReceipt(
                accepted=True,
                action_id="act_1234abcd",
                message="Reprocesso aceito.",
            ),
        },
        {"status": IntentStatus.COMPLETED, "attempts": 0, "receipt": None},
        {
            "status": IntentStatus.COMPLETED,
            "attempts": 1,
            "decision": _policy_result(PolicyDecision.DENY),
            "receipt": ActionReceipt(
                accepted=True,
                action_id="act_1234abcd",
                message="Reprocesso aceito.",
            ),
        },
        {
            "status": IntentStatus.COMPLETED,
            "attempts": 1,
            "error": ApiError(
                category=ApiErrorCategory.TRANSPORT,
                code="CONNECTION_LOST",
                message="Conexão encerrada sem resposta.",
            ),
        },
        {"status": IntentStatus.FAILED, "attempts": 1, "error": None},
        {"status": IntentStatus.UNCERTAIN, "attempts": 1, "error": None},
        {
            "status": IntentStatus.FAILED,
            "attempts": 1,
            "receipt": ActionReceipt(
                accepted=True,
                action_id="act_1234abcd",
                message="Reprocesso aceito.",
            ),
        },
        {
            "status": IntentStatus.UNCERTAIN,
            "attempts": 1,
            "receipt": ActionReceipt(
                accepted=True,
                action_id="act_1234abcd",
                message="Reprocesso aceito.",
            ),
        },
    ],
)
def test_write_intent_rejects_inconsistent_status_fields(changes):
    with pytest.raises(ValidationError, match="status|intenção"):
        _intent(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"payload_hash": "a" * 64},
        {"payload_hash": "sha256:v1:not-hex"},
        {"attempts": -1},
    ],
)
def test_write_intent_rejects_invalid_hash_or_attempt_count(changes):
    with pytest.raises(ValidationError):
        _intent(**changes)


def test_write_models_reject_extra_fields_and_mutation():
    with pytest.raises(ValidationError):
        _scope(client="industrial-api-client")

    intent = _intent()
    with pytest.raises(ValidationError):
        intent.status = IntentStatus.COMPLETED
    with pytest.raises(ValidationError):
        intent.scope.analysis_id = "an_9907"


def test_write_intent_snapshots_mutable_receipt_and_error_contracts():
    receipt = ActionReceipt(
        accepted=True,
        action_id="act_1234abcd",
        message="Reprocesso aceito.",
    )
    error = ApiError(
        category=ApiErrorCategory.TRANSPORT,
        code="CONNECTION_LOST",
        message="Conexão encerrada sem resposta.",
    )
    completed = _intent(
        status=IntentStatus.COMPLETED,
        attempts=1,
        receipt=receipt,
    )
    failed = _intent(
        status=IntentStatus.FAILED,
        attempts=1,
        error=error,
    )
    completed_before = completed.model_dump_json()
    failed_before = failed.model_dump_json()

    receipt.message = "recibo mutado"
    error.message = "erro mutado"

    assert completed.model_dump_json() == completed_before
    assert failed.model_dump_json() == failed_before
    with pytest.raises(ValidationError):
        completed.receipt.message = "outra mutação"
    with pytest.raises(ValidationError):
        failed.error.message = "outra mutação"


def test_write_intent_serializes_as_plain_json_without_runtime_objects():
    intent = _intent()

    serialized = intent.model_dump(mode="json")
    encoded = json.dumps(serialized, allow_nan=False)

    assert '"expires_at": "2026-09-06T09:30:00-03:00"' in encoded
    assert all(
        forbidden not in encoded.casefold()
        for forbidden in ("client", "transport", "token", "golden_set")
    )

    with pytest.raises(ValidationError):
        _intent(receipt=object())


def test_reprocess_proposal_adds_only_the_internal_discriminator():
    proposal_fields = ReprocessProposal.model_json_schema()["properties"]

    assert set(proposal_fields) == {"action", "analysis_id", "justification"}
    assert proposal_fields["action"]["const"] == "reprocess_analysis"
    assert not {
        "execution_id",
        "identity",
        "permissions",
        "idempotency_key",
    } & set(proposal_fields)
