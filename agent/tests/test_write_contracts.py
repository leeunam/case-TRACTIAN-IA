from datetime import datetime, timezone
import json

import pytest
from pydantic import ValidationError

from tractian_agent.contracts import ActionReceipt, ApiError, ApiErrorCategory
from tractian_agent.write_contracts import (
    IntentStatus,
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
) -> WritePolicyResult:
    reason = (
        PolicyReason.AUTHORIZED
        if decision is PolicyDecision.ALLOW
        else PolicyReason.EXPLICIT_APPROVAL_REQUIRED
    )
    return WritePolicyResult(decision=decision, reason=reason)


def _intent(**changes: object) -> WriteIntent:
    data = {
        "intent_id": "intent_018f3a",
        "scope": _scope(),
        "payload_hash": "sha256:v1:" + "a" * 64,
        "decision": _policy_result(),
        "status": IntentStatus.PREPARED,
        "idempotency_key": "tractian-agent:018f3a",
        "expires_at": datetime(2026, 9, 6, tzinfo=timezone.utc),
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
    assert intent.expires_at == datetime(2026, 9, 6, tzinfo=timezone.utc)
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


def test_write_intent_serializes_as_plain_json_without_runtime_objects():
    intent = _intent()

    serialized = intent.model_dump(mode="json")
    encoded = json.dumps(serialized, allow_nan=False)

    assert '"expires_at": "2026-09-06T00:00:00Z"' in encoded
    assert all(
        forbidden not in encoded.casefold()
        for forbidden in ("client", "transport", "token", "golden_set")
    )

    with pytest.raises(ValidationError):
        _intent(receipt=object())


def test_task_one_public_proposal_does_not_expose_trusted_runtime_fields():
    proposal_fields = ReprocessProposal.model_json_schema()["properties"]

    assert set(proposal_fields) == {"analysis_id", "justification"}
    assert not {
        "execution_id",
        "identity",
        "permissions",
        "idempotency_key",
    } & set(proposal_fields)
