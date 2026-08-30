import json

import pytest
from pydantic import JsonValue, ValidationError

from tractian_agent.contracts import Identity, SupportRequest, ToolCall
from tractian_agent.state import (
    AgentDecision,
    AgentState,
    FinalResult,
    PersistedMessage,
    ReviewRecord,
    ReviewStatus,
    StateEvidence,
    ThreadScope,
    ToolObservation,
)
from tractian_agent.tools.observations import (
    ToolArtifact,
    ToolOutcome,
    ToolSource,
)
from tractian_agent.tools.runtime import TrustedIdentity
from tractian_agent.write_contracts import IntentStatus, ReprocessIntentScope, WriteIntent
from tractian_agent.write_policy import (
    ApprovalSource,
    PolicyDecision,
    PolicyReason,
    ReprocessProposal,
    TrustedActionApproval,
    WritePolicyResult,
)


def _request(
    *,
    case_id: str = "case_tkt_inv_04",
    company_id: str = "comp_mineracao_andes",
    user_id: str = "usr_pedro",
) -> SupportRequest:
    return SupportRequest(
        case_id=case_id,
        ticket_id="TKT-INV-04",
        asset_id="asset_G501",
        message="Por que não recebi nenhum aviso?",
        identity=Identity(user_id=user_id, company_id=company_id),
    )


def _identity(
    *,
    company_id: str = "comp_mineracao_andes",
    user_id: str = "usr_pedro",
) -> TrustedIdentity:
    return TrustedIdentity(user_id=user_id, company_id=company_id)


def _scope() -> ThreadScope:
    return ThreadScope(
        thread_id="thread_case_tkt_inv_04",
        case_id="case_tkt_inv_04",
        company_id="comp_mineracao_andes",
        user_id="usr_pedro",
    )


def _state(**changes: object) -> AgentState:
    data = {
        "request": _request(),
        "identity": _identity(),
        "permissions": frozenset({"read", "action_low"}),
        "request_id": "req_01",
        "thread_id": "thread_case_tkt_inv_04",
        "execution_id": "exec_01",
        "thread_scope": _scope(),
        "step_limit": 3,
    }
    data.update(changes)
    return AgentState(**data)


def _intent() -> WriteIntent:
    return WriteIntent(
        intent_id="intent_018f3a",
        scope=ReprocessIntentScope(
            action="reprocess_analysis",
            case_id="case_tkt_inv_04",
            company_id="comp_mineracao_andes",
            user_id="usr_pedro",
            analysis_id="an_9906",
            justification="Rolamento substituído; solicitar novo processamento.",
        ),
        payload_hash="sha256:v1:" + "a" * 64,
        decision=WritePolicyResult(
            decision=PolicyDecision.ALLOW,
            reason=PolicyReason.AUTHORIZED,
        ),
        status=IntentStatus.PREPARED,
        idempotency_key="tractian-agent:018f3a",
        prepared_execution_id="exec_01",
        attempts=0,
    )


def test_agent_state_contains_the_complete_persistable_contract():
    proposal = ReprocessProposal(
        analysis_id="an_9906",
        justification="Rolamento substituído; solicitar novo processamento.",
    )
    approval = TrustedActionApproval(
        action="reprocess_analysis",
        target_id="an_9906",
        source=ApprovalSource.ORIGINAL_REQUEST,
    )
    call = ToolCall[dict[str, JsonValue]](
        call_id="call_01",
        name="get_analysis",
        arguments={"analysis_id": "an_9906"},
    )
    observation = ToolObservation(
        call_id="call_01",
        artifact=ToolArtifact(
            tool_name="get_analysis",
            arguments={"analysis_id": "an_9906"},
            source=ToolSource(kind="industrial_api", resource="/analyses/an_9906"),
            outcome=ToolOutcome(partial_data={"id": "an_9906"}),
        ),
    )
    evidence = StateEvidence(
        evidence_id="evidence_01",
        call_id="call_01",
        value={"analysis_id": "an_9906", "status": "current"},
    )
    state = _state(
        messages=(PersistedMessage(role="user", content="Investigue a análise."),),
        tool_calls=(call,),
        tool_observations=(observation,),
        evidence=(evidence,),
        decision=AgentDecision.ACT,
        step_count=1,
        pending_proposal=proposal,
        approval=approval,
        intents=(_intent(),),
        final_result=FinalResult(
            decision=AgentDecision.ACT,
            message="Reprocesso preparado.",
        ),
        review=ReviewRecord(status=ReviewStatus.NOT_REQUIRED),
    )

    assert state.request.case_id == "case_tkt_inv_04"
    assert state.identity.user_id == "usr_pedro"
    assert state.request_id == "req_01"
    assert state.thread_id == "thread_case_tkt_inv_04"
    assert state.execution_id == "exec_01"
    assert state.messages[0].role.value == "user"
    assert state.tool_calls[0].name == "get_analysis"
    assert state.tool_observations[0].call_id == "call_01"
    assert state.evidence[0].evidence_id == "evidence_01"
    assert state.decision is AgentDecision.ACT
    assert state.step_count == 1
    assert state.step_limit == 3
    assert state.pending_proposal.analysis_id == "an_9906"
    assert state.approval.target_id == "an_9906"
    assert state.intents[0].intent_id == "intent_018f3a"
    assert state.final_result.message == "Reprocesso preparado."
    assert state.review.status is ReviewStatus.NOT_REQUIRED


def test_new_state_starts_with_empty_typed_evidence_and_observable_collections():
    state = _state()

    assert state.messages == ()
    assert state.tool_calls == ()
    assert state.tool_observations == ()
    assert state.evidence == ()
    assert state.intents == ()
    assert state.step_count == 0
    assert state.decision is None
    assert state.pending_proposal is None
    assert state.approval is None
    assert state.final_result is None
    assert state.review is None


def test_same_thread_accepts_new_request_and_execution_for_the_same_scope():
    continued = _state().continue_with(
        request=_request(),
        identity=_identity(),
        permissions=frozenset({"read"}),
        request_id="req_02",
        execution_id="exec_02",
    )

    assert continued.thread_id == "thread_case_tkt_inv_04"
    assert continued.request_id == "req_02"
    assert continued.execution_id == "exec_02"
    assert continued.permissions == frozenset({"read"})


@pytest.mark.parametrize(
    ("support_request", "trusted_identity"),
    [
        (_request(case_id="case_other"), _identity()),
        (
            _request(company_id="comp_other"),
            _identity(company_id="comp_other"),
        ),
        (
            _request(user_id="usr_other"),
            _identity(user_id="usr_other"),
        ),
    ],
)
def test_same_thread_fails_closed_for_another_case_company_or_person(
    support_request,
    trusted_identity,
):
    with pytest.raises(ValidationError):
        _state().continue_with(
            request=support_request,
            identity=trusted_identity,
            permissions=frozenset({"read"}),
            request_id="req_02",
            execution_id="exec_02",
        )


def test_every_continuation_requires_a_new_execution_id():
    with pytest.raises(ValueError, match="execution_id"):
        _state().continue_with(
            request=_request(),
            identity=_identity(),
            permissions=frozenset({"read"}),
            request_id="req_01",
            execution_id="exec_01",
        )


def test_state_rejects_extra_fields_and_mutation():
    with pytest.raises(ValidationError):
        _state(client=object())

    state = _state()
    with pytest.raises(ValidationError):
        state.execution_id = "exec_fabricated"


def test_state_serializes_to_plain_json_without_runtime_or_restricted_data():
    state = _state(
        messages=(PersistedMessage(role="user", content="Consulte o ativo."),),
        tool_calls=(
            ToolCall[dict[str, JsonValue]](
                call_id="call_01",
                name="get_asset",
                arguments={"asset_id": "asset_G501"},
            ),
        ),
        evidence=(
            StateEvidence(
                evidence_id="evidence_01",
                call_id="call_01",
                value={"asset_id": "asset_G501"},
            ),
        ),
    )

    serialized = state.model_dump(mode="json")
    encoded = json.dumps(serialized, allow_nan=False)

    assert isinstance(serialized, dict)
    assert all(
        forbidden not in encoded.casefold()
        for forbidden in ("client", "transport", "token", "golden_set", "expected_paths")
    )

    with pytest.raises(ValidationError):
        _state(
            evidence=(
                {
                    "evidence_id": "evidence_01",
                    "call_id": "call_01",
                    "value": object(),
                },
            ),
        )


@pytest.mark.parametrize(
    "forbidden_name",
    ["client", "transport", "token", "golden_set", "expected_paths"],
)
def test_state_rejects_restricted_names_inside_persisted_json(forbidden_name):
    with pytest.raises(ValidationError):
        _state(
            tool_calls=(
                ToolCall[dict[str, JsonValue]](
                    call_id="call_01",
                    name="get_asset",
                    arguments={forbidden_name: "must-not-persist"},
                ),
            ),
        )


def test_step_limit_is_positive_and_state_cannot_start_beyond_budget():
    with pytest.raises(ValidationError):
        _state(step_limit=0)
    with pytest.raises(ValidationError):
        _state(step_count=4, step_limit=3)


def test_advance_step_stops_at_the_budget_without_mutating_prior_state():
    initial = _state(step_limit=2)

    first = initial.advance_step()
    second = first.advance_step()

    assert initial.step_count == 0
    assert first.step_count == 1
    assert second.step_count == 2
    with pytest.raises(ValueError, match="orçamento"):
        second.advance_step()
