from datetime import datetime, timedelta, timezone
import json

import pytest
from pydantic import JsonValue, ValidationError

from tractian_agent.contracts import (
    ActionReceipt,
    ApiError,
    ApiErrorCategory,
    Identity,
    ResponseMode,
    SupportRequest,
    ToolCall,
)
from tractian_agent.state import (
    AgentDecision,
    AgentState,
    FinalResult,
    JsonSnapshot,
    PersistedToolArtifact,
    PersistedToolCall,
    PersistedMessage,
    PlannerFailureRecord,
    PlannerTerminalRecord,
    ResumeAnchor,
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
from tractian_agent.tools.analyses import (
    AnalysisDetailToolArtifact,
    AnalysisDetailToolOutcome,
    AnalysisListToolArtifact,
    AnalysisListToolOutcome,
)
from tractian_agent.tools.assets import AssetToolArtifact, AssetToolOutcome
from tractian_agent.tools.knowledge import (
    KnowledgeDocumentToolArtifact,
    KnowledgeDocumentToolOutcome,
    KnowledgeSearchToolArtifact,
    KnowledgeSearchToolOutcome,
    ModelToolArtifact,
    ModelToolOutcome,
)
from tractian_agent.tools.runtime import TrustedIdentity
from tractian_agent.tools.technical import (
    BaselineToolArtifact,
    BaselineToolOutcome,
    DataQualityToolArtifact,
    DataQualityToolOutcome,
    RmsToolArtifact,
    RmsToolOutcome,
    SpectrumToolArtifact,
    SpectrumToolOutcome,
)
from tractian_agent.write_contracts import (
    IntentStatus,
    PersistedActionReceipt,
    PersistedApiError,
    ReprocessIntentScope,
    WriteIntent,
)
from tractian_agent.write_policy import (
    ApprovalSource,
    EscalateCaseProposal,
    PolicyDecision,
    PolicyReason,
    ReprocessProposal,
    RequestModelRetrainingProposal,
    RequestSpecialistAnalysisProposal,
    TrustedActionApproval,
    UpdateAssetCriticalityProposal,
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
        expires_at=datetime(
            2026,
            9,
            6,
            9,
            30,
            tzinfo=timezone(timedelta(hours=-3)),
        ),
        prepared_execution_id="exec_01",
        attempts=0,
    )


def test_agent_state_requires_unique_intent_ids_and_one_active_per_request():
    first_data = _intent().model_dump(mode="python")
    first_data.update(
        request_id="req_write_01",
        status=IntentStatus.PROPOSED,
        idempotency_key=None,
        expires_at=None,
        prepared_execution_id=None,
    )
    first = WriteIntent.model_validate(first_data)

    with pytest.raises(ValidationError, match="IDs de intenção.*únicos"):
        _state(intents=(first, first))

    second_data = first.model_dump(mode="python")
    second_data["intent_id"] = "intent_second"
    second = WriteIntent.model_validate(second_data)
    with pytest.raises(ValidationError, match="no máximo uma intenção ativa"):
        _state(intents=(first, second))


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
    assert state.resume_anchor is ResumeAnchor.START
    assert state.planner_terminal is None
    assert state.planner_failure is None


def test_planner_progress_round_trips_as_typed_json_safe_state():
    state = _state(
        resume_anchor=ResumeAnchor.PLANNER_SELECT,
        planner_terminal=PlannerTerminalRecord(
            decision="request_information",
            stop_reason="missing_information",
            missing_information="Informe o ponto de medição.",
        ),
    )

    restored = AgentState.model_validate_json(state.model_dump_json())

    assert restored == state
    assert restored.resume_anchor is ResumeAnchor.PLANNER_SELECT
    assert restored.planner_terminal is not None
    assert restored.planner_terminal.missing_information == (
        "Informe o ponto de medição."
    )
    assert restored.planner_failure is None

    failed = _state(
        resume_anchor=ResumeAnchor.PLANNER_TOOL,
        decision=AgentDecision.REQUIRE_HUMAN_REVIEW,
        final_result=FinalResult(
            decision=AgentDecision.REQUIRE_HUMAN_REVIEW,
            message="O ciclo terminou de forma segura.",
        ),
        review=ReviewRecord(
            status=ReviewStatus.REQUIRED,
            reason="planner:planner_tool:invalid_tool_result",
        ),
        planner_failure=PlannerFailureRecord(
            stage="planner_tool", code="invalid_tool_result"
        ),
    )
    assert AgentState.model_validate_json(failed.model_dump_json()) == failed


def test_planner_failure_requires_a_coherent_safe_terminal_state():
    failure = PlannerFailureRecord(
        stage="planner_tool",
        code="invalid_tool_result",
    )

    with pytest.raises(ValidationError, match="falha do planner exige"):
        _state(planner_failure=failure)

    with pytest.raises(ValidationError, match="mutuamente exclusivos"):
        _state(
            planner_failure=failure,
            planner_terminal=PlannerTerminalRecord(
                decision="guide",
                stop_reason="sufficient_evidence",
            ),
            decision=AgentDecision.REQUIRE_HUMAN_REVIEW,
            final_result=FinalResult(
                decision=AgentDecision.REQUIRE_HUMAN_REVIEW,
                message="O ciclo terminou de forma segura.",
            ),
            review=ReviewRecord(status=ReviewStatus.REQUIRED),
        )


@pytest.mark.parametrize(
    ("anchor", "planner_terminal", "decision"),
    [
        (
            ResumeAnchor.PLANNER_FINALIZE,
            PlannerTerminalRecord(
                decision="guide",
                stop_reason="sufficient_evidence",
            ),
            AgentDecision.ACT,
        ),
        (ResumeAnchor.FINISH, None, AgentDecision.REQUEST_INFORMATION),
    ],
)
def test_terminal_state_rejects_decision_that_diverges_from_anchor_contract(
    anchor: ResumeAnchor,
    planner_terminal: PlannerTerminalRecord | None,
    decision: AgentDecision,
):
    with pytest.raises(ValidationError, match="terminal diverge"):
        _state(
            resume_anchor=anchor,
            planner_terminal=planner_terminal,
            decision=decision,
            final_result=FinalResult(
                decision=decision,
                message="Checkpoint terminal adulterado.",
            ),
        )


def test_terminal_state_rejects_incompatible_write_intent_or_execution_result():
    proposal = ReprocessProposal(
        analysis_id="an_9906",
        justification="A intenção terminal deve manter seu status canônico.",
    )
    denied_data = _intent().model_dump(mode="python")
    denied_data.update(
        request_id="req_01",
        status=IntentStatus.DENIED,
        decision=WritePolicyResult(
            decision=PolicyDecision.DENY,
            reason=PolicyReason.MISSING_PERMISSION,
        ),
        idempotency_key=None,
        expires_at=None,
        prepared_execution_id=None,
    )
    denied = WriteIntent.model_validate(denied_data)
    denial_state = _state(
        pending_proposal=proposal,
        intents=(denied,),
        decision=AgentDecision.GUIDE,
        final_result=FinalResult(
            decision=AgentDecision.GUIDE,
            message="A política recusou a ação.",
        ),
        resume_anchor=ResumeAnchor.WRITE_POLICY,
    )
    denial_wire = denial_state.model_dump(mode="json")
    denial_wire["intents"][0]["status"] = IntentStatus.AWAITING_CONFIRMATION.value
    denial_wire["intents"][0]["decision"] = {
        "decision": PolicyDecision.REQUIRE_CONFIRMATION.value,
        "reason": PolicyReason.EXPLICIT_APPROVAL_REQUIRED.value,
    }

    with pytest.raises(ValidationError, match="terminal diverge"):
        AgentState.model_validate(denial_wire)

    completed_data = _intent().model_dump(mode="python")
    completed_data.update(
        request_id="req_01",
        status=IntentStatus.COMPLETED,
        attempts=1,
        receipt=ActionReceipt(
            accepted=True,
            action_id="act_terminal_state",
            message="Reprocesso concluído.",
        ),
    )
    completed = WriteIntent.model_validate(completed_data)
    execution_state = _state(
        pending_proposal=proposal,
        intents=(completed,),
        decision=AgentDecision.ACT,
        final_result=FinalResult(
            decision=AgentDecision.ACT,
            message="Reprocesso concluído.",
        ),
        resume_anchor=ResumeAnchor.EXECUTE_ACTION,
    )
    execution_wire = execution_state.model_dump(mode="json")
    execution_wire["decision"] = AgentDecision.GUIDE.value
    execution_wire["final_result"]["decision"] = AgentDecision.GUIDE.value

    with pytest.raises(ValidationError, match="terminal diverge"):
        AgentState.model_validate(execution_wire)


def test_new_request_resets_request_bound_planner_progress_and_anchor():
    terminal_prior = _state(
        resume_anchor=ResumeAnchor.PLANNER_TOOL,
        planner_terminal=PlannerTerminalRecord(
            decision="guide",
            stop_reason="sufficient_evidence",
        ),
    )
    failed_prior = _state(
        resume_anchor=ResumeAnchor.PLANNER_SELECT,
        planner_failure=PlannerFailureRecord(
            stage="planner_select",
            code="repeated_tool_call",
        ),
        decision=AgentDecision.REQUIRE_HUMAN_REVIEW,
        final_result=FinalResult(
            decision=AgentDecision.REQUIRE_HUMAN_REVIEW,
            message="O ciclo terminou de forma segura.",
        ),
        review=ReviewRecord(status=ReviewStatus.REQUIRED),
    )

    continued_states = tuple(
        prior.continue_with(
            request=_request(),
            identity=prior.identity,
            permissions=prior.permissions,
            request_id="req_02",
            execution_id="exec_02",
            step_limit=20,
        )
        for prior in (terminal_prior, failed_prior)
    )

    for continued in continued_states:
        assert continued.resume_anchor is ResumeAnchor.START
        assert continued.planner_terminal is None
        assert continued.planner_failure is None


@pytest.mark.parametrize(
    "proposal",
    [
        ReprocessProposal(
            analysis_id="an_9906",
            justification="Há dados novos para reprocessar esta análise.",
        ),
        RequestSpecialistAnalysisProposal(
            analysis_id="an_9906",
            justification="A limitação registrada exige análise especializada.",
        ),
        UpdateAssetCriticalityProposal(
            criticality="critical",
            justification="O impacto operacional exige criticidade mais alta.",
        ),
        RequestModelRetrainingProposal(
            justification="Erros sistemáticos sustentam solicitar novo treinamento.",
        ),
        EscalateCaseProposal(
            justification="O caso ultrapassa o atendimento remoto disponível.",
        ),
    ],
)
def test_pending_proposal_restores_each_discriminated_variant(proposal):
    original = _state(pending_proposal=proposal)

    restored = AgentState.model_validate_json(original.model_dump_json())

    assert restored.pending_proposal == proposal
    assert type(restored.pending_proposal) is type(proposal)


def test_agent_state_restores_the_unambiguous_legacy_reprocess_proposal():
    current = _state(
        pending_proposal=ReprocessProposal(
            analysis_id="an_9906",
            justification="Há dados novos para reprocessar esta análise.",
        )
    )
    legacy_payload = current.model_dump(mode="json")
    del legacy_payload["pending_proposal"]["action"]

    restored = AgentState.model_validate(legacy_payload)

    assert restored.pending_proposal == current.pending_proposal
    assert restored.pending_proposal.action == "reprocess_analysis"


@pytest.mark.parametrize(
    "unknown_proposal",
    [
        {"analysis_id": "an_9906"},
        {"justification": "Há dados novos para reprocessar esta análise."},
        {
            "analysis_id": "an_9906",
            "justification": "Há dados novos para reprocessar esta análise.",
            "criticality": "high",
        },
        {
            "analysis_id": "an_9906",
            "justification": "Há dados novos para reprocessar esta análise.",
            "expected_path": ["POST /analyses/an_9906/reprocess"],
        },
        {
            "action": "unknown_action",
            "analysis_id": "an_9906",
            "justification": "Há dados novos para reprocessar esta análise.",
        },
        {
            "action": "reprocess_analysis",
            "analysis_id": "an_9906",
            "justification": "Há dados novos para reprocessar esta análise.",
            "expected_path": ["POST /analyses/an_9906/reprocess"],
        },
    ],
)
def test_agent_state_does_not_migrate_incomplete_ambiguous_or_unknown_proposals(
    unknown_proposal,
):
    payload = _state().model_dump(mode="json")
    payload["pending_proposal"] = unknown_proposal

    with pytest.raises(ValidationError, match="pending_proposal"):
        AgentState.model_validate(payload)


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


@pytest.mark.parametrize(
    ("scope_field", "other_value"),
    [
        ("case_id", "case_other"),
        ("company_id", "comp_other"),
        ("user_id", "usr_other"),
    ],
)
def test_state_rejects_intent_from_another_thread_scope(scope_field, other_value):
    intent_data = _intent().model_dump(mode="python")
    intent_data["scope"][scope_field] = other_value

    with pytest.raises(ValidationError, match="intenção.*escopo"):
        _state(intents=(WriteIntent.model_validate(intent_data),))


def test_every_continuation_requires_a_new_execution_id():
    with pytest.raises(ValueError, match="execution_id"):
        _state().continue_with(
            request=_request(),
            identity=_identity(),
            permissions=frozenset({"read"}),
            request_id="req_01",
            execution_id="exec_01",
        )


def test_same_request_resume_requires_identical_request_and_preserves_progress():
    approval = TrustedActionApproval(
        action="reprocess_analysis",
        target_id="an_9906",
        source=ApprovalSource.CONFIRMATION,
    )
    state = _state(
        messages=(PersistedMessage(role="user", content="Solicitação original."),),
        evidence=(
            StateEvidence(
                evidence_id="evidence_01",
                call_id="call_01",
                value={"analysis_id": "an_9906"},
            ),
        ),
        decision=AgentDecision.ACT,
        step_count=2,
        pending_proposal=ReprocessProposal(
            analysis_id="an_9906",
            justification="Rolamento substituído; solicitar novo processamento.",
        ),
        approval=approval,
        intents=(_intent(),),
        final_result=FinalResult(
            decision=AgentDecision.ACT,
            message="Aguardando execução.",
        ),
        review=ReviewRecord(status=ReviewStatus.NOT_REQUIRED),
    )

    resumed = state.continue_with(
        request=_request(),
        identity=_identity(),
        permissions=frozenset({"read"}),
        request_id="req_01",
        execution_id="exec_02",
        step_limit=99,
    )

    assert resumed.step_count == 2
    assert resumed.step_limit == 3
    assert resumed.decision is AgentDecision.ACT
    assert resumed.pending_proposal is not None
    assert resumed.approval == approval
    assert resumed.final_result is not None
    assert resumed.review is not None
    assert resumed.messages == state.messages
    assert resumed.evidence == state.evidence
    assert resumed.intents == state.intents

    changed_request = _request().model_copy(update={"message": "Pedido alterado."})
    with pytest.raises(ValueError, match="solicitação.*idêntica"):
        state.continue_with(
            request=changed_request,
            identity=_identity(),
            permissions=frozenset({"read"}),
            request_id="req_01",
            execution_id="exec_03",
        )


def test_new_request_preserves_audit_and_intents_but_resets_transient_state():
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
            outcome=ToolOutcome(partial_data={"status": "current"}),
        ),
    )
    state = _state(
        messages=(PersistedMessage(role="user", content="Solicitação original."),),
        tool_calls=(call,),
        tool_observations=(observation,),
        evidence=(
            StateEvidence(
                evidence_id="evidence_01",
                call_id="call_01",
                value={"analysis_id": "an_9906"},
            ),
        ),
        decision=AgentDecision.ACT,
        step_count=3,
        pending_proposal=ReprocessProposal(
            analysis_id="an_9906",
            justification="Rolamento substituído; solicitar novo processamento.",
        ),
        approval=TrustedActionApproval(
            action="reprocess_analysis",
            target_id="an_9906",
            source=ApprovalSource.CONFIRMATION,
        ),
        intents=(_intent(),),
        final_result=FinalResult(
            decision=AgentDecision.ACT,
            message="Reprocesso preparado.",
        ),
        review=ReviewRecord(status=ReviewStatus.NOT_REQUIRED),
    )
    next_request = _request().model_copy(
        update={"message": "Qual é o estado atual da análise?"}
    )

    continued = state.continue_with(
        request=next_request,
        identity=_identity(),
        permissions=frozenset({"read"}),
        request_id="req_02",
        execution_id="exec_02",
        step_limit=7,
    )

    assert continued.messages == state.messages
    assert continued.tool_calls == state.tool_calls
    assert continued.tool_observations == state.tool_observations
    assert continued.evidence == state.evidence
    assert continued.intents == state.intents
    assert continued.decision is None
    assert continued.step_count == 0
    assert continued.step_limit == 7
    assert continued.pending_proposal is None
    assert continued.approval is None
    assert continued.final_result is None
    assert continued.review is None
    assert continued.advance_step().step_count == 1


def test_state_rejects_extra_fields_and_mutation():
    with pytest.raises(ValidationError):
        _state(client=object())

    state = _state()
    with pytest.raises(ValidationError):
        state.execution_id = "exec_fabricated"


def test_state_is_detached_from_mutable_request_calls_artifact_and_evidence():
    request = _request()
    call = ToolCall[dict[str, JsonValue]](
        call_id="call_01",
        name="get_analysis",
        arguments={"analysis_id": "an_9906"},
    )
    artifact = ToolArtifact(
        tool_name="get_analysis",
        arguments={"analysis_id": "an_9906"},
        source=ToolSource(kind="industrial_api", resource="/analyses/an_9906"),
        outcome=ToolOutcome(partial_data={"status": "current"}),
    )
    evidence_value = {"analysis_id": "an_9906", "status": "current"}
    state = _state(
        request=request,
        tool_calls=(call,),
        tool_observations=(ToolObservation(call_id="call_01", artifact=artifact),),
        evidence=(
            StateEvidence(
                evidence_id="evidence_01",
                call_id="call_01",
                value=evidence_value,
            ),
        ),
    )
    persisted_before = state.model_dump_json()

    request.message = "conteúdo mutado"
    call.arguments["client"] = "leaked"
    artifact.arguments["client"] = "leaked"
    artifact.outcome.partial_data["client"] = "leaked"
    evidence_value["client"] = "leaked"

    assert state.model_dump_json() == persisted_before
    with pytest.raises(ValidationError):
        state.request.message = "outra mutação"
    with pytest.raises(TypeError):
        state.tool_calls[0].arguments["client"] = "leaked"
    with pytest.raises(TypeError):
        state.tool_observations[0].artifact.outcome.partial_data["client"] = "leaked"
    with pytest.raises(TypeError):
        state.evidence[0].value["client"] = "leaked"


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
        tool_observations=(
            ToolObservation(
                call_id="call_01",
                artifact=ToolArtifact(
                    tool_name="get_analysis",
                    arguments={"analysis_id": "an_9906"},
                    source=ToolSource(
                        kind="industrial_api",
                        resource="/analyses/an_9906",
                    ),
                    outcome=ToolOutcome(partial_data={"status": "current"}),
                ),
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


def test_agent_state_round_trips_real_json_with_receipt_error_and_offset():
    prepared_data = _intent().model_dump(mode="python")
    completed_data = {
        **prepared_data,
        "intent_id": "intent_completed",
        "status": IntentStatus.COMPLETED,
        "attempts": 1,
        "receipt": ActionReceipt(
            accepted=True,
            action_id="act_1234abcd",
            message="Reprocesso aceito.",
        ),
    }
    failed_data = {
        **prepared_data,
        "intent_id": "intent_failed",
        "status": IntentStatus.FAILED,
        "attempts": 1,
        "error": ApiError(
            category=ApiErrorCategory.TRANSPORT,
            code="CONNECTION_LOST",
            message="Conexão encerrada sem resposta.",
        ),
    }
    state = _state(
        messages=(PersistedMessage(role="user", content="Investigue a análise."),),
        tool_calls=(
            ToolCall[dict[str, JsonValue]](
                call_id="call_01",
                name="get_analysis",
                arguments={"analysis_id": "an_9906"},
            ),
        ),
        tool_observations=(
            ToolObservation(
                call_id="call_01",
                artifact=ToolArtifact(
                    tool_name="get_analysis",
                    arguments={"analysis_id": "an_9906"},
                    source=ToolSource(
                        kind="industrial_api",
                        resource="/analyses/an_9906",
                    ),
                    outcome=ToolOutcome(partial_data={"status": "current"}),
                ),
            ),
        ),
        evidence=(
            StateEvidence(
                evidence_id="evidence_01",
                call_id="call_01",
                value={"analysis_id": "an_9906", "status": "current"},
            ),
        ),
        intents=(
            WriteIntent.model_validate(completed_data),
            WriteIntent.model_validate(failed_data),
        ),
    )

    restored = AgentState.model_validate_json(state.model_dump_json())

    assert restored == state
    assert isinstance(restored.intents[0].receipt, PersistedActionReceipt)
    assert isinstance(restored.intents[1].error, PersistedApiError)
    assert restored.intents[0].expires_at.utcoffset() == timedelta(hours=-3)


def test_tool_observation_round_trips_json_safe_next_turn_content():
    observation = ToolObservation(
        call_id="call_01",
        content={
            "analysis_id": "an_9906",
            "status": "current",
            "limitations": ["Sinal disponível somente até 10:00."],
        },
        artifact=ToolArtifact(
            tool_name="get_analysis",
            arguments={"analysis_id": "an_9906"},
            source=ToolSource(
                kind="industrial_api",
                resource="/analyses/an_9906",
            ),
            outcome=ToolOutcome(partial_data={"status": "current"}),
        ),
    )

    restored = ToolObservation.model_validate_json(observation.model_dump_json())

    assert restored == observation
    assert restored.content is not None
    assert restored.content.to_python() == {
        "analysis_id": "an_9906",
        "limitations": ["Sinal disponível somente até 10:00."],
        "status": "current",
    }

    api_error = ApiError(
        category=ApiErrorCategory.TIMEOUT,
        code="READ_TIMEOUT",
        message="A consulta excedeu o tempo limite.",
    )
    failed_observation = ToolObservation(
        call_id="call_timeout",
        content={"error": api_error.model_dump(mode="json")},
        artifact=ToolArtifact(
            tool_name="get_analysis",
            arguments={"analysis_id": "an_9906"},
            source=ToolSource(
                kind="industrial_api",
                resource="/analyses/an_9906",
            ),
            outcome=ToolOutcome(error=api_error),
        ),
    )
    restored_failure = ToolObservation.model_validate_json(
        failed_observation.model_dump_json()
    )

    assert restored_failure == failed_observation
    assert isinstance(restored_failure.artifact.outcome.error, PersistedApiError)

    with pytest.raises(ValidationError):
        ToolObservation(
            call_id="call_unsafe",
            content={"raw_http_response": {"token": "must-not-persist"}},
            artifact=observation.artifact,
        )


_READ_ARTIFACT_CASES = [
        (
            "get_asset",
            AssetToolArtifact,
            AssetToolOutcome,
            {"asset_id": "asset_G501"},
            "/assets/asset_G501",
        ),
        (
            "list_asset_analyses",
            AnalysisListToolArtifact,
            AnalysisListToolOutcome,
            {"asset_id": "asset_G501"},
            "/assets/asset_G501/analyses",
        ),
        (
            "get_analysis",
            AnalysisDetailToolArtifact,
            AnalysisDetailToolOutcome,
            {"analysis_id": "an_9906"},
            "/analyses/an_9906",
        ),
        (
            "get_baseline",
            BaselineToolArtifact,
            BaselineToolOutcome,
            {"asset_id": "asset_G501", "point_id": None},
            "/assets/asset_G501/baseline",
        ),
        (
            "get_rms_series",
            RmsToolArtifact,
            RmsToolOutcome,
            {"asset_id": "asset_G501", "point_id": None},
            "/assets/asset_G501/rms",
        ),
        (
            "get_spectrum",
            SpectrumToolArtifact,
            SpectrumToolOutcome,
            {"asset_id": "asset_G501", "point_id": None},
            "/assets/asset_G501/spectrum",
        ),
        (
            "get_data_quality",
            DataQualityToolArtifact,
            DataQualityToolOutcome,
            {"asset_id": "asset_G501", "point_id": None},
            "/assets/asset_G501/data-quality",
        ),
        (
            "get_model",
            ModelToolArtifact,
            ModelToolOutcome,
            {},
            "/models/mdl_vib_v3",
        ),
        (
            "search_knowledge",
            KnowledgeSearchToolArtifact,
            KnowledgeSearchToolOutcome,
            {"query": "rolamento"},
            "/knowledge/search",
        ),
        (
            "get_knowledge_document",
            KnowledgeDocumentToolArtifact,
            KnowledgeDocumentToolOutcome,
            {"document_id": "kb_bearing_guidance"},
            "/knowledge/kb_bearing_guidance",
        ),
]


@pytest.mark.parametrize(
    ("tool_name", "artifact_type", "outcome_type", "arguments", "resource"),
    _READ_ARTIFACT_CASES,
)
def test_tool_observation_rehydrates_exact_read_artifact_after_json_round_trip(
    tool_name,
    artifact_type,
    outcome_type,
    arguments,
    resource,
):
    error = ApiError(
        category=ApiErrorCategory.TIMEOUT,
        code="READ_TIMEOUT",
        message="A consulta excedeu o tempo limite.",
    )
    artifact = artifact_type(
        tool_name=tool_name,
        arguments=arguments,
        source=ToolSource(kind="industrial_api", resource=resource),
        outcome=outcome_type(error=error),
    )
    observation = ToolObservation(
        request_id="req_01",
        call_id=f"call_{tool_name}",
        content={"error": error.model_dump(mode="json")},
        artifact=artifact,
    )

    restored = ToolObservation.model_validate_json(observation.model_dump_json())
    restored_artifact = restored.artifact.validated_read_artifact()

    assert type(restored_artifact) is artifact_type
    assert type(restored_artifact.outcome) is outcome_type
    assert restored_artifact.model_dump(mode="json") == artifact.model_dump(
        mode="json"
    )


@pytest.mark.parametrize(
    ("tool_name", "artifact_type", "outcome_type", "arguments", "resource"),
    _READ_ARTIFACT_CASES,
)
@pytest.mark.parametrize(
    "unsafe_partial_data",
    [
        {"outer": {"identity": {"user_id": "usr_leaked"}}},
        {"outer": {"headers": {"authorization": "leaked"}}},
        {"outer": {"raw_response": {"body": "leaked"}}},
    ],
    ids=["identity", "headers", "raw-response"],
)
def test_degraded_read_artifact_rejects_unsafe_partial_data_before_checkpoint(
    tool_name,
    artifact_type,
    outcome_type,
    arguments,
    resource,
    unsafe_partial_data,
):
    artifact = artifact_type(
        tool_name=tool_name,
        arguments=arguments,
        source=ToolSource(kind="industrial_api", resource=resource),
        outcome=outcome_type(
            mode=ResponseMode.PARTIAL,
            notes="Resposta parcial.",
            partial_data=unsafe_partial_data,
        ),
    )

    with pytest.raises(ValidationError, match="campo proibido"):
        ToolObservation(
            request_id="req_01",
            call_id=f"call_{tool_name}",
            content={
                "mode": "partial",
                "notes": "Resposta parcial.",
                "partial_data": unsafe_partial_data,
            },
            artifact=artifact,
        )


@pytest.mark.parametrize(
    ("tool_name", "artifact_type", "outcome_type", "arguments", "resource"),
    _READ_ARTIFACT_CASES,
)
def test_raw_read_artifact_rejects_coerced_types_before_projection(
    tool_name,
    artifact_type,
    outcome_type,
    arguments,
    resource,
):
    wire = artifact_type(
        tool_name=tool_name,
        arguments=arguments,
        source=ToolSource(kind="industrial_api", resource=resource),
        outcome=outcome_type(
            error=ApiError(
                category=ApiErrorCategory.TIMEOUT,
                code="READ_TIMEOUT",
                message="A consulta excedeu o tempo limite.",
            )
        ),
    ).model_dump(mode="json")
    wire["omitted_items"] = "0"

    with pytest.raises(ValidationError):
        PersistedToolArtifact.model_validate(wire)


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "case_id",
        "company_id",
        "user_id",
        "identity",
        "permissions",
        "approval",
        "thread_id",
        "request_id",
        "execution_id",
        "idempotency_key",
        "central_asset_id",
        "configured_model_id",
        "client",
        "transport",
        "seed",
        "runtime",
        "context",
        "url",
        "method",
        "headers",
        "token",
        "golden_set",
        "expected_paths",
        "test_scenarios",
    ],
)
def test_tool_call_rejects_all_trusted_public_argument_names(forbidden_name):
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


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "case_id",
        "company_id",
        "user_id",
        "identity",
        "permissions",
        "approval",
        "thread_id",
        "request_id",
        "execution_id",
        "idempotency_key",
        "central_asset_id",
        "configured_model_id",
        "client",
        "transport",
        "seed",
        "runtime",
        "context",
        "url",
        "method",
        "headers",
        "token",
        "golden_set",
        "expected_paths",
        "test_scenarios",
    ],
)
def test_artifact_rejects_all_trusted_public_argument_names(forbidden_name):
    with pytest.raises(ValidationError):
        _state(
            tool_observations=(
                ToolObservation(
                    call_id="call_01",
                    artifact=ToolArtifact(
                        tool_name="get_asset",
                        arguments={forbidden_name: "must-not-persist"},
                        source=ToolSource(
                            kind="industrial_api",
                            resource="/assets/asset_G501",
                        ),
                        outcome=ToolOutcome(),
                    ),
                ),
            ),
        )


@pytest.mark.parametrize("boundary", ["call", "artifact"])
@pytest.mark.parametrize(
    "invalid_arguments",
    [
        ["asset_G501"],
        "asset_G501",
        7,
        None,
        {1: "asset_G501"},
    ],
)
def test_persisted_arguments_require_json_object_with_string_keys(
    boundary,
    invalid_arguments,
):
    with pytest.raises(ValidationError, match="arguments|argumentos"):
        if boundary == "call":
            PersistedToolCall(
                call_id="call_01",
                name="get_asset",
                arguments=invalid_arguments,
            )
        else:
            PersistedToolArtifact(
                tool_name="get_asset",
                arguments=invalid_arguments,
                source={"kind": "industrial_api", "resource": "/assets/asset_G501"},
                outcome={},
            )


def test_persisted_argument_objects_keep_nested_mapping_and_round_trip():
    arguments = {
        "asset_id": "asset_G501",
        "filters": {"status": ["current", "stale"]},
    }
    call = PersistedToolCall(
        call_id="call_01",
        name="custom_nested_tool",
        arguments=arguments,
    )
    artifact = PersistedToolArtifact(
        tool_name="custom_nested_tool",
        arguments=arguments,
        source={"kind": "industrial_api", "resource": "/assets/asset_G501"},
        outcome={},
    )

    restored_call = PersistedToolCall.model_validate_json(call.model_dump_json())
    restored_artifact = PersistedToolArtifact.model_validate_json(
        artifact.model_dump_json()
    )

    assert restored_call.arguments.to_python() == arguments
    assert restored_artifact.arguments.to_python() == arguments


def test_json_snapshot_round_trips_its_explicit_wire_representation():
    domain_value = {
        "analysis_id": "an_9906",
        "measurements": [1, {"status": "current"}],
    }
    snapshot = JsonSnapshot.capture(domain_value, forbidden_names=frozenset())

    wire = snapshot.model_dump_json()
    restored = JsonSnapshot.model_validate_json(wire)

    assert json.loads(wire) == {"encoded": snapshot.encoded}
    assert restored == snapshot
    assert restored.to_python() == domain_value


def test_json_snapshot_validation_and_serialization_schemas_match_the_wire():
    validation_schema = JsonSnapshot.model_json_schema(mode="validation")
    serialization_schema = JsonSnapshot.model_json_schema(mode="serialization")

    assert validation_schema == serialization_schema
    assert validation_schema["type"] == "object"
    assert validation_schema["required"] == ["encoded"]
    assert validation_schema["properties"]["encoded"]["type"] == "string"
    assert validation_schema["additionalProperties"] is False


def test_nested_snapshot_uses_the_same_explicit_wire_and_schema():
    call = PersistedToolCall(
        call_id="call_01",
        name="get_asset",
        arguments={"filters": {"status": ["current"]}},
    )

    wire = call.model_dump_json()
    restored = PersistedToolCall.model_validate_json(wire)
    validation_schema = PersistedToolCall.model_json_schema(mode="validation")
    serialization_schema = PersistedToolCall.model_json_schema(mode="serialization")

    assert json.loads(wire)["arguments"] == {"encoded": call.arguments.encoded}
    assert restored == call
    assert validation_schema == serialization_schema


@pytest.mark.parametrize("boundary", ["call", "artifact"])
@pytest.mark.parametrize(
    "forbidden_name",
    [
        "access_token",
        "accessToken",
        "client-secret",
        "clientSecret",
        "trusted_identity",
        "trustedIdentity",
        "action-approval",
        "actionApproval",
        "agent_thread_id",
        "agentThreadId",
        "http_response",
        "httpResponse",
        "response-body",
        "responseBody",
        "reasoning_trace_detail",
        "reasoningTraceDetail",
    ],
)
def test_public_arguments_reject_nested_segmented_sensitive_aliases(
    boundary,
    forbidden_name,
):
    arguments = {"outer": {forbidden_name: "must-not-persist"}}

    with pytest.raises(ValidationError):
        if boundary == "call":
            PersistedToolCall(
                call_id="call_01",
                name="get_asset",
                arguments=arguments,
            )
        else:
            PersistedToolArtifact(
                tool_name="get_asset",
                arguments=arguments,
                source={"kind": "industrial_api", "resource": "/assets/asset_G501"},
                outcome={},
            )


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "access_token",
        "clientSecret",
        "http_response",
        "responseBody",
        "reasoning_trace_detail",
    ],
)
def test_technical_data_rejects_nested_segmented_sensitive_aliases(
    forbidden_name,
):
    with pytest.raises(ValidationError):
        StateEvidence(
            evidence_id="evidence_01",
            call_id="call_01",
            value={"outer": {forbidden_name: "must-not-persist"}},
        )


def test_technical_data_preserves_nested_domain_identifiers_and_names():
    domain_data = {
        "outer": {
            "case_id": "case_tkt_inv_04",
            "companyId": "comp_mineracao_andes",
            "user-id": "usr_pedro",
            "asset_id": "asset_G501",
            "analysisId": "an_9906",
            "machine_runtime_hours": 72,
            "bearingAuthenticity": "verified",
            "response-time": 12,
        }
    }

    evidence = StateEvidence(
        evidence_id="evidence_01",
        call_id="call_01",
        value=domain_data,
    )
    outcome = ToolObservation(
        call_id="call_01",
        artifact=ToolArtifact(
            tool_name="get_analysis",
            arguments={"analysis_id": "an_9906"},
            source=ToolSource(kind="industrial_api", resource="/analyses/an_9906"),
            outcome=ToolOutcome(partial_data=domain_data),
        ),
    ).artifact.outcome

    assert evidence.value.to_python() == domain_data
    assert outcome.partial_data.to_python() == domain_data


def _alias_forms(*segments: str) -> tuple[str, str, str]:
    snake = "_".join(segments)
    kebab = "-".join(segments)
    camel = segments[0] + "".join(segment.title() for segment in segments[1:])
    return snake, kebab, camel


_TECHNICAL_SUFFIX_ALIASES = [
    alias
    for segments in (
        ("token", "value"),
        ("password", "hash"),
        ("credential", "cache"),
        ("credentials", "cache"),
        ("authorization", "context"),
        ("secret", "value"),
        ("cookie", "value"),
        ("evaluation", "result"),
        ("eval", "result"),
        ("golden", "set", "version"),
        ("evaluation", "seed", "value"),
        ("expected", "paths", "digest"),
        ("test", "scenarios", "version"),
    )
    for alias in _alias_forms(*segments)
]

_PUBLIC_ONLY_SUFFIX_ALIASES = [
    alias
    for segments in (
        ("permissions", "snapshot"),
        ("approval", "record"),
        ("identity", "context"),
    )
    for alias in _alias_forms(*segments)
]

_SAFE_AMBIGUOUS_AND_DOMAIN_NAMES = [
    "client_secretary",
    "access_tokenization",
    "http_responsiveness",
    "baseline_reference",
    "processing_state",
    "response_time",
    "bearing_authenticity",
    "machine_runtime_hours",
    "runtime_client_state",
    "asset_id",
    "analysis_id",
]


@pytest.mark.parametrize("forbidden_name", _TECHNICAL_SUFFIX_ALIASES)
def test_technical_suffix_policy_rejects_sensitive_segment_aliases(forbidden_name):
    with pytest.raises(ValidationError):
        StateEvidence(
            evidence_id="evidence_01",
            call_id="call_01",
            value={"outer": {forbidden_name: "must-not-persist"}},
        )


@pytest.mark.parametrize(
    "forbidden_name",
    _TECHNICAL_SUFFIX_ALIASES + _PUBLIC_ONLY_SUFFIX_ALIASES,
)
def test_public_suffix_policy_rejects_sensitive_segment_aliases(forbidden_name):
    with pytest.raises(ValidationError):
        PersistedToolCall(
            call_id="call_01",
            name="get_asset",
            arguments={"outer": {forbidden_name: "must-not-persist"}},
        )


@pytest.mark.parametrize("allowed_name", _PUBLIC_ONLY_SUFFIX_ALIASES)
def test_technical_suffix_policy_allows_public_only_context_names(allowed_name):
    evidence = StateEvidence(
        evidence_id="evidence_01",
        call_id="call_01",
        value={"outer": {allowed_name: "domain-observation"}},
    )

    assert evidence.value.to_python()["outer"][allowed_name] == "domain-observation"


@pytest.mark.parametrize("allowed_name", _SAFE_AMBIGUOUS_AND_DOMAIN_NAMES)
@pytest.mark.parametrize("boundary", ["public", "technical"])
def test_suffix_policies_preserve_ambiguous_and_domain_names(boundary, allowed_name):
    nested_value = {"outer": {allowed_name: "domain-observation"}}

    if boundary == "public":
        snapshot = PersistedToolCall(
            call_id="call_01",
            name="get_asset",
            arguments=nested_value,
        ).arguments
    else:
        snapshot = StateEvidence(
            evidence_id="evidence_01",
            call_id="call_01",
            value=nested_value,
        ).value

    assert snapshot.to_python()["outer"][allowed_name] == "domain-observation"


def test_technical_evidence_and_result_allow_legitimate_domain_names():
    domain_data = {
        "case_id": "case_tkt_inv_04",
        "company_id": "comp_mineracao_andes",
        "user_id": "usr_pedro",
        "request": "inspeção",
        "response": "estável",
        "method": "detecção sintomática",
        "context": "domínio industrial",
        "store": "almoxarifado",
    }

    state = _state(
        tool_observations=(
            ToolObservation(
                call_id="call_01",
                artifact=ToolArtifact(
                    tool_name="get_analysis",
                    arguments={"analysis_id": "an_9906"},
                    source=ToolSource(
                        kind="industrial_api",
                        resource="/analyses/an_9906",
                    ),
                    outcome=ToolOutcome(partial_data=domain_data),
                ),
            ),
        ),
        evidence=(
            StateEvidence(
                evidence_id="evidence_01",
                call_id="call_01",
                value=domain_data,
            ),
        ),
    )

    assert state.evidence[0].value["company_id"] == "comp_mineracao_andes"
    assert (
        state.tool_observations[0].artifact.outcome.partial_data["method"]
        == "detecção sintomática"
    )


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "client",
        "transport",
        "runtime",
        "authorization",
        "api_token",
        "credential",
        "golden_set",
        "golden",
        "eval",
        "evaluation",
        "expected_paths",
        "test_scenarios",
        "evaluation_seed",
        "raw_http_response",
        "reasoning_trace",
    ],
)
def test_technical_evidence_rejects_runtime_credentials_and_evaluation(
    forbidden_name,
):
    with pytest.raises(ValidationError):
        _state(
            evidence=(
                StateEvidence(
                    evidence_id="evidence_01",
                    call_id="call_01",
                    value={forbidden_name: "must-not-persist"},
                ),
            ),
        )


def test_technical_result_rejects_runtime_data():
    with pytest.raises(ValidationError):
        ToolObservation(
            call_id="call_01",
            artifact=ToolArtifact(
                tool_name="get_analysis",
                arguments={"analysis_id": "an_9906"},
                source=ToolSource(
                    kind="industrial_api",
                    resource="/analyses/an_9906",
                ),
                outcome=ToolOutcome(partial_data={"runtime": "must-not-persist"}),
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
