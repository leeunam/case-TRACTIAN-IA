from tractian_agent.contracts import Identity, ResponseMode, SupportRequest
from tractian_agent.evaluation.capture import output_from_agent_state
from tractian_agent.state import (
    AgentDecision,
    AgentState,
    FinalResult,
    PersistedToolCall,
    PersistedToolArtifact,
    ResumeAnchor,
    ThreadScope,
    ToolObservation,
)
from tractian_agent.tools.runtime import TrustedIdentity
from tractian_agent.write_policy import TrustedWriteContext


def test_capture_projects_terminal_state_and_ordered_tool_trace() -> None:
    request = SupportRequest(
        case_id="case_tkt_ctx_02",
        ticket_id="TKT-CTX-02",
        asset_id="asset_B204",
        message="O que significa BPFO?",
        identity=Identity(user_id="usr_lucas", company_id="comp_aurora"),
    )
    call = PersistedToolCall(
        request_id="req_eval_01",
        call_id="call_eval_01",
        name="historical_lookup",
        arguments={"query": "BPFO"},
    )
    observation = ToolObservation(
        request_id="req_eval_01",
        call_id=call.call_id,
        content={"status": "ok"},
        artifact=PersistedToolArtifact(
            tool_name="historical_lookup",
            arguments={"query": "BPFO"},
            source={"kind": "industrial_api", "resource": "/knowledge/search?q=BPFO"},
            outcome={"mode": ResponseMode.COMPLETE},
        ),
    )
    state = AgentState(
        request=request,
        identity=TrustedIdentity(user_id="usr_lucas", company_id="comp_aurora"),
        permissions=frozenset({"action_low", "read"}),
        request_id="req_eval_01",
        thread_id="thread_eval_01",
        execution_id="exec_eval_01",
        thread_scope=ThreadScope(
            thread_id="thread_eval_01",
            case_id=request.case_id,
            company_id=request.identity.company_id,
            user_id=request.identity.user_id,
        ),
        trusted_write_context=TrustedWriteContext(
            central_asset_id="asset_B204",
            current_case_id="case_tkt_ctx_02",
            configured_model_id="mdl_vib_v3",
        ),
        tool_calls=(call,),
        tool_observations=(observation,),
        decision=AgentDecision.GUIDE,
        step_count=5,
        step_limit=12,
        final_result=FinalResult(
            decision=AgentDecision.GUIDE,
            message="BPFO é uma frequência característica do rolamento.",
        ),
        resume_anchor=ResumeAnchor.FINISH,
    )

    output = output_from_agent_state(state, duration_ms=12.5)

    assert output.case_id == request.case_id
    assert output.ticket_id == request.ticket_id
    assert output.permissions == ("action_low", "read")
    assert output.decision == "guide"
    assert output.message == state.final_result.message
    assert output.duration_ms == 12.5
    assert [step.model_dump(mode="json") for step in output.steps] == [
        {
            "ordinal": 1,
            "call_id": "call_eval_01",
            "tool_name": "historical_lookup",
            "arguments": {"query": "BPFO"},
            "method": "GET",
            "resource": "/knowledge/search?q=BPFO",
            "outcome": "success",
            "error_code": None,
        }
    ]
