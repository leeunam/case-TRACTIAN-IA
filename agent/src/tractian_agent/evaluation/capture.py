"""Projeção segura do estado persistido para o trace de avaliação."""

from __future__ import annotations

from typing import cast

from pydantic import JsonValue

from tractian_agent.evaluation.contracts import EvaluationOutput, ObservedStep
from tractian_agent.state import AgentState
from tractian_agent.write_contracts import IntentStatus


def _json_arguments(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TypeError("argumentos persistidos devem formar um objeto JSON")
    return cast(dict[str, JsonValue], value)


def _action_step(state: AgentState, ordinal: int) -> ObservedStep | None:
    intents = tuple(
        intent for intent in state.intents if intent.request_id == state.request_id
    )
    if len(intents) != 1 or intents[0].attempts < 1:
        return None
    intent = intents[0]
    scope = intent.scope
    action = scope.action
    common = {"justification": scope.justification}
    if action == "reprocess_analysis":
        tool_name = "execute_reprocess_analysis"
        method = "POST"
        resource = f"/analyses/{scope.analysis_id}/reprocess"
        arguments = {"analysis_id": scope.analysis_id, **common}
    elif action == "request_specialist_analysis":
        tool_name = "execute_request_specialist_analysis"
        method = "POST"
        resource = f"/analyses/{scope.analysis_id}/request-specialist"
        arguments = {"analysis_id": scope.analysis_id, **common}
    elif action == "update_asset_criticality":
        tool_name = "execute_update_asset_criticality"
        method = "PATCH"
        resource = f"/assets/{scope.asset_id}/criticality"
        arguments = {
            "asset_id": scope.asset_id,
            "criticality": scope.criticality,
            **common,
        }
    elif action == "request_model_retraining":
        tool_name = "execute_request_model_retraining"
        method = "POST"
        resource = f"/models/{scope.model_id}/retrain"
        arguments = {"model_id": scope.model_id, **common}
    elif action == "escalate_case":
        tool_name = "execute_escalate_case"
        method = "POST"
        resource = f"/cases/{scope.case_id}/escalate"
        arguments = {"case_id": scope.case_id, **common}
    else:  # pragma: no cover - união discriminada torna o ramo inalcançável.
        raise ValueError(f"ação persistida desconhecida: {action}")
    succeeded = intent.status is IntentStatus.COMPLETED
    error_code = (
        None
        if succeeded
        else (intent.error.code if intent.error is not None else "ACTION_NOT_COMPLETED")
    )
    return ObservedStep(
        ordinal=ordinal,
        call_id=f"intent:{intent.intent_id}",
        tool_name=tool_name,
        arguments=arguments,
        method=method,
        resource=resource,
        outcome="success" if succeeded else "error",
        error_code=error_code,
    )


def output_from_agent_state(
    state: AgentState,
    *,
    duration_ms: float,
) -> EvaluationOutput:
    """Extrai somente resultado, trajetória validada e contadores observáveis."""

    observations = {
        observation.call_id: observation
        for observation in state.tool_observations
        if observation.request_id == state.request_id
    }
    steps: list[ObservedStep] = []
    failure_codes: list[str] = []
    for call in state.tool_calls:
        if call.request_id != state.request_id:
            continue
        observation = observations.get(call.call_id)
        error = observation.artifact.outcome.error if observation is not None else None
        if error is not None:
            failure_codes.append(error.code)
        steps.append(
            ObservedStep(
                ordinal=len(steps) + 1,
                call_id=call.call_id,
                tool_name=call.name,
                arguments=_json_arguments(call.arguments.to_python()),
                method="GET" if observation is not None else None,
                resource=(
                    observation.artifact.source.resource
                    if observation is not None
                    else None
                ),
                outcome="error" if error is not None else "success",
                error_code=error.code if error is not None else None,
            )
        )
    action_step = _action_step(state, len(steps) + 1)
    if action_step is not None:
        steps.append(action_step)
        if action_step.error_code is not None:
            failure_codes.append(action_step.error_code)
    if state.planner_failure is not None:
        failure_codes.append(state.planner_failure.code)
    if state.writer_failure is not None:
        failure_codes.append(state.writer_failure.code.value)

    final_result = state.final_result
    decision = (
        final_result.decision.value
        if final_result is not None
        else state.decision.value
        if state.decision is not None
        else "require_human_review"
    )
    return EvaluationOutput(
        case_id=state.request.case_id,
        ticket_id=state.request.ticket_id,
        decision=decision,
        message=(
            final_result.message
            if final_result is not None
            else "A execução não produziu resultado público terminal."
        ),
        permissions=tuple(sorted(state.permissions)),
        steps=tuple(steps),
        step_count=state.step_count,
        step_limit=state.step_limit,
        planner_selection_count=state.planner_usage.selection_count,
        planner_finalization_count=state.planner_usage.finalization_count,
        writer_attempts=state.writer_attempts,
        gate_outcome=(
            state.release_gate.outcome.value if state.release_gate is not None else None
        ),
        evidence_ids=final_result.evidence_ids if final_result is not None else (),
        limitation_refs=(
            final_result.limitation_refs if final_result is not None else ()
        ),
        failure_codes=tuple(dict.fromkeys(failure_codes)),
        duration_ms=duration_ms,
    )
