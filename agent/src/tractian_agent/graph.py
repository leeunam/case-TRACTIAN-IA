"""Grafo do planner com fronteiras determinísticas para os efeitos de escrita."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import StateSnapshot, interrupt

from tractian_agent.checkpoint import get_checkpoint_owner
from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import ActionReceipt, ApiError, ApiErrorCategory
from tractian_agent.planner import (
    Planner,
    PlannerDecisionKind,
    PlannerDecisionTurn,
    PlannerProtocolError,
    PlannerToolTurn,
    select_planner_tools,
    validate_planner_read_observation,
)
from tractian_agent.state import (
    AgentDecision,
    AgentState,
    FinalResult,
    MessageRole,
    PersistedMessage,
    PersistedToolArtifact,
    PersistedToolCall,
    PlannerFailureRecord,
    PlannerTerminalRecord,
    ResumeAnchor,
    ReviewRecord,
    ReviewStatus,
    ToolObservation,
)
from tractian_agent.tools.runtime import ReadToolRuntime, WriteToolRuntime
from tractian_agent.tools.writes import (
    WriteProposalArtifact,
    WriteProposalContent,
)
from tractian_agent.write_contracts import (
    ConfirmationReply,
    EscalateCaseIntentScope,
    IntentStatus,
    ReprocessIntentScope,
    RequestModelRetrainingIntentScope,
    RequestSpecialistAnalysisIntentScope,
    UpdateAssetCriticalityIntentScope,
    WriteIntent,
    WriteIntentScope,
    intent_scope_target_id,
)
from tractian_agent.write_operations import (
    execute_escalate_case,
    execute_reprocess_analysis,
    execute_request_model_retraining,
    execute_request_specialist_analysis,
    execute_update_asset_criticality,
)
from tractian_agent.write_policy import (
    EscalateCaseProposal,
    PolicyDecision,
    PolicyReason,
    ReprocessProposal,
    RequestModelRetrainingProposal,
    RequestSpecialistAnalysisProposal,
    TrustedWriteContext,
    UpdateAssetCriticalityProposal,
    WriteProposal,
    WritePolicyResult,
    evaluate_write_policy,
    resolve_action_scope,
)


MINIMAL_GRAPH_STEP_COUNT = 3
REPROCESS_GRAPH_STEP_COUNT = 5
PLANNER_GRAPH_STEP_LIMIT = 20
_PLANNER_FIXED_WRITE_STEPS = 4
_IDEMPOTENCY_TTL = timedelta(days=7)
_AMBIGUOUS_ERROR_CATEGORIES = frozenset(
    {
        ApiErrorCategory.SERVER,
        ApiErrorCategory.TIMEOUT,
        ApiErrorCategory.TRANSPORT,
        ApiErrorCategory.INVALID_RESPONSE,
    }
)
_UNCERTAIN_IDEMPOTENCY_CODES = frozenset(
    {"IDEMPOTENCY_IN_PROGRESS", "IDEMPOTENCY_OUTCOME_UNKNOWN"}
)
_PROPOSAL_ACTION_BY_TOOL = {
    "propose_reprocess_analysis": "reprocess_analysis",
    "propose_request_specialist_analysis": "request_specialist_analysis",
    "propose_update_asset_criticality": "update_asset_criticality",
    "propose_request_model_retraining": "request_model_retraining",
    "propose_escalate_case": "escalate_case",
}


class CompiledAgentGraph:
    """Grafo compilado que reutiliza o owner local do checkpointer."""

    def __init__(
        self,
        graph: CompiledStateGraph,
        *,
        planner_enabled: bool = False,
    ) -> None:
        self._graph = graph
        self._checkpoint_owner = get_checkpoint_owner(graph.checkpointer)
        self._planner_enabled = planner_enabled

    @property
    def planner_enabled(self) -> bool:
        return self._planner_enabled

    def thread_lock(self, thread_id: str) -> AbstractAsyncContextManager[None]:
        return self._checkpoint_owner.thread_lock(thread_id)

    async def aget_state(self, config: RunnableConfig) -> StateSnapshot:
        return await self._graph.aget_state(config)

    async def ainvoke(
        self,
        input: object,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> object:
        return await self._graph.ainvoke(input, config, **kwargs)

    async def aupdate_state(
        self,
        config: RunnableConfig,
        values: object,
        **kwargs: Any,
    ) -> RunnableConfig:
        return await self._graph.aupdate_state(config, values, **kwargs)

    def get_graph(self) -> object:
        return self._graph.get_graph()


def _replace_state(state: AgentState, **changes: object) -> AgentState:
    data = {
        field_name: getattr(state, field_name)
        for field_name in type(state).model_fields
    }
    data.update(changes)
    return AgentState.model_validate(data)


def _checkpoint_update(state: AgentState) -> dict[str, object]:
    return state.model_dump(mode="json")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_payload_body(proposal: WriteProposal) -> dict[str, object]:
    if isinstance(proposal, UpdateAssetCriticalityProposal):
        return {
            "changes": {"criticality": proposal.criticality},
            "justification": proposal.justification,
        }
    return {"justification": proposal.justification}


def _canonical_payload_hash(proposal: WriteProposal) -> str:
    body = json.dumps(
        _canonical_payload_body(proposal),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:v1:{hashlib.sha256(body).hexdigest()}"


def _current_write_proposal(state: AgentState) -> WriteProposal:
    if state.pending_proposal is None:
        raise TypeError("o fluxo de escrita exige proposta persistida")
    return state.pending_proposal


def _trusted_write_context(context: WriteToolRuntime) -> TrustedWriteContext:
    return TrustedWriteContext(
        central_asset_id=context.central_asset_id,
        current_case_id=context.current_case_id,
        configured_model_id=context.configured_model_id,
    )


def _scope_from_proposal(
    state: AgentState,
    proposal: WriteProposal,
    trusted_context: TrustedWriteContext,
) -> WriteIntentScope:
    canonical = resolve_action_scope(
        proposal,
        trusted_context=trusted_context,
    )
    common = {
        "case_id": state.request.case_id,
        "company_id": state.identity.company_id,
        "user_id": state.identity.user_id,
        "justification": proposal.justification,
    }
    if isinstance(proposal, ReprocessProposal):
        return ReprocessIntentScope(
            action=proposal.action,
            analysis_id=canonical.target_id,
            **common,
        )
    if isinstance(proposal, RequestSpecialistAnalysisProposal):
        return RequestSpecialistAnalysisIntentScope(
            action=proposal.action,
            analysis_id=canonical.target_id,
            **common,
        )
    if isinstance(proposal, UpdateAssetCriticalityProposal):
        return UpdateAssetCriticalityIntentScope(
            action=proposal.action,
            asset_id=canonical.target_id,
            criticality=proposal.criticality,
            **common,
        )
    if isinstance(proposal, RequestModelRetrainingProposal):
        return RequestModelRetrainingIntentScope(
            action=proposal.action,
            model_id=canonical.target_id,
            **common,
        )
    return EscalateCaseIntentScope(action=proposal.action, **common)


def _runtime_matches_state(
    state: AgentState,
    context: WriteToolRuntime,
) -> bool:
    return (
        context.identity == state.identity
        and context.current_case_id == state.request.case_id
        and (
            state.request.asset_id is None
            or context.central_asset_id == state.request.asset_id
        )
    )


def _current_intent(state: AgentState) -> WriteIntent:
    matches = tuple(
        intent for intent in state.intents if intent.request_id == state.request_id
    )
    if len(matches) != 1:
        raise ValueError("a solicitação deve possuir exatamente uma intenção")
    return matches[0]


def _updated_intent(intent: WriteIntent, **changes: object) -> WriteIntent:
    data = intent.model_dump(mode="python")
    data.update(changes)
    return WriteIntent.model_validate(data)


def _replace_intent(state: AgentState, replacement: WriteIntent) -> AgentState:
    found = any(
        intent.intent_id == replacement.intent_id for intent in state.intents
    )
    replacements = tuple(
        replacement if intent.intent_id == replacement.intent_id else intent
        for intent in state.intents
    )
    if not found:
        raise ValueError("intenção a atualizar não pertence ao estado")
    return _replace_state(state, intents=replacements)


def _terminal_result(
    state: AgentState,
    *,
    decision: AgentDecision,
    message: str,
) -> AgentState:
    return _replace_state(
        state,
        decision=decision,
        final_result=FinalResult(decision=decision, message=message),
    )


def _planner_failure_result(
    state: AgentState,
    *,
    stage: Literal["planner_select", "planner_tool", "planner_finalize"],
    code: str,
    anchor: ResumeAnchor,
) -> AgentState:
    decision = AgentDecision.REQUIRE_HUMAN_REVIEW
    return _replace_state(
        state,
        resume_anchor=anchor,
        planner_terminal=None,
        planner_failure=PlannerFailureRecord(stage=stage, code=code),
        decision=decision,
        final_result=FinalResult(
            decision=decision,
            message="O ciclo do planner terminou de forma segura e exige revisão.",
        ),
        review=ReviewRecord(
            status=ReviewStatus.REQUIRED,
            reason=f"planner:{stage}:{code}",
        ),
    )


def _planner_failure_update(
    state: AgentState,
    *,
    stage: Literal["planner_select", "planner_tool", "planner_finalize"],
    code: str,
    anchor: ResumeAnchor,
) -> dict[str, object]:
    return _checkpoint_update(
        _planner_failure_result(
            state,
            stage=stage,
            code=code,
            anchor=anchor,
        )
    )


def _successful_action_decision(action: str) -> AgentDecision:
    return (
        AgentDecision.ESCALATE
        if action == "escalate_case"
        else AgentDecision.ACT
    )


def _ingest(
    state: AgentState,
    runtime: Runtime[ReadToolRuntime],
) -> dict[str, object]:
    context = runtime.context
    if not isinstance(context, ReadToolRuntime) or not isinstance(
        context.client,
        IndustrialApiClient,
    ):
        raise TypeError("contexto confiável do grafo é obrigatório")
    if context.identity != state.identity or context.permissions != state.permissions:
        raise ValueError("contexto confiável diverge do estado persistido")
    if (
        state.request.asset_id is not None
        and context.central_asset_id != state.request.asset_id
    ):
        raise ValueError("ativo central do contexto diverge da solicitação")
    advanced = state.advance_step()
    updated = _replace_state(
        advanced,
        resume_anchor=ResumeAnchor.INGEST,
        messages=(
            *advanced.messages,
            PersistedMessage(role=MessageRole.USER, content=advanced.request.message),
        ),
    )
    return _checkpoint_update(updated)


def _route(state: AgentState) -> dict[str, object]:
    advanced = state.advance_step()
    return _checkpoint_update(
        _replace_state(
            advanced,
            decision=AgentDecision.GUIDE,
            resume_anchor=ResumeAnchor.ROUTE,
        )
    )


def _finish(state: AgentState) -> dict[str, object]:
    advanced = state.advance_step()
    result = FinalResult(
        decision=AgentDecision.GUIDE,
        message="Fluxo determinístico de leitura concluído sem LLM.",
    )
    return _checkpoint_update(
        _replace_state(
            advanced,
            final_result=result,
            resume_anchor=ResumeAnchor.FINISH,
        )
    )


def _after_ingest(state: AgentState) -> Literal["read", "write"]:
    return "write" if state.pending_proposal is not None else "read"


def _after_ingest_with_planner(
    state: AgentState,
) -> Literal["planner", "write"]:
    return "write" if state.pending_proposal is not None else "planner"


def _planner_select_node(planner: Planner):
    async def planner_select(
        state: AgentState,
        runtime: Runtime[ReadToolRuntime],
    ) -> dict[str, object]:
        try:
            advanced = state.advance_step()
        except ValueError:
            return _planner_failure_update(
                state,
                stage="planner_select",
                code="step_limit_exhausted",
                anchor=ResumeAnchor.PLANNER_SELECT,
            )
        context = runtime.context
        if not isinstance(context, ReadToolRuntime):
            return _planner_failure_update(
                advanced,
                stage="planner_select",
                code="runtime_required",
                anchor=ResumeAnchor.PLANNER_SELECT,
            )
        try:
            offered_tools = select_planner_tools(advanced, context)
            turn = await planner.ainvoke(
                advanced.request,
                offered_tools=offered_tools,
                request_id=advanced.request_id,
                usage=advanced.planner_usage,
                tool_calls=advanced.tool_calls,
                tool_observations=advanced.tool_observations,
            )
        except PlannerProtocolError as error:
            failed_state = advanced
            if error.usage is not None:
                failed_state = _replace_state(
                    failed_state,
                    planner_usage=error.usage,
                )
            return _planner_failure_update(
                failed_state,
                stage="planner_select",
                code=error.code.value,
                anchor=ResumeAnchor.PLANNER_SELECT,
            )
        except Exception:
            return _planner_failure_update(
                advanced,
                stage="planner_select",
                code="model_failure",
                anchor=ResumeAnchor.PLANNER_SELECT,
            )

        updated = _replace_state(
            advanced,
            planner_usage=turn.usage,
            planner_failure=None,
            resume_anchor=ResumeAnchor.PLANNER_SELECT,
        )
        remaining_steps = updated.step_limit - updated.step_count
        if isinstance(turn, PlannerToolTurn):
            is_proposal = turn.tool_call.name in _PROPOSAL_ACTION_BY_TOOL
            required_steps = 1 + (
                _PLANNER_FIXED_WRITE_STEPS if is_proposal else 0
            )
            if remaining_steps < required_steps:
                return _planner_failure_update(
                    updated,
                    stage="planner_select",
                    code="step_limit_exhausted",
                    anchor=ResumeAnchor.PLANNER_SELECT,
                )
            return _checkpoint_update(
                _replace_state(
                    updated,
                    planner_terminal=None,
                    tool_calls=(*updated.tool_calls, turn.tool_call),
                )
            )
        if not isinstance(turn, PlannerDecisionTurn):
            return _planner_failure_update(
                updated,
                stage="planner_select",
                code="invalid_planner_turn",
                anchor=ResumeAnchor.PLANNER_SELECT,
            )
        if remaining_steps < 1:
            return _planner_failure_update(
                updated,
                stage="planner_select",
                code="step_limit_exhausted",
                anchor=ResumeAnchor.PLANNER_SELECT,
            )
        terminal = PlannerTerminalRecord.model_validate(
            turn.decision.model_dump(mode="json")
        )
        return _checkpoint_update(
            _replace_state(updated, planner_terminal=terminal)
        )

    return planner_select


def _after_planner_select(
    state: AgentState,
) -> Literal["end", "finalize", "tool"]:
    if state.final_result is not None:
        return "end"
    if state.planner_terminal is not None:
        return "finalize"
    return "tool"


def _pending_planner_tool(
    state: AgentState,
    runtime: ReadToolRuntime,
) -> tuple[PersistedToolCall, BaseTool]:
    calls = tuple(
        call for call in state.tool_calls if call.request_id == state.request_id
    )
    observations = tuple(
        observation
        for observation in state.tool_observations
        if observation.request_id == state.request_id
    )
    if (
        not calls
        or state.tool_calls[-1] != calls[-1]
        or len(calls) != len(observations) + 1
        or any(
            call.call_id != observation.call_id
            for call, observation in zip(calls[:-1], observations, strict=True)
        )
    ):
        raise ValueError("estado não possui uma única tool pendente")
    pending = calls[-1]
    prior = _replace_state(state, tool_calls=state.tool_calls[:-1])
    offered_by_name = {
        tool.name: tool for tool in select_planner_tools(prior, runtime)
    }
    selected_tool = offered_by_name.get(pending.name)
    if selected_tool is None:
        raise ValueError("tool pendente não pertence ao catálogo autorizado")
    validated_arguments = selected_tool.tool_call_schema.model_validate(
        pending.arguments.to_python()
    ).model_dump(mode="json")
    if validated_arguments != pending.arguments.to_python():
        raise ValueError("argumentos persistidos divergem do schema público")
    return pending, selected_tool


def _proposal_matches_call(
    call_arguments: object,
    content: WriteProposalContent,
    artifact: WriteProposalArtifact,
) -> bool:
    expected_action = _PROPOSAL_ACTION_BY_TOOL.get(artifact.tool_name)
    proposal_arguments = artifact.proposal.model_dump(
        mode="json",
        exclude={"action"},
    )
    return (
        expected_action == artifact.proposal.action
        and artifact.effect_executed is False
        and content.proposal == artifact.proposal
        and proposal_arguments == call_arguments
    )


async def _planner_tool(
    state: AgentState,
    runtime: Runtime[ReadToolRuntime],
) -> dict[str, object]:
    try:
        advanced = state.advance_step()
    except ValueError:
        return _planner_failure_update(
            state,
            stage="planner_tool",
            code="step_limit_exhausted",
            anchor=ResumeAnchor.PLANNER_TOOL,
        )
    context = runtime.context
    if not isinstance(context, ReadToolRuntime):
        return _planner_failure_update(
            advanced,
            stage="planner_tool",
            code="runtime_required",
            anchor=ResumeAnchor.PLANNER_TOOL,
        )
    try:
        call, selected_tool = _pending_planner_tool(advanced, context)
        raw_output = await ToolNode(
            (selected_tool,),
            handle_tool_errors=False,
        ).ainvoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": call.name,
                                "args": call.arguments.to_python(),
                                "id": call.call_id,
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            },
            runtime=runtime,
        )
        if not isinstance(raw_output, Mapping):
            raise ValueError("ToolNode devolveu envelope inválido")
        messages = raw_output.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 1
            or not isinstance(messages[0], ToolMessage)
        ):
            raise ValueError("ToolNode deve devolver um único ToolMessage")
        message = messages[0]
        if (
            message.status != "success"
            or message.name != call.name
            or message.tool_call_id != call.call_id
            or not isinstance(message.content, str)
        ):
            raise ValueError("ToolMessage diverge da chamada persistida")
        parsed_content = json.loads(message.content)

        if call.name in _PROPOSAL_ACTION_BY_TOOL:
            content = WriteProposalContent.model_validate(parsed_content)
            artifact = WriteProposalArtifact.model_validate(message.artifact)
            if artifact.tool_name != call.name or not _proposal_matches_call(
                call.arguments.to_python(),
                content,
                artifact,
            ):
                raise ValueError("proposta diverge da tool selecionada")
            updated = _replace_state(
                advanced,
                pending_proposal=artifact.proposal,
                planner_terminal=None,
                planner_failure=None,
                resume_anchor=ResumeAnchor.PLANNER_TOOL,
            )
            if updated.step_limit - updated.step_count < _PLANNER_FIXED_WRITE_STEPS:
                return _planner_failure_update(
                    updated,
                    stage="planner_tool",
                    code="write_step_budget_exhausted",
                    anchor=ResumeAnchor.PLANNER_TOOL,
                )
            return _checkpoint_update(updated)

        artifact = PersistedToolArtifact.model_validate(message.artifact)
        observation = ToolObservation(
            request_id=advanced.request_id,
            call_id=call.call_id,
            content=parsed_content,
            artifact=artifact,
        )
        validate_planner_read_observation(advanced, context, observation)
        updated = _replace_state(
            advanced,
            tool_observations=(*advanced.tool_observations, observation),
            planner_terminal=None,
            planner_failure=None,
            resume_anchor=ResumeAnchor.PLANNER_TOOL,
        )
        if updated.step_count >= updated.step_limit:
            return _planner_failure_update(
                updated,
                stage="planner_tool",
                code="step_limit_exhausted",
                anchor=ResumeAnchor.PLANNER_TOOL,
            )
        return _checkpoint_update(updated)
    except (PlannerProtocolError, TypeError, ValueError):
        return _planner_failure_update(
            advanced,
            stage="planner_tool",
            code="invalid_tool_result",
            anchor=ResumeAnchor.PLANNER_TOOL,
        )
    except Exception:
        return _planner_failure_update(
            advanced,
            stage="planner_tool",
            code="tool_execution_failed",
            anchor=ResumeAnchor.PLANNER_TOOL,
        )


def _after_planner_tool(
    state: AgentState,
) -> Literal["end", "planner", "write"]:
    if state.final_result is not None:
        return "end"
    return "write" if state.pending_proposal is not None else "planner"


def _planner_finalize(state: AgentState) -> dict[str, object]:
    try:
        advanced = state.advance_step()
    except ValueError:
        return _planner_failure_update(
            state,
            stage="planner_finalize",
            code="step_limit_exhausted",
            anchor=ResumeAnchor.PLANNER_FINALIZE,
        )
    terminal = advanced.planner_terminal
    if terminal is None:
        return _planner_failure_update(
            advanced,
            stage="planner_finalize",
            code="missing_terminal_decision",
            anchor=ResumeAnchor.PLANNER_FINALIZE,
        )
    decision = AgentDecision(terminal.decision)
    messages = {
        PlannerDecisionKind.GUIDE.value: (
            "O planner encerrou com evidência suficiente; o writer ainda "
            "não foi implementado."
        ),
        PlannerDecisionKind.REQUEST_INFORMATION.value: (
            f"Informação adicional necessária: {terminal.missing_information}"
        ),
        PlannerDecisionKind.REQUIRE_HUMAN_REVIEW.value: (
            "O planner determinou que o caso exige revisão humana."
        ),
    }
    updated = _replace_state(
        advanced,
        resume_anchor=ResumeAnchor.PLANNER_FINALIZE,
        decision=decision,
        final_result=FinalResult(
            decision=decision,
            message=messages[terminal.decision],
        ),
        review=(
            ReviewRecord(
                status=ReviewStatus.REQUIRED,
                reason="planner:human_review_required",
            )
            if decision is AgentDecision.REQUIRE_HUMAN_REVIEW
            else advanced.review
        ),
    )
    return _checkpoint_update(updated)


def _write_policy(
    state: AgentState,
    runtime: Runtime[ReadToolRuntime],
) -> dict[str, object]:
    context = runtime.context
    if not isinstance(context, WriteToolRuntime):
        raise TypeError("runtime de escrita é obrigatório para avaliar ação")
    advanced = _replace_state(
        state.advance_step(),
        resume_anchor=ResumeAnchor.WRITE_POLICY,
    )
    proposal = _current_write_proposal(advanced)
    trusted_context = _trusted_write_context(context)
    policy = evaluate_write_policy(
        proposal,
        permissions=advanced.permissions,
        approval=advanced.approval,
        trusted_context=trusted_context,
    )
    status = {
        PolicyDecision.ALLOW: IntentStatus.PROPOSED,
        PolicyDecision.REQUIRE_CONFIRMATION: IntentStatus.AWAITING_CONFIRMATION,
        PolicyDecision.DENY: IntentStatus.DENIED,
    }[policy.decision]
    intent = WriteIntent(
        intent_id=str(uuid4()),
        request_id=advanced.request_id,
        scope=_scope_from_proposal(advanced, proposal, trusted_context),
        payload_hash=_canonical_payload_hash(proposal),
        decision=policy,
        status=status,
    )
    updated = _replace_state(advanced, intents=(*advanced.intents, intent))
    if status is IntentStatus.DENIED:
        updated = _terminal_result(
            updated,
            decision=AgentDecision.GUIDE,
            message="A política determinística recusou a ação proposta.",
        )
    elif status is IntentStatus.AWAITING_CONFIRMATION:
        updated = _replace_state(
            updated,
            decision=AgentDecision.REQUEST_CONFIRMATION,
        )
    else:
        updated = _replace_state(
            updated,
            decision=_successful_action_decision(proposal.action),
        )
    return _checkpoint_update(updated)


def _after_write_policy(state: AgentState) -> Literal["end", "gate"]:
    return "end" if state.final_result is not None else "gate"


def _confirmation_prompt(
    intent: WriteIntent,
    proposal: WriteProposal,
) -> dict[str, object]:
    prompt: dict[str, object] = {
        "intent_id": intent.intent_id,
        "action": proposal.action,
        "target_id": intent_scope_target_id(intent.scope),
        "justification": proposal.justification,
        "payload_hash": intent.payload_hash,
    }
    if isinstance(intent.scope, UpdateAssetCriticalityIntentScope):
        prompt["criticality"] = intent.scope.criticality
    return prompt


def _confirmation_gate(
    state: AgentState,
    runtime: Runtime[ReadToolRuntime],
) -> dict[str, object]:
    context = runtime.context
    if not isinstance(context, WriteToolRuntime):
        raise TypeError("runtime de escrita é obrigatório para confirmar ação")
    proposal = _current_write_proposal(state)
    intent = _current_intent(state)
    reply: ConfirmationReply | None = None
    if intent.status is IntentStatus.AWAITING_CONFIRMATION:
        reply = ConfirmationReply.model_validate(
            interrupt(_confirmation_prompt(intent, proposal))
        )
        if reply.intent_id != intent.intent_id:
            raise ValueError("a confirmação não corresponde à intenção persistida")

    advanced = _replace_state(
        state.advance_step(),
        resume_anchor=ResumeAnchor.CONFIRMATION_GATE,
    )
    if reply is not None and reply.decision == "deny":
        denied_policy = WritePolicyResult(
            decision=PolicyDecision.DENY,
            reason=PolicyReason.CONFIRMATION_REJECTED,
        )
        denied = _updated_intent(
            intent,
            decision=denied_policy,
            status=IntentStatus.DENIED,
        )
        return _checkpoint_update(
            _terminal_result(
                _replace_intent(advanced, denied),
                decision=AgentDecision.GUIDE,
                message="A pessoa usuária recusou a ação proposta.",
            )
        )

    policy = evaluate_write_policy(
        proposal,
        permissions=advanced.permissions,
        approval=advanced.approval,
        trusted_context=_trusted_write_context(context),
    )
    if policy.decision is not PolicyDecision.ALLOW:
        denied = _updated_intent(
            intent,
            decision=(
                policy
                if policy.decision is PolicyDecision.DENY
                else WritePolicyResult(
                    decision=PolicyDecision.DENY,
                    reason=PolicyReason.CONFIRMATION_REJECTED,
                )
            ),
            status=IntentStatus.DENIED,
        )
        return _checkpoint_update(
            _terminal_result(
                _replace_intent(advanced, denied),
                decision=AgentDecision.GUIDE,
                message="A confirmação não liberou a ação proposta.",
            )
        )

    allowed = _updated_intent(
        intent,
        decision=policy,
        status=IntentStatus.PROPOSED,
    )
    return _checkpoint_update(
        _replace_state(
            _replace_intent(advanced, allowed),
            decision=_successful_action_decision(proposal.action),
        )
    )


def _after_confirmation(state: AgentState) -> Literal["end", "prepare"]:
    return "end" if state.final_result is not None else "prepare"


def _prepare_intent(state: AgentState) -> dict[str, object]:
    advanced = _replace_state(
        state.advance_step(),
        resume_anchor=ResumeAnchor.PREPARE_INTENT,
    )
    intent = _current_intent(advanced)
    if intent.status is not IntentStatus.PROPOSED:
        raise ValueError("somente intenção autorizada pode ser preparada")
    changes: dict[str, object] = {
        "status": IntentStatus.PREPARED,
        "prepared_execution_id": advanced.execution_id,
        "attempts": 0,
    }
    if isinstance(intent.scope, ReprocessIntentScope):
        changes.update(
            idempotency_key=f"tractian-agent:{intent.intent_id}",
            expires_at=_utc_now() + _IDEMPOTENCY_TTL,
        )
    prepared = _updated_intent(intent, **changes)
    return _checkpoint_update(_replace_intent(advanced, prepared))


def _expiration_error(code: str) -> ApiError:
    return ApiError(
        category=ApiErrorCategory.API,
        code=code,
        message="A chave idempotente persistida expirou.",
    )


def _local_intent_error(code: str, message: str) -> ApiError:
    return ApiError(
        category=ApiErrorCategory.API,
        code=code,
        message=message,
    )


def _failed_before_dispatch(
    state: AgentState,
    intent: WriteIntent,
    error: ApiError,
) -> dict[str, object]:
    failed = _updated_intent(
        intent,
        status=IntentStatus.FAILED,
        attempts=0,
        error=error,
    )
    return _checkpoint_update(
        _terminal_result(
            _replace_intent(state, failed),
            decision=AgentDecision.REQUIRE_HUMAN_REVIEW,
            message="A intenção persistida falhou na validação pré-despacho.",
        )
    )


def _authorization_changed_before_dispatch(
    state: AgentState,
    intent: WriteIntent,
) -> dict[str, object]:
    same_execution = intent.prepared_execution_id == state.execution_id
    terminal = _updated_intent(
        intent,
        status=(
            IntentStatus.FAILED
            if same_execution
            else IntentStatus.UNCERTAIN
        ),
        attempts=0,
        error=_local_intent_error(
            (
                "AUTHORIZATION_CHANGED_BEFORE_DISPATCH"
                if same_execution
                else "AUTHORIZATION_CHANGED_OUTCOME_UNKNOWN"
            ),
            "A autorização atual não permite a intenção preparada.",
        ),
    )
    return _checkpoint_update(
        _terminal_result(
            _replace_intent(state, terminal),
            decision=AgentDecision.REQUIRE_HUMAN_REVIEW,
            message="A autorização mudou antes da ação preparada.",
        )
    )


def _non_idempotent_resume_unknown(
    state: AgentState,
    intent: WriteIntent,
) -> dict[str, object]:
    uncertain = _updated_intent(
        intent,
        status=IntentStatus.UNCERTAIN,
        attempts=0,
        error=_local_intent_error(
            "NON_IDEMPOTENT_OUTCOME_UNKNOWN_AFTER_RESUME",
            "A execução preparadora terminou sem resultado terminal observável.",
        ),
    )
    return _checkpoint_update(
        _terminal_result(
            _replace_intent(state, uncertain),
            decision=AgentDecision.REQUIRE_HUMAN_REVIEW,
            message=(
                "O resultado remoto da ação é desconhecido e ela não será "
                "reenviada automaticamente."
            ),
        )
    )


def _status_for_first_error(error: ApiError) -> IntentStatus:
    if error.code in _UNCERTAIN_IDEMPOTENCY_CODES:
        return IntentStatus.UNCERTAIN
    return IntentStatus.FAILED


def _is_ambiguous_operation_error(error: ApiError) -> bool:
    if error.status_code is not None and 400 <= error.status_code < 500:
        return False
    return error.category in _AMBIGUOUS_ERROR_CATEGORIES


def _terminal_from_operation(
    state: AgentState,
    intent: WriteIntent,
    result: ActionReceipt | ApiError,
    *,
    attempts: int,
    first_was_ambiguous: bool,
) -> AgentState:
    if isinstance(result, ActionReceipt):
        status = (
            IntentStatus.COMPLETED if result.accepted else IntentStatus.FAILED
        )
        terminal_intent = _updated_intent(
            intent,
            status=status,
            attempts=attempts,
            receipt=result,
        )
    else:
        status = (
            IntentStatus.UNCERTAIN
            if first_was_ambiguous
            else _status_for_first_error(result)
        )
        terminal_intent = _updated_intent(
            intent,
            status=status,
            attempts=attempts,
            error=result,
        )
    requires_review = (
        status is IntentStatus.UNCERTAIN
        or (
            isinstance(result, ApiError)
            and result.code == "IDEMPOTENCY_PAYLOAD_CONFLICT"
        )
    )
    decision = (
        AgentDecision.ACT
        if status is IntentStatus.COMPLETED
        else AgentDecision.REQUIRE_HUMAN_REVIEW
        if requires_review
        else AgentDecision.GUIDE
    )
    return _terminal_result(
        _replace_intent(state, terminal_intent),
        decision=decision,
        message={
            IntentStatus.COMPLETED: "Reprocesso concluído pela plataforma.",
            IntentStatus.FAILED: "O reprocesso não foi concluído.",
            IntentStatus.UNCERTAIN: "O resultado remoto do reprocesso é incerto.",
        }[status],
    )


def _terminal_from_non_idempotent_operation(
    state: AgentState,
    intent: WriteIntent,
    result: ActionReceipt | ApiError,
) -> AgentState:
    if isinstance(result, ActionReceipt):
        status = (
            IntentStatus.COMPLETED if result.accepted else IntentStatus.FAILED
        )
        terminal_intent = _updated_intent(
            intent,
            status=status,
            attempts=1,
            receipt=result,
        )
    else:
        status = (
            IntentStatus.UNCERTAIN
            if _is_ambiguous_operation_error(result)
            else IntentStatus.FAILED
        )
        terminal_intent = _updated_intent(
            intent,
            status=status,
            attempts=1,
            error=result,
        )
    if status is IntentStatus.UNCERTAIN:
        decision = AgentDecision.REQUIRE_HUMAN_REVIEW
    elif status is IntentStatus.COMPLETED:
        decision = _successful_action_decision(intent.scope.action)
    else:
        decision = AgentDecision.GUIDE
    return _terminal_result(
        _replace_intent(state, terminal_intent),
        decision=decision,
        message={
            IntentStatus.COMPLETED: "A ação foi concluída pela plataforma.",
            IntentStatus.FAILED: "A ação não foi concluída.",
            IntentStatus.UNCERTAIN: (
                "O resultado remoto da ação é incerto e não haverá reenvio "
                "automático."
            ),
        }[status],
    )


async def _dispatch_non_idempotent_action(
    proposal: WriteProposal,
    context: WriteToolRuntime,
) -> ActionReceipt | ApiError:
    if isinstance(proposal, RequestSpecialistAnalysisProposal):
        return await execute_request_specialist_analysis(proposal, context)
    if isinstance(proposal, UpdateAssetCriticalityProposal):
        return await execute_update_asset_criticality(proposal, context)
    if isinstance(proposal, RequestModelRetrainingProposal):
        return await execute_request_model_retraining(proposal, context)
    if isinstance(proposal, EscalateCaseProposal):
        return await execute_escalate_case(proposal, context)
    raise TypeError("o dispatch não idempotente recebeu proposta incompatível")


async def _execute_action(
    state: AgentState,
    runtime: Runtime[ReadToolRuntime],
) -> dict[str, object]:
    context = runtime.context
    if not isinstance(context, WriteToolRuntime):
        raise TypeError("runtime de escrita é obrigatório para executar ação")
    advanced = _replace_state(
        state.advance_step(),
        resume_anchor=ResumeAnchor.EXECUTE_ACTION,
    )
    proposal = _current_write_proposal(advanced)
    intent = _current_intent(advanced)
    if intent.status is not IntentStatus.PREPARED:
        raise ValueError("somente intenção preparada pode executar a ação")
    is_reprocess = isinstance(intent.scope, ReprocessIntentScope)
    if (
        not is_reprocess
        and intent.prepared_execution_id != advanced.execution_id
    ):
        return _non_idempotent_resume_unknown(advanced, intent)

    if not is_reprocess and not _runtime_matches_state(advanced, context):
        return _failed_before_dispatch(
            advanced,
            intent,
            _local_intent_error(
                "INTENT_SCOPE_MISMATCH",
                "O runtime confiável diverge do escopo persistido.",
            ),
        )
    trusted_context = _trusted_write_context(context)
    expected_scope = _scope_from_proposal(
        advanced,
        proposal,
        trusted_context,
    )
    if intent.scope != expected_scope:
        return _failed_before_dispatch(
            advanced,
            intent,
            _local_intent_error(
                "INTENT_SCOPE_MISMATCH",
                "O escopo persistido diverge da proposta confiável.",
            ),
        )
    if intent.payload_hash != _canonical_payload_hash(proposal):
        return _failed_before_dispatch(
            advanced,
            intent,
            _local_intent_error(
                "PAYLOAD_HASH_MISMATCH",
                "O hash persistido diverge do corpo canônico.",
            ),
        )
    if is_reprocess:
        if intent.expires_at is None or intent.idempotency_key is None:
            raise ValueError("intenção preparada não possui chave e expiração")
        if intent.idempotency_key != f"tractian-agent:{intent.intent_id}":
            return _failed_before_dispatch(
                advanced,
                intent,
                _local_intent_error(
                    "IDEMPOTENCY_KEY_INTENT_MISMATCH",
                    "A chave persistida não deriva da intenção atual.",
                ),
            )

        if intent.expires_at <= _utc_now():
            same_execution = intent.prepared_execution_id == advanced.execution_id
            expired = _updated_intent(
                intent,
                status=(
                    IntentStatus.FAILED
                    if same_execution
                    else IntentStatus.UNCERTAIN
                ),
                attempts=0,
                error=_expiration_error(
                    "IDEMPOTENCY_KEY_EXPIRED"
                    if same_execution
                    else "IDEMPOTENCY_KEY_EXPIRED_OUTCOME_UNKNOWN"
                ),
            )
            return _checkpoint_update(
                _terminal_result(
                    _replace_intent(advanced, expired),
                    decision=(
                        AgentDecision.GUIDE
                        if same_execution
                        else AgentDecision.REQUIRE_HUMAN_REVIEW
                    ),
                    message="A chave idempotente preparada expirou sem novo envio.",
                )
            )

    current_policy = evaluate_write_policy(
        proposal,
        permissions=advanced.permissions,
        approval=advanced.approval,
        trusted_context=trusted_context,
    )
    if current_policy.decision is not PolicyDecision.ALLOW:
        return _authorization_changed_before_dispatch(advanced, intent)

    if not is_reprocess:
        result = await _dispatch_non_idempotent_action(proposal, context)
        return _checkpoint_update(
            _terminal_from_non_idempotent_operation(advanced, intent, result)
        )

    if not isinstance(proposal, ReprocessProposal) or intent.idempotency_key is None:
        raise TypeError("intenção de reprocesso diverge da proposta persistida")
    first = await execute_reprocess_analysis(
        proposal,
        context,
        idempotency_key=intent.idempotency_key,
    )
    first_was_ambiguous = (
        isinstance(first, ApiError)
        and _is_ambiguous_operation_error(first)
    )
    if not first_was_ambiguous:
        terminal = _terminal_from_operation(
            advanced,
            intent,
            first,
            attempts=1,
            first_was_ambiguous=False,
        )
        return _checkpoint_update(terminal)

    second = await execute_reprocess_analysis(
        proposal,
        context,
        idempotency_key=intent.idempotency_key,
    )
    terminal = _terminal_from_operation(
        advanced,
        intent,
        second,
        attempts=2,
        first_was_ambiguous=True,
    )
    return _checkpoint_update(terminal)


def build_agent_graph(
    checkpointer: BaseCheckpointSaver[str],
    *,
    planner: Planner | None = None,
) -> CompiledAgentGraph:
    """Compila o fallback determinístico ou o ciclo opt-in do planner."""
    builder = StateGraph(AgentState, context_schema=ReadToolRuntime)
    builder.add_node("ingest", _ingest)
    builder.add_node("write_policy", _write_policy)
    builder.add_node("confirmation_gate", _confirmation_gate)
    builder.add_node("prepare_intent", _prepare_intent)
    builder.add_node("execute_action", _execute_action)
    builder.add_edge(START, "ingest")
    if planner is None:
        builder.add_node("route", _route)
        builder.add_node("finish", _finish)
        builder.add_conditional_edges(
            "ingest",
            _after_ingest,
            {"read": "route", "write": "write_policy"},
        )
        builder.add_edge("route", "finish")
        builder.add_edge("finish", END)
    else:
        builder.add_node("planner_select", _planner_select_node(planner))
        builder.add_node("planner_tool", _planner_tool)
        builder.add_node("planner_finalize", _planner_finalize)
        builder.add_conditional_edges(
            "ingest",
            _after_ingest_with_planner,
            {"planner": "planner_select", "write": "write_policy"},
        )
        builder.add_conditional_edges(
            "planner_select",
            _after_planner_select,
            {"end": END, "finalize": "planner_finalize", "tool": "planner_tool"},
        )
        builder.add_conditional_edges(
            "planner_tool",
            _after_planner_tool,
            {"end": END, "planner": "planner_select", "write": "write_policy"},
        )
        builder.add_edge("planner_finalize", END)
    builder.add_conditional_edges(
        "write_policy",
        _after_write_policy,
        {"end": END, "gate": "confirmation_gate"},
    )
    builder.add_conditional_edges(
        "confirmation_gate",
        _after_confirmation,
        {"end": END, "prepare": "prepare_intent"},
    )
    builder.add_edge("prepare_intent", "execute_action")
    builder.add_edge("execute_action", END)
    return CompiledAgentGraph(
        builder.compile(checkpointer=checkpointer),
        planner_enabled=planner is not None,
    )
