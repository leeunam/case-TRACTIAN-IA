"""Grafo determinístico de leitura e dos fluxos verticais de escrita."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any, Literal
from uuid import uuid4

from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import StateSnapshot, interrupt

from tractian_agent.checkpoint import get_checkpoint_owner
from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import ActionReceipt, ApiError, ApiErrorCategory
from tractian_agent.state import (
    AgentDecision,
    AgentState,
    FinalResult,
    MessageRole,
    PersistedMessage,
)
from tractian_agent.tools.runtime import ReadToolRuntime, WriteToolRuntime
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


class CompiledAgentGraph:
    """Grafo compilado que reutiliza o owner local do checkpointer."""

    def __init__(self, graph: CompiledStateGraph) -> None:
        self._graph = graph
        self._checkpoint_owner = get_checkpoint_owner(graph.checkpointer)

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
        messages=(
            *advanced.messages,
            PersistedMessage(role=MessageRole.USER, content=advanced.request.message),
        ),
    )
    return _checkpoint_update(updated)


def _route(state: AgentState) -> dict[str, object]:
    advanced = state.advance_step()
    return _checkpoint_update(_replace_state(advanced, decision=AgentDecision.GUIDE))


def _finish(state: AgentState) -> dict[str, object]:
    advanced = state.advance_step()
    result = FinalResult(
        decision=AgentDecision.GUIDE,
        message="Fluxo determinístico de leitura concluído sem LLM.",
    )
    return _checkpoint_update(_replace_state(advanced, final_result=result))


def _after_ingest(state: AgentState) -> Literal["read", "write"]:
    return "write" if state.pending_proposal is not None else "read"


def _write_policy(
    state: AgentState,
    runtime: Runtime[ReadToolRuntime],
) -> dict[str, object]:
    context = runtime.context
    if not isinstance(context, WriteToolRuntime):
        raise TypeError("runtime de escrita é obrigatório para avaliar ação")
    advanced = state.advance_step()
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

    advanced = state.advance_step()
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
    advanced = state.advance_step()
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
    advanced = state.advance_step()
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
) -> CompiledAgentGraph:
    """Compila leitura e os fluxos verticais de escrita."""
    builder = StateGraph(AgentState, context_schema=ReadToolRuntime)
    builder.add_node("ingest", _ingest)
    builder.add_node("route", _route)
    builder.add_node("finish", _finish)
    builder.add_node("write_policy", _write_policy)
    builder.add_node("confirmation_gate", _confirmation_gate)
    builder.add_node("prepare_intent", _prepare_intent)
    builder.add_node("execute_action", _execute_action)
    builder.add_edge(START, "ingest")
    builder.add_conditional_edges(
        "ingest",
        _after_ingest,
        {"read": "route", "write": "write_policy"},
    )
    builder.add_edge("route", "finish")
    builder.add_edge("finish", END)
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
    return CompiledAgentGraph(builder.compile(checkpointer=checkpointer))
