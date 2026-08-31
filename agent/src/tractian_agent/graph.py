"""Grafo determinístico de leitura e do reprocesso vertical seguro."""

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
    IntentStatus,
    ReprocessIntentScope,
    WriteIntent,
)
from tractian_agent.write_operations import execute_reprocess_analysis
from tractian_agent.write_policy import (
    PolicyDecision,
    PolicyReason,
    ReprocessProposal,
    WritePolicyResult,
    evaluate_reprocess_policy,
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


def _canonical_payload_hash(proposal: ReprocessProposal) -> str:
    body = json.dumps(
        {"justification": proposal.justification},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:v1:{hashlib.sha256(body).hexdigest()}"


def _current_reprocess_proposal(state: AgentState) -> ReprocessProposal:
    if not isinstance(state.pending_proposal, ReprocessProposal):
        raise TypeError("o fluxo de reprocesso exige proposta persistida")
    return state.pending_proposal


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


def _write_policy(state: AgentState) -> dict[str, object]:
    advanced = state.advance_step()
    proposal = _current_reprocess_proposal(advanced)
    policy = evaluate_reprocess_policy(
        proposal,
        permissions=advanced.permissions,
        approval=advanced.approval,
    )
    status = {
        PolicyDecision.ALLOW: IntentStatus.PROPOSED,
        PolicyDecision.REQUIRE_CONFIRMATION: IntentStatus.AWAITING_CONFIRMATION,
        PolicyDecision.DENY: IntentStatus.DENIED,
    }[policy.decision]
    intent = WriteIntent(
        intent_id=str(uuid4()),
        request_id=advanced.request_id,
        scope=ReprocessIntentScope(
            action="reprocess_analysis",
            case_id=advanced.request.case_id,
            company_id=advanced.identity.company_id,
            user_id=advanced.identity.user_id,
            analysis_id=proposal.analysis_id,
            justification=proposal.justification,
        ),
        payload_hash=_canonical_payload_hash(proposal),
        decision=policy,
        status=status,
    )
    updated = _replace_state(advanced, intents=(*advanced.intents, intent))
    if status is IntentStatus.DENIED:
        updated = _terminal_result(
            updated,
            decision=AgentDecision.GUIDE,
            message="A política determinística recusou o reprocesso.",
        )
    elif status is IntentStatus.AWAITING_CONFIRMATION:
        updated = _replace_state(
            updated,
            decision=AgentDecision.REQUEST_CONFIRMATION,
        )
    else:
        updated = _replace_state(updated, decision=AgentDecision.ACT)
    return _checkpoint_update(updated)


def _after_write_policy(state: AgentState) -> Literal["end", "gate"]:
    return "end" if state.final_result is not None else "gate"


def _confirmation_prompt(
    intent: WriteIntent,
    proposal: ReprocessProposal,
) -> dict[str, str]:
    return {
        "intent_id": intent.intent_id,
        "action": proposal.action,
        "target_id": proposal.analysis_id,
        "justification": proposal.justification,
        "payload_hash": intent.payload_hash,
    }


def _confirmation_gate(state: AgentState) -> dict[str, object]:
    proposal = _current_reprocess_proposal(state)
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
                message="A pessoa usuária recusou o reprocesso proposto.",
            )
        )

    policy = evaluate_reprocess_policy(
        proposal,
        permissions=advanced.permissions,
        approval=advanced.approval,
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
                message="A confirmação não liberou o reprocesso.",
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
            decision=AgentDecision.ACT,
        )
    )


def _after_confirmation(state: AgentState) -> Literal["end", "prepare"]:
    return "end" if state.final_result is not None else "prepare"


def _prepare_intent(state: AgentState) -> dict[str, object]:
    advanced = state.advance_step()
    intent = _current_intent(advanced)
    if intent.status is not IntentStatus.PROPOSED:
        raise ValueError("somente intenção autorizada pode ser preparada")
    prepared = _updated_intent(
        intent,
        status=IntentStatus.PREPARED,
        idempotency_key=f"tractian-agent:{intent.intent_id}",
        expires_at=_utc_now() + _IDEMPOTENCY_TTL,
        prepared_execution_id=advanced.execution_id,
    )
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


def _status_for_first_error(error: ApiError) -> IntentStatus:
    if error.code in _UNCERTAIN_IDEMPOTENCY_CODES:
        return IntentStatus.UNCERTAIN
    return IntentStatus.FAILED


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


async def _execute_action(
    state: AgentState,
    runtime: Runtime[ReadToolRuntime],
) -> dict[str, object]:
    context = runtime.context
    if not isinstance(context, WriteToolRuntime):
        raise TypeError("runtime de escrita é obrigatório para executar ação")
    advanced = state.advance_step()
    proposal = _current_reprocess_proposal(advanced)
    intent = _current_intent(advanced)
    if intent.status is not IntentStatus.PREPARED:
        raise ValueError("somente intenção preparada pode executar a ação")
    if intent.expires_at is None or intent.idempotency_key is None:
        raise ValueError("intenção preparada não possui chave e expiração")

    expected_scope = ReprocessIntentScope(
        action="reprocess_analysis",
        case_id=advanced.request.case_id,
        company_id=advanced.identity.company_id,
        user_id=advanced.identity.user_id,
        analysis_id=proposal.analysis_id,
        justification=proposal.justification,
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

    first = await execute_reprocess_analysis(
        proposal,
        context,
        idempotency_key=intent.idempotency_key,
    )
    first_was_ambiguous = (
        isinstance(first, ApiError)
        and first.category in _AMBIGUOUS_ERROR_CATEGORIES
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
    """Compila leitura e o primeiro fluxo vertical de escrita."""
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
