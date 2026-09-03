"""Fronteira Python assíncrona do grafo de atendimento."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from langgraph.graph import START
from langgraph.types import Command

from tractian_agent.contracts import SupportRequest
from tractian_agent.graph import (
    MINIMAL_GRAPH_STEP_COUNT,
    PLANNER_GRAPH_STEP_LIMIT,
    PLANNER_MINIMAL_GRAPH_STEP_COUNT,
    PLANNER_WRITE_GRAPH_STEP_COUNT,
    REPROCESS_GRAPH_STEP_COUNT,
)
from tractian_agent.human_review import (
    ReviewApproveReply,
    ReviewEditReply,
    ReviewRejectReply,
    ReviewResumeEnvelope,
    ReviewerIdentity,
    build_reviewed_draft,
    canonical_digest,
    render_review_expired_result,
    review_resolution_subject_digest,
)
from tractian_agent.observability import (
    AgentTelemetry,
    ErrorCode,
    ExecutionCorrelations,
    NullTelemetry,
    Outcome,
    ResponseSpanAttributes,
    SpanName,
    TraceId,
)
from tractian_agent.state import (
    AgentDecision,
    AgentState,
    ResumeAnchor,
    ReviewExpiry,
    ReviewRecord,
    ReviewReply,
    ReviewStatus,
    ThreadScope,
)
from tractian_agent.tools.runtime import ReadToolRuntime, WriteToolRuntime
from tractian_agent.write_contracts import (
    ConfirmationReply,
    IntentStatus,
    WriteIntent,
    approval_matches_write_intent,
    intent_scope_material_parameters,
    intent_scope_target_id,
)
from tractian_agent.write_policy import (
    ApprovalSource,
    PolicyReason,
    TrustedActionApproval,
    TrustedWriteContext,
    WriteProposal,
)


class AgentInvocationProtocolError(RuntimeError):
    """Checkpoint válido, mas incompatível com a operação solicitada."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class AgentInvocationResult(BaseModel):
    """Envelope não persistível que expõe somente estado e correlação técnica."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    state: AgentState
    trace_id: TraceId


class GraphStateSnapshot(Protocol):
    values: Mapping[str, object]
    config: dict[str, object]
    next: tuple[str, ...]
    interrupts: tuple[Any, ...]


class AgentGraph(Protocol):
    @property
    def planner_enabled(self) -> bool: ...

    def thread_lock(self, thread_id: str) -> AbstractAsyncContextManager[None]: ...

    async def aget_state(
        self,
        config: dict[str, object],
    ) -> GraphStateSnapshot: ...

    async def ainvoke(
        self,
        input: object,
        config: dict[str, object],
        *,
        context: ReadToolRuntime,
        durability: Literal["sync"],
    ) -> Mapping[str, object]: ...

    async def aupdate_state(
        self,
        config: dict[str, object],
        values: dict[str, object],
        *,
        as_node: str,
    ) -> dict[str, object]: ...


_COMMON_RESUME_PAIRS = {
    (ResumeAnchor.START, "ingest"),
    (ResumeAnchor.INGEST, "write_policy"),
    (ResumeAnchor.WRITE_POLICY, "confirmation_gate"),
    (ResumeAnchor.CONFIRMATION_GATE, "prepare_intent"),
    (ResumeAnchor.PREPARE_INTENT, "execute_action"),
}
_FALLBACK_RESUME_PAIRS = {
    (ResumeAnchor.INGEST, "route"),
    (ResumeAnchor.ROUTE, "finish"),
}
_PLANNER_RESUME_PAIRS = {
    (ResumeAnchor.INGEST, "planner_select"),
    (ResumeAnchor.PLANNER_SELECT, "planner_tool"),
    (ResumeAnchor.PLANNER_SELECT, "planner_finalize"),
    (ResumeAnchor.PLANNER_TOOL, "planner_select"),
    (ResumeAnchor.PLANNER_TOOL, "write_policy"),
    (ResumeAnchor.WRITE_POLICY, "writer"),
    (ResumeAnchor.CONFIRMATION_GATE, "writer"),
    (ResumeAnchor.EXECUTE_ACTION, "writer"),
    (ResumeAnchor.PLANNER_FINALIZE, "writer"),
    (ResumeAnchor.WRITER, "writer"),
    (ResumeAnchor.WRITER, "release_gate"),
    (ResumeAnchor.RELEASE_GATE, "await_human_review"),
    (ResumeAnchor.AWAIT_HUMAN_REVIEW, "release_gate"),
}
_ALLOWED_RESUME_PAIRS = (
    _COMMON_RESUME_PAIRS | _FALLBACK_RESUME_PAIRS | _PLANNER_RESUME_PAIRS
)
_KNOWN_PENDING_NODES = frozenset(node for _, node in _ALLOWED_RESUME_PAIRS)

_ACTIVE_INTENT_STATUSES = frozenset(
    {
        IntentStatus.PROPOSED,
        IntentStatus.AWAITING_CONFIRMATION,
        IntentStatus.PREPARED,
    }
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_opaque_id(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{name} é obrigatório e não aceita espaços")
    return value


def _pending_node(next_nodes: tuple[str, ...]) -> str:
    if not next_nodes:
        raise AgentInvocationProtocolError(
            "NON_TERMINAL_WITHOUT_PENDING_WORK",
            "checkpoint não terminal não possui trabalho pendente",
        )
    if len(next_nodes) != 1:
        raise AgentInvocationProtocolError(
            "MULTIPLE_PENDING_NODES",
            "checkpoint possui múltiplos nós pendentes",
        )
    pending_node = next_nodes[0]
    if pending_node not in _KNOWN_PENDING_NODES:
        raise AgentInvocationProtocolError(
            "UNKNOWN_PENDING_NODE",
            f"checkpoint possui nó pendente desconhecido: {pending_node}",
        )
    return pending_node


def _required_resume_anchor(
    persisted_values: Mapping[str, object],
) -> ResumeAnchor:
    if (
        "resume_anchor" not in persisted_values
        or persisted_values.get("resume_anchor") is None
    ):
        raise AgentInvocationProtocolError(
            "MISSING_RESUME_ANCHOR",
            "checkpoint não registra o último nó concluído",
        )
    raw_anchor = persisted_values["resume_anchor"]
    try:
        return ResumeAnchor(raw_anchor)
    except (TypeError, ValueError) as error:
        raise AgentInvocationProtocolError(
            "UNKNOWN_RESUME_ANCHOR",
            "checkpoint registra uma âncora desconhecida",
        ) from error


def _resume_predecessor(
    anchor: ResumeAnchor,
    next_nodes: tuple[str, ...],
    *,
    planner_enabled: bool,
    state: AgentState,
) -> str:
    pending_node = _pending_node(next_nodes)
    if (anchor, pending_node) not in _ALLOWED_RESUME_PAIRS:
        raise AgentInvocationProtocolError(
            "RESUME_ANCHOR_MISMATCH",
            "a âncora persistida diverge do próximo nó",
        )
    topology_pairs = _COMMON_RESUME_PAIRS | (
        _PLANNER_RESUME_PAIRS if planner_enabled else _FALLBACK_RESUME_PAIRS
    )
    if (anchor, pending_node) not in topology_pairs:
        raise AgentInvocationProtocolError(
            "RESUME_TOPOLOGY_MISMATCH",
            "checkpoint parcial pertence a outra topologia de grafo",
        )
    if anchor is ResumeAnchor.WRITER:
        expected_writer_node = (
            "writer"
            if (
                state.writer_draft is None
                and state.writer_failure is not None
                and state.writer_failure.repairable
                and state.writer_attempts == 1
            )
            else "release_gate"
        )
        if pending_node != expected_writer_node:
            raise AgentInvocationProtocolError(
                "RESUME_ANCHOR_MISMATCH",
                "o resultado persistido do writer diverge do próximo nó",
            )
    elif pending_node == "writer" and (
        state.writer_attempts != 0
        or state.writer_draft is not None
        or state.writer_failure is not None
    ):
        raise AgentInvocationProtocolError(
            "RESUME_ANCHOR_MISMATCH",
            "o writer pendente exige contador e resultado vazios",
        )
    return anchor.value


def _validate_terminal_resume_anchor(
    state: AgentState,
    anchor: ResumeAnchor,
) -> None:
    if anchor is not state.resume_anchor or not state.has_coherent_terminal_result():
        raise AgentInvocationProtocolError(
            "RESUME_ANCHOR_MISMATCH",
            "a âncora terminal diverge do resultado persistido",
        )


def _replace_state(state: AgentState, **changes: object) -> AgentState:
    data = state.model_dump(mode="python")
    data.update(changes)
    return AgentState.model_validate(data)


def _required_step_count(
    state: AgentState,
    *,
    planner_enabled: bool,
) -> int:
    if planner_enabled:
        return (
            PLANNER_WRITE_GRAPH_STEP_COUNT
            if state.pending_proposal is not None
            else PLANNER_MINIMAL_GRAPH_STEP_COUNT
        )
    return (
        REPROCESS_GRAPH_STEP_COUNT
        if state.pending_proposal is not None
        else MINIMAL_GRAPH_STEP_COUNT
    )


def _current_request_intents(state: AgentState) -> tuple[WriteIntent, ...]:
    return tuple(
        intent for intent in state.intents if intent.request_id == state.request_id
    )


def _request_id_has_history(state: AgentState, request_id: str) -> bool:
    return (
        any(intent.request_id == request_id for intent in state.intents)
        or any(call.request_id == request_id for call in state.tool_calls)
        or any(
            observation.request_id == request_id
            for observation in state.tool_observations
        )
    )


def _terminal_confirmation_replay(
    state: AgentState,
    confirmation: ConfirmationReply,
) -> bool:
    matches = _current_request_intents(state)
    if (
        len(matches) != 1
        or matches[0].intent_id != confirmation.intent_id
        or matches[0].status in _ACTIVE_INTENT_STATUSES
    ):
        return False
    intent = matches[0]
    if confirmation.decision == "approve":
        return (
            intent.approval_source is ApprovalSource.CONFIRMATION
            and state.pending_proposal is not None
            and approval_matches_write_intent(
                state.pending_proposal,
                intent,
                approval=state.approval,
                trusted_context=state.trusted_write_context,
            )
        )
    return (
        intent.status is IntentStatus.DENIED
        and intent.decision.reason is PolicyReason.CONFIRMATION_REJECTED
        and state.approval is None
    )


def _validate_persisted_write_inputs(
    state: AgentState,
    *,
    proposal: WriteProposal | None,
    original_approval: TrustedActionApproval | None,
) -> None:
    if proposal is not None and proposal != state.pending_proposal:
        raise AgentInvocationProtocolError(
            "PROPOSAL_DRIFT",
            "a proposta diverge da solicitação persistida",
        )
    if original_approval is not None and original_approval != state.approval:
        raise AgentInvocationProtocolError(
            "ORIGINAL_APPROVAL_DRIFT",
            "a aprovação original diverge da solicitação persistida",
        )


def _validate_write_boundary(
    *,
    request: SupportRequest,
    runtime: ReadToolRuntime,
    proposal: WriteProposal | None,
    confirmation: ConfirmationReply | None,
    original_approval: TrustedActionApproval | None,
    planner_enabled: bool = False,
) -> None:
    write_requested = (
        proposal is not None
        or confirmation is not None
        or original_approval is not None
    )
    if not write_requested:
        return
    if not isinstance(runtime, WriteToolRuntime):
        raise AgentInvocationProtocolError(
            "WRITE_RUNTIME_REQUIRED",
            "o fluxo de escrita exige contexto confiável de escrita",
        )
    if original_approval is not None and proposal is None and not planner_enabled:
        raise AgentInvocationProtocolError(
            "ORIGINAL_APPROVAL_WITHOUT_PROPOSAL",
            "aprovação original exige proposta na mesma entrada",
        )
    if (
        original_approval is not None
        and original_approval.source is not ApprovalSource.ORIGINAL_REQUEST
    ):
        raise AgentInvocationProtocolError(
            "INVALID_ORIGINAL_APPROVAL_SOURCE",
            "aprovação original exige proveniência da solicitação original",
        )
    if runtime.current_case_id != request.case_id:
        raise AgentInvocationProtocolError(
            "WRITE_CASE_SCOPE_MISMATCH",
            "o caso atual do runtime diverge da solicitação",
        )
    if runtime.identity.model_dump() != request.identity.model_dump():
        raise AgentInvocationProtocolError(
            "WRITE_IDENTITY_SCOPE_MISMATCH",
            "a identidade confiável diverge da solicitação",
        )
    if request.asset_id is not None and runtime.central_asset_id != request.asset_id:
        raise AgentInvocationProtocolError(
            "WRITE_ASSET_SCOPE_MISMATCH",
            "o ativo central do runtime diverge da solicitação",
        )


def _validate_runtime_request_scope(
    request: SupportRequest,
    runtime: ReadToolRuntime,
) -> None:
    _validate_runtime_identity_scope(request, runtime)
    if request.asset_id is not None and runtime.central_asset_id != request.asset_id:
        raise AgentInvocationProtocolError(
            "RUNTIME_ASSET_SCOPE_MISMATCH",
            "o ativo central do runtime diverge da solicitação",
        )


def _validate_runtime_identity_scope(
    request: SupportRequest,
    runtime: ReadToolRuntime,
) -> None:
    if runtime.identity.model_dump() != request.identity.model_dump():
        raise AgentInvocationProtocolError(
            "RUNTIME_IDENTITY_SCOPE_MISMATCH",
            "a identidade confiável diverge da solicitação",
        )


def _validate_persisted_thread_boundary(
    *,
    state: AgentState,
    request: SupportRequest,
    runtime: ReadToolRuntime,
    thread_id: str,
    request_id: str,
    execution_id: str,
    allow_conservative_target_drift: bool,
) -> None:
    """Autoriza a continuação sem alterar ou revelar o checkpoint restaurado."""
    scope = state.thread_scope
    if thread_id != state.thread_id or thread_id != scope.thread_id:
        raise AgentInvocationProtocolError(
            "THREAD_SCOPE_MISMATCH",
            "o thread autenticado não corresponde ao checkpoint",
        )
    if execution_id == state.execution_id:
        raise AgentInvocationProtocolError(
            "EXECUTION_ID_ALREADY_USED",
            "cada continuação exige novo execution_id",
        )
    if (
        request.case_id != scope.case_id
        or request.identity.company_id != scope.company_id
        or request.identity.user_id != scope.user_id
        or runtime.identity.company_id != scope.company_id
        or runtime.identity.user_id != scope.user_id
    ):
        raise AgentInvocationProtocolError(
            "THREAD_SCOPE_MISMATCH",
            "a solicitação não pode continuar este thread",
        )
    trusted_context = state.trusted_write_context
    if trusted_context is None:
        raise AgentInvocationProtocolError(
            "TRUSTED_WRITE_CONTEXT_MISSING",
            "o checkpoint não possui o alvo confiável persistido",
        )
    if request.asset_id is not None and (
        request.asset_id != trusted_context.central_asset_id
    ):
        raise AgentInvocationProtocolError(
            "THREAD_SCOPE_MISMATCH",
            "o alvo não pode mudar dentro deste thread",
        )
    if (
        not allow_conservative_target_drift
        and request.asset_id is not None
        and runtime.central_asset_id != request.asset_id
    ):
        raise AgentInvocationProtocolError(
            "RUNTIME_ASSET_SCOPE_MISMATCH",
            "o ativo central do runtime diverge da solicitação",
        )
    if not allow_conservative_target_drift and (
        runtime.central_asset_id != trusted_context.central_asset_id
        or runtime.configured_model_id != trusted_context.configured_model_id
    ):
        raise AgentInvocationProtocolError(
            "TRUSTED_WRITE_CONTEXT_DRIFT",
            "o runtime atual diverge do alvo ou modelo persistido",
        )
    if (
        request_id == state.request_id
        and type(state.request).model_validate(request) != state.request
    ):
        raise AgentInvocationProtocolError(
            "REQUEST_ID_PAYLOAD_MISMATCH",
            "a mesma request_id exige solicitação idêntica",
        )


def _trusted_write_context(
    request: SupportRequest,
    runtime: ReadToolRuntime,
) -> TrustedWriteContext:
    """Persiste somente os alvos confiáveis necessários à política e ao gate."""
    return TrustedWriteContext(
        central_asset_id=runtime.central_asset_id,
        current_case_id=request.case_id,
        configured_model_id=runtime.configured_model_id,
    )


def _validate_persisted_trusted_write_context(
    state: AgentState,
    runtime: ReadToolRuntime,
) -> None:
    """Impede que uma retomada substitua o escopo confiável já persistido."""

    expected = _trusted_write_context(state.request, runtime)
    if state.trusted_write_context != expected:
        raise AgentInvocationProtocolError(
            "TRUSTED_WRITE_CONTEXT_DRIFT",
            "o runtime atual diverge do contexto confiável persistido",
        )


def _is_conservative_non_idempotent_resume(
    state: AgentState,
    *,
    checkpoint_anchor: ResumeAnchor,
    pending_nodes: tuple[str, ...],
    execution_id: str,
    new_request: bool,
    proposal: WriteProposal | None,
    original_approval: TrustedActionApproval | None,
    confirmation: ConfirmationReply | None,
) -> bool:
    """Reconhece somente o salto PREPARED que deve terminar como incerto."""

    intents = _current_request_intents(state)
    return (
        not new_request
        and state.final_result is None
        and checkpoint_anchor is ResumeAnchor.PREPARE_INTENT
        and pending_nodes == ("execute_action",)
        and proposal is None
        and original_approval is None
        and confirmation is None
        and state.pending_proposal is not None
        and len(intents) == 1
        and intents[0].status is IntentStatus.PREPARED
        and intents[0].scope.action != "reprocess_analysis"
        and intents[0].prepared_execution_id is not None
        and intents[0].prepared_execution_id != execution_id
    )


def _validate_current_read_access(
    state: AgentState,
    runtime: ReadToolRuntime,
    *,
    planner_enabled: bool,
    include_current_flow: bool,
    allow_review_resume: bool = False,
) -> None:
    if "read" in runtime.permissions:
        return
    if allow_review_resume and state.review_request is not None:
        return
    contains_read_artifact = bool(state.tool_observations)
    current_intents = _current_request_intents(state)
    fallback_read_result = include_current_flow and (
        state.resume_anchor is ResumeAnchor.FINISH
        or (
            not planner_enabled
            and state.pending_proposal is None
            and state.approval is None
            and not current_intents
        )
    )
    if contains_read_artifact or fallback_read_result:
        raise AgentInvocationProtocolError(
            "READ_PERMISSION_REQUIRED",
            "o runtime atual não pode acessar resultado de leitura",
        )


def _validate_legacy_intents_for_write(state: AgentState) -> None:
    if any(
        intent.request_id is None and intent.status in _ACTIVE_INTENT_STATUSES
        for intent in state.intents
    ):
        raise AgentInvocationProtocolError(
            "LEGACY_INTENT_REQUIRES_REVIEW",
            "uma intenção legada ativa exige revisão antes de nova escrita",
        )


def _confirmation_command(
    *,
    snapshot: GraphStateSnapshot,
    state: AgentState,
    confirmation: ConfirmationReply,
) -> Command:
    if snapshot.next != ("confirmation_gate",):
        raise AgentInvocationProtocolError(
            "STALE_CONFIRMATION",
            "a intenção não está aguardando esta confirmação",
        )
    if len(snapshot.interrupts) != 1:
        raise AgentInvocationProtocolError(
            "AMBIGUOUS_CONFIRMATION",
            "o checkpoint deve possuir exatamente um interrupt",
        )
    matches = tuple(
        intent
        for intent in _current_request_intents(state)
        if intent.status is IntentStatus.AWAITING_CONFIRMATION
    )
    if len(matches) != 1 or matches[0].intent_id != confirmation.intent_id:
        raise AgentInvocationProtocolError(
            "STALE_CONFIRMATION",
            "a confirmação não corresponde à intenção pendente",
        )
    intent = matches[0]
    approval = (
        TrustedActionApproval(
            action=intent.scope.action,
            target_id=intent_scope_target_id(intent.scope),
            material_parameters=intent_scope_material_parameters(intent.scope),
            source=ApprovalSource.CONFIRMATION,
        )
        if confirmation.decision == "approve"
        else None
    )
    continued = _replace_state(state, approval=approval)
    interrupt_id = snapshot.interrupts[0].id
    return Command(
        resume={interrupt_id: confirmation.model_dump(mode="json")},
        update=continued.model_dump(mode="json"),
    )


def _review_command(
    *,
    snapshot: GraphStateSnapshot,
    state: AgentState,
    reply: ReviewReply,
    reviewer: ReviewerIdentity,
    received_at: datetime,
) -> Command:
    if snapshot.next != ("await_human_review",):
        raise AgentInvocationProtocolError(
            "STALE_REVIEW", "o checkpoint não aguarda esta revisão"
        )
    if len(snapshot.interrupts) != 1 or state.review_request is None:
        raise AgentInvocationProtocolError(
            "AMBIGUOUS_REVIEW", "o checkpoint deve possuir uma revisão pendente"
        )
    request = state.review_request
    if received_at < request.created_at:
        raise AgentInvocationProtocolError(
            "INVALID_REVIEW_TIME", "o relógio confiável precede a revisão"
        )
    if reply.review_id != request.review_id:
        raise AgentInvocationProtocolError(
            "STALE_REVIEW", "o reply pertence a outra revisão"
        )
    if reviewer.company_id != state.identity.company_id:
        raise AgentInvocationProtocolError(
            "REVIEW_COMPANY_MISMATCH", "o revisor pertence a outra empresa"
        )
    if received_at < request.expires_at:
        if reply.operation not in {
            operation.value for operation in request.allowed_operations
        }:
            raise AgentInvocationProtocolError(
                "INVALID_REVIEW_OPERATION", "a operação não é permitida"
            )
        if isinstance(reply, (ReviewApproveReply, ReviewEditReply)):
            try:
                build_reviewed_draft(request, reply, state.ledger)
            except ValueError as error:
                raise AgentInvocationProtocolError(
                    "INVALID_REVIEW_EDIT",
                    "a edição não passa pelo contrato estrutural",
                ) from error
    envelope = ReviewResumeEnvelope(
        reply=reply,
        reviewer=reviewer,
        received_at=received_at,
    )
    return Command(
        resume={snapshot.interrupts[0].id: envelope.model_dump(mode="json")},
        update=state.model_dump(mode="json"),
    )


def _terminal_review_replay(
    state: AgentState,
    reply: ReviewReply,
    reviewer: ReviewerIdentity,
) -> bool:
    if state.review_request is None:
        return False
    resolution_digest = review_resolution_subject_digest(reply, reviewer)
    if state.review_expiry is not None:
        return (
            state.review_expiry.trigger == "reply"
            and reply.review_id == state.review_expiry.review_id
            and resolution_digest == state.review_expiry.resolution_digest
        )
    audit = state.review_audit
    return bool(
        audit is not None
        and reply.review_id == audit.review_id
        and state.review_resolution is not None
        and resolution_digest
        == review_resolution_subject_digest(
            state.review_resolution.reply,
            state.review_resolution.reviewer,
        )
    )


def _expire_review_for_new_request(
    state: AgentState,
    *,
    expired_at: datetime,
) -> AgentState:
    request = state.review_request
    if request is None or expired_at < request.expires_at:
        raise ValueError("somente revisão vencida pode ser encerrada internamente")
    data = state.advance_step().model_dump(mode="json")
    data.update(
        resume_anchor=ResumeAnchor.AWAIT_HUMAN_REVIEW.value,
        decision=AgentDecision.REQUIRE_HUMAN_REVIEW.value,
        final_result=render_review_expired_result().model_dump(mode="json"),
        release_gate=request.gate_basis.model_dump(mode="json"),
        review=ReviewRecord(
            status=ReviewStatus.REQUIRED,
            reason="human_review:expired",
        ).model_dump(mode="json"),
        review_expiry=ReviewExpiry(
            review_id=request.review_id,
            review_digest=canonical_digest(request),
            trigger="new_request",
            resolution_digest=None,
            expired_at=expired_at,
        ).model_dump(mode="json"),
        review_continuation=None,
    )
    return AgentState.model_validate(data)


async def _invoke_agent_unobserved(
    graph: AgentGraph,
    *,
    request: SupportRequest,
    runtime: ReadToolRuntime,
    thread_id: str,
    request_id: str,
    execution_id: str,
    step_limit: int | None = None,
    proposal: WriteProposal | None = None,
    original_approval: TrustedActionApproval | None = None,
    confirmation: ConfirmationReply | None = None,
    review_reply: ReviewReply | None = None,
    reviewer: ReviewerIdentity | None = None,
) -> AgentState:
    """Cria ou continua estado confiável e executa com checkpoint síncrono."""
    if not isinstance(runtime, ReadToolRuntime):
        raise TypeError("runtime autenticado é obrigatório")
    thread_id = _require_opaque_id(thread_id, name="thread_id")
    request_id = _require_opaque_id(request_id, name="request_id")
    execution_id = _require_opaque_id(execution_id, name="execution_id")
    if (review_reply is None) != (reviewer is None):
        raise AgentInvocationProtocolError(
            "INVALID_REVIEW_ENVELOPE",
            "reply e identidade confiável do revisor são obrigatórios juntos",
        )
    if review_reply is not None and not isinstance(
        review_reply, (ReviewApproveReply, ReviewEditReply, ReviewRejectReply)
    ):
        raise TypeError("review_reply deve usar o contrato discriminado")
    if reviewer is not None and not isinstance(reviewer, ReviewerIdentity):
        raise TypeError("reviewer autenticado é obrigatório")
    planner_enabled = bool(getattr(graph, "planner_enabled", False))
    _validate_write_boundary(
        request=request,
        runtime=runtime,
        proposal=proposal,
        confirmation=confirmation,
        original_approval=original_approval,
        planner_enabled=planner_enabled,
    )
    _validate_runtime_identity_scope(request, runtime)
    trusted_write_context = _trusted_write_context(request, runtime)
    resolved_step_limit = step_limit
    if resolved_step_limit is None:
        if planner_enabled:
            resolved_step_limit = PLANNER_GRAPH_STEP_LIMIT
        else:
            resolved_step_limit = (
                REPROCESS_GRAPH_STEP_COUNT
                if proposal is not None or confirmation is not None
                else MINIMAL_GRAPH_STEP_COUNT
            )
    if planner_enabled and (
        isinstance(resolved_step_limit, bool)
        or not isinstance(resolved_step_limit, int)
        or resolved_step_limit > PLANNER_GRAPH_STEP_LIMIT
    ):
        raise AgentInvocationProtocolError(
            "PLANNER_STEP_LIMIT_EXCEEDED",
            "o caminho do planner aceita no máximo 24 passos",
        )
    config: dict[str, object] = {"configurable": {"thread_id": thread_id}}

    async with graph.thread_lock(thread_id):
        snapshot = await graph.aget_state(config)
        persisted_values = snapshot.values
        if persisted_values:
            checkpoint_anchor = _required_resume_anchor(persisted_values)
            persisted = AgentState.model_validate(persisted_values)
            new_request = request_id != persisted.request_id
            conservative_resume = _is_conservative_non_idempotent_resume(
                persisted,
                checkpoint_anchor=checkpoint_anchor,
                pending_nodes=snapshot.next,
                execution_id=execution_id,
                new_request=new_request,
                proposal=proposal,
                original_approval=original_approval,
                confirmation=confirmation,
            )
            _validate_persisted_thread_boundary(
                state=persisted,
                request=request,
                runtime=runtime,
                thread_id=thread_id,
                request_id=request_id,
                execution_id=execution_id,
                allow_conservative_target_drift=conservative_resume,
            )
            if (
                not new_request
                and persisted.final_result is None
                and persisted.review_audit is not None
                and review_reply is not None
                and reviewer is not None
            ):
                if not _terminal_review_replay(persisted, review_reply, reviewer):
                    raise AgentInvocationProtocolError(
                        "DIVERGENT_REVIEW",
                        "o julgamento diverge da auditoria persistida",
                    )
                # Retry do mesmo envelope após o checkpoint da auditoria:
                # continua o gate pendente sem gravar uma segunda auditoria.
                review_reply = None
                reviewer = None
            internal_review_continuation = (
                not new_request
                and persisted.final_result is None
                and persisted.review_audit is not None
                and snapshot.next == ("release_gate",)
            )
            pending_review_new_request = (
                new_request
                and review_reply is None
                and persisted.review_request is not None
                and persisted.review_audit is None
                and persisted.review_expiry is None
            )
            expiry_boundary_time = _utc_now() if pending_review_new_request else None
            close_expired_review = bool(
                expiry_boundary_time is not None
                and persisted.review_request is not None
                and expiry_boundary_time >= persisted.review_request.expires_at
            )
            if persisted.final_result is not None:
                _validate_terminal_resume_anchor(persisted, checkpoint_anchor)
                if snapshot.next:
                    raise AgentInvocationProtocolError(
                        "TERMINAL_WITH_PENDING_WORK",
                        "checkpoint terminal ainda possui trabalho pendente",
                    )
                resume_predecessor = None
            else:
                resume_predecessor = (
                    _resume_predecessor(
                        checkpoint_anchor,
                        snapshot.next,
                        planner_enabled=planner_enabled,
                        state=persisted,
                    )
                    if snapshot.next
                    else None
                )
            if not conservative_resume:
                _validate_runtime_request_scope(request, runtime)
            _validate_current_read_access(
                persisted,
                runtime,
                planner_enabled=planner_enabled,
                include_current_flow=not new_request,
                allow_review_resume=(
                    review_reply is not None
                    or internal_review_continuation
                    or close_expired_review
                ),
            )
            if (
                planner_enabled
                and not new_request
                and persisted.step_limit > PLANNER_GRAPH_STEP_LIMIT
            ):
                raise AgentInvocationProtocolError(
                    "PLANNER_STEP_LIMIT_EXCEEDED",
                    "checkpoint do planner excede o teto de 24 passos",
                )
            persisted_original_approval = (
                persisted.approval
                if (
                    not new_request
                    and persisted.approval is not None
                    and persisted.approval.source is ApprovalSource.ORIGINAL_REQUEST
                )
                else None
            )
            write_flow = (
                proposal is not None
                or confirmation is not None
                or original_approval is not None
                or persisted_original_approval is not None
                or (not new_request and persisted.pending_proposal is not None)
            )
            if write_flow:
                if not isinstance(runtime, WriteToolRuntime):
                    raise AgentInvocationProtocolError(
                        "WRITE_RUNTIME_REQUIRED",
                        "o fluxo de escrita exige contexto confiável de escrita",
                    )
                if not conservative_resume:
                    _validate_write_boundary(
                        request=request,
                        runtime=runtime,
                        proposal=(
                            proposal
                            if proposal is not None
                            else persisted.pending_proposal
                        ),
                        confirmation=confirmation,
                        original_approval=persisted_original_approval,
                        planner_enabled=planner_enabled,
                    )
                _validate_legacy_intents_for_write(persisted)
                if not new_request and not conservative_resume:
                    _validate_persisted_trusted_write_context(
                        persisted,
                        runtime,
                    )
            if new_request and _request_id_has_history(persisted, request_id):
                raise AgentInvocationProtocolError(
                    "REQUEST_ID_ALREADY_USED",
                    "request_id já pertence ao histórico do thread",
                )
            if close_expired_review:
                assert expiry_boundary_time is not None
                persisted = _expire_review_for_new_request(
                    persisted,
                    expired_at=expiry_boundary_time,
                )
                config = await graph.aupdate_state(
                    snapshot.config,
                    persisted.model_dump(mode="json"),
                    as_node="await_human_review",
                )
                snapshot = await graph.aget_state(config)
                persisted = AgentState.model_validate(snapshot.values)
                if snapshot.next:
                    raise AgentInvocationProtocolError(
                        "EXPIRED_REVIEW_WITH_PENDING_WORK",
                        "a revisão vencida não encerrou canonicamente",
                    )
            state = persisted.continue_with(
                request=request,
                identity=runtime.identity,
                permissions=runtime.permissions,
                request_id=request_id,
                execution_id=execution_id,
                step_limit=resolved_step_limit,
                trusted_write_context=trusted_write_context,
            )
            if not new_request:
                _validate_persisted_write_inputs(
                    persisted,
                    proposal=proposal,
                    original_approval=original_approval,
                )
            if not new_request and persisted.final_result is not None:
                if review_reply is not None and reviewer is not None:
                    if _terminal_review_replay(persisted, review_reply, reviewer):
                        _validate_current_read_access(
                            persisted,
                            runtime,
                            planner_enabled=planner_enabled,
                            include_current_flow=True,
                        )
                        return persisted
                    raise AgentInvocationProtocolError(
                        "DIVERGENT_REVIEW",
                        "o julgamento diverge da revisão terminal persistida",
                    )
                if confirmation is not None and not _terminal_confirmation_replay(
                    persisted,
                    confirmation,
                ):
                    raise AgentInvocationProtocolError(
                        "STALE_CONFIRMATION",
                        "a confirmação não corresponde ao resultado terminal",
                    )
                return persisted
            if new_request:
                if persisted.review_request is not None and (
                    persisted.review_audit is None and persisted.review_expiry is None
                ):
                    raise AgentInvocationProtocolError(
                        "PENDING_REVIEW_BLOCKS_NEW_REQUEST",
                        "uma revisão pendente bloqueia nova solicitação",
                    )
                if review_reply is not None:
                    raise AgentInvocationProtocolError(
                        "STALE_REVIEW",
                        "um reply não pode iniciar uma nova solicitação",
                    )
                if confirmation is not None:
                    raise AgentInvocationProtocolError(
                        "STALE_CONFIRMATION",
                        "confirmação não pode iniciar uma nova solicitação",
                    )
                if any(
                    intent.status in _ACTIVE_INTENT_STATUSES
                    for intent in persisted.intents
                ):
                    raise AgentInvocationProtocolError(
                        "ACTIVE_INTENT_BLOCKS_NEW_REQUEST",
                        "uma intenção não terminal bloqueia nova solicitação",
                    )
                state = _replace_state(
                    state,
                    pending_proposal=proposal,
                    approval=original_approval,
                )
                _validate_current_read_access(
                    state,
                    runtime,
                    planner_enabled=planner_enabled,
                    include_current_flow=True,
                )
                if (
                    state.pending_proposal is not None
                    and state.step_limit
                    < _required_step_count(
                        state,
                        planner_enabled=planner_enabled,
                    )
                ):
                    raise AgentInvocationProtocolError(
                        "STEP_LIMIT_EXHAUSTED",
                        "o orçamento não comporta o fluxo solicitado",
                    )
                config = await graph.aupdate_state(
                    snapshot.config,
                    state.model_dump(mode="json"),
                    as_node=START,
                )
                invocation_input = None
            else:
                if state.step_limit < _required_step_count(
                    state,
                    planner_enabled=planner_enabled,
                ):
                    raise AgentInvocationProtocolError(
                        "STEP_LIMIT_EXHAUSTED",
                        "checkpoint parcial esgotou o orçamento de passos",
                    )
                if review_reply is not None and reviewer is not None:
                    invocation_input = _review_command(
                        snapshot=snapshot,
                        state=state,
                        reply=review_reply,
                        reviewer=reviewer,
                        received_at=_utc_now(),
                    )
                    config = snapshot.config
                elif confirmation is not None:
                    invocation_input = _confirmation_command(
                        snapshot=snapshot,
                        state=state,
                        confirmation=confirmation,
                    )
                    config = snapshot.config
                else:
                    if snapshot.interrupts:
                        raise AgentInvocationProtocolError(
                            (
                                "REVIEW_REQUIRED"
                                if state.review_request is not None
                                else "CONFIRMATION_REQUIRED"
                            ),
                            "o fluxo aguarda uma resposta estruturada",
                        )
                    predecessor = (
                        resume_predecessor
                        if resume_predecessor is not None
                        else _resume_predecessor(
                            checkpoint_anchor,
                            snapshot.next,
                            planner_enabled=planner_enabled,
                            state=state,
                        )
                    )
                    config = await graph.aupdate_state(
                        snapshot.config,
                        state.model_dump(mode="json"),
                        as_node=predecessor,
                    )
                    invocation_input = None
        else:
            _validate_runtime_request_scope(request, runtime)
            if confirmation is not None:
                raise AgentInvocationProtocolError(
                    "STALE_CONFIRMATION",
                    "não existe intenção persistida para confirmar",
                )
            if review_reply is not None:
                raise AgentInvocationProtocolError(
                    "STALE_REVIEW", "não existe revisão persistida para responder"
                )
            if proposal is not None and resolved_step_limit < (
                PLANNER_WRITE_GRAPH_STEP_COUNT
                if planner_enabled
                else REPROCESS_GRAPH_STEP_COUNT
            ):
                raise AgentInvocationProtocolError(
                    "STEP_LIMIT_EXHAUSTED",
                    "o orçamento não comporta o fluxo solicitado",
                )
            state = AgentState(
                request=request,
                identity=runtime.identity,
                permissions=runtime.permissions,
                request_id=request_id,
                thread_id=thread_id,
                execution_id=execution_id,
                thread_scope=ThreadScope(
                    thread_id=thread_id,
                    case_id=request.case_id,
                    company_id=runtime.identity.company_id,
                    user_id=runtime.identity.user_id,
                ),
                trusted_write_context=trusted_write_context,
                step_limit=resolved_step_limit,
                pending_proposal=proposal,
                approval=original_approval,
            )
            _validate_current_read_access(
                state,
                runtime,
                planner_enabled=planner_enabled,
                include_current_flow=True,
            )
            invocation_input = state.model_dump(mode="json")

        await graph.ainvoke(
            invocation_input,
            config,
            context=runtime,
            durability="sync",
        )
        final_snapshot = await graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        if not final_snapshot.values:
            raise TypeError("o grafo não persistiu estado após a execução")
        final_state = AgentState.model_validate(final_snapshot.values)
        _validate_current_read_access(
            final_state,
            runtime,
            planner_enabled=planner_enabled,
            include_current_flow=True,
        )
    return final_state


def _safe_execution_trace(
    graph: AgentGraph,
    *,
    request: SupportRequest,
    runtime: ReadToolRuntime,
    thread_id: str,
    request_id: str,
    execution_id: str,
    review_resumed: bool,
):
    telemetry = getattr(graph, "telemetry", None)
    if not isinstance(telemetry, AgentTelemetry):
        telemetry = NullTelemetry()
    try:
        correlations = ExecutionCorrelations(
            request_id=request_id,
            thread_id=thread_id,
            execution_id=execution_id,
            case_id=request.case_id,
            company_id=runtime.identity.company_id,
            user_id=runtime.identity.user_id,
            planner_enabled=bool(getattr(graph, "planner_enabled", False)),
            review_resumed=review_resumed,
        )
        return telemetry.start_execution(correlations)
    except Exception:
        # Telemetria nunca amplia a validação da fronteira de negócio.
        return NullTelemetry().start_execution(
            ExecutionCorrelations(
                request_id="unavailable",
                thread_id="unavailable",
                execution_id="unavailable",
                case_id="unavailable",
                company_id="unavailable",
                user_id="unavailable",
                planner_enabled=bool(getattr(graph, "planner_enabled", False)),
                review_resumed=review_resumed,
            )
        )


async def invoke_agent_observed(
    graph: AgentGraph,
    *,
    request: SupportRequest,
    runtime: ReadToolRuntime,
    thread_id: str,
    request_id: str,
    execution_id: str,
    step_limit: int | None = None,
    proposal: WriteProposal | None = None,
    original_approval: TrustedActionApproval | None = None,
    confirmation: ConfirmationReply | None = None,
    review_reply: ReviewReply | None = None,
    reviewer: ReviewerIdentity | None = None,
) -> AgentInvocationResult:
    """Executa uma invocação técnica e devolve correlação fora do checkpoint."""

    trace = _safe_execution_trace(
        graph,
        request=request,
        runtime=runtime,
        thread_id=thread_id,
        request_id=request_id,
        execution_id=execution_id,
        review_resumed=review_reply is not None,
    )
    planner_enabled = bool(getattr(graph, "planner_enabled", False))
    with trace.activate():
        with trace.span(SpanName.REQUEST, trace.request_attributes) as request_span:
            try:
                state = await _invoke_agent_unobserved(
                    graph,
                    request=request,
                    runtime=runtime,
                    thread_id=thread_id,
                    request_id=request_id,
                    execution_id=execution_id,
                    step_limit=step_limit,
                    proposal=proposal,
                    original_approval=original_approval,
                    confirmation=confirmation,
                    review_reply=review_reply,
                    reviewer=reviewer,
                )
                replayed = state.execution_id != execution_id
                outcome = (
                    Outcome.REPLAYED
                    if replayed
                    else Outcome.SUSPENDED
                    if state.final_result is None
                    else Outcome.OK
                )
                with trace.span(
                    SpanName.RESPONSE,
                    ResponseSpanAttributes(
                        planner_enabled=planner_enabled,
                        replayed=replayed,
                    ),
                ) as response_span:
                    response_span.finish(outcome)
                request_span.finish(outcome)
                return AgentInvocationResult(state=state, trace_id=trace.trace_id)
            except BaseException as error:
                cancelled = isinstance(error, asyncio.CancelledError)
                error_code = (
                    ErrorCode.CANCELLED
                    if cancelled
                    else ErrorCode.PROTOCOL
                    if isinstance(error, AgentInvocationProtocolError)
                    else ErrorCode.RUNTIME
                )
                with trace.span(
                    SpanName.RESPONSE,
                    ResponseSpanAttributes(
                        planner_enabled=planner_enabled,
                        replayed=False,
                    ),
                ) as response_span:
                    response_span.finish(
                        Outcome.CANCELLED if cancelled else Outcome.ERROR,
                        error_code,
                    )
                request_span.finish(
                    Outcome.CANCELLED if cancelled else Outcome.ERROR,
                    error_code,
                )
                raise


async def invoke_agent(
    graph: AgentGraph,
    *,
    request: SupportRequest,
    runtime: ReadToolRuntime,
    thread_id: str,
    request_id: str,
    execution_id: str,
    step_limit: int | None = None,
    proposal: WriteProposal | None = None,
    original_approval: TrustedActionApproval | None = None,
    confirmation: ConfirmationReply | None = None,
    review_reply: ReviewReply | None = None,
    reviewer: ReviewerIdentity | None = None,
) -> AgentState:
    """API compatível: mantém o retorno histórico e ativa telemetria injetada."""

    result = await invoke_agent_observed(
        graph,
        request=request,
        runtime=runtime,
        thread_id=thread_id,
        request_id=request_id,
        execution_id=execution_id,
        step_limit=step_limit,
        proposal=proposal,
        original_approval=original_approval,
        confirmation=confirmation,
        review_reply=review_reply,
        reviewer=reviewer,
    )
    return result.state
