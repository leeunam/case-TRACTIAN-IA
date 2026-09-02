"""Fronteira Python assíncrona do grafo de atendimento."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from typing import Any, Literal, Protocol

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
from tractian_agent.state import AgentState, ResumeAnchor, ThreadScope
from tractian_agent.tools.runtime import ReadToolRuntime, WriteToolRuntime
from tractian_agent.write_contracts import (
    ConfirmationReply,
    IntentStatus,
    WriteIntent,
    intent_scope_material_parameters,
    intent_scope_target_id,
)
from tractian_agent.write_policy import (
    ApprovalSource,
    PolicyReason,
    TrustedActionApproval,
    WriteProposal,
)


class AgentInvocationProtocolError(RuntimeError):
    """Checkpoint válido, mas incompatível com a operação solicitada."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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


def _require_opaque_id(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value or any(
        character.isspace() for character in value
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
    if "resume_anchor" not in persisted_values or persisted_values.get(
        "resume_anchor"
    ) is None:
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
        expected_approval = TrustedActionApproval(
            action=intent.scope.action,
            target_id=intent_scope_target_id(intent.scope),
            material_parameters=intent_scope_material_parameters(intent.scope),
            source=ApprovalSource.CONFIRMATION,
        )
        return state.approval == expected_approval
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
    if (
        original_approval is not None
        and proposal is None
        and not planner_enabled
    ):
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
    if (
        request.asset_id is not None
        and runtime.central_asset_id != request.asset_id
    ):
        raise AgentInvocationProtocolError(
            "WRITE_ASSET_SCOPE_MISMATCH",
            "o ativo central do runtime diverge da solicitação",
        )


def _validate_runtime_request_scope(
    request: SupportRequest,
    runtime: ReadToolRuntime,
) -> None:
    if runtime.identity.model_dump() != request.identity.model_dump():
        raise AgentInvocationProtocolError(
            "RUNTIME_IDENTITY_SCOPE_MISMATCH",
            "a identidade confiável diverge da solicitação",
        )
    if (
        request.asset_id is not None
        and runtime.central_asset_id != request.asset_id
    ):
        raise AgentInvocationProtocolError(
            "RUNTIME_ASSET_SCOPE_MISMATCH",
            "o ativo central do runtime diverge da solicitação",
        )


def _validate_current_read_access(
    state: AgentState,
    runtime: ReadToolRuntime,
    *,
    planner_enabled: bool,
    include_current_flow: bool,
) -> None:
    if "read" in runtime.permissions:
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
) -> AgentState:
    """Cria ou continua estado confiável e executa com checkpoint síncrono."""
    if not isinstance(runtime, ReadToolRuntime):
        raise TypeError("runtime autenticado é obrigatório")
    thread_id = _require_opaque_id(thread_id, name="thread_id")
    request_id = _require_opaque_id(request_id, name="request_id")
    execution_id = _require_opaque_id(execution_id, name="execution_id")
    planner_enabled = bool(getattr(graph, "planner_enabled", False))
    _validate_write_boundary(
        request=request,
        runtime=runtime,
        proposal=proposal,
        confirmation=confirmation,
        original_approval=original_approval,
        planner_enabled=planner_enabled,
    )
    _validate_runtime_request_scope(request, runtime)
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
                    )
                    if snapshot.next
                    else None
                )
            _validate_current_read_access(
                persisted,
                runtime,
                planner_enabled=planner_enabled,
                include_current_flow=not new_request,
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
                    and persisted.approval.source
                    is ApprovalSource.ORIGINAL_REQUEST
                )
                else None
            )
            write_flow = (
                proposal is not None
                or confirmation is not None
                or original_approval is not None
                or persisted_original_approval is not None
                or (
                    not new_request
                    and persisted.pending_proposal is not None
                )
            )
            if write_flow:
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
            if new_request and _request_id_has_history(persisted, request_id):
                raise AgentInvocationProtocolError(
                    "REQUEST_ID_ALREADY_USED",
                    "request_id já pertence ao histórico do thread",
                )
            state = persisted.continue_with(
                request=request,
                identity=runtime.identity,
                permissions=runtime.permissions,
                request_id=request_id,
                execution_id=execution_id,
                step_limit=resolved_step_limit,
            )
            if not new_request:
                _validate_persisted_write_inputs(
                    persisted,
                    proposal=proposal,
                    original_approval=original_approval,
                )
            if not new_request and persisted.final_result is not None:
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
                if confirmation is not None:
                    invocation_input = _confirmation_command(
                        snapshot=snapshot,
                        state=state,
                        confirmation=confirmation,
                    )
                    config = snapshot.config
                else:
                    if snapshot.interrupts:
                        raise AgentInvocationProtocolError(
                            "CONFIRMATION_REQUIRED",
                            "o fluxo aguarda uma confirmação estruturada",
                        )
                    predecessor = (
                        resume_predecessor
                        if resume_predecessor is not None
                        else _resume_predecessor(
                            checkpoint_anchor,
                            snapshot.next,
                            planner_enabled=planner_enabled,
                        )
                    )
                    config = await graph.aupdate_state(
                        snapshot.config,
                        state.model_dump(mode="json"),
                        as_node=predecessor,
                    )
                    invocation_input = None
        else:
            if confirmation is not None:
                raise AgentInvocationProtocolError(
                    "STALE_CONFIRMATION",
                    "não existe intenção persistida para confirmar",
                )
            if (
                proposal is not None
                and resolved_step_limit
                < (
                    PLANNER_WRITE_GRAPH_STEP_COUNT
                    if planner_enabled
                    else REPROCESS_GRAPH_STEP_COUNT
                )
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
