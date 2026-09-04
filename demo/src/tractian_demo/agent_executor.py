from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import httpx
from pydantic import TypeAdapter

from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import SupportRequest
from tractian_agent.entrypoint import AgentGraph, invoke_agent_observed
from tractian_agent.human_review import ReviewApproveReply, ReviewRejectReply
from tractian_agent.state import ReviewerIdentity
from tractian_agent.tools.runtime import WriteToolRuntime
from tractian_agent.write_contracts import (
    ConfirmationReply,
    IntentStatus,
    intent_scope_target_id,
)
from tractian_agent.write_policy import (
    ApprovalSource,
    DelegatedApprovalAttestation,
    TrustedActionApproval,
    WriteMaterialParameters,
    WriteProposal,
    delegated_subject_digest,
)

from tractian_demo.contracts import (
    AgentRunProjection,
    CaseMessage,
    DecisionCandidate,
    DecisionStatus,
    DemoCase,
    Execution,
)
from tractian_demo.decisions import route_decision
from tractian_demo.repository import DemoRepository
from tractian_demo.provider_router import FallbackTracker


_PROPOSAL_ADAPTER = TypeAdapter(WriteProposal)


class LiveAgentExecutor:
    """Adapta o entrypoint público do agente à projeção mínima da demonstração."""

    def __init__(
        self,
        *,
        graph: AgentGraph,
        industrial_client: IndustrialApiClient,
        identity_client: httpx.AsyncClient,
        provider: str,
        repository: DemoRepository | None = None,
        fallback_tracker: FallbackTracker | None = None,
        fallback_provider: str | None = None,
    ) -> None:
        self._graph = graph
        self._industrial_client = industrial_client
        self._identity_client = identity_client
        self._provider = provider
        self._repository = repository
        self._fallback_tracker = fallback_tracker
        self._fallback_provider = fallback_provider

    async def _runtime(self, case: DemoCase) -> WriteToolRuntime:
        response = await self._identity_client.get(
            "/users/me", headers={"x-user-id": case.requester_id}
        )
        response.raise_for_status()
        profile: Mapping[str, object] = response.json()
        if (
            profile.get("id") != case.requester_id
            or profile.get("company_id") != case.company_id
        ):
            raise RuntimeError("IDENTITY_SCOPE_MISMATCH")
        permissions = frozenset(str(item) for item in profile.get("permissions", ()))
        industrial_case_id = case.source_case_id or case.id
        return WriteToolRuntime.create(
            user_id=case.requester_id,
            company_id=case.company_id,
            permissions=permissions,
            central_asset_id=case.asset_id,
            current_case_id=industrial_case_id,
            client=self._industrial_client,
            seed=(
                "complete"
                if case.simulation_mode == "complete"
                else case.seed
                if case.simulation_mode == "custom_seed"
                else None
            ),
        )

    async def execute(
        self, *, case: DemoCase, message: CaseMessage, execution: Execution
    ) -> AgentRunProjection:
        if self._fallback_tracker is not None:
            self._fallback_tracker.reset()
        runtime = await self._runtime(case)
        invocation_message = message
        request_id = message.id
        confirmation = None
        review_reply = None
        reviewer = None
        proposal = None
        delegated_approval = None
        if execution.resume_decision_id is not None:
            if self._repository is None:
                raise RuntimeError("RESUME_REPOSITORY_REQUIRED")
            decision = self._repository.get_decision(execution.resume_decision_id)
            original_execution = self._repository.get_execution(decision.execution_id)
            invocation_message = self._repository.get_message(
                original_execution.message_id
            )
            approved = decision.status is DecisionStatus.APPROVED
            if execution.resume_kind == "confirmation":
                request_id = invocation_message.id
                confirmation = ConfirmationReply(
                    intent_id=str(decision.scope["intent_id"]),
                    decision="approve" if approved else "deny",
                )
            elif execution.resume_kind == "technical_review":
                request_id = invocation_message.id
                review_id = str(decision.scope["review_id"])
                review_reply = (
                    ReviewApproveReply(review_id=review_id, operation="approve")
                    if approved
                    else ReviewRejectReply(review_id=review_id, operation="reject")
                )
                reviewer = ReviewerIdentity(
                    reviewer_id=decision.resolved_by or "unknown_reviewer",
                    company_id=case.company_id,
                    permission="review",
                )
            elif execution.resume_kind == "delegated_action":
                if not approved:
                    return AgentRunProjection(
                        assistant_message="A autoridade rejeitou a ação proposta.",
                        decision="guide",
                        trace_id=f"decision:{decision.id}",
                        provider=self._provider,
                        fallback_reason=None,
                        evidence_count=0,
                        limitation_count=0,
                        tool_names=(),
                    )
                request_id = message.id
                if decision.required_permission is None or decision.resolved_at is None:
                    raise RuntimeError("INVALID_DELEGATED_DECISION")
                proposal = _PROPOSAL_ADAPTER.validate_python(decision.scope["proposal"])
                material = WriteMaterialParameters.model_validate(
                    decision.scope["material_parameters"]
                )
                subject = {
                    "action": proposal.action,
                    "target_id": decision.scope["target_id"],
                    "material_parameters": material.model_dump(mode="json"),
                    "company_id": case.company_id,
                }
                delegated_approval = TrustedActionApproval(
                    action=proposal.action,
                    target_id=str(decision.scope["target_id"]),
                    material_parameters=material,
                    source=ApprovalSource.DELEGATED,
                    delegation=DelegatedApprovalAttestation(
                        decision_id=decision.id,
                        approver_id=decision.resolved_by or "unknown_approver",
                        company_id=case.company_id,
                        permission=decision.required_permission,
                        subject_digest=delegated_subject_digest(subject),
                        approved_at=decision.resolved_at,
                        expires_at=decision.expires_at,
                    ),
                )
            else:
                raise RuntimeError("UNKNOWN_RESUME_KIND")
        industrial_case_id = case.source_case_id or case.id
        result = await invoke_agent_observed(
            self._graph,
            request=SupportRequest(
                case_id=industrial_case_id,
                ticket_id=case.ticket_id,
                asset_id=case.asset_id,
                message=invocation_message.content,
                identity={"user_id": case.requester_id, "company_id": case.company_id},
            ),
            runtime=runtime,
            thread_id=case.id,
            request_id=request_id,
            execution_id=execution.id,
            proposal=proposal,
            original_approval=delegated_approval,
            confirmation=confirmation,
            review_reply=review_reply,
            reviewer=reviewer,
        )
        state = result.state
        final = state.final_result
        candidate = self._decision_candidate(state)
        if final is None and candidate is None:
            raise RuntimeError("AGENT_WITHOUT_PUBLIC_RESULT")
        tool_names = tuple(
            sorted({item.artifact.tool_name for item in state.tool_observations})
        )
        limitations = (
            sum(
                len(item.artifact.outcome.notes or "") > 0
                for item in state.tool_observations
            )
            + len(state.ledger.gaps)
            + len(state.ledger.conflicts)
        )
        fallback_reason = (
            self._fallback_tracker.reason
            if self._fallback_tracker is not None
            else None
        )
        provider = (
            self._fallback_provider if fallback_reason is not None else self._provider
        )
        return AgentRunProjection(
            assistant_message=(
                final.message
                if final is not None
                else "A resposta foi pausada para uma decisão humana segura."
            ),
            decision=(
                final.decision.value if final is not None else "require_human_review"
            ),
            trace_id=result.trace_id.value,
            provider=provider,
            fallback_reason=fallback_reason,
            evidence_count=len(final.evidence_ids) if final is not None else 0,
            limitation_count=limitations,
            tool_names=tool_names,
            decision_candidate=candidate,
        )

    @staticmethod
    def _decision_candidate(state) -> DecisionCandidate | None:
        if state.review_request is not None and state.review_audit is None:
            request = state.review_request
            routed = route_decision(
                action=None,
                requester_permissions=state.permissions,
                technical_review=True,
            )
            return routed.model_copy(
                update={
                    "summary": "Revisar bloqueio técnico antes de liberar a resposta.",
                    "scope": {"review_id": request.review_id},
                    "expires_at": request.expires_at,
                }
            )
        if (
            state.final_result is not None
            and state.final_result.decision.value == "require_human_review"
        ):
            return DecisionCandidate(
                audience="tractian",
                kind="technical_review",
                summary="Avaliar a execução encerrada de forma conservadora.",
                scope={"reason": "agent_conservative_terminal"},
                required_permission=None,
                resume_kind="acknowledgement",
                expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
            )
        current = [
            item for item in state.intents if item.request_id == state.request_id
        ]
        if not current or state.pending_proposal is None:
            return None
        intent = current[-1]
        needs_decision = intent.status is IntentStatus.AWAITING_CONFIRMATION or (
            intent.status is IntentStatus.DENIED
            and intent.decision.reason.value == "missing_permission"
        )
        if not needs_decision:
            return None
        proposal = state.pending_proposal
        routed = route_decision(
            action=proposal.action, requester_permissions=state.permissions
        )
        material = (
            {"criticality": getattr(intent.scope, "criticality")}
            if proposal.action == "update_asset_criticality"
            else {"criticality": None}
        )
        return routed.model_copy(
            update={
                "summary": f"Autorizar {proposal.action} para {intent_scope_target_id(intent.scope)}.",
                "scope": {
                    "intent_id": intent.intent_id,
                    "proposal": proposal.model_dump(mode="json"),
                    "target_id": intent_scope_target_id(intent.scope),
                    "material_parameters": material,
                },
                "expires_at": datetime.now(timezone.utc) + timedelta(hours=24),
            }
        )
