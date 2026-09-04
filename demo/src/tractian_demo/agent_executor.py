from __future__ import annotations

from collections.abc import Mapping
import httpx

from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import SupportRequest
from tractian_agent.entrypoint import AgentGraph, invoke_agent_observed
from tractian_agent.tools.runtime import WriteToolRuntime

from tractian_demo.contracts import AgentRunProjection, CaseMessage, DemoCase, Execution


class LiveAgentExecutor:
    """Adapta o entrypoint público do agente à projeção mínima da demonstração."""

    def __init__(
        self,
        *,
        graph: AgentGraph,
        industrial_client: IndustrialApiClient,
        identity_client: httpx.AsyncClient,
        provider: str,
    ) -> None:
        self._graph = graph
        self._industrial_client = industrial_client
        self._identity_client = identity_client
        self._provider = provider

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
        )

    async def execute(
        self, *, case: DemoCase, message: CaseMessage, execution: Execution
    ) -> AgentRunProjection:
        runtime = await self._runtime(case)
        industrial_case_id = case.source_case_id or case.id
        result = await invoke_agent_observed(
            self._graph,
            request=SupportRequest(
                case_id=industrial_case_id,
                ticket_id=case.ticket_id,
                asset_id=case.asset_id,
                message=message.content,
                identity={"user_id": case.requester_id, "company_id": case.company_id},
            ),
            runtime=runtime,
            thread_id=case.id,
            request_id=message.id,
            execution_id=execution.id,
        )
        state = result.state
        final = state.final_result
        if final is None:
            raise RuntimeError("AGENT_WITHOUT_PUBLIC_RESULT")
        tool_names = tuple(
            sorted({item.artifact.tool_name for item in state.tool_observations})
        )
        limitations = sum(
            len(item.artifact.outcome.notes or "") > 0
            for item in state.tool_observations
        ) + len(state.ledger.gaps) + len(state.ledger.conflicts)
        return AgentRunProjection(
            assistant_message=final.message,
            decision=final.decision.value,
            trace_id=str(result.trace_id),
            provider=self._provider,
            fallback_reason=None,
            evidence_count=len(final.evidence_ids),
            limitation_count=limitations,
            tool_names=tool_names,
        )
