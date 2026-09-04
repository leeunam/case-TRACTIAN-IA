from __future__ import annotations

from typing import Protocol

from tractian_demo.contracts import AgentRunProjection, CaseMessage, DemoCase, Execution
from tractian_demo.repository import DemoRepository


class AgentExecutor(Protocol):
    async def execute(
        self, *, case: DemoCase, message: CaseMessage, execution: Execution
    ) -> AgentRunProjection: ...


class DemoWorker:
    def __init__(
        self, repository: DemoRepository, executor: AgentExecutor, *, worker_id: str
    ) -> None:
        self._repository = repository
        self._executor = executor
        self._worker_id = worker_id

    async def run_once(self) -> bool:
        execution = self._repository.claim_execution(worker_id=self._worker_id)
        if execution is None:
            return False
        case = self._repository.get_case(execution.case_id)
        message = self._repository.get_message(execution.message_id)
        self._repository.append_event(
            case.id, "planner.started", {"execution_id": execution.id}
        )
        try:
            projection = await self._executor.execute(
                case=case, message=message, execution=execution
            )
        except BaseException:
            self._repository.fail_execution(
                execution.id, error_code="AGENT_EXECUTION_FAILED"
            )
            return True
        if projection.decision_candidate is not None:
            self._repository.wait_for_decision(execution.id, projection)
        else:
            self._repository.complete_execution(execution.id, projection)
        return True
