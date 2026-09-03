"""Orquestra execução e avaliação mantendo o golden set fora do runtime."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import asyncio
import hashlib
import inspect
from pathlib import Path
from time import perf_counter

from pydantic_evals import Dataset
from pydantic_evals.reporting import EvaluationReport

from tractian_agent.evaluation.contracts import (
    BenchmarkInput,
    EvaluationOutput,
    ExpectedCase,
)
from tractian_agent.evaluation.capture import output_from_agent_state
from tractian_agent.evaluation.dataset import load_reference_cases
from tractian_agent.contracts import Identity, SupportRequest
from tractian_agent.entrypoint import AgentGraph, invoke_agent
from tractian_agent.tools.runtime import ReadToolRuntime


CaseTask = Callable[[BenchmarkInput], Awaitable[EvaluationOutput]]
RuntimeFactory = Callable[
    [BenchmarkInput],
    ReadToolRuntime | Awaitable[ReadToolRuntime],
]


def _execution_identifier(prefix: str, *parts: str) -> str:
    payload = "\0".join(parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:24]}"


class AgentCaseExecutor:
    """Adapta cada caso sanitizado à fronteira pública real do agente."""

    def __init__(
        self,
        *,
        graph: AgentGraph,
        runtime_factory: RuntimeFactory,
        experiment_id: str,
        step_limit: int,
    ) -> None:
        if not experiment_id or any(character.isspace() for character in experiment_id):
            raise ValueError("experiment_id é obrigatório e não aceita espaços")
        if step_limit < 1:
            raise ValueError("step_limit deve ser positivo")
        self._graph = graph
        self._runtime_factory = runtime_factory
        self._experiment_id = experiment_id
        self._step_limit = step_limit
        self._repetitions: dict[str, int] = {}
        self._repetition_lock = asyncio.Lock()

    async def _next_repetition(self, case_id: str) -> int:
        async with self._repetition_lock:
            value = self._repetitions.get(case_id, 0) + 1
            self._repetitions[case_id] = value
            return value

    async def execute(self, inputs: BenchmarkInput) -> EvaluationOutput:
        """Executa um caso; método nomeado preserva detecção async do Pydantic Evals."""

        repetition = await self._next_repetition(inputs.id)
        run_scope = f"{self._experiment_id}\0{inputs.id}\0{repetition}"
        request = SupportRequest(
            case_id=inputs.id,
            ticket_id=inputs.ticket_id,
            asset_id=inputs.asset_id,
            message=inputs.message,
            identity=Identity(
                user_id=inputs.user_id,
                company_id=inputs.company_id,
            ),
        )
        started = perf_counter()
        runtime = self._runtime_factory(inputs)
        if inspect.isawaitable(runtime):
            runtime = await runtime
        state = await invoke_agent(
            self._graph,
            request=request,
            runtime=runtime,
            thread_id=_execution_identifier("thread_eval", run_scope),
            request_id=_execution_identifier("request_eval", run_scope),
            execution_id=_execution_identifier("execution_eval", run_scope),
            step_limit=self._step_limit,
        )
        duration_ms = (perf_counter() - started) * 1_000
        return output_from_agent_state(state, duration_ms=duration_ms)

    async def __call__(self, inputs: BenchmarkInput) -> EvaluationOutput:
        return await self.execute(inputs)


@dataclass(frozen=True)
class CompletedExecutionBatch:
    """Execuções concluídas e gabarito carregado posteriormente."""

    execution_report: EvaluationReport[BenchmarkInput, EvaluationOutput, None]
    references: dict[str, ExpectedCase]


async def execute_before_loading_references(
    dataset: Dataset[BenchmarkInput, EvaluationOutput, None],
    task: CaseTask,
    *,
    reference_path: Path,
    repeat: int = 1,
    max_concurrency: int = 1,
) -> CompletedExecutionBatch:
    """Executa todos os casos e só então abre o gabarito dos avaliadores."""

    if repeat < 1:
        raise ValueError("repeat deve ser positivo")
    if max_concurrency < 1:
        raise ValueError("max_concurrency deve ser positivo")
    report = await dataset.evaluate(
        task,
        progress=False,
        repeat=repeat,
        max_concurrency=max_concurrency,
    )
    references = load_reference_cases(reference_path)
    executed_case_ids = {
        result.inputs.id for result in (*report.cases, *report.failures)
    }
    if executed_case_ids != set(references):
        raise ValueError("o conjunto de casos executados diverge do gabarito")
    return CompletedExecutionBatch(
        execution_report=report,
        references=references,
    )
