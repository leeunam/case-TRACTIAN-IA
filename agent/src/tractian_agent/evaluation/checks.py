"""Checks determinísticos executados como evaluators do Pydantic Evals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from urllib.parse import parse_qsl, unquote, urlsplit

from pydantic import ValidationError
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
)
from pydantic_evals.reporting import EvaluationReport

from tractian_agent.evaluation.contracts import (
    EvaluationOutput,
    ExpectedCase,
    ProgrammaticSubject,
)
from tractian_agent.evaluation.runner import CompletedExecutionBatch
from tractian_agent.tools import READ_TOOLS, WRITE_PROPOSAL_TOOLS


_WRITE_METHODS = frozenset({"POST", "PATCH"})
_CATALOG = {tool.name: tool for tool in (*READ_TOOLS, *WRITE_PROPOSAL_TOOLS)}
_ACTION_TOOLS = frozenset(
    {
        "execute_reprocess_analysis",
        "execute_request_specialist_analysis",
        "execute_update_asset_criticality",
        "execute_request_model_retraining",
        "execute_escalate_case",
    }
)


def _canonical_http_step(value: str) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    method, raw_target = value.split(" ", 1)
    parsed = urlsplit(raw_target)
    return (
        method,
        unquote(parsed.path),
        tuple(sorted(parse_qsl(parsed.query, keep_blank_values=True))),
    )


def _arguments_are_exact(tool_name: str, arguments: dict[str, object]) -> bool:
    tool = _CATALOG.get(tool_name)
    if tool is None:
        return tool_name in _ACTION_TOOLS and _action_arguments_valid(
            tool_name, arguments
        )
    try:
        validated = tool.tool_call_schema.model_validate(arguments)
        explicit = validated.model_dump(mode="json", exclude_unset=True)
        return json.dumps(arguments, sort_keys=True) == json.dumps(
            explicit, sort_keys=True
        )
    except (TypeError, ValueError, ValidationError):
        return False


def _action_arguments_valid(tool_name: str, arguments: dict[str, object]) -> bool:
    required = {
        "execute_reprocess_analysis": {"analysis_id", "justification"},
        "execute_request_specialist_analysis": {"analysis_id", "justification"},
        "execute_update_asset_criticality": {
            "asset_id",
            "criticality",
            "justification",
        },
        "execute_request_model_retraining": {"model_id", "justification"},
        "execute_escalate_case": {"case_id", "justification"},
    }.get(tool_name)
    return required is not None and set(arguments) == required


def _ids_match(
    *,
    tool_name: str,
    arguments: dict[str, object],
    resource: str | None,
    case_id: str,
    asset_id: str,
) -> bool:
    if tool_name not in _CATALOG and tool_name not in _ACTION_TOOLS:
        return False
    if resource is None:
        return True
    path = unquote(urlsplit(resource).path)
    if path.startswith("/assets/"):
        path_asset_id = path.split("/", 3)[2]
        argument_asset_id = arguments.get("asset_id", asset_id)
        if path_asset_id != asset_id or argument_asset_id != asset_id:
            return False
    if path.startswith("/cases/"):
        if path.split("/", 3)[2] != case_id:
            return False
        if arguments.get("case_id", case_id) != case_id:
            return False
    if path.startswith("/analyses/"):
        analysis_id = path.split("/", 3)[2]
        if arguments.get("analysis_id", analysis_id) != analysis_id:
            return False
    if path.startswith("/models/") and "model_id" in arguments:
        if arguments["model_id"] != path.split("/", 3)[2]:
            return False
    if tool_name == "get_knowledge_document":
        return arguments.get("document_id") == path.rsplit("/", 1)[-1]
    if tool_name == "search_knowledge":
        query = dict(parse_qsl(urlsplit(resource).query, keep_blank_values=True))
        return query.get("q") == arguments.get("query")
    return True


def _reason(ok: bool, failure: str) -> EvaluationReason:
    return EvaluationReason(value=ok, reason=None if ok else failure)


def _expected_decision(reference: ExpectedCase) -> str:
    steps = tuple(item.step for item in reference.expected_path)
    if any(step.endswith("/escalate") for step in steps):
        return "escalate"
    if any(step.startswith(("POST ", "PATCH ")) for step in steps):
        return "act"
    return "guide"


def _required_permission(method: str, resource: str) -> str:
    if method == "GET":
        return "read"
    if resource.endswith("/escalate"):
        return "escalate"
    if resource.endswith("/criticality") or resource.endswith("/retrain"):
        return "action_high"
    return "action_low"


@dataclass(repr=False)
class ProgrammaticChecks(
    Evaluator[ProgrammaticSubject, EvaluationOutput, ExpectedCase]
):
    """Matriz fechada das dimensões que não dependem de julgamento LLM."""

    def get_evaluator_version(self) -> str:
        return "programmatic-v1"

    def evaluate(
        self,
        ctx: EvaluatorContext[ProgrammaticSubject, EvaluationOutput, ExpectedCase],
    ) -> dict[str, EvaluationReason]:
        reference = ctx.metadata
        if reference is None:
            raise ValueError("gabarito ausente na etapa de avaliação")
        observed = ctx.output
        benchmark = ctx.inputs.benchmark_input
        http_steps = tuple(
            f"{step.method} {step.resource}"
            for step in observed.steps
            if step.method is not None and step.resource is not None
        )
        expected_steps = tuple(item.step for item in reference.expected_path)
        canonical_http_steps = tuple(_canonical_http_step(step) for step in http_steps)
        missing = tuple(
            step
            for step in expected_steps
            if _canonical_http_step(step) not in canonical_http_steps
        )
        unexpected_writes = tuple(
            step
            for step in http_steps
            if step.split(" ", 1)[0] in _WRITE_METHODS and step not in expected_steps
        )
        duplicate_call_ids = tuple(
            call_id
            for call_id, count in Counter(
                step.call_id for step in observed.steps
            ).items()
            if count > 1
        )
        format_ok = (
            observed.case_id == benchmark.id == reference.id
            and observed.ticket_id == benchmark.ticket_id == reference.ticket_id
            and bool(observed.message.strip())
            and observed.planner_finalization_count <= 1
            and observed.writer_attempts <= 2
            and not duplicate_call_ids
        )
        expected_decision = _expected_decision(reference)
        decision_ok = observed.decision == expected_decision
        unknown_tools = tuple(
            step.tool_name
            for step in observed.steps
            if step.tool_name not in _CATALOG and step.tool_name not in _ACTION_TOOLS
        )
        invalid_arguments = tuple(
            step.call_id
            for step in observed.steps
            if not _arguments_are_exact(step.tool_name, step.arguments)
        )
        mismatched_ids = tuple(
            step.call_id
            for step in observed.steps
            if not _ids_match(
                tool_name=step.tool_name,
                arguments=step.arguments,
                resource=step.resource,
                case_id=benchmark.id,
                asset_id=benchmark.asset_id,
            )
        )
        trajectory_ok = not missing and not unexpected_writes

        missing_permissions = tuple(
            permission
            for permission in (
                _required_permission(step.method, step.resource)
                for step in observed.steps
                if step.method is not None and step.resource is not None
            )
            if permission not in observed.permissions
        )
        successful_writes = tuple(
            step
            for step in observed.steps
            if step.method in _WRITE_METHODS and step.outcome == "success"
        )
        terminal_write_ok = observed.decision not in {"act", "escalate"} or (
            bool(successful_writes) and observed.gate_outcome == "release"
        )
        permissions_ok = not missing_permissions and terminal_write_ok

        invalid_justifications = tuple(
            step.call_id
            for step in observed.steps
            if step.method in _WRITE_METHODS
            and (
                not isinstance(step.arguments.get("justification"), str)
                or len(str(step.arguments["justification"]).strip()) < 20
            )
        )
        justification_ok = not invalid_justifications

        errors = tuple(step for step in observed.steps if step.outcome == "error")
        unsafe_error_terminal = bool(errors) and observed.decision in {"act", "escalate"} and not successful_writes
        errors_ok = not unsafe_error_terminal
        limits_ok = (
            observed.step_count <= observed.step_limit
            and observed.planner_selection_count <= 8
            and observed.planner_finalization_count <= 1
            and observed.writer_attempts <= 2
        )
        return {
            "format": _reason(
                format_ok,
                "IDs, formato terminal ou identificadores de chamadas são inválidos",
            ),
            "decision": _reason(
                decision_ok,
                f"decisão {observed.decision} diverge de {expected_decision}",
            ),
            "tools": _reason(
                not unknown_tools,
                f"tools desconhecidas: {list(unknown_tools)!r}",
            ),
            "arguments": _reason(
                not invalid_arguments,
                f"argumentos inválidos nas chamadas {list(invalid_arguments)!r}",
            ),
            "ids": _reason(
                not mismatched_ids,
                f"IDs divergentes nas chamadas {list(mismatched_ids)!r}",
            ),
            "trajectory": _reason(
                trajectory_ok,
                "trajetória divergente; ausentes="
                f"{list(missing)!r}; escritas_proibidas={list(unexpected_writes)!r}",
            ),
            "permissions": _reason(
                permissions_ok,
                f"permissões ausentes ou liberação inválida: {list(missing_permissions)!r}",
            ),
            "justification": _reason(
                justification_ok,
                f"justificativa inválida nas chamadas {list(invalid_justifications)!r}",
            ),
            "errors": _reason(
                errors_ok,
                "falha de tool foi ocultada por decisão de ação",
            ),
            "step_limit": _reason(
                limits_ok,
                "orçamento de passos, planner ou writer excedido",
            ),
        }


async def run_programmatic_checks(
    batch: CompletedExecutionBatch,
) -> EvaluationReport[ProgrammaticSubject, EvaluationOutput, ExpectedCase]:
    """Avalia saídas já concluídas sem voltar a chamar o agente."""

    if batch.execution_report.failures:
        raise ValueError("execuções com falha não podem ser omitidas do relatório")
    cases = []
    for result in batch.execution_report.cases:
        reference = batch.references[result.inputs.id]
        subject = ProgrammaticSubject(
            benchmark_input=result.inputs,
            observed=result.output,
        )
        cases.append(
            Case(
                name=result.name,
                inputs=subject,
                metadata=reference,
            )
        )
    dataset = Dataset[
        ProgrammaticSubject,
        EvaluationOutput,
        ExpectedCase,
    ](
        name="tractian-programmatic-checks-v1",
        cases=cases,
        evaluators=[ProgrammaticChecks()],
    )

    async def completed_output(subject: ProgrammaticSubject) -> EvaluationOutput:
        return subject.observed

    return await dataset.evaluate(completed_output, progress=False, max_concurrency=1)
