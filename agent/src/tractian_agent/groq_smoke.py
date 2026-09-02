"""Smoke opt-in e sem persistência para comparar os dois modelos Groq."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Final, Literal, TextIO

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import ConfigDict

from tractian_agent.contracts import StrictModel
from tractian_agent.groq_provider import GroqModelProvider
from tractian_agent.model_provider import ModelConfig, ModelProvider
from tractian_agent.planner import (
    PlannerDecisionKind,
    PlannerStopReason,
    PlannerTerminalDecision,
)


SMOKE_MODEL_IDS: Final = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
)
_SMOKE_CONFIG: Final = {
    "temperature": 0.0,
    "timeout_seconds": 30.0,
    "max_output_tokens": 512,
}
_PORTUGUESE_OBJECTIVE: Final = "continuar o atendimento industrial em português"


class SmokeToolArguments(StrictModel):
    """Argumentos sintéticos que comprovam tool calling em português."""

    model_config = ConfigDict(extra="forbid", strict=True)

    idioma: Literal["pt-BR"]
    objetivo: Literal[_PORTUGUESE_OBJECTIVE]


def _validate_portuguese_tool(idioma: Literal["pt-BR"], objetivo: str) -> str:
    return objetivo


_PORTUGUESE_TOOL: Final = StructuredTool.from_function(
    func=_validate_portuguese_tool,
    name="validar_atendimento_em_portugues",
    args_schema=SmokeToolArguments,
    description=(
        "Confirma que o atendimento industrial deve continuar em português. "
        "Use o argumento objetivo com uma frase curta em português."
    ),
)


@dataclass(frozen=True)
class _SmokeResult:
    model_id: str
    passed: bool
    portuguese: bool
    tool: bool
    arguments: bool
    pydantic: bool
    calls: int
    latency_ms: int
    signature: tuple[object, ...] | None


def _safe_result_line(
    *,
    model_id: str,
    status: str,
    portuguese: bool,
    tool: bool,
    arguments: bool,
    pydantic: bool,
    calls: int,
    latency_ms: int,
    runs: int,
    stable: Literal["true", "false", "not_measured"],
) -> str:
    return (
        f"model={model_id} status={status} portuguese={str(portuguese).lower()} "
        f"tool={str(tool).lower()} arguments={str(arguments).lower()} "
        f"pydantic={str(pydantic).lower()} calls={calls} "
        f"latency_ms={latency_ms} runs={runs} stable={stable}"
    )


def _failed_result(
    model_id: str,
    *,
    portuguese: bool = False,
    tool: bool = False,
    arguments: bool = False,
    calls: int = 0,
    latency_ms: int = 0,
) -> _SmokeResult:
    return _SmokeResult(
        model_id=model_id,
        passed=False,
        portuguese=portuguese,
        tool=tool,
        arguments=arguments,
        pydantic=False,
        calls=calls,
        latency_ms=latency_ms,
        signature=None,
    )


async def _run_model(provider: ModelProvider, model_id: str) -> _SmokeResult:
    started_at = perf_counter()
    successful_calls = 0
    portuguese = tool_selected = arguments_valid = terminal_valid = False
    try:
        model = provider.create_chat_model(ModelConfig(model_id=model_id, **_SMOKE_CONFIG))
        selection = await model.bind_tools((_PORTUGUESE_TOOL,)).ainvoke(
            [
                SystemMessage(
                    content="Você valida contratos de atendimento industrial."
                ),
                HumanMessage(
                    content=(
                        "Em português, escolha a tool disponível e informe um "
                        "objetivo curto para continuar o atendimento."
                    )
                ),
            ]
        )
        successful_calls += 1
        if len(selection.tool_calls) == 1:
            selected = selection.tool_calls[0]
            tool_selected = selected.get("name") == _PORTUGUESE_TOOL.name
            if tool_selected:
                validated = SmokeToolArguments.model_validate(selected.get("args"))
                arguments_valid = bool(validated.objetivo.strip())
                portuguese = arguments_valid and validated.idioma == "pt-BR"

        terminal = await model.with_structured_output(
            PlannerTerminalDecision,
            include_raw=False,
        ).ainvoke(
            [
                HumanMessage(
                    content=(
                        "Finalize o contrato do planner com decision=guide, "
                        "stop_reason=sufficient_evidence e "
                        "missing_information=null, sem explicações adicionais."
                    )
                )
            ]
        )
        successful_calls += 1
        validated_terminal = PlannerTerminalDecision.model_validate(terminal)
        terminal_valid = (
            validated_terminal.decision is PlannerDecisionKind.GUIDE
            and validated_terminal.stop_reason
            is PlannerStopReason.SUFFICIENT_EVIDENCE
            and validated_terminal.missing_information is None
        )
        passed = portuguese and tool_selected and arguments_valid and terminal_valid
        signature = (
            selected.get("name") if tool_selected else None,
            validated.model_dump_json() if tool_selected else None,
            validated_terminal.model_dump_json(),
        )
    except Exception:
        return _failed_result(
            model_id,
            portuguese=portuguese,
            tool=tool_selected,
            arguments=arguments_valid,
            calls=successful_calls,
            latency_ms=int((perf_counter() - started_at) * 1000),
        )
    latency_ms = int((perf_counter() - started_at) * 1000)
    return _SmokeResult(
        model_id=model_id,
        passed=passed,
        portuguese=portuguese,
        tool=tool_selected,
        arguments=arguments_valid,
        pydantic=terminal_valid,
        calls=successful_calls,
        latency_ms=latency_ms,
        signature=signature if passed else None,
    )


def _configured_runs(environment: Mapping[str, str]) -> int | None:
    raw_runs = environment.get("GROQ_SMOKE_RUNS", "1")
    try:
        runs = int(raw_runs)
    except (TypeError, ValueError):
        return None
    return runs if runs >= 1 else None


def _aggregate_result(
    model_id: str,
    results: tuple[_SmokeResult, ...],
    *,
    runs: int,
) -> str:
    if not results:
        return _safe_result_line(
            model_id=model_id,
            status="failed",
            portuguese=False,
            tool=False,
            arguments=False,
            pydantic=False,
            calls=0,
            latency_ms=0,
            runs=runs,
            stable="not_measured",
        )
    contracts_passed = all(result.passed for result in results)
    stable: Literal["true", "false", "not_measured"]
    if runs == 1:
        stable = "not_measured"
    elif contracts_passed and len({result.signature for result in results}) == 1:
        stable = "true"
    else:
        stable = "false"
    return _safe_result_line(
        model_id=model_id,
        status="passed" if contracts_passed else "failed",
        portuguese=all(result.portuguese for result in results),
        tool=all(result.tool for result in results),
        arguments=all(result.arguments for result in results),
        pydantic=all(result.pydantic for result in results),
        calls=sum(result.calls for result in results),
        latency_ms=sum(result.latency_ms for result in results),
        runs=runs,
        stable=stable,
    )


def run_smoke(
    *,
    environment: Mapping[str, str],
    output: TextIO,
    provider_factory: type[GroqModelProvider] = GroqModelProvider,
) -> int:
    """Executa uma rodada por modelo, ou faz skip seguro se não houver chave."""
    api_key = environment.get("GROQ_API_KEY")
    if not isinstance(api_key, str) or not api_key.strip():
        print("status=skipped reason=missing_groq_api_key", file=output)
        return 0

    runs = _configured_runs(environment)
    if runs is None:
        lines = [
            _aggregate_result(model_id, (), runs=0)
            for model_id in SMOKE_MODEL_IDS
        ]
        for line in lines:
            print(line, file=output)
        return 1
    try:
        provider = provider_factory.from_env(environment)
    except Exception:
        lines = [
            _aggregate_result(model_id, (), runs=runs)
            for model_id in SMOKE_MODEL_IDS
        ]
        for line in lines:
            print(line, file=output)
        return 1
    lines = []
    for model_id in SMOKE_MODEL_IDS:
        results = tuple(
            asyncio.run(_run_model(provider, model_id)) for _ in range(runs)
        )
        lines.append(_aggregate_result(model_id, results, runs=runs))
    for line in lines:
        print(line, file=output)
    return 0 if all("status=passed" in line for line in lines) else 1


def main() -> int:
    import os
    import sys

    return run_smoke(environment=os.environ, output=sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
