"""Smoke opt-in e sem persistência para comparar os dois modelos Groq."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from time import perf_counter
from typing import Final, Literal, TextIO

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import ConfigDict

from tractian_agent.contracts import StrictModel
from tractian_agent.groq_provider import GroqModelProvider
from tractian_agent.model_provider import ModelConfig, ModelProvider


SMOKE_MODEL_IDS: Final = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
)
_SMOKE_CONFIG: Final = {
    "temperature": 0.0,
    "timeout_seconds": 30.0,
    "max_output_tokens": 128,
}


class SmokeToolArguments(StrictModel):
    """Argumentos sintéticos que comprovam tool calling em português."""

    model_config = ConfigDict(extra="forbid", strict=True)

    idioma: Literal["pt-BR"]
    objetivo: str


class SmokeTerminalDecision(StrictModel):
    """Finalização Pydantic independente da escolha da tool."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: str


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
    stable: bool,
) -> str:
    return (
        f"model={model_id} status={status} portuguese={str(portuguese).lower()} "
        f"tool={str(tool).lower()} arguments={str(arguments).lower()} "
        f"pydantic={str(pydantic).lower()} calls={calls} "
        f"latency_ms={latency_ms} stable={str(stable).lower()}"
    )


async def _run_model(provider: ModelProvider, model_id: str) -> str:
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
            SmokeTerminalDecision,
            include_raw=False,
        ).ainvoke(
            [
                HumanMessage(
                    content=(
                        "Finalize em português com o schema solicitado, sem "
                        "explicações adicionais."
                    )
                )
            ]
        )
        successful_calls += 1
        terminal_valid = (
            SmokeTerminalDecision.model_validate(terminal).status.strip() != ""
        )
        passed = portuguese and tool_selected and arguments_valid and terminal_valid
        status = "passed" if passed else "failed"
    except Exception:
        status = "failed"
        successful_calls = 0
        portuguese = tool_selected = arguments_valid = terminal_valid = False
    latency_ms = int((perf_counter() - started_at) * 1000)
    return _safe_result_line(
        model_id=model_id,
        status=status,
        portuguese=portuguese,
        tool=tool_selected,
        arguments=arguments_valid,
        pydantic=terminal_valid,
        calls=successful_calls,
        latency_ms=latency_ms,
        stable=(status == "passed"),
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

    provider = provider_factory.from_env(environment)
    lines = [asyncio.run(_run_model(provider, model_id)) for model_id in SMOKE_MODEL_IDS]
    for line in lines:
        print(line, file=output)
    return 0 if all("status=passed" in line for line in lines) else 1


def main() -> int:
    import os
    import sys

    return run_smoke(environment=os.environ, output=sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
