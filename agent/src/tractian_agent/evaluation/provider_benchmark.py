"""Benchmark versionável dos papéis planner e writer entre providers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Final, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import ConfigDict, Field

from tractian_agent.contracts import StrictModel
from tractian_agent.evaluation.contracts import EvaluationModel
from tractian_agent.model_provider import ModelConfig, ModelProvider
from tractian_agent.planner import PlannerTerminalDecision
from tractian_agent.state import AgentDecision, WriterDraft, WriterNextStep


PROVIDER_BENCHMARK_VERSION: Final = "provider-benchmark-v1"
_PORTUGUESE_OBJECTIVE: Final = "continuar o atendimento industrial em português"


class _ProbeToolArguments(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    idioma: Literal["pt-BR"]
    objetivo: Literal[_PORTUGUESE_OBJECTIVE]


def _validate_portuguese(idioma: Literal["pt-BR"], objetivo: str) -> str:
    return objetivo


_PORTUGUESE_TOOL: Final = StructuredTool.from_function(
    func=_validate_portuguese,
    name="validar_atendimento_em_portugues",
    args_schema=_ProbeToolArguments,
    description="Confirma por tool calling que o atendimento continuará em português.",
)


class WriterProbeOutput(EvaluationModel):
    """Envelope sintético que testa idioma e o contrato real do writer."""

    idioma: Literal["pt-BR"]
    draft: WriterDraft


class ProviderBenchmarkSpec(EvaluationModel):
    provider: str = Field(min_length=1, pattern=r"^\S+$")
    planner: ModelConfig
    writer: ModelConfig
    repetitions: int = Field(ge=1, le=10, strict=True)
    context_characters: int = Field(ge=1, le=48_000, strict=True)
    input_cost_per_million_tokens: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    output_cost_per_million_tokens: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )


class ProviderRoleReport(EvaluationModel):
    role: Literal["planner", "writer"]
    model_id: str = Field(min_length=1, pattern=r"^\S+$")
    passed: bool
    portuguese: bool
    tool_calling: Literal["passed", "failed", "not_applicable"]
    structured_output: bool
    stability: Literal["stable", "unstable", "not_measured"]
    runs: int = Field(ge=1, strict=True)
    successful_calls: int = Field(ge=0, strict=True)
    context_characters: int = Field(ge=1, strict=True)
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    input_tokens: int | None = Field(default=None, ge=0, strict=True)
    output_tokens: int | None = Field(default=None, ge=0, strict=True)
    estimated_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class ProviderBenchmarkReport(EvaluationModel):
    version: Literal["provider-benchmark-v1"] = PROVIDER_BENCHMARK_VERSION
    provider: str = Field(min_length=1, pattern=r"^\S+$")
    roles: tuple[ProviderRoleReport, ProviderRoleReport]


class ProviderComparisonReport(EvaluationModel):
    version: Literal["provider-comparison-v1"] = "provider-comparison-v1"
    providers: tuple[ProviderBenchmarkReport, ProviderBenchmarkReport]
    comparable: bool
    recommended_provider: str | None = Field(default=None, pattern=r"^\S+$")
    rationale: str = Field(min_length=1, pattern=r"\S")


@dataclass(frozen=True)
class _ProbeSample:
    passed: bool
    portuguese: bool
    tool_calling: bool | None
    structured: bool
    successful_calls: int
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    signature: str | None


def _context_payload(characters: int) -> str:
    token = "contexto industrial seguro em português. "
    return (token * ((characters // len(token)) + 1))[:characters]


def _usage(message: object) -> tuple[int, int] | None:
    if not isinstance(message, AIMessage) or message.usage_metadata is None:
        return None
    return (
        message.usage_metadata["input_tokens"],
        message.usage_metadata["output_tokens"],
    )


def _structured_result(value: object) -> tuple[object, object] | None:
    if not isinstance(value, dict):
        return None
    if value.get("parsing_error") is not None or value.get("parsed") is None:
        return None
    return value["raw"], value["parsed"]


async def _planner_sample(model: object, context: str) -> _ProbeSample:
    started = perf_counter()
    calls = 0
    usages: list[tuple[int, int]] = []
    tool_ok = portuguese = structured = False
    signature: str | None = None
    try:
        selection = await model.bind_tools((_PORTUGUESE_TOOL,)).ainvoke(
            [
                SystemMessage(content=context),
                HumanMessage(
                    content="Escolha a tool disponível e mantenha o atendimento em português."
                ),
            ]
        )
        calls += 1
        usage = _usage(selection)
        if usage is not None:
            usages.append(usage)
        if isinstance(selection, AIMessage) and len(selection.tool_calls) == 1:
            call = selection.tool_calls[0]
            tool_ok = call.get("name") == _PORTUGUESE_TOOL.name
            if tool_ok:
                arguments = _ProbeToolArguments.model_validate(call.get("args"))
                portuguese = arguments.idioma == "pt-BR"
        terminal_wire = await model.with_structured_output(
            PlannerTerminalDecision,
            include_raw=True,
        ).ainvoke(
            [
                HumanMessage(
                    content=(
                        "Finalize com decision=guide, stop_reason=sufficient_evidence "
                        "e missing_information=null."
                    )
                )
            ]
        )
        calls += 1
        parts = _structured_result(terminal_wire)
        if parts is not None:
            raw, parsed = parts
            usage = _usage(raw)
            if usage is not None:
                usages.append(usage)
            terminal = PlannerTerminalDecision.model_validate(parsed)
            structured = (
                terminal.decision.value == "guide"
                and terminal.stop_reason.value == "sufficient_evidence"
                and terminal.missing_information is None
            )
            signature = (
                f"{arguments.model_dump_json()}|{terminal.model_dump_json()}"
                if tool_ok
                else None
            )
    except Exception:
        pass
    usage_complete = len(usages) == calls
    return _ProbeSample(
        passed=tool_ok and portuguese and structured,
        portuguese=portuguese,
        tool_calling=tool_ok,
        structured=structured,
        successful_calls=calls,
        latency_ms=(perf_counter() - started) * 1_000,
        input_tokens=sum(item[0] for item in usages) if usage_complete else None,
        output_tokens=sum(item[1] for item in usages) if usage_complete else None,
        signature=signature,
    )


async def _writer_sample(model: object, context: str) -> _ProbeSample:
    started = perf_counter()
    calls = 0
    usages: list[tuple[int, int]] = []
    portuguese = structured = False
    signature: str | None = None
    try:
        wire = await model.with_structured_output(
            WriterProbeOutput,
            include_raw=True,
        ).ainvoke(
            [
                SystemMessage(content=context),
                HumanMessage(
                    content=(
                        "Em português, devolva idioma=pt-BR e um draft GUIDE sem "
                        "evidências, sem limitações e com next_step=monitor."
                    )
                ),
            ]
        )
        calls += 1
        parts = _structured_result(wire)
        if parts is not None:
            raw, parsed = parts
            usage = _usage(raw)
            if usage is not None:
                usages.append(usage)
            output = WriterProbeOutput.model_validate(parsed)
            portuguese = output.idioma == "pt-BR"
            structured = output.draft == WriterDraft(
                decision=AgentDecision.GUIDE,
                evidence_ids=(),
                limitation_refs=(),
                next_step=WriterNextStep.MONITOR,
            )
            signature = output.model_dump_json()
    except Exception:
        pass
    usage_complete = len(usages) == calls
    return _ProbeSample(
        passed=portuguese and structured,
        portuguese=portuguese,
        tool_calling=None,
        structured=structured,
        successful_calls=calls,
        latency_ms=(perf_counter() - started) * 1_000,
        input_tokens=sum(item[0] for item in usages) if usage_complete else None,
        output_tokens=sum(item[1] for item in usages) if usage_complete else None,
        signature=signature,
    )


def _aggregate_role(
    *,
    role: Literal["planner", "writer"],
    model_id: str,
    samples: tuple[_ProbeSample, ...],
    spec: ProviderBenchmarkSpec,
) -> ProviderRoleReport:
    passed = all(sample.passed for sample in samples)
    if len(samples) == 1:
        stability: Literal["stable", "unstable", "not_measured"] = "not_measured"
    elif passed and len({sample.signature for sample in samples}) == 1:
        stability = "stable"
    else:
        stability = "unstable"
    input_tokens = (
        sum(
            sample.input_tokens for sample in samples if sample.input_tokens is not None
        )
        if all(sample.input_tokens is not None for sample in samples)
        else None
    )
    output_tokens = (
        sum(
            sample.output_tokens
            for sample in samples
            if sample.output_tokens is not None
        )
        if all(sample.output_tokens is not None for sample in samples)
        else None
    )
    cost = None
    if (
        input_tokens is not None
        and output_tokens is not None
        and spec.input_cost_per_million_tokens is not None
        and spec.output_cost_per_million_tokens is not None
    ):
        cost = (
            input_tokens * spec.input_cost_per_million_tokens
            + output_tokens * spec.output_cost_per_million_tokens
        ) / 1_000_000
    return ProviderRoleReport(
        role=role,
        model_id=model_id,
        passed=passed,
        portuguese=all(sample.portuguese for sample in samples),
        tool_calling=(
            "not_applicable"
            if role == "writer"
            else "passed"
            if all(sample.tool_calling for sample in samples)
            else "failed"
        ),
        structured_output=all(sample.structured for sample in samples),
        stability=stability,
        runs=len(samples),
        successful_calls=sum(sample.successful_calls for sample in samples),
        context_characters=spec.context_characters,
        latency_ms=sum(sample.latency_ms for sample in samples),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
    )


async def run_provider_benchmark(
    provider: ModelProvider,
    spec: ProviderBenchmarkSpec,
) -> ProviderBenchmarkReport:
    """Executa os mesmos probes de planner e writer no provider escolhido."""

    context = _context_payload(spec.context_characters)
    planner_model = provider.create_chat_model(spec.planner)
    writer_model = provider.create_chat_model(spec.writer)
    planner_samples = tuple(
        [await _planner_sample(planner_model, context) for _ in range(spec.repetitions)]
    )
    writer_samples = tuple(
        [await _writer_sample(writer_model, context) for _ in range(spec.repetitions)]
    )
    return ProviderBenchmarkReport(
        provider=spec.provider,
        roles=(
            _aggregate_role(
                role="planner",
                model_id=spec.planner.model_id,
                samples=planner_samples,
                spec=spec,
            ),
            _aggregate_role(
                role="writer",
                model_id=spec.writer.model_id,
                samples=writer_samples,
                spec=spec,
            ),
        ),
    )


def compare_provider_reports(
    reports: tuple[ProviderBenchmarkReport, ProviderBenchmarkReport],
) -> ProviderComparisonReport:
    """Recomenda somente quando modelos e condições realmente coincidem."""

    conditions = {
        tuple(
            (
                role.role,
                role.model_id,
                role.runs,
                role.context_characters,
            )
            for role in report.roles
        )
        for report in reports
    }
    comparable = len(conditions) == 1
    if not comparable:
        return ProviderComparisonReport(
            providers=reports,
            comparable=False,
            rationale=(
                "Os providers foram executados com modelos, repetições ou contexto "
                "diferentes; não há recomendação causal de provider."
            ),
        )
    eligible = tuple(
        report
        for report in reports
        if all(role.passed and role.stability != "unstable" for role in report.roles)
    )
    if len(eligible) == 1:
        selected = eligible[0]
        return ProviderComparisonReport(
            providers=reports,
            comparable=True,
            recommended_provider=selected.provider,
            rationale=(
                f"{selected.provider} foi o único provider que passou os contratos "
                "de planner e writer nas condições registradas."
            ),
        )
    if len(eligible) != 2:
        return ProviderComparisonReport(
            providers=reports,
            comparable=True,
            rationale="Nenhum provider passou todos os contratos; não há recomendação.",
        )
    latencies = {
        report.provider: sum(role.latency_ms for role in report.roles)
        for report in eligible
    }
    fastest = min(eligible, key=lambda report: latencies[report.provider])
    slowest = max(eligible, key=lambda report: latencies[report.provider])
    if latencies[fastest.provider] <= latencies[slowest.provider] * 0.95:
        return ProviderComparisonReport(
            providers=reports,
            comparable=True,
            recommended_provider=fastest.provider,
            rationale=(
                f"{fastest.provider} preservou os contratos e apresentou menor "
                "latência observada no benchmark versionado."
            ),
        )
    costs = {
        report.provider: sum(role.estimated_cost_usd for role in report.roles)
        for report in eligible
        if all(role.estimated_cost_usd is not None for role in report.roles)
    }
    if len(costs) == 2 and len(set(costs.values())) > 1:
        cheapest = min(eligible, key=lambda report: costs[report.provider])
        return ProviderComparisonReport(
            providers=reports,
            comparable=True,
            recommended_provider=cheapest.provider,
            rationale=(
                f"{cheapest.provider} preservou os contratos e teve menor custo "
                "estimado sob as tarifas registradas."
            ),
        )
    return ProviderComparisonReport(
        providers=reports,
        comparable=True,
        rationale=(
            "Os dois providers preservaram os contratos sem diferença material de "
            "latência e sem custo comparável; não há preferência fundamentada."
        ),
    )
