import asyncio
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import PrivateAttr

from tractian_agent.evaluation.provider_benchmark import (
    ProviderBenchmarkSpec,
    ProviderBenchmarkReport,
    ProviderRoleReport,
    WriterProbeOutput,
    compare_provider_reports,
    run_provider_benchmark,
)
from tractian_agent.model_provider import ModelConfig
from tractian_agent.planner import (
    PlannerDecisionKind,
    PlannerStopReason,
    PlannerTerminalDecision,
)
from tractian_agent.state import AgentDecision, WriterDraft, WriterNextStep


class _ProbeModel(BaseChatModel):
    _messages: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "provider-probe-model"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise AssertionError("probe usa interfaces vinculadas")

    def bind_tools(self, *args: Any, **kwargs: Any) -> RunnableLambda:
        async def invoke(messages: list[BaseMessage]) -> AIMessage:
            self._messages.append(list(messages))
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "validar_atendimento_em_portugues",
                        "args": {
                            "idioma": "pt-BR",
                            "objetivo": "continuar o atendimento industrial em português",
                        },
                        "id": "probe_call",
                        "type": "tool_call",
                    }
                ],
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                },
            )

        return RunnableLambda(invoke)

    def with_structured_output(
        self,
        schema: object,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> RunnableLambda:
        assert include_raw is True

        async def invoke(messages: list[BaseMessage]) -> dict[str, object]:
            self._messages.append(list(messages))
            raw = AIMessage(
                content="{}",
                usage_metadata={
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "total_tokens": 60,
                },
            )
            if schema is PlannerTerminalDecision:
                parsed: object = PlannerTerminalDecision(
                    decision=PlannerDecisionKind.GUIDE,
                    stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
                    missing_information=None,
                )
            elif schema is WriterProbeOutput:
                parsed = WriterProbeOutput(
                    idioma="pt-BR",
                    draft=WriterDraft(
                        decision=AgentDecision.GUIDE,
                        evidence_ids=(),
                        limitation_refs=(),
                        next_step=WriterNextStep.MONITOR,
                    ),
                )
            else:
                raise AssertionError(f"schema inesperado: {schema}")
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(invoke)


class _ProbeProvider:
    def __init__(self) -> None:
        self.configs: list[ModelConfig] = []
        self.models: list[_ProbeModel] = []

    def create_chat_model(self, config: ModelConfig) -> BaseChatModel:
        self.configs.append(config)
        model = _ProbeModel()
        self.models.append(model)
        return model


def test_provider_benchmark_measures_planner_and_writer_separately() -> None:
    provider = _ProbeProvider()
    spec = ProviderBenchmarkSpec(
        provider="fake-nim",
        planner=ModelConfig(
            model_id="planner-model",
            temperature=0.0,
            timeout_seconds=30.0,
            max_output_tokens=512,
        ),
        writer=ModelConfig(
            model_id="writer-model",
            temperature=0.0,
            timeout_seconds=30.0,
            max_output_tokens=512,
        ),
        repetitions=2,
        context_characters=256,
        input_cost_per_million_tokens=1.0,
        output_cost_per_million_tokens=2.0,
    )

    report = asyncio.run(run_provider_benchmark(provider, spec))

    assert [role.role for role in report.roles] == ["planner", "writer"]
    planner, writer = report.roles
    assert planner.tool_calling == "passed"
    assert writer.tool_calling == "not_applicable"
    assert planner.structured_output is True
    assert writer.structured_output is True
    assert planner.portuguese is True
    assert writer.portuguese is True
    assert planner.stability == "stable"
    assert writer.stability == "stable"
    assert planner.runs == writer.runs == 2
    assert planner.input_tokens == 300
    assert planner.output_tokens == 60
    assert writer.input_tokens == 100
    assert writer.output_tokens == 20
    assert planner.estimated_cost_usd == 0.00042
    assert writer.estimated_cost_usd == 0.00014
    assert [config.model_id for config in provider.configs] == [
        "planner-model",
        "writer-model",
    ]
    assert all(
        any(len(str(message.content)) >= 256 for message in model._messages[0])
        for model in provider.models
    )


def test_provider_recommendation_requires_comparable_real_conditions() -> None:
    def provider_report(name: str, latency: float) -> ProviderBenchmarkReport:
        return ProviderBenchmarkReport(
            provider=name,
            roles=tuple(
                ProviderRoleReport(
                    role=role,
                    model_id="openai/gpt-oss-20b",
                    passed=True,
                    portuguese=True,
                    tool_calling=("passed" if role == "planner" else "not_applicable"),
                    structured_output=True,
                    stability="stable",
                    runs=2,
                    successful_calls=4 if role == "planner" else 2,
                    context_characters=8000,
                    latency_ms=latency,
                )
                for role in ("planner", "writer")
            ),
        )

    comparison = compare_provider_reports(
        (provider_report("groq", 1000), provider_report("nvidia-nim", 700))
    )

    assert comparison.comparable is True
    assert comparison.recommended_provider == "nvidia-nim"
    assert "latência observada" in comparison.rationale
