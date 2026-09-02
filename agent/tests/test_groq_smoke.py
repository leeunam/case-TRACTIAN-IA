from __future__ import annotations

from io import StringIO
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from tractian_agent.groq_smoke import SmokeTerminalDecision, run_smoke
from tractian_agent.model_provider import ModelConfig


class _FakeSmokeModel:
    def bind_tools(self, tools: object) -> RunnableLambda:
        async def select(_: object) -> AIMessage:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "validar_atendimento_em_portugues",
                        "args": {
                            "idioma": "pt-BR",
                            "objetivo": "Continuar o atendimento em português",
                        },
                        "id": "provider-only-id",
                        "type": "tool_call",
                    }
                ],
            )

        return RunnableLambda(select)

    def with_structured_output(self, schema: object, **kwargs: Any) -> RunnableLambda:
        assert schema is SmokeTerminalDecision
        assert kwargs == {"include_raw": False}

        async def finalize(_: object) -> SmokeTerminalDecision:
            return SmokeTerminalDecision(status="concluido")

        return RunnableLambda(finalize)


class _FakeSmokeProvider:
    constructed = False
    received_configs: list[ModelConfig] = []

    @classmethod
    def from_env(cls, environment: object) -> "_FakeSmokeProvider":
        cls.constructed = True
        return cls()

    def create_chat_model(self, config: ModelConfig) -> _FakeSmokeModel:
        self.received_configs.append(config)
        return _FakeSmokeModel()


def test_groq_smoke_skips_without_key_without_constructing_provider():
    output = StringIO()

    class ForbiddenProvider:
        @classmethod
        def from_env(cls, environment: object):
            raise AssertionError("skip não pode construir provider ou acessar rede")

    exit_code = run_smoke(
        environment={},
        output=output,
        provider_factory=ForbiddenProvider,  # type: ignore[arg-type]
    )

    assert exit_code == 0
    assert output.getvalue() == "status=skipped reason=missing_groq_api_key\n"


def test_groq_smoke_reports_only_safe_aggregate_metrics_with_fake_provider():
    _FakeSmokeProvider.constructed = False
    _FakeSmokeProvider.received_configs = []
    output = StringIO()

    exit_code = run_smoke(
        environment={"GROQ_API_KEY": "not-printed-test-key"},
        output=output,
        provider_factory=_FakeSmokeProvider,  # type: ignore[arg-type]
    )

    lines = output.getvalue().splitlines()
    assert exit_code == 0
    assert _FakeSmokeProvider.constructed is True
    assert [config.model_id for config in _FakeSmokeProvider.received_configs] == [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]
    assert len(lines) == 2
    assert all("status=passed" in line and "calls=2" in line for line in lines)
    assert all("not-printed-test-key" not in line for line in lines)
    assert all("provider-only-id" not in line for line in lines)
    assert all("Continuar o atendimento" not in line for line in lines)
