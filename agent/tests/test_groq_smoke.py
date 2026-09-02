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
                            "objetivo": "continuar o atendimento industrial em português",
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
            return SmokeTerminalDecision(status="concluído")

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
    assert all(
        "status=passed" in line
        and "calls=2" in line
        and "runs=1" in line
        and "stable=not_measured" in line
        for line in lines
    )
    assert all("not-printed-test-key" not in line for line in lines)
    assert all("provider-only-id" not in line for line in lines)
    assert all("Continuar o atendimento" not in line for line in lines)


class _EnglishSmokeModel(_FakeSmokeModel):
    def bind_tools(self, tools: object) -> RunnableLambda:
        async def select(_: object) -> AIMessage:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "validar_atendimento_em_portugues",
                        "args": {
                            "idioma": "pt-BR",
                            "objetivo": "continue industrial support in English",
                        },
                        "id": "provider-only-id",
                        "type": "tool_call",
                    }
                ],
            )

        return RunnableLambda(select)


class _EnglishSmokeProvider(_FakeSmokeProvider):
    def create_chat_model(self, config: ModelConfig) -> _EnglishSmokeModel:
        self.received_configs.append(config)
        return _EnglishSmokeModel()


def test_groq_smoke_rejects_english_tool_argument_without_exposing_it():
    output = StringIO()

    exit_code = run_smoke(
        environment={"GROQ_API_KEY": "not-printed-test-key"},
        output=output,
        provider_factory=_EnglishSmokeProvider,  # type: ignore[arg-type]
    )

    lines = output.getvalue().splitlines()
    assert exit_code == 1
    assert len(lines) == 2
    assert all("status=failed" in line and "portuguese=false" in line for line in lines)
    assert "continue industrial support in English" not in output.getvalue()


class _TerminalFailureSmokeModel(_FakeSmokeModel):
    def with_structured_output(self, schema: object, **kwargs: Any) -> RunnableLambda:
        raise RuntimeError("SENTINEL_SECRET raw response provider-only-id")


class _TerminalFailureSmokeProvider(_FakeSmokeProvider):
    def create_chat_model(self, config: ModelConfig) -> _TerminalFailureSmokeModel:
        self.received_configs.append(config)
        return _TerminalFailureSmokeModel()


def test_groq_smoke_preserves_safe_selection_progress_when_finalization_fails():
    _TerminalFailureSmokeProvider.received_configs = []
    output = StringIO()

    exit_code = run_smoke(
        environment={"GROQ_API_KEY": "not-printed-test-key"},
        output=output,
        provider_factory=_TerminalFailureSmokeProvider,  # type: ignore[arg-type]
    )

    lines = output.getvalue().splitlines()
    assert exit_code == 1
    assert len(lines) == 2
    assert all(
        "status=failed" in line
        and "portuguese=true" in line
        and "tool=true" in line
        and "arguments=true" in line
        and "pydantic=false" in line
        and "calls=1" in line
        and "runs=1" in line
        and "stable=not_measured" in line
        for line in lines
    )
    assert "SENTINEL_SECRET" not in output.getvalue()
    assert "raw response" not in output.getvalue()
    assert "provider-only-id" not in output.getvalue()


class _SequencedSmokeProvider(_FakeSmokeProvider):
    def __init__(self, *, diverges: bool) -> None:
        self._diverges = diverges
        self._calls_by_model: dict[str, int] = {}

    @classmethod
    def from_env(cls, environment: object) -> "_SequencedSmokeProvider":
        return cls(diverges=False)

    def create_chat_model(self, config: ModelConfig) -> _FakeSmokeModel:
        run_index = self._calls_by_model.get(config.model_id, 0)
        self._calls_by_model[config.model_id] = run_index + 1
        if self._diverges and run_index == 1:
            return _EnglishSmokeModel()
        return _FakeSmokeModel()


def test_groq_smoke_marks_stability_true_only_after_matching_repetitions():
    output = StringIO()

    exit_code = run_smoke(
        environment={"GROQ_API_KEY": "test-key", "GROQ_SMOKE_RUNS": "2"},
        output=output,
        provider_factory=_SequencedSmokeProvider,  # type: ignore[arg-type]
    )

    assert exit_code == 0
    assert all(
        "status=passed" in line and "runs=2" in line and "stable=true" in line
        for line in output.getvalue().splitlines()
    )


def test_groq_smoke_marks_divergent_repetitions_unstable_without_text():
    output = StringIO()

    class DivergentProvider(_SequencedSmokeProvider):
        @classmethod
        def from_env(cls, environment: object) -> "DivergentProvider":
            return cls(diverges=True)

    exit_code = run_smoke(
        environment={"GROQ_API_KEY": "test-key", "GROQ_SMOKE_RUNS": "2"},
        output=output,
        provider_factory=DivergentProvider,  # type: ignore[arg-type]
    )

    assert exit_code == 1
    assert all(
        "status=failed" in line and "runs=2" in line and "stable=false" in line
        for line in output.getvalue().splitlines()
    )
    assert "continue industrial support in English" not in output.getvalue()


def test_groq_smoke_sanitizes_provider_initialization_failure(capsys):
    output = StringIO()

    class FailingProvider:
        @classmethod
        def from_env(cls, environment: object):
            raise RuntimeError("SENTINEL_SECRET raw response and traceback")

    exit_code = run_smoke(
        environment={"GROQ_API_KEY": "test-key"},
        output=output,
        provider_factory=FailingProvider,  # type: ignore[arg-type]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == ""
    lines = output.getvalue().splitlines()
    assert len(lines) == 2
    assert all(
        "status=failed" in line
        and "portuguese=false" in line
        and "tool=false" in line
        and "arguments=false" in line
        and "pydantic=false" in line
        and "calls=0" in line
        for line in lines
    )
    assert "SENTINEL_SECRET" not in output.getvalue()
    assert "traceback" not in output.getvalue()


class _FirstCallFailureSmokeModel:
    def bind_tools(self, tools: object) -> RunnableLambda:
        async def fail(_: object) -> AIMessage:
            raise RuntimeError("SENTINEL_SECRET raw response")

        return RunnableLambda(fail)


class _FirstCallFailureSmokeProvider(_FakeSmokeProvider):
    def create_chat_model(self, config: ModelConfig) -> _FirstCallFailureSmokeModel:
        self.received_configs.append(config)
        return _FirstCallFailureSmokeModel()


def test_groq_smoke_reports_no_progress_when_the_first_call_fails():
    _FirstCallFailureSmokeProvider.received_configs = []
    output = StringIO()

    exit_code = run_smoke(
        environment={"GROQ_API_KEY": "not-printed-test-key"},
        output=output,
        provider_factory=_FirstCallFailureSmokeProvider,  # type: ignore[arg-type]
    )

    lines = output.getvalue().splitlines()
    assert exit_code == 1
    assert len(lines) == 2
    assert all(
        "status=failed" in line
        and "portuguese=false" in line
        and "tool=false" in line
        and "arguments=false" in line
        and "pydantic=false" in line
        and "calls=0" in line
        for line in lines
    )
    assert "SENTINEL_SECRET" not in output.getvalue()
    assert "raw response" not in output.getvalue()
