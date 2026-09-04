from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from tractian_agent.model_provider import ModelConfig
from tractian_agent.nvidia_nim_provider import NvidiaNimModelProvider
from tractian_agent.planner import PlannerTerminalDecision


def _config() -> ModelConfig:
    return ModelConfig(
        model_id="openai/gpt-oss-20b",
        temperature=0.0,
        timeout_seconds=30.0,
        max_output_tokens=512,
    )


def test_nim_remote_endpoint_requires_api_key() -> None:
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        NvidiaNimModelProvider.from_env(
            {"NVIDIA_NIM_BASE_URL": "https://integrate.api.nvidia.com/v1"}
        )


def test_nim_defaults_to_hosted_endpoint_when_api_key_is_present() -> None:
    provider = NvidiaNimModelProvider.from_env({"NVIDIA_API_KEY": "nvapi-secret-value"})

    with patch(
        "tractian_agent.nvidia_nim_provider._StrictJsonSchemaChatOpenAI"
    ) as chat_model:
        provider.create_chat_model(_config())

    assert chat_model.call_args.kwargs["base_url"] == (
        "https://integrate.api.nvidia.com/v1"
    )


def test_nim_local_endpoint_does_not_require_external_credentials() -> None:
    provider = NvidiaNimModelProvider.from_env(
        {"NVIDIA_NIM_BASE_URL": "http://127.0.0.1:8000/v1"}
    )

    with patch(
        "tractian_agent.nvidia_nim_provider._StrictJsonSchemaChatOpenAI"
    ) as chat_model:
        provider.create_chat_model(_config())

    kwargs = chat_model.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-oss-20b"
    assert kwargs["temperature"] == 0.0
    assert kwargs["timeout"] == 30.0
    assert kwargs["max_tokens"] == 512
    assert kwargs["max_retries"] == 0
    assert kwargs["base_url"] == "http://127.0.0.1:8000/v1"
    assert isinstance(kwargs["api_key"], SecretStr)
    assert kwargs["api_key"].get_secret_value() == "local-nim-no-auth"


def test_nim_remote_provider_passes_secret_without_exposing_it() -> None:
    provider = NvidiaNimModelProvider.from_env(
        {
            "NVIDIA_NIM_BASE_URL": "https://integrate.api.nvidia.com/v1",
            "NVIDIA_API_KEY": "nvapi-secret-value",
        }
    )

    with patch(
        "tractian_agent.nvidia_nim_provider._StrictJsonSchemaChatOpenAI"
    ) as chat_model:
        provider.create_chat_model(_config())

    key = chat_model.call_args.kwargs["api_key"]
    assert isinstance(key, SecretStr)
    assert key.get_secret_value() == "nvapi-secret-value"
    assert "nvapi-secret-value" not in repr(provider)


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        "ftp://nim.example/v1",
        "https://user:password@nim.example/v1",
        "https://nim.example/v1?token=secret",
        "https://nim.example/v1#fragment",
    ],
)
def test_nim_rejects_unsafe_or_ambiguous_base_urls(base_url: str) -> None:
    with pytest.raises(ValueError, match="NVIDIA_NIM_BASE_URL"):
        NvidiaNimModelProvider(
            base_url=base_url,
            api_key=SecretStr("secret"),
        )


def test_nim_provider_uses_strict_json_schema_for_structured_output() -> None:
    model = NvidiaNimModelProvider(
        base_url="http://127.0.0.1:8000/v1"
    ).create_chat_model(_config())
    upstream = RunnableLambda(lambda _: {"raw": AIMessage(content="{}")})

    with patch.object(
        ChatOpenAI,
        "with_structured_output",
        return_value=upstream,
    ) as method:
        model.with_structured_output(PlannerTerminalDecision, include_raw=False)

    method.assert_called_once_with(
        PlannerTerminalDecision,
        method="json_schema",
        strict=True,
        include_raw=True,
    )
