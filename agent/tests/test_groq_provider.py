from unittest.mock import patch

import pytest
from langchain_groq import ChatGroq
from pydantic import SecretStr

from tractian_agent.groq_provider import GroqModelProvider
from tractian_agent.model_provider import ModelConfig
from tractian_agent.planner import PlannerTerminalDecision


def test_groq_provider_masks_a_direct_string_key_in_its_state():
    secret = "direct-test-groq-secret"

    provider = GroqModelProvider(api_key=secret)

    assert secret not in repr(provider)
    assert secret not in repr(vars(provider))
    assert isinstance(vars(provider)["_api_key"], SecretStr)


@pytest.mark.parametrize("raw_key", ["", "   ", "\t"])
def test_groq_provider_rejects_blank_secret_str_without_leaking(raw_key):
    with pytest.raises(ValueError) as exc_info:
        GroqModelProvider(api_key=SecretStr(raw_key))

    assert str(exc_info.value) == "GROQ_API_KEY must be set and non-empty"
    if raw_key:
        assert raw_key not in str(exc_info.value)


@pytest.mark.parametrize(
    "environment",
    [{}, {"GROQ_API_KEY": ""}, {"GROQ_API_KEY": "   "}],
)
def test_groq_provider_from_env_rejects_missing_or_empty_key(environment):
    with pytest.raises(ValueError) as exc_info:
        GroqModelProvider.from_env(environment)

    assert str(exc_info.value) == "GROQ_API_KEY must be set and non-empty"


def test_groq_provider_from_env_keeps_secret_out_of_public_state():
    secret = "test-groq-secret-value"
    provider = GroqModelProvider.from_env({"GROQ_API_KEY": secret})
    config = ModelConfig(
        model_id="provider/test-model",
        temperature=0.0,
        timeout_seconds=10.0,
        max_output_tokens=512,
    )

    model = provider.create_chat_model(config)

    public_representations = (
        repr(provider),
        repr(vars(provider)),
        repr(config),
        config.model_dump_json(),
        repr(model),
    )
    assert all(secret not in value for value in public_representations)
    assert model.groq_api_key == SecretStr(secret)


def test_groq_provider_maps_common_config_without_network():
    provider = GroqModelProvider(api_key=SecretStr("test-groq-key"))
    config = ModelConfig(
        model_id="provider/test-model",
        temperature=0.0,
        timeout_seconds=10.0,
        max_output_tokens=512,
    )

    model = provider.create_chat_model(config)

    assert isinstance(model, ChatGroq)
    assert model.model_name == "provider/test-model"
    assert model.temperature == pytest.approx(0.0, abs=1e-8)
    assert model.request_timeout == 10.0
    assert model.max_tokens == 512
    assert model.max_retries == 0
    assert model.groq_api_key == SecretStr("test-groq-key")
    assert "test-groq-key" not in repr(model)


def test_groq_provider_uses_strict_json_schema_for_public_structured_output():
    provider = GroqModelProvider(api_key=SecretStr("test-groq-key"))
    model = provider.create_chat_model(
        ModelConfig(
            model_id="openai/gpt-oss-20b",
            temperature=0.0,
            timeout_seconds=10.0,
            max_output_tokens=512,
        )
    )

    with patch.object(ChatGroq, "with_structured_output", return_value=object()) as method:
        model.with_structured_output(PlannerTerminalDecision, include_raw=False)

    method.assert_called_once_with(
        PlannerTerminalDecision,
        method="json_schema",
        strict=True,
        include_raw=False,
    )
