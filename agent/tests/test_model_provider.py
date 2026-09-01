import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import ValidationError

from tractian_agent.model_provider import (
    INITIAL_PLANNER_MODEL_CONFIG,
    ModelConfig,
    ModelProvider,
)


class FakeModelProvider:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model
        self.received_config: ModelConfig | None = None

    def create_chat_model(self, config: ModelConfig) -> BaseChatModel:
        self.received_config = config
        return self.model


def _create_model_through_contract(
    provider: ModelProvider,
    config: ModelConfig,
) -> BaseChatModel:
    return provider.create_chat_model(config)


def test_model_config_accepts_explicit_valid_values():
    config = ModelConfig(
        model_id="provider/test-model",
        temperature=0.0,
        timeout_seconds=10.0,
        max_output_tokens=512,
    )

    assert config.model_id == "provider/test-model"
    assert config.temperature == 0.0
    assert config.timeout_seconds == 10.0
    assert config.max_output_tokens == 512


def test_initial_planner_model_config_records_the_selected_limits():
    assert INITIAL_PLANNER_MODEL_CONFIG.model_dump() == {
        "model_id": "openai/gpt-oss-120b",
        "temperature": 0.0,
        "timeout_seconds": 30.0,
        "max_output_tokens": 512,
    }


def test_model_config_rejects_blank_model_id():
    with pytest.raises(ValidationError):
        ModelConfig(
            model_id="   ",
            temperature=0.0,
            timeout_seconds=10.0,
            max_output_tokens=512,
        )


@pytest.mark.parametrize(
    "model_id",
    [
        " provider/test-model",
        "provider/test model",
        "provider/test-model\t",
        "provider/\ntest-model",
        "provider/\N{NO-BREAK SPACE}test-model",
    ],
)
def test_model_config_rejects_model_id_with_whitespace(model_id):
    with pytest.raises(ValidationError):
        ModelConfig(
            model_id=model_id,
            temperature=0.0,
            timeout_seconds=10.0,
            max_output_tokens=512,
        )


@pytest.mark.parametrize(
    "temperature",
    [-0.1, 2.1, float("nan"), float("inf"), float("-inf")],
)
def test_model_config_rejects_invalid_temperature(temperature):
    with pytest.raises(ValidationError):
        ModelConfig(
            model_id="provider/test-model",
            temperature=temperature,
            timeout_seconds=10.0,
            max_output_tokens=512,
        )


@pytest.mark.parametrize(
    "timeout_seconds",
    [0.0, -0.1, float("nan"), float("inf"), float("-inf")],
)
def test_model_config_rejects_invalid_timeout(timeout_seconds):
    with pytest.raises(ValidationError):
        ModelConfig(
            model_id="provider/test-model",
            temperature=0.0,
            timeout_seconds=timeout_seconds,
            max_output_tokens=512,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("temperature", False),
        ("temperature", "0.0"),
        ("timeout_seconds", True),
        ("timeout_seconds", "10.0"),
    ],
)
def test_model_config_rejects_numeric_coercion(field_name, value):
    values = {
        "model_id": "provider/test-model",
        "temperature": 0.0,
        "timeout_seconds": 10.0,
        "max_output_tokens": 512,
    }
    values[field_name] = value

    with pytest.raises(ValidationError):
        ModelConfig(**values)


@pytest.mark.parametrize(
    "max_output_tokens",
    [0, -1, 1.5, "512", True, float("nan"), float("inf"), float("-inf")],
)
def test_model_config_rejects_invalid_output_token_limit(max_output_tokens):
    with pytest.raises(ValidationError):
        ModelConfig(
            model_id="provider/test-model",
            temperature=0.0,
            timeout_seconds=10.0,
            max_output_tokens=max_output_tokens,
        )


def test_model_config_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ModelConfig(
            model_id="provider/test-model",
            temperature=0.0,
            timeout_seconds=10.0,
            max_output_tokens=512,
            api_key="must-not-enter-the-config",
        )


def test_model_config_is_immutable():
    config = ModelConfig(
        model_id="provider/test-model",
        temperature=0.0,
        timeout_seconds=10.0,
        max_output_tokens=512,
    )

    with pytest.raises(ValidationError):
        config.temperature = 0.5


def test_model_provider_can_be_replaced_by_fake():
    config = ModelConfig(
        model_id="provider/test-model",
        temperature=0.0,
        timeout_seconds=10.0,
        max_output_tokens=512,
    )
    fake_model = FakeListChatModel(responses=["ok"])
    provider = FakeModelProvider(fake_model)

    returned_model = _create_model_through_contract(provider, config)

    assert provider.received_config is config
    assert returned_model is fake_model
