import asyncio
from unittest.mock import AsyncMock, patch

from groq import BadRequestError
import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from pydantic import SecretStr, ValidationError

from tractian_agent.groq_provider import GroqModelProvider
from tractian_agent.model_provider import ModelConfig
from tractian_agent.planner import (
    PlannerDecisionKind,
    PlannerStopReason,
    PlannerTerminalDecision,
)


def _fake_chat_result(content: str) -> ChatResult:
    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content=content))]
    )


def _create_test_model():
    return GroqModelProvider(
        api_key=SecretStr("test-groq-key")
    ).create_chat_model(
        ModelConfig(
            model_id="openai/gpt-oss-20b",
            temperature=0.0,
            timeout_seconds=10.0,
            max_output_tokens=512,
        )
    )


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


def test_groq_provider_applies_explicit_retry_budget_without_network():
    provider = GroqModelProvider(
        api_key=SecretStr("test-groq-key"),
        max_retries=3,
    )

    model = provider.create_chat_model(
        ModelConfig(
            model_id="provider/test-model",
            temperature=0.0,
            timeout_seconds=10.0,
            max_output_tokens=512,
        )
    )

    assert model.max_retries == 3


def test_groq_provider_retries_only_sanitized_output_parse_failures():
    provider = GroqModelProvider(
        api_key=SecretStr("test-groq-key"),
        output_parse_retries=2,
    )
    model = provider.create_chat_model(
        ModelConfig(
            model_id="provider/test-model",
            temperature=0.0,
            timeout_seconds=10.0,
            max_output_tokens=512,
        )
    )
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    parse_error = BadRequestError(
        "provider output omitted",
        response=response,
        body={
            "error": {
                "type": "invalid_request_error",
                "code": "output_parse_failed",
                "failed_generation": "must never be retained",
            }
        },
    )
    expected = _fake_chat_result("")

    with patch.object(
        ChatGroq,
        "_agenerate",
        new=AsyncMock(side_effect=[parse_error, parse_error, expected]),
    ) as generate:
        result = asyncio.run(model._agenerate([]))

    assert result is expected
    assert generate.await_count == 3
    assert "must never be retained" not in repr(vars(provider))
    assert "must never be retained" not in repr(model)


def test_groq_provider_uses_strict_json_schema_for_public_structured_output():
    model = _create_test_model()

    upstream = RunnableLambda(lambda _: {"raw": AIMessage(content="{}")})
    with patch.object(
        ChatGroq,
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


def test_groq_provider_validates_strict_pydantic_output_from_json_wire():
    model = _create_test_model()
    response = _fake_chat_result(
        '{"decision":"guide","stop_reason":"sufficient_evidence",'
        '"missing_information":null}'
    )

    with patch.object(model, "_generate", return_value=response):
        result = model.with_structured_output(
            PlannerTerminalDecision,
            include_raw=False,
        ).invoke([HumanMessage(content="Finalize.")])

    assert result == PlannerTerminalDecision(
        decision=PlannerDecisionKind.GUIDE,
        stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
        missing_information=None,
    )


def test_groq_provider_validates_pydantic_json_through_async_public_interface():
    model = _create_test_model()
    response = _fake_chat_result(
        '{"decision":"guide","stop_reason":"sufficient_evidence",'
        '"missing_information":null}'
    )

    async def invoke_structured_output():
        with patch.object(
            model,
            "_agenerate",
            new=AsyncMock(return_value=response),
        ):
            return await asyncio.wait_for(
                model.with_structured_output(
                    PlannerTerminalDecision,
                    include_raw=False,
                ).ainvoke([HumanMessage(content="Finalize.")]),
                timeout=2.0,
            )

    result = asyncio.run(invoke_structured_output())

    assert result == PlannerTerminalDecision(
        decision=PlannerDecisionKind.GUIDE,
        stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
        missing_information=None,
    )


def test_groq_provider_preserves_include_raw_while_parsing_json_wire():
    model = _create_test_model()
    response = _fake_chat_result(
        '{"decision":"guide","stop_reason":"sufficient_evidence",'
        '"missing_information":null}'
    )

    with patch.object(model, "_generate", return_value=response):
        result = model.with_structured_output(
            PlannerTerminalDecision,
            include_raw=True,
        ).invoke([HumanMessage(content="Finalize.")])

    assert result["raw"] == response.generations[0].message
    assert result["parsed"] == PlannerTerminalDecision(
        decision=PlannerDecisionKind.GUIDE,
        stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
        missing_information=None,
    )
    assert result["parsing_error"] is None


def test_groq_provider_include_raw_captures_invalid_pydantic_json():
    model = _create_test_model()
    response = _fake_chat_result(
        '{"decision":"invented","stop_reason":"sufficient_evidence",'
        '"missing_information":null}'
    )

    with patch.object(model, "_generate", return_value=response):
        result = model.with_structured_output(
            PlannerTerminalDecision,
            include_raw=True,
        ).invoke([HumanMessage(content="Finalize.")])

    assert result["raw"] == response.generations[0].message
    assert result["parsed"] is None
    assert isinstance(result["parsing_error"], ValidationError)


@pytest.mark.parametrize(
    "content",
    [
        (
            '{"decision":"invented","stop_reason":"sufficient_evidence",'
            '"missing_information":null}'
        ),
        (
            '{"decision":"guide","stop_reason":"missing_information",'
            '"missing_information":"Informe o ponto de medição."}'
        ),
    ],
    ids=["unknown-enum", "incoherent-contract"],
)
def test_groq_provider_fails_closed_for_invalid_terminal_json(content):
    model = _create_test_model()

    with patch.object(model, "_generate", return_value=_fake_chat_result(content)):
        structured_model = model.with_structured_output(
            PlannerTerminalDecision,
            include_raw=False,
        )
        with pytest.raises(ValidationError):
            structured_model.invoke([HumanMessage(content="Finalize.")])


def test_groq_provider_preserves_include_raw_for_json_schema_dict():
    model = _create_test_model()
    schema = {
        "title": "SmokeStatus",
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
        "additionalProperties": False,
    }
    response = _fake_chat_result('{"status":"ok"}')

    with patch.object(model, "_generate", return_value=response):
        result = model.with_structured_output(schema, include_raw=True).invoke(
            [HumanMessage(content="Finalize.")]
        )

    assert result["raw"] == response.generations[0].message
    assert result["parsed"] == {"status": "ok"}
    assert result["parsing_error"] is None
