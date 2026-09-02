"""Adapter Groq para o contrato comum de modelos."""

from collections.abc import Mapping
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from pydantic import BaseModel, SecretStr, ValidationError

from tractian_agent.model_provider import ModelConfig


class _StrictJsonSchemaChatGroq(ChatGroq):
    """ChatGroq que fixa o transporte de saída estruturada do adapter."""

    def with_structured_output(
        self,
        schema: object = None,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Usa o JSON Schema estrito nativo da Groq para contratos Pydantic."""
        kwargs.pop("method", None)
        kwargs.pop("strict", None)
        is_pydantic_schema = isinstance(schema, type) and issubclass(
            schema,
            BaseModel,
        )
        structured_model = super().with_structured_output(
            schema,
            method="json_schema",
            strict=True,
            include_raw=True if is_pydantic_schema else include_raw,
            **kwargs,
        )
        if not is_pydantic_schema:
            return structured_model

        pydantic_schema = cast(type[BaseModel], schema)

        def parse_json_wire(result: object) -> object:
            if not isinstance(result, Mapping):
                raise TypeError("structured output must contain a raw message")
            raw = result.get("raw")
            parsing_error: ValidationError | TypeError | None = None
            if not isinstance(raw, BaseMessage) or not isinstance(raw.content, str):
                parsing_error = TypeError(
                    "structured output raw content must be JSON text"
                )
                parsed = None
            else:
                try:
                    parsed = pydantic_schema.model_validate_json(raw.content)
                except ValidationError as error:
                    parsing_error = error
                    parsed = None
            if parsing_error is not None:
                if include_raw:
                    return {
                        "raw": raw,
                        "parsed": None,
                        "parsing_error": parsing_error,
                    }
                raise parsing_error
            if include_raw:
                return {
                    "raw": raw,
                    "parsed": parsed,
                    "parsing_error": None,
                }
            return parsed

        async def parse_json_wire_async(result: object) -> object:
            return parse_json_wire(result)

        return structured_model | RunnableLambda(
            parse_json_wire,
            afunc=parse_json_wire_async,
        )


class GroqModelProvider:
    """Cria modelos LangChain hospedados pela Groq."""

    def __init__(self, *, api_key: str | SecretStr) -> None:
        raw_api_key = (
            api_key.get_secret_value()
            if isinstance(api_key, SecretStr)
            else api_key
        )
        if not isinstance(raw_api_key, str) or not raw_api_key.strip():
            raise ValueError("GROQ_API_KEY must be set and non-empty")
        self._api_key = SecretStr(raw_api_key)

    @classmethod
    def from_env(cls, environment: Mapping[str, str]) -> "GroqModelProvider":
        """Constrói o adapter a partir de um ambiente fornecido explicitamente."""
        api_key = environment.get("GROQ_API_KEY")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("GROQ_API_KEY must be set and non-empty")
        return cls(api_key=SecretStr(api_key))

    def create_chat_model(self, config: ModelConfig) -> BaseChatModel:
        """Traduz a configuração comum para o adapter oficial da Groq."""
        return _StrictJsonSchemaChatGroq(
            model=config.model_id,
            temperature=config.temperature,
            timeout=config.timeout_seconds,
            max_tokens=config.max_output_tokens,
            max_retries=0,
            api_key=self._api_key,
        )
