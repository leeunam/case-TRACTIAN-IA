"""Adapter Groq para o contrato comum de modelos."""

from collections.abc import Mapping
from typing import Any, cast

from groq import BadRequestError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from pydantic import BaseModel, SecretStr, ValidationError

from tractian_agent.model_provider import ModelConfig


class _StrictJsonSchemaChatGroq(ChatGroq):
    """ChatGroq que fixa o transporte de saída estruturada do adapter."""

    output_parse_retries: int = 0

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        for attempt in range(self.output_parse_retries + 1):
            try:
                return super()._generate(
                    messages,
                    stop=stop,
                    run_manager=run_manager,
                    **kwargs,
                )
            except BadRequestError as error:
                if attempt >= self.output_parse_retries or not _output_parse_failed(
                    error
                ):
                    raise
        raise AssertionError("output parse retry loop ended without result")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        for attempt in range(self.output_parse_retries + 1):
            try:
                return await super()._agenerate(
                    messages,
                    stop=stop,
                    run_manager=run_manager,
                    **kwargs,
                )
            except BadRequestError as error:
                if attempt >= self.output_parse_retries or not _output_parse_failed(
                    error
                ):
                    raise
        raise AssertionError("output parse retry loop ended without result")

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

    def __init__(
        self,
        *,
        api_key: str | SecretStr,
        max_retries: int = 0,
        output_parse_retries: int = 0,
    ) -> None:
        raw_api_key = (
            api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        )
        if not isinstance(raw_api_key, str) or not raw_api_key.strip():
            raise ValueError("GROQ_API_KEY must be set and non-empty")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool):
            raise TypeError("max_retries must be an integer")
        if not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be between zero and five")
        if not isinstance(output_parse_retries, int) or isinstance(
            output_parse_retries, bool
        ):
            raise TypeError("output_parse_retries must be an integer")
        if not 0 <= output_parse_retries <= 3:
            raise ValueError("output_parse_retries must be between zero and three")
        self._api_key = SecretStr(raw_api_key)
        self._max_retries = max_retries
        self._output_parse_retries = output_parse_retries

    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str],
        *,
        max_retries: int = 0,
        output_parse_retries: int = 0,
    ) -> "GroqModelProvider":
        """Constrói o adapter a partir de um ambiente fornecido explicitamente."""
        api_key = environment.get("GROQ_API_KEY")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("GROQ_API_KEY must be set and non-empty")
        return cls(
            api_key=SecretStr(api_key),
            max_retries=max_retries,
            output_parse_retries=output_parse_retries,
        )

    def create_chat_model(self, config: ModelConfig) -> BaseChatModel:
        """Traduz a configuração comum para o adapter oficial da Groq."""
        return _StrictJsonSchemaChatGroq(
            model=config.model_id,
            temperature=config.temperature,
            timeout=config.timeout_seconds,
            max_tokens=config.max_output_tokens,
            max_retries=self._max_retries,
            output_parse_retries=self._output_parse_retries,
            api_key=self._api_key,
        )


def _output_parse_failed(error: BadRequestError) -> bool:
    """Reconhece só o código seguro, sem preservar a geração recusada."""

    body = error.body
    if not isinstance(body, Mapping):
        return False
    detail = body.get("error")
    return isinstance(detail, Mapping) and detail.get("code") in {
        "output_parse_failed",
        "tool_use_failed",
    }
