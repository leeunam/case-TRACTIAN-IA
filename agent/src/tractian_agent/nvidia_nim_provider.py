"""Adapter NVIDIA NIM sobre a API de chat OpenAI-compatible."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import urlsplit

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr, ValidationError

from tractian_agent.model_provider import ModelConfig


_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
HOSTED_NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _validated_base_url(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("NVIDIA_NIM_BASE_URL deve ser uma URL absoluta")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "NVIDIA_NIM_BASE_URL deve ser uma URL HTTP(S) sem credenciais ou query"
        )
    if parsed.hostname not in _LOCAL_HOSTS and parsed.scheme != "https":
        raise ValueError("NVIDIA_NIM_BASE_URL remota deve usar HTTPS")
    return value.rstrip("/")


def _secret_value(value: str | SecretStr | None, *, required: bool) -> SecretStr:
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    if raw is None and not required:
        return SecretStr("local-nim-no-auth")
    if (
        not isinstance(raw, str)
        or not raw
        or len(raw.encode("utf-8")) > 4096
        or any(character.isspace() for character in raw)
        or "," in raw
    ):
        raise ValueError("NVIDIA_API_KEY deve ser única, não vazia e sem espaços")
    return SecretStr(raw)


class _StrictJsonSchemaChatOpenAI(ChatOpenAI):
    """ChatOpenAI que mantém validação Pydantic estrita no wire JSON."""

    def with_structured_output(
        self,
        schema: object = None,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("method", None)
        kwargs.pop("strict", None)
        is_pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)
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
                    return {"raw": raw, "parsed": None, "parsing_error": parsing_error}
                raise parsing_error
            if include_raw:
                return {"raw": raw, "parsed": parsed, "parsing_error": None}
            return parsed

        async def parse_json_wire_async(result: object) -> object:
            return parse_json_wire(result)

        return structured_model | RunnableLambda(
            parse_json_wire,
            afunc=parse_json_wire_async,
        )


class NvidiaNimModelProvider:
    """Cria modelos LangChain contra um NIM local ou hospedado."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | SecretStr | None = None,
    ) -> None:
        self._base_url = _validated_base_url(base_url)
        hostname = urlsplit(self._base_url).hostname
        self._api_key = _secret_value(
            api_key,
            required=hostname not in _LOCAL_HOSTS,
        )

    @classmethod
    def from_env(cls, environment: Mapping[str, str]) -> "NvidiaNimModelProvider":
        """Copia configuração explícita sem conservar o mapping mutável."""

        base_url = environment.get(
            "NVIDIA_NIM_BASE_URL",
            HOSTED_NVIDIA_NIM_BASE_URL,
        )
        api_key = environment.get("NVIDIA_API_KEY")
        return cls(base_url=str(base_url), api_key=api_key)

    def create_chat_model(self, config: ModelConfig) -> BaseChatModel:
        """Traduz o contrato comum sem retry oculto."""

        return _StrictJsonSchemaChatOpenAI(
            model=config.model_id,
            temperature=config.temperature,
            timeout=config.timeout_seconds,
            max_tokens=config.max_output_tokens,
            max_retries=0,
            base_url=self._base_url,
            api_key=self._api_key,
        )
