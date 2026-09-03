"""Configuração de modelo exclusiva dos juízes offline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from pydantic import SecretStr

from tractian_agent.model_provider import ModelConfig


class _JsonModeChatGroq(ChatGroq):
    """Evita tool/schema server-side e mantém validação Pydantic local."""

    def with_structured_output(
        self,
        schema: object = None,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("method", None)
        kwargs.pop("strict", None)
        return super().with_structured_output(
            schema,
            method="json_mode",
            include_raw=include_raw,
            **kwargs,
        )


class GroqJudgeModelProvider:
    """Provider Groq em JSON mode, restrito a avaliação offline."""

    def __init__(self, *, api_key: str | SecretStr) -> None:
        raw = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("GROQ_API_KEY must be set and non-empty")
        self._api_key = SecretStr(raw)

    @classmethod
    def from_env(cls, environment: Mapping[str, str]) -> "GroqJudgeModelProvider":
        key = environment.get("GROQ_API_KEY")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("GROQ_API_KEY must be set and non-empty")
        return cls(api_key=key)

    def create_chat_model(self, config: ModelConfig) -> BaseChatModel:
        return _JsonModeChatGroq(
            model=config.model_id,
            temperature=config.temperature,
            timeout=config.timeout_seconds,
            max_tokens=config.max_output_tokens,
            max_retries=0,
            api_key=self._api_key,
        )
