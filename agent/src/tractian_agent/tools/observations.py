"""Resultado estruturado compartilhado entre tools, trace e futuro ledger."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field, JsonValue

from tractian_agent.contracts import ApiError, ResponseMode, StrictModel


_FORBIDDEN_PARTIAL_KEYS = frozenset(
    {
        "userid",
        "xuserid",
        "identity",
        "permissions",
        "client",
        "seed",
        "authorization",
        "apikey",
        "password",
        "passwd",
        "secret",
        "cookie",
        "centralassetid",
        "context",
        "runtime",
        "config",
        "store",
        "toolcallid",
        "headers",
        "responseheaders",
        "rawresponse",
        "url",
        "method",
        "request",
        "response",
    }
)


def _normalize_partial_key(key: str) -> str:
    """Compara chaves sem depender de caixa, hífen, espaço ou sublinhado."""
    return "".join(character for character in key.casefold() if character.isalnum())


def assert_safe_partial_json(value: JsonValue) -> None:
    """Recusa recursivamente contexto confiável e envelope HTTP em JSON parcial.

    A regra inspeciona somente nomes de campos, nunca conteúdo textual, para não
    tentar inferir segredos ou dados de domínio pelos seus valores.
    """
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            normalized_key = _normalize_partial_key(key)
            if (
                normalized_key in _FORBIDDEN_PARTIAL_KEYS
                or "token" in normalized_key
            ):
                raise ValueError("A resposta degradada contém um campo proibido.")
            assert_safe_partial_json(nested_value)
    elif isinstance(value, list):
        for item in value:
            assert_safe_partial_json(item)


class ToolSource(StrictModel):
    kind: Literal["industrial_api"]
    resource: str = Field(min_length=1, pattern=r"^/")


class ToolOutcome(StrictModel):
    mode: ResponseMode | None = None
    notes: str | None = None
    partial_data: JsonValue | None = None
    error: ApiError | None = None


class ToolArtifact(StrictModel):
    tool_name: str = Field(min_length=1, pattern=r"^\S+$")
    arguments: dict[str, JsonValue]
    source: ToolSource
    outcome: ToolOutcome
    truncated: bool = False
    omitted_items: int = Field(default=0, ge=0)
