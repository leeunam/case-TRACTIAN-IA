"""Resultado estruturado compartilhado entre tools, trace e futuro ledger."""
from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Literal

from pydantic import Field, JsonValue

from tractian_agent.contracts import ApiError, ResponseMode, StrictModel


_SENSITIVE_PARTIAL_KEY_FRAGMENTS = frozenset(
    {
        "userid",
        "xuserid",
        "authorization",
        "apikey",
        "password",
        "passwd",
        "secret",
        "cookie",
        "centralassetid",
        "toolcallid",
        "headers",
        "responseheaders",
        "rawresponse",
        "token",
    }
)

_GENERIC_FORBIDDEN_PARTIAL_KEYS = frozenset(
    {
        "identity",
        "permissions",
        "client",
        "seed",
        "context",
        "runtime",
        "config",
        "store",
        "url",
        "method",
        "request",
        "response",
    }
)

_ALWAYS_SENSITIVE_KEY_SEGMENTS = frozenset(
    {
        "identity",
        "permissions",
        "client",
        "seed",
        "evaluation",
        "eval",
        "golden",
    }
)

_KNOWN_SENSITIVE_COMPOUNDS = frozenset(
    {
        "trustedidentity",
        "apiclient",
        "evaluationseed",
        "runtimecontext",
        "goldenset",
        "expectedpaths",
        "testscenarios",
        "httpresponse",
        "responsebody",
        "rawhttpbody",
        "requesturl",
    }
)


def _normalize_partial_key(key: str) -> str:
    """Compara chaves sem depender de caixa, hífen, espaço ou sublinhado."""
    return "".join(character for character in key.casefold() if character.isalnum())


def _partial_key_segments(key: str) -> frozenset[str]:
    """Separa snake/kebab/espaços e fronteiras camel/Pascal semânticas."""
    separated_acronyms = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", key)
    separated_camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", separated_acronyms)
    return frozenset(
        segment.casefold()
        for segment in re.findall(r"[A-Za-z0-9]+", separated_camel)
    )


def _is_forbidden_partial_key(key: str) -> bool:
    normalized_key = _normalize_partial_key(key)
    segments = _partial_key_segments(key)

    if normalized_key in _GENERIC_FORBIDDEN_PARTIAL_KEYS:
        return True
    if any(
        fragment in normalized_key for fragment in _SENSITIVE_PARTIAL_KEY_FRAGMENTS
    ):
        return True
    if segments & _ALWAYS_SENSITIVE_KEY_SEGMENTS:
        return True
    if any(compound in normalized_key for compound in _KNOWN_SENSITIVE_COMPOUNDS):
        return True
    if normalized_key.startswith(("evaluation", "eval", "golden")):
        return True
    if "runtime" in segments and segments & {
        "agent",
        "config",
        "configuration",
        "context",
        "identity",
        "store",
        "trusted",
    }:
        return True
    if "trusted" in segments and "context" in segments:
        return True
    if "expected" in segments and segments & {"path", "paths"}:
        return True
    if "test" in segments and segments & {"scenario", "scenarios"}:
        return True
    if "http" in segments and segments & {
        "body",
        "headers",
        "request",
        "response",
        "url",
    }:
        return True
    if "response" in segments and segments & {"body", "headers"}:
        return True
    if "request" in segments and segments & {"body", "headers", "method", "url"}:
        return True
    if "raw" in segments and segments & {"body", "http", "request", "response"}:
        return True
    return False


def assert_safe_partial_json(value: JsonValue) -> None:
    """Recusa recursivamente contexto confiável e envelope HTTP em JSON parcial.

    A regra inspeciona nomes de campos e a finitude de números, mas nunca tenta
    inferir segredos ou dados de domínio a partir de conteúdo textual.
    """
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if _is_forbidden_partial_key(key):
                raise ValueError("A resposta degradada contém um campo proibido.")
            assert_safe_partial_json(nested_value)
    elif isinstance(value, list):
        for item in value:
            assert_safe_partial_json(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("A resposta degradada contém número não finito.")


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
