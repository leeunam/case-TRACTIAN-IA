"""Resultado estruturado compartilhado entre tools, trace e futuro ledger."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from tractian_agent.contracts import ApiError, ResponseMode, StrictModel


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
