"""Modelos Pydantic para validação de entrada/saída (espelham o contrato OpenAPI)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ActionRequest(BaseModel):
    justification: str = Field(..., min_length=20, description="Justificativa (>=20 chars)")
    params: dict[str, Any] | None = None
    changes: dict[str, Any] | None = None


class AssetConfigUpdate(ActionRequest):
    changes: dict[str, Any] | None = None


class ActionResult(BaseModel):
    accepted: bool = True
    action_id: str
    message: str
