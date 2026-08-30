"""Contexto confiável, validado e imutável para tools de leitura."""
from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import Identity, StrictModel

from .identifiers import AssetId

Permission = Literal["read", "action_low", "action_high", "escalate"]


class ReadToolRuntime(StrictModel):
    """Dados injetados pela fronteira de entrada; nunca argumentos do modelo."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    identity: Identity
    permissions: frozenset[Permission]
    central_asset_id: AssetId
    client: IndustrialApiClient
    seed: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")

    @classmethod
    def create(
        cls,
        *,
        user_id: str,
        company_id: str,
        permissions: frozenset[Permission],
        central_asset_id: str,
        client: IndustrialApiClient,
        seed: str | None = None,
    ) -> ReadToolRuntime:
        return cls(
            identity=Identity(user_id=user_id, company_id=company_id),
            permissions=permissions,
            central_asset_id=central_asset_id,
            client=client,
            seed=seed,
        )
