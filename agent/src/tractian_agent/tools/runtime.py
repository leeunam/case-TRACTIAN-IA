"""Contexto confiável, validado e imutável para tools de leitura."""
from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import Identity, StrictModel

from .identifiers import AssetId

Permission = Literal["read", "action_low", "action_high", "escalate"]


class TrustedIdentity(Identity):
    """Identidade da fronteira de entrada que não pode ser alterada pela tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ReadToolRuntime(StrictModel):
    """Dados injetados pela fronteira de entrada; nunca argumentos do modelo."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    identity: TrustedIdentity
    permissions: frozenset[Permission]
    central_asset_id: AssetId
    client: IndustrialApiClient
    seed: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")

    @field_validator("identity", mode="before")
    @classmethod
    def _freeze_identity(cls, value: Identity | dict[str, str]) -> Identity | dict[str, str]:
        if isinstance(value, Identity):
            return value.model_dump()
        return value

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
            identity=TrustedIdentity(user_id=user_id, company_id=company_id),
            permissions=permissions,
            central_asset_id=central_asset_id,
            client=client,
            seed=seed,
        )
