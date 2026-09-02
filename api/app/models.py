"""Modelos Pydantic para validação de entrada/saída (espelham o contrato OpenAPI)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ActionRequest(BaseModel):
    justification: str = Field(..., min_length=20, description="Justificativa (>=20 chars)")
    params: dict[str, Any] | None = None
    changes: dict[str, Any] | None = None


class BearingSpecsUpdate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        json_schema_extra={"minProperties": 1},
    )

    # ``default_factory`` permite omitir o campo sem anunciá-lo como nullable no
    # JSON Schema; se o cliente enviar null explicitamente, Pydantic rejeita.
    part_number: str = Field(default_factory=lambda: None)
    bpfo_hz: float = Field(default_factory=lambda: None, ge=0)
    bpfi_hz: float = Field(default_factory=lambda: None, ge=0)
    bsf_hz: float = Field(default_factory=lambda: None, ge=0)
    ftf_hz: float = Field(default_factory=lambda: None, ge=0)

    @model_validator(mode="after")
    def require_field(self) -> BearingSpecsUpdate:
        if not self.model_fields_set:
            raise ValueError("bearing_specs exige ao menos um valor não nulo")
        return self


class AssetTechnicalConfigUpdate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        json_schema_extra={"minProperties": 1},
    )

    machine_type: Literal[
        "compressor",
        "fan",
        "gearbox",
        "mill",
        "motor_dc",
        "motor_induction",
        "pump",
        "spindle",
    ] = Field(default_factory=lambda: None)
    rotation_rpm: float = Field(default_factory=lambda: None, gt=0)
    bearing_specs: BearingSpecsUpdate = Field(default_factory=lambda: None)
    line_frequency_hz: float = Field(default_factory=lambda: None, gt=0)

    @model_validator(mode="after")
    def require_field(self) -> AssetTechnicalConfigUpdate:
        if not self.model_fields_set:
            raise ValueError("config exige ao menos um valor não nulo")
        return self


class AssetChanges(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        json_schema_extra={"minProperties": 1},
    )

    criticality: Literal["low", "medium", "high", "critical"] = Field(
        default_factory=lambda: None
    )
    config: AssetTechnicalConfigUpdate = Field(default_factory=lambda: None)

    @model_validator(mode="after")
    def require_change(self) -> AssetChanges:
        if self.criticality is None and self.config is None:
            raise ValueError("changes deve conter ao menos uma alteração")
        return self


class AssetConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    justification: str = Field(min_length=20)
    changes: AssetChanges

    @field_validator("justification")
    @classmethod
    def validate_justification(cls, value: str) -> str:
        if len(value.strip()) < 20:
            raise ValueError("justification deve ter ao menos 20 caracteres")
        return value


class ActionResult(BaseModel):
    accepted: bool = True
    action_id: str
    message: str


class Error(BaseModel):
    code: str
    message: str
