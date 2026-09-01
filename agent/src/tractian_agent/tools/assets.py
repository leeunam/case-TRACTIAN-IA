"""Tool de leitura do cadastro técnico de um ativo."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from langchain.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import ConfigDict, Field, JsonValue

from tractian_agent.contracts import ApiError, ResponseMode, StrictModel

from .identifiers import AssetId
from .observations import (
    ToolArtifact,
    ToolOutcome,
    ToolSource,
    assert_safe_partial_json,
)
from .runtime import ReadToolRuntime


class _AssetPointWire(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1, pattern=r"^pt_[A-Za-z0-9_-]{1,64}$")
    asset_id: AssetId
    location: str = Field(min_length=1, pattern=r"\S")
    sensor_status: str = Field(min_length=1, pattern=r"\S")


class _AssetWire(StrictModel):
    """Payload plano realmente emitido pelo simulador, não o OpenAPI idealizado."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: AssetId
    name: str = Field(min_length=1, pattern=r"\S")
    company_id: str = Field(min_length=1, pattern=r"^comp_[A-Za-z0-9_-]{1,64}$")
    criticality: Literal["low", "medium", "high", "critical"]
    plant: str = Field(min_length=1, pattern=r"\S")
    line: str = Field(min_length=1, pattern=r"\S")
    parent_asset_id: AssetId | None
    machine_type: str = Field(min_length=1, pattern=r"\S")
    rotation_rpm: float = Field(ge=0, allow_inf_nan=False)
    bearing_pn: str | None
    bpfo_hz: float | None = Field(allow_inf_nan=False)
    bpfi_hz: float | None = Field(allow_inf_nan=False)
    bsf_hz: float | None = Field(allow_inf_nan=False)
    ftf_hz: float | None = Field(allow_inf_nan=False)
    line_frequency_hz: float | None = Field(allow_inf_nan=False)
    sensor_status: str = Field(min_length=1, pattern=r"\S")
    points: list[_AssetPointWire]


class AssetHierarchy(StrictModel):
    plant: str
    line: str
    parent_asset_id: AssetId | None


class BearingSpecifications(StrictModel):
    part_number: str | None
    bpfo_hz: float | None
    bpfi_hz: float | None
    bsf_hz: float | None
    ftf_hz: float | None


class TechnicalConfiguration(StrictModel):
    machine_type: str
    rotation_rpm: float
    bearing_specs: BearingSpecifications
    line_frequency_hz: float | None


class AssetPoint(StrictModel):
    id: str
    location: str
    sensor_status: str


class AssetArtifact(StrictModel):
    """Forma estável que corrige a divergência plana do simulador e OpenAPI."""

    id: AssetId
    name: str
    company_id: str
    criticality: Literal["low", "medium", "high", "critical"]
    hierarchy: AssetHierarchy
    points: list[AssetPoint]
    technical_configuration: TechnicalConfiguration
    sensor_status: str


class AssetModelContent(StrictModel):
    """Recorte mínimo que o modelo pode usar para decidir a próxima consulta."""

    id: AssetId
    name: str
    criticality: Literal["low", "medium", "high", "critical"]
    machine_type: str
    rotation_rpm: float
    sensor_status: str
    points: list[AssetPoint]


class AssetToolOutcome(ToolOutcome):
    asset: AssetArtifact | None = None


class AssetToolArtifact(ToolArtifact):
    outcome: AssetToolOutcome


class GetAssetResult(StrictModel):
    content: AssetModelContent | None
    artifact: AssetToolArtifact
    error: ApiError | None = None


class _GetAssetToolArguments(StrictModel):
    """Inclui contexto apenas para a injeção; o schema público o remove."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    asset_id: AssetId
    runtime: ToolRuntime[ReadToolRuntime]


def _normalize_asset(asset: _AssetWire) -> AssetArtifact:
    return AssetArtifact(
        id=asset.id,
        name=asset.name,
        company_id=asset.company_id,
        criticality=asset.criticality,
        hierarchy=AssetHierarchy(
            plant=asset.plant,
            line=asset.line,
            parent_asset_id=asset.parent_asset_id,
        ),
        points=[
            AssetPoint(
                id=point.id,
                location=point.location,
                sensor_status=point.sensor_status,
            )
            for point in asset.points
        ],
        technical_configuration=TechnicalConfiguration(
            machine_type=asset.machine_type,
            rotation_rpm=asset.rotation_rpm,
            bearing_specs=BearingSpecifications(
                part_number=asset.bearing_pn,
                bpfo_hz=asset.bpfo_hz,
                bpfi_hz=asset.bpfi_hz,
                bsf_hz=asset.bsf_hz,
                ftf_hz=asset.ftf_hz,
            ),
            line_frequency_hz=asset.line_frequency_hz,
        ),
        sensor_status=asset.sensor_status,
    )


def _model_content(asset: AssetArtifact) -> AssetModelContent:
    return AssetModelContent(
        id=asset.id,
        name=asset.name,
        criticality=asset.criticality,
        machine_type=asset.technical_configuration.machine_type,
        rotation_rpm=asset.technical_configuration.rotation_rpm,
        sensor_status=asset.sensor_status,
        points=asset.points,
    )


def _assert_scope(asset_id: str, runtime: ReadToolRuntime) -> None:
    if "read" not in runtime.permissions:
        raise PermissionError("A permissão 'read' é necessária para consultar ativos.")
    if asset_id != runtime.central_asset_id:
        raise ValueError("A consulta deve usar o ativo central confiável.")


def _assert_returned_scope(
    *, asset_id: object | None, company_id: object | None, runtime: ReadToolRuntime
) -> None:
    if asset_id is not None and asset_id != runtime.central_asset_id:
        raise ValueError("A API retornou um identificador de ativo fora do escopo.")
    if company_id is not None and company_id != runtime.identity.company_id:
        raise ValueError("A API retornou um ativo de outra empresa.")


def _assert_complete_point_scope(asset: _AssetWire, runtime: ReadToolRuntime) -> None:
    for point in asset.points:
        if point.asset_id != asset.id or point.asset_id != runtime.central_asset_id:
            raise ValueError("A API retornou um ponto de outro ativo.")


def validate_degraded_asset_scope(
    data: JsonValue,
    *,
    asset_id: str,
    company_id: str,
) -> None:
    """Valida o escopo parcial com a mesma regra usada pelo executor."""
    assert_safe_partial_json(data)
    if isinstance(data, Mapping):
        if "id" in data:
            if data["id"] is None:
                raise ValueError("A resposta degradada contém um identificador nulo.")
            if data["id"] != asset_id:
                raise ValueError(
                    "A API retornou um identificador de ativo fora do escopo."
                )
        if "company_id" in data:
            if data["company_id"] is None:
                raise ValueError("A resposta degradada contém uma empresa nula.")
            if data["company_id"] != company_id:
                raise ValueError("A API retornou um ativo de outra empresa.")
        if "points" in data:
            points = data["points"]
            if not isinstance(points, list):
                raise ValueError("A resposta degradada contém pontos inválidos.")
            for point in points:
                if not isinstance(point, Mapping):
                    raise ValueError("A resposta degradada contém um ponto inválido.")
                if "asset_id" in point:
                    if point["asset_id"] is None or point["asset_id"] != asset_id:
                        raise ValueError("A resposta degradada contém um ponto de outro ativo.")

    pending: list[JsonValue] = [data]
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            for key, nested_value in current.items():
                normalized_key = "".join(
                    character for character in key.casefold() if character.isalnum()
                )
                if normalized_key == "assetid":
                    if nested_value is None:
                        raise ValueError(
                            "A resposta degradada contém um ativo não verificável."
                        )
                    if nested_value != asset_id:
                        raise ValueError(
                            "A API retornou um identificador de ativo fora do escopo."
                        )
                elif normalized_key == "companyid":
                    if nested_value is None:
                        raise ValueError(
                            "A resposta degradada contém uma empresa não verificável."
                        )
                    if nested_value != company_id:
                        raise ValueError("A API retornou um ativo de outra empresa.")
                pending.append(nested_value)
        elif isinstance(current, list):
            pending.extend(current)


def _assert_degraded_scope(data: JsonValue, runtime: ReadToolRuntime) -> None:
    validate_degraded_asset_scope(
        data,
        asset_id=runtime.central_asset_id,
        company_id=runtime.identity.company_id,
    )


async def execute_get_asset(
    asset_id: AssetId, runtime: ReadToolRuntime
) -> GetAssetResult:
    _assert_scope(asset_id, runtime)
    path = f"/assets/{asset_id}"
    params = {"seed": runtime.seed} if runtime.seed is not None else None
    result = await runtime.client.query(
        path,
        response_model=_AssetWire,
        identity=runtime.identity,
        params=params,
    )
    source = ToolSource(kind="industrial_api", resource=path)
    common = {
        "tool_name": "get_asset",
        "arguments": {"asset_id": asset_id},
        "source": source,
        "truncated": False,
        "omitted_items": 0,
    }
    if isinstance(result, ApiError):
        return GetAssetResult(
            content=None,
            error=result,
            artifact=AssetToolArtifact(
                **common,
                outcome=AssetToolOutcome(error=result),
            ),
        )
    if result.mode is ResponseMode.COMPLETE:
        asset = result.data
        if not isinstance(asset, _AssetWire):  # Defensive narrowing for the generic API client.
            raise TypeError("A resposta completa do ativo não foi validada.")
        _assert_returned_scope(
            asset_id=asset.id,
            company_id=asset.company_id,
            runtime=runtime,
        )
        _assert_complete_point_scope(asset, runtime)
        normalized = _normalize_asset(asset)
        return GetAssetResult(
            content=_model_content(normalized),
            artifact=AssetToolArtifact(
                **common,
                outcome=AssetToolOutcome(
                    mode=result.mode,
                    notes=result.notes,
                    asset=normalized,
                ),
            ),
        )
    _assert_degraded_scope(result.data, runtime)
    return GetAssetResult(
        content=None,
        artifact=AssetToolArtifact(
            **common,
            outcome=AssetToolOutcome(
                mode=result.mode,
                notes=result.notes,
                partial_data=result.data,
            ),
        ),
    )


@tool(
    "get_asset",
    args_schema=_GetAssetToolArguments,
    response_format="content_and_artifact",
    description=(
        "Consulta o cadastro técnico e os pontos de medição de um ativo. "
        "Não use para análises, baseline, RMS, espectro ou qualidade dos dados."
    ),
)
async def get_asset(
    asset_id: AssetId, runtime: ToolRuntime[ReadToolRuntime]
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Consulta o ativo central, com contexto confiável injetado pelo runtime."""
    result = await execute_get_asset(asset_id, runtime.context)
    if result.error is not None:
        content: dict[str, JsonValue] = {"error": result.error.model_dump(mode="json")}
    elif result.content is not None:
        content = result.content.model_dump(mode="json")
    else:
        outcome = result.artifact.outcome
        content = {
            "mode": outcome.mode.value if outcome.mode is not None else None,
            "notes": outcome.notes,
            "partial_data": outcome.partial_data,
        }
    return content, result.artifact.model_dump(mode="json")
