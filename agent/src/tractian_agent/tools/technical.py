"""Tools de leitura para baseline e sinais técnicos de um ativo."""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Literal, TypeVar

from langchain.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from tractian_agent.contracts import ApiError, ResponseMode, StrictModel

from .identifiers import AssetId, PointId
from .observations import (
    ToolArtifact,
    ToolOutcome,
    ToolSource,
    assert_safe_partial_json,
)
from .runtime import ReadToolRuntime
from .timestamps import parse_aware_iso_timestamp as _parse_timestamp


ItemT = TypeVar("ItemT")
_POINT_ID_ADAPTER = TypeAdapter(PointId)


def _validate_optional_point_id(value: object) -> PointId | None:
    if value is None:
        return None
    return _POINT_ID_ADAPTER.validate_python(value, strict=True)


def _assert_read_scope(asset_id: str, runtime: ReadToolRuntime) -> None:
    if "read" not in runtime.permissions:
        raise PermissionError("A permissão 'read' é necessária para consultar dados técnicos.")
    if asset_id != runtime.central_asset_id:
        raise ValueError("A consulta deve usar o ativo central confiável.")


def _assert_complete_scope(
    *,
    asset_id: AssetId,
    point_id: PointId | None,
    requested_point_id: PointId | None,
    runtime: ReadToolRuntime,
) -> None:
    if asset_id != runtime.central_asset_id:
        raise ValueError("A API retornou um identificador de ativo fora do escopo.")
    if point_id is None:
        raise ValueError("A API retornou um ponto não verificável.")
    if requested_point_id is not None and point_id != requested_point_id:
        raise ValueError("A API retornou um ponto diferente do solicitado.")


def _assert_degraded_scope(
    data: JsonValue,
    *,
    requested_point_id: PointId | None,
    runtime: ReadToolRuntime,
) -> None:
    assert_safe_partial_json(data)
    pending: list[JsonValue] = [data]
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            if "asset_id" in current:
                asset_id = current["asset_id"]
                if asset_id is None or asset_id != runtime.central_asset_id:
                    raise ValueError("A resposta degradada contém um ativo fora do escopo.")
            if "point_id" in current:
                try:
                    point_id = _validate_optional_point_id(current["point_id"])
                except ValidationError as exc:
                    raise ValueError("A resposta degradada contém um ponto inválido.") from exc
                if point_id is None:
                    raise ValueError("A resposta degradada contém um ponto inválido.")
                if requested_point_id is not None and point_id != requested_point_id:
                    raise ValueError("A resposta degradada contém um ponto diferente do solicitado.")
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)


def _query_params(
    *, point_id: PointId | None, runtime: ReadToolRuntime
) -> dict[str, str] | None:
    params: dict[str, str] = {}
    if point_id is not None:
        params["point_id"] = point_id
    if runtime.seed is not None:
        params["seed"] = runtime.seed
    return params or None


def _arguments(asset_id: AssetId, point_id: PointId | None) -> dict[str, JsonValue]:
    arguments: dict[str, JsonValue] = {"asset_id": asset_id}
    if point_id is not None:
        arguments["point_id"] = point_id
    return arguments


def _content_and_artifact(
    *,
    content: StrictModel | None,
    artifact: ToolArtifact,
    error: ApiError | None,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    if error is not None:
        model_content: dict[str, JsonValue] = {"error": error.model_dump(mode="json")}
    elif content is not None:
        model_content = content.model_dump(mode="json")
    else:
        outcome = artifact.outcome
        model_content = {
            "mode": outcome.mode.value if outcome.mode is not None else None,
            "notes": outcome.notes,
            "partial_data": outcome.partial_data,
        }
    return model_content, artifact.model_dump(mode="json")


class _TechnicalToolArguments(StrictModel):
    """Inclui o runtime somente para a injeção do LangGraph."""

    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)

    asset_id: AssetId
    point_id: PointId | None = None
    runtime: ToolRuntime[ReadToolRuntime]


class _BaselineFeatureWire(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    feature: str = Field(min_length=1, pattern=r"\S")
    reference: float = Field(ge=0, allow_inf_nan=False)
    tolerance: float = Field(ge=0, allow_inf_nan=False)


class _BaselineWire(StrictModel):
    """Payload plano emitido pelo simulador para baseline."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1, pattern=r"^bs_[A-Za-z0-9_-]{1,64}$")
    asset_id: AssetId
    point_id: PointId
    state: Literal["learning", "established", "invalidated"]
    detection_mode: Literal["baseline", "symptom"]
    learnable: bool
    established_at: str | None
    invalidated_at: str | None
    invalidation_reason: str | None
    features: list[_BaselineFeatureWire]

    @field_validator("established_at", "invalidated_at")
    @classmethod
    def _requires_timestamp_with_timezone(cls, value: str | None) -> str | None:
        if value is not None:
            _parse_timestamp(value)
        return value


class BaselineFeature(StrictModel):
    feature: str
    reference: float
    tolerance: float


class BaselineArtifact(StrictModel):
    id: str
    asset_id: AssetId
    point_id: PointId
    state: Literal["learning", "established", "invalidated"]
    detection_mode: Literal["baseline", "symptom"]
    learnable: bool
    established_at: str | None
    invalidated_at: str | None
    invalidation_reason: str | None
    features: list[BaselineFeature]
    alarm_threshold: float | None


class BaselineToolOutcome(ToolOutcome):
    baseline: BaselineArtifact | None = None


class BaselineToolArtifact(ToolArtifact):
    outcome: BaselineToolOutcome


class GetBaselineResult(StrictModel):
    content: BaselineArtifact | None
    artifact: BaselineToolArtifact
    error: ApiError | None = None


def _derive_rms_alarm_threshold(features: list[BaselineFeature]) -> float | None:
    for feature in features:
        if feature.feature == "rms_mm_s":
            threshold = feature.reference + feature.tolerance
            if not math.isfinite(threshold):
                raise ValueError("O limiar RMS derivado não é finito.")
            return round(threshold, 3)
    return None


def _normalize_baseline(baseline: _BaselineWire) -> BaselineArtifact:
    features = [
        BaselineFeature(
            feature=feature.feature,
            reference=feature.reference,
            tolerance=feature.tolerance,
        )
        for feature in baseline.features
    ]
    return BaselineArtifact(
        id=baseline.id,
        asset_id=baseline.asset_id,
        point_id=baseline.point_id,
        state=baseline.state,
        detection_mode=baseline.detection_mode,
        learnable=baseline.learnable,
        established_at=baseline.established_at,
        invalidated_at=baseline.invalidated_at,
        invalidation_reason=baseline.invalidation_reason,
        features=features,
        alarm_threshold=_derive_rms_alarm_threshold(features),
    )


async def execute_get_baseline(
    asset_id: AssetId,
    point_id: PointId | None,
    runtime: ReadToolRuntime,
) -> GetBaselineResult:
    point_id = _validate_optional_point_id(point_id)
    _assert_read_scope(asset_id, runtime)
    path = f"/assets/{asset_id}/baseline"
    result = await runtime.client.query(
        path,
        response_model=_BaselineWire,
        identity=runtime.identity,
        params=_query_params(point_id=point_id, runtime=runtime),
    )
    common = {
        "tool_name": "get_baseline",
        "arguments": _arguments(asset_id, point_id),
        "source": ToolSource(kind="industrial_api", resource=path),
    }
    if isinstance(result, ApiError):
        return GetBaselineResult(
            content=None,
            error=result,
            artifact=BaselineToolArtifact(
                **common, outcome=BaselineToolOutcome(error=result)
            ),
        )
    if result.mode is ResponseMode.COMPLETE:
        baseline = result.data
        if not isinstance(baseline, _BaselineWire):
            raise TypeError("A resposta completa do baseline não foi validada.")
        _assert_complete_scope(
            asset_id=baseline.asset_id,
            point_id=baseline.point_id,
            requested_point_id=point_id,
            runtime=runtime,
        )
        normalized = _normalize_baseline(baseline)
        return GetBaselineResult(
            content=normalized,
            artifact=BaselineToolArtifact(
                **common,
                outcome=BaselineToolOutcome(
                    mode=result.mode, notes=result.notes, baseline=normalized
                ),
            ),
        )
    _assert_degraded_scope(result.data, requested_point_id=point_id, runtime=runtime)
    return GetBaselineResult(
        content=None,
        artifact=BaselineToolArtifact(
            **common,
            outcome=BaselineToolOutcome(
                mode=result.mode, notes=result.notes, partial_data=result.data
            ),
        ),
    )


@tool(
    "get_baseline",
    args_schema=_TechnicalToolArguments,
    response_format="content_and_artifact",
    description=(
        "Consulta o baseline aprendido do ativo e ponto de medição, incluindo "
        "estado, modo de detecção e limiar RMS derivado quando disponível."
    ),
)
async def get_baseline(
    asset_id: AssetId,
    runtime: ToolRuntime[ReadToolRuntime],
    point_id: PointId | None = None,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    result = await execute_get_baseline(asset_id, point_id, runtime.context)
    return _content_and_artifact(
        content=result.content, artifact=result.artifact, error=result.error
    )


class _RmsSampleWire(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    ts: str = Field(min_length=1)
    value: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("ts")
    @classmethod
    def _requires_iso_timestamp(cls, value: str) -> str:
        try:
            _parse_timestamp(value)
        except ValueError as exc:
            raise ValueError("O timestamp RMS deve usar ISO 8601.") from exc
        return value


class _RmsSeriesWire(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    asset_id: AssetId
    point_id: PointId
    unit: Literal["mm/s"]
    baseline_reference: float | None = Field(ge=0, allow_inf_nan=False)
    baseline_state: Literal["learning", "established", "invalidated", "not_applicable"]
    alarm_threshold: float | None = Field(ge=0, allow_inf_nan=False)
    samples: list[_RmsSampleWire]


class RmsSample(StrictModel):
    ts: str
    value: float


class RmsArtifact(StrictModel):
    asset_id: AssetId
    point_id: PointId | None
    unit: Literal["mm/s"]
    baseline_reference: float | None
    baseline_state: Literal["learning", "established", "invalidated", "not_applicable"]
    alarm_threshold: float | None
    samples: list[RmsSample]
    total_samples: int = Field(ge=0)


class RmsModelContent(RmsArtifact):
    omitted_samples: int = Field(ge=0)


class RmsToolOutcome(ToolOutcome):
    rms: RmsArtifact | None = None


class RmsToolArtifact(ToolArtifact):
    outcome: RmsToolOutcome
    model_content: RmsModelContent | None = None


class GetRmsSeriesResult(StrictModel):
    content: RmsModelContent | None
    artifact: RmsToolArtifact
    error: ApiError | None = None


def _chronological_samples(samples: list[_RmsSampleWire]) -> list[RmsSample]:
    return [
        RmsSample(ts=sample.ts, value=sample.value)
        for _, sample in sorted(
            enumerate(samples),
            key=lambda item: (
                _parse_timestamp(item[1].ts),
                item[0],
            ),
        )
    ]


def _edge_preserving_projection(items: list[ItemT], limit: int) -> list[ItemT]:
    if len(items) <= limit:
        return list(items)
    return [items[(index * (len(items) - 1)) // (limit - 1)] for index in range(limit)]


def _normalize_rms(
    rms: _RmsSeriesWire,
) -> tuple[RmsModelContent, RmsArtifact, int]:
    samples = _chronological_samples(rms.samples)
    artifact_samples = _edge_preserving_projection(samples, 1000)
    artifact_omitted = len(samples) - len(artifact_samples)
    prompt_samples = _edge_preserving_projection(samples, 100)
    return (
        RmsModelContent(
            asset_id=rms.asset_id,
            point_id=rms.point_id,
            unit=rms.unit,
            baseline_reference=rms.baseline_reference,
            baseline_state=rms.baseline_state,
            alarm_threshold=rms.alarm_threshold,
            samples=prompt_samples,
            total_samples=len(samples),
            omitted_samples=len(samples) - len(prompt_samples),
        ),
        RmsArtifact(
            asset_id=rms.asset_id,
            point_id=rms.point_id,
            unit=rms.unit,
            baseline_reference=rms.baseline_reference,
            baseline_state=rms.baseline_state,
            alarm_threshold=rms.alarm_threshold,
            samples=artifact_samples,
            total_samples=len(samples),
        ),
        artifact_omitted,
    )


async def execute_get_rms_series(
    asset_id: AssetId,
    point_id: PointId | None,
    runtime: ReadToolRuntime,
) -> GetRmsSeriesResult:
    point_id = _validate_optional_point_id(point_id)
    _assert_read_scope(asset_id, runtime)
    path = f"/assets/{asset_id}/rms"
    result = await runtime.client.query(
        path,
        response_model=_RmsSeriesWire,
        identity=runtime.identity,
        params=_query_params(point_id=point_id, runtime=runtime),
    )
    common = {
        "tool_name": "get_rms_series",
        "arguments": _arguments(asset_id, point_id),
        "source": ToolSource(kind="industrial_api", resource=path),
    }
    if isinstance(result, ApiError):
        return GetRmsSeriesResult(
            content=None,
            error=result,
            artifact=RmsToolArtifact(**common, outcome=RmsToolOutcome(error=result)),
        )
    if result.mode is ResponseMode.COMPLETE:
        rms = result.data
        if not isinstance(rms, _RmsSeriesWire):
            raise TypeError("A resposta completa da série RMS não foi validada.")
        _assert_complete_scope(
            asset_id=rms.asset_id,
            point_id=rms.point_id,
            requested_point_id=point_id,
            runtime=runtime,
        )
        content, normalized, artifact_omitted = _normalize_rms(rms)
        return GetRmsSeriesResult(
            content=content,
            artifact=RmsToolArtifact(
                **common,
                outcome=RmsToolOutcome(
                    mode=result.mode, notes=result.notes, rms=normalized
                ),
                model_content=content,
                truncated=artifact_omitted > 0,
                omitted_items=artifact_omitted,
            ),
        )
    _assert_degraded_scope(result.data, requested_point_id=point_id, runtime=runtime)
    return GetRmsSeriesResult(
        content=None,
        artifact=RmsToolArtifact(
            **common,
            outcome=RmsToolOutcome(
                mode=result.mode, notes=result.notes, partial_data=result.data
            ),
        ),
    )


@tool(
    "get_rms_series",
    args_schema=_TechnicalToolArguments,
    response_format="content_and_artifact",
    description=(
        "Consulta a série temporal RMS de um ativo e ponto de medição. "
        "A amostra enviada ao modelo é limitada e declara itens omitidos."
    ),
)
async def get_rms_series(
    asset_id: AssetId,
    runtime: ToolRuntime[ReadToolRuntime],
    point_id: PointId | None = None,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    result = await execute_get_rms_series(asset_id, point_id, runtime.context)
    return _content_and_artifact(
        content=result.content, artifact=result.artifact, error=result.error
    )


class _SpectrumPeakWire(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    freq_hz: float = Field(ge=0, allow_inf_nan=False)
    amplitude_mm_s: float = Field(ge=0, allow_inf_nan=False)
    note: str | None = None


class _SpectrumWire(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    asset_id: AssetId
    point_id: PointId
    peaks: list[_SpectrumPeakWire]
    bands_missing: list[str]
    collected_at: str = Field(min_length=1)

    @field_validator("collected_at")
    @classmethod
    def _requires_iso_timestamp(cls, value: str) -> str:
        try:
            _parse_timestamp(value)
        except ValueError as exc:
            raise ValueError("A coleta do espectro deve usar ISO 8601.") from exc
        return value


class SpectrumPeak(StrictModel):
    freq_hz: float
    amplitude_mm_s: float
    note: str | None


class SpectrumArtifact(StrictModel):
    asset_id: AssetId
    point_id: PointId | None
    peaks: list[SpectrumPeak]
    bands_missing: list[str]
    collected_at: str
    total_peaks: int = Field(ge=0)


class SpectrumModelContent(SpectrumArtifact):
    omitted_peaks: int = Field(ge=0)


class SpectrumToolOutcome(ToolOutcome):
    spectrum: SpectrumArtifact | None = None


class SpectrumToolArtifact(ToolArtifact):
    outcome: SpectrumToolOutcome
    model_content: SpectrumModelContent | None = None


class GetSpectrumResult(StrictModel):
    content: SpectrumModelContent | None
    artifact: SpectrumToolArtifact
    error: ApiError | None = None


def _stable_spectrum_peaks(peaks: list[_SpectrumPeakWire]) -> list[SpectrumPeak]:
    return [
        SpectrumPeak(
            freq_hz=peak.freq_hz,
            amplitude_mm_s=peak.amplitude_mm_s,
            note=peak.note,
        )
        for peak in sorted(peaks, key=lambda peak: peak.freq_hz)
    ]


def _normalize_spectrum(
    spectrum: _SpectrumWire,
) -> tuple[SpectrumModelContent, SpectrumArtifact, int]:
    peaks = _stable_spectrum_peaks(spectrum.peaks)
    artifact_peaks = peaks[:200]
    artifact_omitted = len(peaks) - len(artifact_peaks)
    prompt_peaks = _edge_preserving_projection(peaks, 20)
    base = {
        "asset_id": spectrum.asset_id,
        "point_id": spectrum.point_id,
        "bands_missing": spectrum.bands_missing,
        "collected_at": spectrum.collected_at,
        "total_peaks": len(peaks),
    }
    return (
        SpectrumModelContent(
            **base,
            peaks=prompt_peaks,
            omitted_peaks=len(peaks) - len(prompt_peaks),
        ),
        SpectrumArtifact(
            **base,
            peaks=artifact_peaks,
        ),
        artifact_omitted,
    )


async def execute_get_spectrum(
    asset_id: AssetId,
    point_id: PointId | None,
    runtime: ReadToolRuntime,
) -> GetSpectrumResult:
    point_id = _validate_optional_point_id(point_id)
    _assert_read_scope(asset_id, runtime)
    path = f"/assets/{asset_id}/spectrum"
    result = await runtime.client.query(
        path,
        response_model=_SpectrumWire,
        identity=runtime.identity,
        params=_query_params(point_id=point_id, runtime=runtime),
    )
    common = {
        "tool_name": "get_spectrum",
        "arguments": _arguments(asset_id, point_id),
        "source": ToolSource(kind="industrial_api", resource=path),
    }
    if isinstance(result, ApiError):
        return GetSpectrumResult(
            content=None,
            error=result,
            artifact=SpectrumToolArtifact(
                **common, outcome=SpectrumToolOutcome(error=result)
            ),
        )
    if result.mode is ResponseMode.COMPLETE:
        spectrum = result.data
        if not isinstance(spectrum, _SpectrumWire):
            raise TypeError("A resposta completa do espectro não foi validada.")
        _assert_complete_scope(
            asset_id=spectrum.asset_id,
            point_id=spectrum.point_id,
            requested_point_id=point_id,
            runtime=runtime,
        )
        content, normalized, artifact_omitted = _normalize_spectrum(spectrum)
        return GetSpectrumResult(
            content=content,
            artifact=SpectrumToolArtifact(
                **common,
                outcome=SpectrumToolOutcome(
                    mode=result.mode, notes=result.notes, spectrum=normalized
                ),
                model_content=content,
                truncated=artifact_omitted > 0,
                omitted_items=artifact_omitted,
            ),
        )
    _assert_degraded_scope(result.data, requested_point_id=point_id, runtime=runtime)
    return GetSpectrumResult(
        content=None,
        artifact=SpectrumToolArtifact(
            **common,
            outcome=SpectrumToolOutcome(
                mode=result.mode, notes=result.notes, partial_data=result.data
            ),
        ),
    )


@tool(
    "get_spectrum",
    args_schema=_TechnicalToolArguments,
    response_format="content_and_artifact",
    description=(
        "Consulta o espectro de vibração de um ativo e ponto de medição, "
        "incluindo picos e bandas ausentes declaradas pela API."
    ),
)
async def get_spectrum(
    asset_id: AssetId,
    runtime: ToolRuntime[ReadToolRuntime],
    point_id: PointId | None = None,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    result = await execute_get_spectrum(asset_id, point_id, runtime.context)
    return _content_and_artifact(
        content=result.content, artifact=result.artifact, error=result.error
    )


class _DataQualityWire(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    asset_id: AssetId
    point_id: PointId
    completeness: float = Field(ge=0, le=1, allow_inf_nan=False)
    freshness_minutes: int = Field(ge=0)
    snr_db: float = Field(allow_inf_nan=False)
    staleness_flag: bool


class DataQualityArtifact(StrictModel):
    asset_id: AssetId
    point_id: PointId | None
    completeness: float
    freshness_minutes: int
    snr_db: float
    staleness_flag: bool


class DataQualityToolOutcome(ToolOutcome):
    data_quality: DataQualityArtifact | None = None


class DataQualityToolArtifact(ToolArtifact):
    outcome: DataQualityToolOutcome


class GetDataQualityResult(StrictModel):
    content: DataQualityArtifact | None
    artifact: DataQualityToolArtifact
    error: ApiError | None = None


def _normalize_data_quality(data_quality: _DataQualityWire) -> DataQualityArtifact:
    return DataQualityArtifact(
        asset_id=data_quality.asset_id,
        point_id=data_quality.point_id,
        completeness=data_quality.completeness,
        freshness_minutes=data_quality.freshness_minutes,
        snr_db=data_quality.snr_db,
        staleness_flag=data_quality.staleness_flag,
    )


async def execute_get_data_quality(
    asset_id: AssetId,
    point_id: PointId | None,
    runtime: ReadToolRuntime,
) -> GetDataQualityResult:
    point_id = _validate_optional_point_id(point_id)
    _assert_read_scope(asset_id, runtime)
    path = f"/assets/{asset_id}/data-quality"
    result = await runtime.client.query(
        path,
        response_model=_DataQualityWire,
        identity=runtime.identity,
        params=_query_params(point_id=point_id, runtime=runtime),
    )
    common = {
        "tool_name": "get_data_quality",
        "arguments": _arguments(asset_id, point_id),
        "source": ToolSource(kind="industrial_api", resource=path),
    }
    if isinstance(result, ApiError):
        return GetDataQualityResult(
            content=None,
            error=result,
            artifact=DataQualityToolArtifact(
                **common, outcome=DataQualityToolOutcome(error=result)
            ),
        )
    if result.mode is ResponseMode.COMPLETE:
        data_quality = result.data
        if not isinstance(data_quality, _DataQualityWire):
            raise TypeError("A resposta completa da qualidade dos dados não foi validada.")
        _assert_complete_scope(
            asset_id=data_quality.asset_id,
            point_id=data_quality.point_id,
            requested_point_id=point_id,
            runtime=runtime,
        )
        normalized = _normalize_data_quality(data_quality)
        return GetDataQualityResult(
            content=normalized,
            artifact=DataQualityToolArtifact(
                **common,
                outcome=DataQualityToolOutcome(
                    mode=result.mode, notes=result.notes, data_quality=normalized
                ),
            ),
        )
    _assert_degraded_scope(result.data, requested_point_id=point_id, runtime=runtime)
    return GetDataQualityResult(
        content=None,
        artifact=DataQualityToolArtifact(
            **common,
            outcome=DataQualityToolOutcome(
                mode=result.mode, notes=result.notes, partial_data=result.data
            ),
        ),
    )


@tool(
    "get_data_quality",
    args_schema=_TechnicalToolArguments,
    response_format="content_and_artifact",
    description=(
        "Consulta completude, frescor, relação sinal-ruído e obsolescência dos "
        "dados de um ativo e ponto de medição."
    ),
)
async def get_data_quality(
    asset_id: AssetId,
    runtime: ToolRuntime[ReadToolRuntime],
    point_id: PointId | None = None,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    result = await execute_get_data_quality(asset_id, point_id, runtime.context)
    return _content_and_artifact(
        content=result.content, artifact=result.artifact, error=result.error
    )
