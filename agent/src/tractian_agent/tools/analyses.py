"""Tools de leitura para listar e detalhar análises de um ativo."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import math
from typing import Literal

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

from .identifiers import AnalysisId, AssetId, PointId
from .observations import ToolArtifact, ToolOutcome, ToolSource, assert_safe_partial_json
from .runtime import ReadToolRuntime
from .timestamps import parse_aware_iso_timestamp as _parse_timestamp

AnalysisStatus = Literal["current", "stale", "pending", "inconclusive"]
AnalysisType = Literal[
    "none",
    "imbalance",
    "misalignment",
    "bearing_fault",
    "electrical_fault",
    "looseness",
    "lubrication",
]
AnalysisSeverity = Literal["none", "low", "medium", "high", "critical"]
BaselineStateAtDetection = Literal[
    "learning", "established", "invalidated", "not_applicable"
]
_ANALYSIS_ID = TypeAdapter(AnalysisId)
_ASSET_ID = TypeAdapter(AssetId)
_POINT_ID = TypeAdapter(PointId)
_STATUS = TypeAdapter(AnalysisStatus)
_TYPE = TypeAdapter(AnalysisType)
_SEVERITY = TypeAdapter(AnalysisSeverity)
_DEGRADED_LIST_FLAGS = frozenset({"conflict", "inconclusive"})


def _validate_asset_id(value: object) -> AssetId:
    return _ASSET_ID.validate_python(value, strict=True)


def _validate_analysis_id(value: object) -> str:
    return _ANALYSIS_ID.validate_python(value, strict=True)


def _validate_optional_status(value: object) -> AnalysisStatus | None:
    if value is None:
        return None
    return _STATUS.validate_python(value, strict=True)


class _AnalysisEvidenceWire(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    metric: str = Field(min_length=1, pattern=r"\S")
    value: float = Field(allow_inf_nan=False)
    reference: float | None = Field(allow_inf_nan=False)
    note: str = Field(min_length=1, pattern=r"\S")


class _AnalysisWire(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: AnalysisId
    asset_id: AssetId
    point_id: PointId
    type: AnalysisType
    detection_mode: Literal["baseline", "symptom"]
    severity: AnalysisSeverity
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    baseline_state_at_detection: BaselineStateAtDetection
    evidence: list[_AnalysisEvidenceWire]
    limitations: list[str]
    model_version: str = Field(min_length=1, pattern=r"\S")
    created_at: str = Field(min_length=1)
    status: AnalysisStatus

    @field_validator("created_at")
    @classmethod
    def _requires_timestamp_with_timezone(cls, value: str) -> str:
        _parse_timestamp(value)
        return value


class _AnalysisListWire(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    analyses: list[_AnalysisWire]


class AnalysisEvidence(StrictModel):
    metric: str
    value: float
    reference: float | None
    note: str


class AnalysisArtifact(StrictModel):
    id: str
    asset_id: AssetId
    point_id: PointId
    type: AnalysisType
    detection_mode: Literal["baseline", "symptom"]
    severity: AnalysisSeverity
    confidence: float
    baseline_state_at_detection: BaselineStateAtDetection
    evidence: list[AnalysisEvidence]
    limitations: list[str]
    model_version: str
    created_at: str
    status: AnalysisStatus


class AnalysisSummary(StrictModel):
    """Campos de lista permitidos no contexto do modelo."""

    id: str
    asset_id: AssetId
    point_id: PointId
    type: AnalysisType
    severity: AnalysisSeverity
    confidence: float
    status: AnalysisStatus
    created_at: str
    limitations: list[str]


class AnalysisListModelContent(StrictModel):
    analyses: list[AnalysisSummary]
    total_analyses: int = Field(ge=0)
    returned_analyses: int = Field(ge=0)
    omitted_analyses: int = Field(ge=0)
    truncated: bool


class DegradedAnalysisListModelContent(StrictModel):
    mode: ResponseMode
    notes: str | None
    analyses: list[dict[str, JsonValue]]
    total_analyses: int = Field(ge=0)
    returned_analyses: int = Field(ge=0)
    omitted_analyses: int = Field(ge=0)
    truncated: bool
    partial_data: JsonValue


class AnalysisListToolOutcome(ToolOutcome):
    analyses: list[AnalysisArtifact] | list[dict[str, JsonValue]] | None = None
    total_analyses: int | None = Field(default=None, ge=0)
    returned_analyses: int | None = Field(default=None, ge=0)
    omitted_analyses: int | None = Field(default=None, ge=0)


class AnalysisListToolArtifact(ToolArtifact):
    outcome: AnalysisListToolOutcome


class ListAssetAnalysesResult(StrictModel):
    content: AnalysisListModelContent | DegradedAnalysisListModelContent | None
    artifact: AnalysisListToolArtifact
    error: ApiError | None = None


class AnalysisDetailToolOutcome(ToolOutcome):
    analysis: AnalysisArtifact | None = None


class AnalysisDetailToolArtifact(ToolArtifact):
    outcome: AnalysisDetailToolOutcome


class GetAnalysisResult(StrictModel):
    content: AnalysisArtifact | None
    artifact: AnalysisDetailToolArtifact
    error: ApiError | None = None


class _ListAssetAnalysesToolArguments(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)

    asset_id: AssetId
    status: AnalysisStatus | None = None
    runtime: ToolRuntime[ReadToolRuntime]


class _GetAnalysisToolArguments(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)

    analysis_id: AnalysisId
    runtime: ToolRuntime[ReadToolRuntime]


def _assert_read_permission(runtime: ReadToolRuntime) -> None:
    if "read" not in runtime.permissions:
        raise PermissionError("A permissão 'read' é necessária para consultar análises.")


def _assert_list_scope(asset_id: AssetId, runtime: ReadToolRuntime) -> None:
    _assert_read_permission(runtime)
    if asset_id != runtime.central_asset_id:
        raise ValueError("A consulta deve usar o ativo central confiável.")


def _assert_analysis_asset(asset_id: AssetId, runtime: ReadToolRuntime) -> None:
    if asset_id != runtime.central_asset_id:
        raise ValueError("A API retornou um ativo fora do escopo.")


def _normalize_analysis(analysis: _AnalysisWire) -> AnalysisArtifact:
    return AnalysisArtifact(
        id=analysis.id,
        asset_id=analysis.asset_id,
        point_id=analysis.point_id,
        type=analysis.type,
        detection_mode=analysis.detection_mode,
        severity=analysis.severity,
        confidence=analysis.confidence,
        baseline_state_at_detection=analysis.baseline_state_at_detection,
        evidence=[
            AnalysisEvidence(
                metric=evidence.metric,
                value=evidence.value,
                reference=evidence.reference,
                note=evidence.note,
            )
            for evidence in analysis.evidence
        ],
        limitations=list(analysis.limitations),
        model_version=analysis.model_version,
        created_at=analysis.created_at,
        status=analysis.status,
    )


def _ordered_analyses(analyses: list[_AnalysisWire]) -> list[_AnalysisWire]:
    return [
        analysis
        for _, analysis in sorted(
            enumerate(analyses),
            key=lambda item: (_parse_timestamp(item[1].created_at), -item[0]),
            reverse=True,
        )
    ]


def _summary(analysis: AnalysisArtifact) -> AnalysisSummary:
    return AnalysisSummary(
        id=analysis.id,
        asset_id=analysis.asset_id,
        point_id=analysis.point_id,
        type=analysis.type,
        severity=analysis.severity,
        confidence=analysis.confidence,
        status=analysis.status,
        created_at=analysis.created_at,
        limitations=analysis.limitations,
    )


def _list_params(status: AnalysisStatus | None, runtime: ReadToolRuntime) -> dict[str, str] | None:
    params: dict[str, str] = {}
    if status is not None:
        params["status"] = status
    if runtime.seed is not None:
        params["seed"] = runtime.seed
    return params or None


def _list_common(asset_id: AssetId, status: AnalysisStatus | None, path: str) -> dict[str, object]:
    arguments: dict[str, JsonValue] = {"asset_id": asset_id}
    if status is not None:
        arguments["status"] = status
    return {
        "tool_name": "list_asset_analyses",
        "arguments": arguments,
        "source": ToolSource(kind="industrial_api", resource=path),
    }


def _detail_common(analysis_id: str, path: str) -> dict[str, object]:
    return {
        "tool_name": "get_analysis",
        "arguments": {"analysis_id": analysis_id},
        "source": ToolSource(kind="industrial_api", resource=path),
    }


def _assert_degraded_scope(data: JsonValue, *, runtime: ReadToolRuntime) -> None:
    assert_safe_partial_json(data)
    pending: list[JsonValue] = [data]
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            if "asset_id" in current:
                asset_id = current["asset_id"]
                if asset_id is None or asset_id != runtime.central_asset_id:
                    raise ValueError("A resposta degradada contém um ativo fora do escopo.")
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)


def _assert_degraded_detail_id(data: JsonValue, requested_analysis_id: str) -> None:
    """Valida somente IDs do recurso principal, nunca IDs de objetos aninhados."""
    if not isinstance(data, Mapping):
        return
    for key in ("id", "analysis_id"):
        if key not in data:
            continue
        analysis_id = data[key]
        if analysis_id is None:
            raise ValueError("A resposta degradada contém um identificador nulo.")
        try:
            validated_id = _validate_analysis_id(analysis_id)
        except ValidationError as exc:
            raise ValueError("A resposta degradada contém um identificador inválido.") from exc
        if validated_id != requested_analysis_id:
            raise ValueError("A resposta degradada contém um identificador diferente do solicitado.")


def _validate_degraded_summary_item(item: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Projeta somente os campos seguros que estiverem presentes em uma linha parcial."""
    projected: dict[str, JsonValue] = {}
    try:
        if "id" in item:
            projected["id"] = _validate_analysis_id(item["id"])
        if "asset_id" in item:
            projected["asset_id"] = _validate_asset_id(item["asset_id"])
        if "point_id" in item:
            projected["point_id"] = _POINT_ID.validate_python(item["point_id"], strict=True)
        if "type" in item:
            projected["type"] = _TYPE.validate_python(item["type"], strict=True)
        if "severity" in item:
            projected["severity"] = _SEVERITY.validate_python(item["severity"], strict=True)
        if "confidence" in item:
            confidence = TypeAdapter(float).validate_python(item["confidence"], strict=True)
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("A resposta degradada contém confiança inválida.")
            projected["confidence"] = confidence
        if "status" in item:
            try:
                projected["status"] = _STATUS.validate_python(
                    item["status"], strict=True
                )
            except ValidationError as exc:
                raise ValueError("A resposta degradada contém status inválido.") from exc
        if "created_at" in item:
            created_at = item["created_at"]
            if not isinstance(created_at, str):
                raise ValueError("A resposta degradada contém timestamp inválido.")
            _parse_timestamp(created_at)
            projected["created_at"] = created_at
        if "limitations" in item:
            limitations = item["limitations"]
            if not isinstance(limitations, list) or not all(
                isinstance(value, str) for value in limitations
            ):
                raise ValueError("A resposta degradada contém limitações inválidas.")
            projected["limitations"] = limitations
    except ValidationError as exc:
        raise ValueError("A resposta degradada contém campo de resumo inválido.") from exc
    return projected


def _ordered_degraded_summaries(
    summaries: list[dict[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    dated: list[tuple[int, datetime, dict[str, JsonValue]]] = []
    undated: list[dict[str, JsonValue]] = []
    for index, summary in enumerate(summaries):
        created_at = summary.get("created_at")
        if isinstance(created_at, str):
            dated.append((index, _parse_timestamp(created_at), summary))
        else:
            undated.append(summary)
    return [
        summary
        for _, _, summary in sorted(dated, key=lambda item: (item[1], -item[0]), reverse=True)
    ] + undated


def _normalize_degraded_list(
    data: JsonValue,
    *,
    status: AnalysisStatus | None,
    runtime: ReadToolRuntime,
) -> tuple[list[dict[str, JsonValue]], JsonValue]:
    if not isinstance(data, Mapping):
        raise ValueError("A resposta degradada contém uma lista de análises inválida.")
    analyses = data["analyses"]
    if not isinstance(analyses, list):
        raise ValueError("A resposta degradada contém uma lista de análises inválida.")
    unexpected_keys = set(data) - {"analyses", *_DEGRADED_LIST_FLAGS}
    if unexpected_keys:
        raise ValueError("A resposta degradada contém um campo de topo inesperado.")
    seen_ids: set[str] = set()
    summaries: list[dict[str, JsonValue]] = []
    for item in analyses:
        if not isinstance(item, Mapping):
            raise ValueError("A resposta degradada contém uma análise inválida.")
        summary = _validate_degraded_summary_item(item)
        if "asset_id" in summary:
            _assert_analysis_asset(summary["asset_id"], runtime)
        if "id" in summary:
            analysis_id = summary["id"]
            if analysis_id in seen_ids:
                raise ValueError("A resposta degradada contém um identificador de análise duplicado.")
            seen_ids.add(analysis_id)
        if status is not None:
            if "status" not in summary:
                raise ValueError("A resposta degradada não informa o status solicitado.")
            if summary["status"] != status:
                raise ValueError("A resposta degradada contém uma análise que não respeita o filtro de status.")
        summaries.append(summary)
    flags: dict[str, JsonValue] = {}
    for key in _DEGRADED_LIST_FLAGS:
        if key not in data:
            continue
        value = data[key]
        if not isinstance(value, bool):
            raise ValueError("A resposta degradada contém uma flag inválida.")
        flags[key] = value
    return _ordered_degraded_summaries(summaries), flags


async def execute_list_asset_analyses(
    asset_id: object,
    status: object,
    runtime: ReadToolRuntime,
) -> ListAssetAnalysesResult:
    asset_id = _validate_asset_id(asset_id)
    status = _validate_optional_status(status)
    _assert_list_scope(asset_id, runtime)
    path = f"/assets/{asset_id}/analyses"
    result = await runtime.client.query(
        path,
        response_model=_AnalysisListWire,
        identity=runtime.identity,
        params=_list_params(status, runtime),
    )
    common = _list_common(asset_id, status, path)
    if isinstance(result, ApiError):
        return ListAssetAnalysesResult(
            content=None,
            error=result,
            artifact=AnalysisListToolArtifact(
                **common, outcome=AnalysisListToolOutcome(error=result)
            ),
        )
    if result.mode is not ResponseMode.COMPLETE:
        _assert_degraded_scope(result.data, runtime=runtime)
        if isinstance(result.data, Mapping) and "analyses" in result.data:
            summaries, flags = _normalize_degraded_list(
                result.data, status=status, runtime=runtime
            )
            total = len(summaries)
            artifact_analyses = summaries[:200]
            prompt_analyses = summaries[:20]
            artifact_omitted = total - len(artifact_analyses)
            prompt_omitted = total - len(prompt_analyses)
            return ListAssetAnalysesResult(
                content=DegradedAnalysisListModelContent(
                    mode=result.mode,
                    notes=result.notes,
                    analyses=prompt_analyses,
                    total_analyses=total,
                    returned_analyses=len(prompt_analyses),
                    omitted_analyses=prompt_omitted,
                    truncated=prompt_omitted > 0,
                    partial_data=flags,
                ),
                artifact=AnalysisListToolArtifact(
                    **common,
                    outcome=AnalysisListToolOutcome(
                        mode=result.mode,
                        notes=result.notes,
                        partial_data=flags,
                        analyses=artifact_analyses,
                        total_analyses=total,
                        returned_analyses=len(artifact_analyses),
                        omitted_analyses=artifact_omitted,
                    ),
                    truncated=artifact_omitted > 0,
                    omitted_items=artifact_omitted,
                ),
            )
        return ListAssetAnalysesResult(
            content=None,
            artifact=AnalysisListToolArtifact(
                **common,
                outcome=AnalysisListToolOutcome(
                    mode=result.mode, notes=result.notes, partial_data=result.data
                ),
            ),
        )
    payload = result.data
    if not isinstance(payload, _AnalysisListWire):
        raise TypeError("A resposta completa da lista de análises não foi validada.")
    ordered = _ordered_analyses(payload.analyses)
    seen_ids: set[str] = set()
    normalized: list[AnalysisArtifact] = []
    for analysis in ordered:
        _assert_analysis_asset(analysis.asset_id, runtime)
        if analysis.id in seen_ids:
            raise ValueError("A API retornou um identificador de análise duplicado.")
        if status is not None and analysis.status != status:
            raise ValueError("A API retornou uma análise que não respeita o filtro de status.")
        seen_ids.add(analysis.id)
        normalized.append(_normalize_analysis(analysis))
    total = len(normalized)
    artifact_analyses = normalized[:200]
    prompt_analyses = [_summary(analysis) for analysis in normalized[:20]]
    artifact_omitted = total - len(artifact_analyses)
    prompt_omitted = total - len(prompt_analyses)
    return ListAssetAnalysesResult(
        content=AnalysisListModelContent(
            analyses=prompt_analyses,
            total_analyses=total,
            returned_analyses=len(prompt_analyses),
            omitted_analyses=prompt_omitted,
            truncated=prompt_omitted > 0,
        ),
        artifact=AnalysisListToolArtifact(
            **common,
            outcome=AnalysisListToolOutcome(
                mode=result.mode,
                notes=result.notes,
                analyses=artifact_analyses,
                total_analyses=total,
                returned_analyses=len(artifact_analyses),
                omitted_analyses=artifact_omitted,
            ),
            truncated=artifact_omitted > 0,
            omitted_items=artifact_omitted,
        ),
    )


async def execute_get_analysis(
    analysis_id: object,
    runtime: ReadToolRuntime,
) -> GetAnalysisResult:
    analysis_id = _validate_analysis_id(analysis_id)
    _assert_read_permission(runtime)
    path = f"/analyses/{analysis_id}"
    params = {"seed": runtime.seed} if runtime.seed is not None else None
    result = await runtime.client.query(
        path,
        response_model=_AnalysisWire,
        identity=runtime.identity,
        params=params,
    )
    common = _detail_common(analysis_id, path)
    if isinstance(result, ApiError):
        return GetAnalysisResult(
            content=None,
            error=result,
            artifact=AnalysisDetailToolArtifact(
                **common, outcome=AnalysisDetailToolOutcome(error=result)
            ),
        )
    if result.mode is not ResponseMode.COMPLETE:
        _assert_degraded_scope(result.data, runtime=runtime)
        _assert_degraded_detail_id(result.data, analysis_id)
        return GetAnalysisResult(
            content=None,
            artifact=AnalysisDetailToolArtifact(
                **common,
                outcome=AnalysisDetailToolOutcome(
                    mode=result.mode, notes=result.notes, partial_data=result.data
                ),
            ),
        )
    analysis = result.data
    if not isinstance(analysis, _AnalysisWire):
        raise TypeError("A resposta completa da análise não foi validada.")
    if analysis.id != analysis_id:
        raise ValueError("A API retornou um identificador de análise diferente do solicitado.")
    _assert_analysis_asset(analysis.asset_id, runtime)
    normalized = _normalize_analysis(analysis)
    return GetAnalysisResult(
        content=normalized,
        artifact=AnalysisDetailToolArtifact(
            **common,
            outcome=AnalysisDetailToolOutcome(
                mode=result.mode, notes=result.notes, analysis=normalized
            ),
        ),
    )


def _content_and_artifact(
    *, content: StrictModel | None, artifact: ToolArtifact, error: ApiError | None
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


@tool(
    "list_asset_analyses",
    args_schema=_ListAssetAnalysesToolArguments,
    response_format="content_and_artifact",
    description=(
        "Lista análises do ativo central, com filtro opcional de status. "
        "Retorna ao modelo somente resumos recentes e contagens explícitas."
    ),
)
async def list_asset_analyses(
    asset_id: AssetId,
    runtime: ToolRuntime[ReadToolRuntime],
    status: AnalysisStatus | None = None,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    result = await execute_list_asset_analyses(asset_id, status, runtime.context)
    return _content_and_artifact(
        content=result.content, artifact=result.artifact, error=result.error
    )


@tool(
    "get_analysis",
    args_schema=_GetAnalysisToolArguments,
    response_format="content_and_artifact",
    description=(
        "Consulta o detalhe completo de uma análise do ativo central, incluindo "
        "evidências, limitações, baseline e versão do modelo."
    ),
)
async def get_analysis(
    analysis_id: AnalysisId,
    runtime: ToolRuntime[ReadToolRuntime],
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    result = await execute_get_analysis(analysis_id, runtime.context)
    return _content_and_artifact(
        content=result.content, artifact=result.artifact, error=result.error
    )
