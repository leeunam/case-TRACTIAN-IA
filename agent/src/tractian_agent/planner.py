"""Primeira fronteira isolada do planner, ainda fora do LangGraph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import math
import re
from types import MappingProxyType
from typing import Final, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import (
    ConfigDict,
    Field,
    ValidationError,
    TypeAdapter,
    field_validator,
    model_validator,
)

from tractian_agent.contracts import ResponseMode, StrictModel, SupportRequest
from tractian_agent.state import (
    AgentState,
    PlannerUsage,
    PersistedSupportRequest,
    PersistedToolCall,
    ToolObservation,
)
from tractian_agent.tools import READ_TOOLS, WRITE_PROPOSAL_TOOLS
from tractian_agent.tools.analyses import (
    AnalysisArtifact,
    AnalysisDetailToolArtifact,
    AnalysisListModelContent,
    AnalysisListToolArtifact,
    AnalysisSummary,
    DegradedAnalysisListModelContent,
)
from tractian_agent.tools.assets import (
    AssetModelContent,
    AssetToolArtifact,
    validate_degraded_asset_scope,
)
from tractian_agent.tools.identifiers import (
    AnalysisId,
    CompanyId,
    KnowledgeDocumentId,
    ModelId,
    PointId,
)
from tractian_agent.tools.runtime import ReadToolRuntime, WriteToolRuntime
from tractian_agent.tools.knowledge import (
    DegradedKnowledgeDocumentContent,
    DegradedKnowledgeSearchModelContent,
    DegradedModelContent,
    KnowledgeDocumentContent,
    KnowledgeDocumentToolArtifact,
    KnowledgeSearchItem,
    KnowledgeSearchModelContent,
    KnowledgeSearchToolArtifact,
    ModelArtifact,
    ModelToolArtifact,
)
from tractian_agent.tools.technical import (
    BaselineArtifact,
    BaselineToolArtifact,
    DataQualityArtifact,
    DataQualityToolArtifact,
    RmsModelContent,
    RmsToolArtifact,
    SpectrumModelContent,
    SpectrumToolArtifact,
)
from tractian_agent.tools.timestamps import parse_aware_iso_timestamp


PLANNER_SYSTEM_PROMPT_VERSION: Final = "planner-v1"
PLANNER_SYSTEM_PROMPT: Final = f"""\
prompt_version: {PLANNER_SYSTEM_PROMPT_VERSION}

Você é o planner do atendimento industrial, separado do writer. Sua função é
escolher a próxima tool oferecida ou encerrar com uma decisão estruturada; não
redija a resposta destinada ao cliente.

Use no máximo uma tool por turno e somente uma tool explicitamente oferecida.
Não invente evidência nem transforme hipótese em fato. Proposal tools apenas
registram propostas e não executam efeito industrial.

Não revele nem devolva raciocínio interno. Produza somente a chamada de tool
solicitada pela etapa de seleção ou os campos do schema da etapa terminal.
"""


class PlannerLimits(StrictModel):
    """Orçamento fixo da fatia isolada do planner."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tool_calls: Literal[7] = 7
    selections: Literal[8] = 8
    finalizations: Literal[1] = 1
    context_characters: Literal[48_000] = 48_000


PLANNER_LIMITS: Final = PlannerLimits()

_ANALYSIS_ID_ADAPTER: Final = TypeAdapter(AnalysisId)
_COMPANY_ID_ADAPTER: Final = TypeAdapter(CompanyId)
_KNOWLEDGE_ID_ADAPTER: Final = TypeAdapter(KnowledgeDocumentId)
_MODEL_ID_ADAPTER: Final = TypeAdapter(ModelId)
_POINT_ID_ADAPTER: Final = TypeAdapter(PointId)
_ANALYSIS_ID_PATTERN: Final = re.compile(
    r"(?<![A-Za-z0-9_-])an_[A-Za-z0-9_-]{1,64}(?![A-Za-z0-9_-])"
)
_KNOWLEDGE_ID_PATTERN: Final = re.compile(
    r"(?<![A-Za-z0-9_-])kb_[A-Za-z0-9_-]{1,64}(?![A-Za-z0-9_-])"
)
_MODEL_ID_PATTERN: Final = re.compile(
    r"(?<![A-Za-z0-9_-])mdl_[A-Za-z0-9_-]{1,64}(?![A-Za-z0-9_-])"
)
_POINT_ID_PATTERN: Final = re.compile(
    r"(?<![A-Za-z0-9_-])pt_[A-Za-z0-9_-]{1,64}(?![A-Za-z0-9_-])"
)
_ASSET_ARGUMENT_TOOL_NAMES: Final = frozenset(
    {
        "get_asset",
        "list_asset_analyses",
        "get_baseline",
        "get_rms_series",
        "get_spectrum",
        "get_data_quality",
    }
)
_POINT_ARGUMENT_TOOL_NAMES: Final = frozenset(
    {"get_baseline", "get_rms_series", "get_spectrum", "get_data_quality"}
)
_ANALYSIS_ARGUMENT_TOOL_NAMES: Final = frozenset(
    {
        "get_analysis",
        "propose_reprocess_analysis",
        "propose_request_specialist_analysis",
    }
)
_PLANNER_CATALOG: Final = (*READ_TOOLS, *WRITE_PROPOSAL_TOOLS)
_PLANNER_TOOLS_BY_NAME: Final[Mapping[str, BaseTool]] = MappingProxyType(
    {tool.name: tool for tool in _PLANNER_CATALOG}
)
if len(_PLANNER_TOOLS_BY_NAME) != len(_PLANNER_CATALOG):
    raise RuntimeError("Os catálogos do planner contêm nomes de tool duplicados.")


def _canonical_catalog_arguments(
    tool_name: str,
    arguments: object,
) -> dict[str, object] | None:
    tool = _PLANNER_TOOLS_BY_NAME.get(tool_name)
    if tool is None or not isinstance(arguments, Mapping):
        return None
    try:
        validated = tool.tool_call_schema.model_validate(arguments)
        explicit_wire = validated.model_dump(mode="json", exclude_unset=True)
        canonical_wire = validated.model_dump(mode="json")
        persisted_wire = json.dumps(
            arguments,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        validated_wire = json.dumps(
            explicit_wire,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, ValidationError):
        return None
    if persisted_wire != validated_wire:
        return None
    return canonical_wire


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _exact_model_wire(
    schema: type[StrictModel],
    value: object,
) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    try:
        validated = schema.model_validate(value)
        canonical = validated.model_dump(mode="json")
        if _canonical_json(value) != _canonical_json(canonical):
            return None
    except (TypeError, ValueError, ValidationError):
        return None
    return canonical


def _expected_read_resource(
    tool_name: str,
    arguments: Mapping[str, object],
    actual_resource: str,
    *,
    configured_model_id: str | None,
) -> str | None:
    asset_id = arguments.get("asset_id")
    analysis_id = arguments.get("analysis_id")
    document_id = arguments.get("document_id")
    resources = {
        "get_asset": f"/assets/{asset_id}",
        "list_asset_analyses": f"/assets/{asset_id}/analyses",
        "get_analysis": f"/analyses/{analysis_id}",
        "get_baseline": f"/assets/{asset_id}/baseline",
        "get_rms_series": f"/assets/{asset_id}/rms",
        "get_spectrum": f"/assets/{asset_id}/spectrum",
        "get_data_quality": f"/assets/{asset_id}/data-quality",
        "search_knowledge": "/knowledge/search",
        "get_knowledge_document": f"/knowledge/{document_id}",
    }
    if tool_name != "get_model":
        return resources.get(tool_name)
    if configured_model_id is not None:
        model_id = _validated_id(configured_model_id, _MODEL_ID_ADAPTER)
        return f"/models/{model_id}" if model_id is not None else None
    prefix = "/models/"
    if not actual_resource.startswith(prefix):
        return None
    model_id = _validated_id(actual_resource.removeprefix(prefix), _MODEL_ID_ADAPTER)
    return f"/models/{model_id}" if model_id is not None else None


def _analysis_summary(analysis: AnalysisArtifact) -> AnalysisSummary:
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


def _edge_preserving_projection(
    items: Sequence[object],
    limit: int,
) -> list[object]:
    """Replica a projeção pública usada pelas tools técnicas."""
    if len(items) <= limit:
        return list(items)
    return [
        items[(index * (len(items) - 1)) // (limit - 1)]
        for index in range(limit)
    ]


def _edge_projection_source_indices(total: int, limit: int) -> list[int]:
    if total <= limit:
        return list(range(total))
    return [
        (index * (total - 1)) // (limit - 1)
        for index in range(limit)
    ]


def _shared_projection_items_match(
    first: Sequence[object],
    first_indices: Sequence[int],
    second: Sequence[object],
    second_indices: Sequence[int],
) -> bool:
    first_by_source = dict(zip(first_indices, first, strict=True))
    second_by_source = dict(zip(second_indices, second, strict=True))
    return all(
        first_by_source[source_index] == second_by_source[source_index]
        for source_index in first_by_source.keys() & second_by_source.keys()
    )


def _nonblank(value: object) -> bool:
    return isinstance(value, str) and bool(value) and bool(value.strip())


def _finite_number(value: object, *, minimum: float | None = None) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and (minimum is None or value >= minimum)
    )


def _asset_artifact_is_concrete(artifact: AssetToolArtifact) -> bool:
    asset = artifact.outcome.asset
    if asset is None:
        return False
    technical = asset.technical_configuration
    bearing = technical.bearing_specs
    optional_frequencies = (
        bearing.bpfo_hz,
        bearing.bpfi_hz,
        bearing.bsf_hz,
        bearing.ftf_hz,
        technical.line_frequency_hz,
    )
    return (
        _validated_id(asset.company_id, _COMPANY_ID_ADAPTER) is not None
        and _nonblank(asset.name)
        and _nonblank(asset.hierarchy.plant)
        and _nonblank(asset.hierarchy.line)
        and _nonblank(technical.machine_type)
        and _finite_number(technical.rotation_rpm, minimum=0)
        and _nonblank(asset.sensor_status)
        and all(
            value is None or _finite_number(value)
            for value in optional_frequencies
        )
        and all(
            _validated_id(point.id, _POINT_ID_ADAPTER) is not None
            and _nonblank(point.location)
            and _nonblank(point.sensor_status)
            for point in asset.points
        )
    )


def _analysis_artifact_is_concrete(analysis: AnalysisArtifact) -> bool:
    return (
        _validated_id(analysis.id, _ANALYSIS_ID_ADAPTER) is not None
        and _finite_number(analysis.confidence, minimum=0)
        and analysis.confidence <= 1
        and _nonblank(analysis.model_version)
        and _is_canonical_timestamp(analysis.created_at)
        and all(
            _nonblank(item.metric)
            and _finite_number(item.value)
            and (item.reference is None or _finite_number(item.reference))
            and _nonblank(item.note)
            for item in analysis.evidence
        )
    )


def _baseline_artifact_is_concrete(baseline: BaselineArtifact) -> bool:
    if (
        re.fullmatch(r"bs_[A-Za-z0-9_-]{1,64}", baseline.id) is None
        or (
            baseline.established_at is not None
            and not _is_canonical_timestamp(baseline.established_at)
        )
        or (
            baseline.invalidated_at is not None
            and not _is_canonical_timestamp(baseline.invalidated_at)
        )
        or not all(
            _nonblank(feature.feature)
            and _finite_number(feature.reference, minimum=0)
            and _finite_number(feature.tolerance, minimum=0)
            for feature in baseline.features
        )
    ):
        return False
    rms_feature = next(
        (
            feature
            for feature in baseline.features
            if feature.feature == "rms_mm_s"
        ),
        None,
    )
    expected_threshold: float | None = None
    if rms_feature is not None:
        total = rms_feature.reference + rms_feature.tolerance
        if not math.isfinite(total):
            return False
        expected_threshold = round(total, 3)
    if expected_threshold is None:
        return baseline.alarm_threshold is None
    return (
        _finite_number(baseline.alarm_threshold, minimum=0)
        and baseline.alarm_threshold == expected_threshold
    )


def _rms_artifact_is_concrete(artifact: RmsToolArtifact) -> bool:
    rms = artifact.outcome.rms
    if rms is None:
        return False
    sample_groups = [rms.samples]
    if artifact.model_content is not None:
        sample_groups.append(artifact.model_content.samples)
    return (
        _validated_id(rms.point_id, _POINT_ID_ADAPTER) is not None
        and (
            rms.baseline_reference is None
            or _finite_number(rms.baseline_reference, minimum=0)
        )
        and (
            rms.alarm_threshold is None
            or _finite_number(rms.alarm_threshold, minimum=0)
        )
        and all(
            all(
                _is_canonical_timestamp(sample.ts)
                and _finite_number(sample.value, minimum=0)
                for sample in samples
            )
            and all(
                datetime.fromisoformat(left.ts.replace("Z", "+00:00"))
                <= datetime.fromisoformat(right.ts.replace("Z", "+00:00"))
                for left, right in zip(samples, samples[1:])
            )
            for samples in sample_groups
        )
    )


def _spectrum_artifact_is_concrete(artifact: SpectrumToolArtifact) -> bool:
    spectrum = artifact.outcome.spectrum
    if spectrum is None:
        return False
    peak_groups = [spectrum.peaks]
    if artifact.model_content is not None:
        peak_groups.append(artifact.model_content.peaks)
    return (
        _validated_id(spectrum.point_id, _POINT_ID_ADAPTER) is not None
        and _is_canonical_timestamp(spectrum.collected_at)
        and all(
            all(
                _finite_number(peak.freq_hz, minimum=0)
                and _finite_number(peak.amplitude_mm_s, minimum=0)
                for peak in peaks
            )
            and all(
                left.freq_hz <= right.freq_hz
                for left, right in zip(peaks, peaks[1:])
            )
            for peaks in peak_groups
        )
    )


def _complete_content_matches_artifact(
    tool_name: str,
    content: object,
    artifact: ToolArtifact,
) -> bool:
    outcome = artifact.outcome
    if tool_name == "get_asset" and isinstance(artifact, AssetToolArtifact):
        asset = artifact.outcome.asset
        if asset is None or not _asset_artifact_is_concrete(artifact):
            return False
        expected = AssetModelContent(
            id=asset.id,
            name=asset.name,
            criticality=asset.criticality,
            machine_type=asset.technical_configuration.machine_type,
            rotation_rpm=asset.technical_configuration.rotation_rpm,
            sensor_status=asset.sensor_status,
            points=asset.points,
        ).model_dump(mode="json")
        return _exact_model_wire(AssetModelContent, content) == expected
    if tool_name == "list_asset_analyses" and isinstance(
        artifact, AnalysisListToolArtifact
    ):
        analyses = artifact.outcome.analyses
        if analyses is None:
            return False
        typed_analyses: list[AnalysisArtifact] = []
        for item in analyses:
            item_wire = _exact_model_wire(AnalysisArtifact, item)
            if item_wire is None:
                return False
            typed_item = AnalysisArtifact.model_validate(item_wire)
            if not _analysis_artifact_is_concrete(typed_item):
                return False
            typed_analyses.append(typed_item)
        if any(
            datetime.fromisoformat(left.created_at.replace("Z", "+00:00"))
            < datetime.fromisoformat(right.created_at.replace("Z", "+00:00"))
            for left, right in zip(typed_analyses, typed_analyses[1:])
        ):
            return False
        total = artifact.outcome.total_analyses
        if total is None:
            return False
        expected = AnalysisListModelContent(
            analyses=[_analysis_summary(item) for item in typed_analyses[:20]],
            total_analyses=total,
            returned_analyses=min(total, 20),
            omitted_analyses=max(total - 20, 0),
            truncated=total > 20,
        ).model_dump(mode="json")
        return _exact_model_wire(AnalysisListModelContent, content) == expected
    if tool_name == "get_analysis" and isinstance(
        artifact, AnalysisDetailToolArtifact
    ):
        analysis = artifact.outcome.analysis
        return (
            analysis is not None
            and _analysis_artifact_is_concrete(analysis)
            and _exact_model_wire(
                AnalysisArtifact,
                content,
            )
            == analysis.model_dump(mode="json")
        )
    if tool_name == "get_baseline" and isinstance(
        artifact, BaselineToolArtifact
    ):
        baseline = artifact.outcome.baseline
        return (
            baseline is not None
            and _baseline_artifact_is_concrete(baseline)
            and _exact_model_wire(
                BaselineArtifact,
                content,
            )
            == baseline.model_dump(mode="json")
        )
    if tool_name == "get_data_quality" and isinstance(
        artifact, DataQualityToolArtifact
    ):
        data_quality = artifact.outcome.data_quality
        return (
            data_quality is not None
            and _validated_id(data_quality.point_id, _POINT_ID_ADAPTER) is not None
            and _finite_number(data_quality.completeness, minimum=0)
            and data_quality.completeness <= 1
            and not isinstance(data_quality.freshness_minutes, bool)
            and isinstance(data_quality.freshness_minutes, int)
            and data_quality.freshness_minutes >= 0
            and _finite_number(data_quality.snr_db)
            and isinstance(data_quality.staleness_flag, bool)
            and _exact_model_wire(
                DataQualityArtifact,
                content,
            )
            == data_quality.model_dump(mode="json")
        )
    if tool_name == "get_model" and isinstance(artifact, ModelToolArtifact):
        model = artifact.outcome.model
        model_wire = _exact_model_wire(ModelArtifact, model)
        return (
            model_wire is not None
            and _degraded_model_is_concrete(model_wire)
            and _exact_model_wire(
                ModelArtifact,
                content,
            )
            == model_wire
        )
    if tool_name == "search_knowledge" and isinstance(
        artifact, KnowledgeSearchToolArtifact
    ):
        results = artifact.outcome.results
        total = artifact.outcome.total_results
        if results is None or total is None:
            return False
        typed_results: list[KnowledgeSearchItem] = []
        for item in results:
            item_wire = _exact_model_wire(KnowledgeSearchItem, item)
            if item_wire is None:
                return False
            typed_results.append(KnowledgeSearchItem.model_validate(item_wire))
        expected = KnowledgeSearchModelContent(
            results=typed_results,
            total_results=total,
            returned_results=len(results),
            omitted_results=max(total - len(results), 0),
            truncated=total > len(results),
        ).model_dump(mode="json")
        return _exact_model_wire(KnowledgeSearchModelContent, content) == expected
    if tool_name == "get_knowledge_document" and isinstance(
        artifact, KnowledgeDocumentToolArtifact
    ):
        document = artifact.outcome.document
        document_wire = _exact_model_wire(KnowledgeDocumentContent, document)
        if document_wire is None:
            return False
        document = KnowledgeDocumentContent.model_validate(document_wire)
        total = (
            document.returned_body_characters
            + document.omitted_body_characters
        )
        if (
            not _nonblank(document.title)
            or any(
                not isinstance(tag, str) or not tag.strip()
                for tag in document.tags
            )
            or total < 1
            or document.returned_body_characters != len(document.body)
            or not 0 < document.returned_body_characters <= 32_000
            or (
                document.omitted_body_characters > 0
                and document.returned_body_characters != 32_000
            )
            or document.truncated != (document.omitted_body_characters > 0)
        ):
            return False
        body = document.body[:8_000]
        expected = KnowledgeDocumentContent(
            id=document.id,
            type=document.type,
            title=document.title,
            body=body,
            tags=document.tags,
            returned_body_characters=len(body),
            omitted_body_characters=total - len(body),
            truncated=total > len(body),
        ).model_dump(mode="json")
        return _exact_model_wire(KnowledgeDocumentContent, content) == expected
    if tool_name == "get_rms_series" and isinstance(artifact, RmsToolArtifact):
        rms = artifact.outcome.rms
        wire = _exact_model_wire(RmsModelContent, content)
        if rms is None or wire is None or not _rms_artifact_is_concrete(artifact):
            return False
        expected_content_count = min(rms.total_samples, 100)
        if len(wire["samples"]) != expected_content_count:
            return False
        samples_wire = [sample.model_dump(mode="json") for sample in rms.samples]
        persisted_content_wire = (
            artifact.model_content.model_dump(mode="json")
            if artifact.model_content is not None
            else None
        )
        if persisted_content_wire is not None:
            if wire != persisted_content_wire:
                return False
        elif rms.total_samples <= 1_000:
            expected_samples = _edge_preserving_projection(samples_wire, 100)
            if wire["samples"] != expected_samples:
                return False
        else:
            return False
        return all(
            wire[field] == getattr(rms, field)
            for field in (
                "asset_id",
                "point_id",
                "unit",
                "baseline_reference",
                "baseline_state",
                "alarm_threshold",
                "total_samples",
            )
        ) and wire["omitted_samples"] == rms.total_samples - expected_content_count
    if tool_name == "get_spectrum" and isinstance(
        artifact, SpectrumToolArtifact
    ):
        spectrum = artifact.outcome.spectrum
        wire = _exact_model_wire(SpectrumModelContent, content)
        if (
            spectrum is None
            or wire is None
            or not _spectrum_artifact_is_concrete(artifact)
        ):
            return False
        expected_content_count = min(spectrum.total_peaks, 20)
        if len(wire["peaks"]) != expected_content_count:
            return False
        peaks_wire = [peak.model_dump(mode="json") for peak in spectrum.peaks]
        persisted_content_wire = (
            artifact.model_content.model_dump(mode="json")
            if artifact.model_content is not None
            else None
        )
        if persisted_content_wire is not None:
            if wire != persisted_content_wire:
                return False
        elif spectrum.total_peaks <= 200:
            expected_peaks = _edge_preserving_projection(peaks_wire, 20)
            if wire["peaks"] != expected_peaks:
                return False
        else:
            return False
        if any(
            left["freq_hz"] > right["freq_hz"]
            for left, right in zip(wire["peaks"], wire["peaks"][1:])
        ):
            return False
        return all(
            wire[field] == getattr(spectrum, field)
            for field in (
                "asset_id",
                "point_id",
                "bands_missing",
                "collected_at",
                "total_peaks",
            )
        ) and wire["omitted_peaks"] == spectrum.total_peaks - expected_content_count
    return False


def _specialized_outcome_is_empty(artifact: ToolArtifact) -> bool:
    base_fields = set(type(artifact.outcome).__mro__[1].model_fields)
    specialized_fields = set(type(artifact.outcome).model_fields) - base_fields
    return all(
        getattr(artifact.outcome, field_name) is None
        for field_name in specialized_fields
    )


def _recursive_scope_matches(
    value: object,
    *,
    asset_id: str | None,
    point_id: str | None,
    company_id: str | None = None,
    allow_null_point: bool = False,
) -> bool:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            for key, nested in current.items():
                if key == "asset_id" and (
                    not isinstance(nested, str) or nested != asset_id
                ):
                    return False
                if key == "point_id":
                    if nested is None and allow_null_point:
                        continue
                    validated_point = _validated_id(nested, _POINT_ID_ADAPTER)
                    if validated_point is None or (
                        point_id is not None and validated_point != point_id
                    ):
                        return False
                if key == "company_id" and (
                    not isinstance(nested, str) or nested != company_id
                ):
                    return False
                pending.append(nested)
        elif isinstance(current, list):
            pending.extend(current)
    return True


def _degraded_flags_are_valid(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) <= {"conflict", "inconclusive"}
        and all(isinstance(item, bool) for item in value.values())
    )


def _is_canonical_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parse_aware_iso_timestamp(value)
    except ValueError:
        return False
    return True


def _degraded_analysis_rows_are_concrete(rows: object) -> bool:
    if not isinstance(rows, list):
        return False
    allowed = {
        "id",
        "asset_id",
        "point_id",
        "type",
        "severity",
        "confidence",
        "status",
        "created_at",
        "limitations",
    }
    analysis_types = {
        "none",
        "imbalance",
        "misalignment",
        "bearing_fault",
        "electrical_fault",
        "looseness",
        "lubrication",
    }
    severities = {"none", "low", "medium", "high", "critical"}
    statuses = {"current", "stale", "pending", "inconclusive"}
    dated: list[datetime] = []
    saw_undated = False
    for row in rows:
        if not isinstance(row, Mapping) or set(row) - allowed:
            return False
        if "confidence" in row:
            confidence = row["confidence"]
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                return False
        if "type" in row and row["type"] not in analysis_types:
            return False
        if "severity" in row and row["severity"] not in severities:
            return False
        if "status" in row and row["status"] not in statuses:
            return False
        if "limitations" in row:
            limitations = row["limitations"]
            if not isinstance(limitations, list) or not all(
                isinstance(item, str) for item in limitations
            ):
                return False
        if "created_at" in row:
            if saw_undated or not _is_canonical_timestamp(row["created_at"]):
                return False
            dated.append(
                datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
            )
        else:
            saw_undated = True
    return all(left >= right for left, right in zip(dated, dated[1:]))


def _degraded_search_rows_are_concrete(rows: object) -> bool:
    if not isinstance(rows, list):
        return False
    allowed = {"id", "type", "title", "tags", "snippet"}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) - allowed:
            return False
        if "type" in row and row["type"] not in {
            "procedure",
            "glossary",
            "guidance",
        }:
            return False
        if "title" in row:
            title = row["title"]
            if not isinstance(title, str) or not title.strip():
                return False
        if "tags" in row:
            tags = row["tags"]
            if not isinstance(tags, list) or not all(
                isinstance(tag, str) and tag.strip() for tag in tags
            ):
                return False
        if "snippet" in row:
            snippet = row["snippet"]
            if (
                not isinstance(snippet, str)
                or len(snippet) > 240
                or re.sub(r"\s+", " ", snippet).strip() != snippet
            ):
                return False
    return True


def _degraded_model_is_concrete(model: object) -> bool:
    if not isinstance(model, Mapping) or set(model) - {
        "id",
        "version",
        "coverage",
        "requirements",
        "processing_state",
        "last_run_at",
    }:
        return False
    if "version" in model:
        version = model["version"]
        if not isinstance(version, str) or not version.strip():
            return False
    if "processing_state" in model and model["processing_state"] not in {
        "idle",
        "running",
        "pending",
        "delayed",
        "failed",
    }:
        return False
    if "last_run_at" in model and model["last_run_at"] is not None and not (
        _is_canonical_timestamp(model["last_run_at"])
    ):
        return False
    if "coverage" in model:
        coverage = model["coverage"]
        if not isinstance(coverage, list):
            return False
        seen_machine_types: set[str] = set()
        for item in coverage:
            if not isinstance(item, Mapping) or set(item) - {
                "machine_type",
                "supported",
                "can_learn_baseline",
                "note",
            }:
                return False
            machine_type = item.get("machine_type")
            if machine_type is not None:
                if (
                    not isinstance(machine_type, str)
                    or not machine_type.strip()
                    or machine_type in seen_machine_types
                ):
                    return False
                seen_machine_types.add(machine_type)
            for flag in ("supported", "can_learn_baseline"):
                if flag in item and not isinstance(item[flag], bool):
                    return False
            note = item.get("note")
            if note is not None and (
                not isinstance(note, str) or not note.strip()
            ):
                return False
    if "requirements" in model:
        requirements = model["requirements"]
        if not isinstance(requirements, Mapping) or set(requirements) - {
            "min_completeness",
            "min_snr_db",
            "min_rotation_rpm",
        }:
            return False
        for key, maximum in (
            ("min_completeness", 1.0),
            ("min_snr_db", None),
            ("min_rotation_rpm", None),
        ):
            if key not in requirements:
                continue
            value = requirements[key]
            if value is None and key == "min_rotation_rpm":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (maximum is not None and value > maximum)
            ):
                return False
    return True


def _degraded_document_is_concrete(document: object) -> bool:
    if not isinstance(document, Mapping) or set(document) - {
        "id",
        "type",
        "title",
        "body",
        "tags",
        "returned_body_characters",
        "omitted_body_characters",
        "truncated",
    }:
        return False
    if "type" in document and document["type"] not in {
        "procedure",
        "glossary",
        "guidance",
    }:
        return False
    if "title" in document:
        title = document["title"]
        if not isinstance(title, str) or not title.strip():
            return False
    if "tags" in document:
        tags = document["tags"]
        if not isinstance(tags, list) or not all(
            isinstance(tag, str) and tag.strip() for tag in tags
        ):
            return False
    body_fields = {
        "returned_body_characters",
        "omitted_body_characters",
        "truncated",
    }
    if "body" not in document:
        return body_fields.isdisjoint(document)
    body = document["body"]
    returned = document.get("returned_body_characters")
    omitted = document.get("omitted_body_characters")
    truncated = document.get("truncated")
    if (
        not isinstance(body, str)
        or not isinstance(returned, int)
        or isinstance(returned, bool)
        or not isinstance(omitted, int)
        or isinstance(omitted, bool)
        or omitted < 0
        or not isinstance(truncated, bool)
    ):
        return False
    total = returned + omitted
    return (
        returned == len(body) == min(total, 32_000)
        and truncated == (omitted > 0)
    )


def _artifact_scope_matches(
    call: PersistedToolCall,
    artifact: ToolArtifact,
    request: SupportRequest | PersistedSupportRequest,
    *,
    configured_model_id: str | None,
) -> bool:
    arguments = call.arguments.to_python()
    if not isinstance(arguments, Mapping):
        return False
    requested_asset = arguments.get("asset_id")
    requested_point = arguments.get("point_id")
    if call.name == "get_asset" and isinstance(artifact, AssetToolArtifact):
        asset = artifact.outcome.asset
        return (
            asset is not None
            and asset.id == requested_asset == request.asset_id
            and asset.company_id == request.identity.company_id
            and all(
                _validated_id(point.id, _POINT_ID_ADAPTER) is not None
                for point in asset.points
            )
        )
    if call.name == "list_asset_analyses" and isinstance(
        artifact, AnalysisListToolArtifact
    ):
        analyses = artifact.outcome.analyses
        complete = artifact.outcome.mode is ResponseMode.COMPLETE
        if analyses is None or (
            not complete and not _degraded_analysis_rows_are_concrete(analyses)
        ):
            return False
        seen_ids: set[str] = set()
        for item in analyses:
            if complete:
                complete_item_wire = _exact_model_wire(AnalysisArtifact, item)
                if complete_item_wire is None:
                    return False
                item_wire = complete_item_wire
            elif isinstance(item, Mapping):
                item_wire = item
            else:
                return False
            item_id = item_wire.get("id")
            if item_id is not None:
                validated_id = _validated_id(item_id, _ANALYSIS_ID_ADAPTER)
                if validated_id is None or validated_id in seen_ids:
                    return False
                seen_ids.add(validated_id)
            if (
                item_wire.get("asset_id", requested_asset) != requested_asset
                or item_wire.get("asset_id", request.asset_id) != request.asset_id
            ):
                return False
            if "point_id" in item_wire and _validated_id(
                item_wire["point_id"],
                _POINT_ID_ADAPTER,
            ) is None:
                return False
            status = arguments.get("status")
            if status is not None and item_wire.get("status") != status:
                return False
        return requested_asset == request.asset_id
    if call.name == "get_analysis" and isinstance(
        artifact, AnalysisDetailToolArtifact
    ):
        analysis = artifact.outcome.analysis
        return (
            analysis is not None
            and analysis.id == arguments.get("analysis_id")
            and analysis.asset_id == request.asset_id
            and _validated_id(analysis.point_id, _POINT_ID_ADAPTER) is not None
        )
    if call.name in {
        "get_baseline",
        "get_rms_series",
        "get_spectrum",
        "get_data_quality",
    }:
        field_name = {
            "get_baseline": "baseline",
            "get_rms_series": "rms",
            "get_spectrum": "spectrum",
            "get_data_quality": "data_quality",
        }[call.name]
        result = getattr(artifact.outcome, field_name, None)
        return (
            result is not None
            and result.asset_id == requested_asset == request.asset_id
            and _validated_id(result.point_id, _POINT_ID_ADAPTER) is not None
            and (requested_point is None or result.point_id == requested_point)
        )
    if call.name == "get_model" and isinstance(artifact, ModelToolArtifact):
        model = artifact.outcome.model
        expected_model_id = configured_model_id
        if expected_model_id is None:
            expected_model_id = artifact.source.resource.removeprefix("/models/")
        if isinstance(model, ModelArtifact):
            return model.id == expected_model_id
        if not _degraded_model_is_concrete(model):
            return False
        returned_id = model.get("id")
        return returned_id is None or (
            _validated_id(returned_id, _MODEL_ID_ADAPTER) == expected_model_id
        )
    if call.name == "search_knowledge" and isinstance(
        artifact, KnowledgeSearchToolArtifact
    ):
        results = artifact.outcome.results
        if results is None or not _degraded_search_rows_are_concrete(results):
            return False
        seen_ids: set[str] = set()
        requested_type = arguments.get("document_type")
        for item in results:
            item_wire = (
                item.model_dump(mode="json")
                if isinstance(item, KnowledgeSearchItem)
                else item
            )
            if not isinstance(item_wire, Mapping) or set(item_wire) - {
                "id",
                "type",
                "title",
                "tags",
                "snippet",
            }:
                return False
            item_id = item_wire.get("id")
            if item_id is not None:
                validated_id = _validated_id(item_id, _KNOWLEDGE_ID_ADAPTER)
                if validated_id is None or validated_id in seen_ids:
                    return False
                seen_ids.add(validated_id)
            if requested_type is not None and item_wire.get("type") != requested_type:
                return False
        return True
    if call.name == "get_knowledge_document" and isinstance(
        artifact, KnowledgeDocumentToolArtifact
    ):
        document = artifact.outcome.document
        if isinstance(document, KnowledgeDocumentContent):
            document_id = document.id
        elif isinstance(document, Mapping):
            document_id = document.get("id")
        else:
            return False
        return (
            _degraded_document_is_concrete(document)
            and document_id in {None, arguments.get("document_id")}
        )
    return False


def _artifact_projection_metadata_matches(
    tool_name: str,
    artifact: ToolArtifact,
) -> bool:
    outcome = artifact.outcome
    if tool_name == "list_asset_analyses" and isinstance(
        artifact, AnalysisListToolArtifact
    ):
        items = outcome.analyses
        total = outcome.total_analyses
        return (
            items is not None
            and total is not None
            and len(items) == min(total, 200)
            and outcome.returned_analyses == len(items)
            and outcome.omitted_analyses == total - len(items)
            and artifact.omitted_items == total - len(items)
            and artifact.truncated == (total > len(items))
        )
    if tool_name == "search_knowledge" and isinstance(
        artifact, KnowledgeSearchToolArtifact
    ):
        items = outcome.results
        total = outcome.total_results
        return (
            items is not None
            and total is not None
            and len(items) == min(total, 10)
            and outcome.returned_results == len(items)
            and outcome.omitted_results == total - len(items)
            and artifact.omitted_items == total - len(items)
            and artifact.truncated == (total > len(items))
        )
    if tool_name == "get_knowledge_document" and isinstance(
        artifact,
        KnowledgeDocumentToolArtifact,
    ):
        document = artifact.outcome.document
        return artifact.omitted_items == 0 and (
            not isinstance(document, Mapping)
            or artifact.truncated == bool(document.get("truncated", False))
        ) and (
            not isinstance(document, KnowledgeDocumentContent)
            or artifact.truncated == document.truncated
        )
    if tool_name == "get_rms_series" and isinstance(artifact, RmsToolArtifact):
        rms = outcome.rms
        if rms is None:
            return False
        total = rms.total_samples
        artifact_indices = _edge_projection_source_indices(total, 1_000)
        if (
            len(rms.samples) != min(total, 1_000)
            or artifact.omitted_items != total - len(rms.samples)
            or artifact.truncated != (total > len(rms.samples))
        ):
            return False
        model_content = artifact.model_content
        if model_content is None:
            return total <= 1_000
        prompt_indices = _edge_projection_source_indices(total, 100)
        return (
            model_content.total_samples == total
            and len(model_content.samples) == min(total, 100)
            and model_content.omitted_samples == total - len(model_content.samples)
            and _shared_projection_items_match(
                rms.samples,
                artifact_indices,
                model_content.samples,
                prompt_indices,
            )
            and (
                total > 1_000
                or model_content.samples
                == _edge_preserving_projection(rms.samples, 100)
            )
        )
    if tool_name == "get_spectrum" and isinstance(
        artifact, SpectrumToolArtifact
    ):
        spectrum = outcome.spectrum
        if spectrum is None:
            return False
        total = spectrum.total_peaks
        if (
            len(spectrum.peaks) != min(total, 200)
            or artifact.omitted_items != total - len(spectrum.peaks)
            or artifact.truncated != (total > len(spectrum.peaks))
        ):
            return False
        model_content = artifact.model_content
        if model_content is None:
            return total <= 200
        prompt_indices = _edge_projection_source_indices(total, 20)
        return (
            model_content.total_peaks == total
            and len(model_content.peaks) == min(total, 20)
            and model_content.omitted_peaks == total - len(model_content.peaks)
            and _shared_projection_items_match(
                spectrum.peaks,
                range(len(spectrum.peaks)),
                model_content.peaks,
                prompt_indices,
            )
            and (
                total > 200
                or model_content.peaks
                == _edge_preserving_projection(spectrum.peaks, 20)
            )
        )
    return not artifact.truncated and artifact.omitted_items == 0


def _generic_degraded_content(artifact: ToolArtifact) -> dict[str, object]:
    outcome = artifact.outcome
    return {
        "mode": outcome.mode.value if outcome.mode is not None else None,
        "notes": outcome.notes,
        "partial_data": outcome.partial_data,
    }


def _bounded_model_content_is_empty(artifact: ToolArtifact) -> bool:
    return not isinstance(
        artifact,
        (RmsToolArtifact, SpectrumToolArtifact),
    ) or artifact.model_content is None


def _degraded_content_matches_artifact(
    tool_name: str,
    content: object,
    artifact: ToolArtifact,
) -> bool:
    outcome = artifact.outcome
    if tool_name in {
        "get_asset",
        "get_analysis",
        "get_baseline",
        "get_rms_series",
        "get_spectrum",
        "get_data_quality",
    }:
        return (
            _specialized_outcome_is_empty(artifact)
            and _canonical_json(content)
            == _canonical_json(_generic_degraded_content(artifact))
        )
    if tool_name == "list_asset_analyses" and isinstance(
        artifact, AnalysisListToolArtifact
    ):
        analyses = outcome.analyses
        if analyses is None:
            return _canonical_json(content) == _canonical_json(
                _generic_degraded_content(artifact)
            )
        total = outcome.total_analyses
        if total is None:
            return False
        if not _degraded_flags_are_valid(outcome.partial_data):
            return False
        expected = DegradedAnalysisListModelContent(
            mode=outcome.mode,
            notes=outcome.notes,
            analyses=analyses[:20],
            total_analyses=total,
            returned_analyses=min(total, 20),
            omitted_analyses=max(total - 20, 0),
            truncated=total > 20,
            partial_data=outcome.partial_data,
        ).model_dump(mode="json")
        return _exact_model_wire(DegradedAnalysisListModelContent, content) == expected
    if tool_name == "get_model" and isinstance(artifact, ModelToolArtifact):
        model = outcome.model
        if (
            not _degraded_model_is_concrete(model)
            or not _degraded_flags_are_valid(outcome.partial_data)
        ):
            return False
        expected = DegradedModelContent(
            mode=outcome.mode,
            notes=outcome.notes,
            model=model,
            partial_data=outcome.partial_data,
        ).model_dump(mode="json")
        return _exact_model_wire(DegradedModelContent, content) == expected
    if tool_name == "search_knowledge" and isinstance(
        artifact, KnowledgeSearchToolArtifact
    ):
        results = outcome.results
        total = outcome.total_results
        if results is None or total is None:
            return False
        if (
            not _degraded_search_rows_are_concrete(results)
            or not _degraded_flags_are_valid(outcome.partial_data)
        ):
            return False
        expected = DegradedKnowledgeSearchModelContent(
            mode=outcome.mode,
            notes=outcome.notes,
            results=results,
            total_results=total,
            returned_results=len(results),
            omitted_results=max(total - len(results), 0),
            truncated=total > len(results),
            partial_data=outcome.partial_data,
        ).model_dump(mode="json")
        return _exact_model_wire(
            DegradedKnowledgeSearchModelContent,
            content,
        ) == expected
    if tool_name == "get_knowledge_document" and isinstance(
        artifact, KnowledgeDocumentToolArtifact
    ):
        document = outcome.document
        if (
            not _degraded_document_is_concrete(document)
            or not _degraded_flags_are_valid(outcome.partial_data)
        ):
            return False
        content_document = dict(document)
        body = content_document.get("body")
        if isinstance(body, str):
            total = len(body) + int(
                content_document.get("omitted_body_characters", 0)
            )
            trimmed = body[:8_000]
            content_document.update(
                body=trimmed,
                returned_body_characters=len(trimmed),
                omitted_body_characters=total - len(trimmed),
                truncated=total > len(trimmed),
            )
        expected = DegradedKnowledgeDocumentContent(
            mode=outcome.mode,
            notes=outcome.notes,
            document=content_document,
            partial_data=outcome.partial_data,
        ).model_dump(mode="json")
        return _exact_model_wire(
            DegradedKnowledgeDocumentContent,
            content,
        ) == expected
    return False


def _read_observation_is_semantically_valid(
    call: PersistedToolCall,
    observation: ToolObservation,
    request: SupportRequest | PersistedSupportRequest,
    *,
    configured_model_id: str | None,
) -> bool:
    try:
        artifact = observation.artifact.validated_read_artifact()
        arguments = call.arguments.to_python()
        content = observation.content.to_python() if observation.content else None
        if artifact is None or not isinstance(arguments, Mapping):
            return False
        expected_resource = _expected_read_resource(
            call.name,
            arguments,
            artifact.source.resource,
            configured_model_id=configured_model_id,
        )
        if artifact.source.resource != expected_resource:
            return False
        outcome = artifact.outcome
        if outcome.error is not None:
            return (
                outcome.mode is None
                and outcome.notes is None
                and outcome.partial_data is None
                and _specialized_outcome_is_empty(artifact)
                and _bounded_model_content_is_empty(artifact)
                and not artifact.truncated
                and artifact.omitted_items == 0
                and _canonical_json(content)
                == _canonical_json(
                    {"error": outcome.error.model_dump(mode="json")}
                )
            )
        if outcome.mode is ResponseMode.COMPLETE:
            complete_partial_data_is_valid = outcome.partial_data is None
            if call.name == "search_knowledge":
                complete_partial_data_is_valid = _degraded_flags_are_valid(
                    outcome.partial_data
                )
            return (
                complete_partial_data_is_valid
                and _complete_content_matches_artifact(
                    call.name,
                    content,
                    artifact,
                )
                and _artifact_scope_matches(
                    call,
                    artifact,
                    request,
                    configured_model_id=configured_model_id,
                )
                and _artifact_projection_metadata_matches(call.name, artifact)
            )
        if outcome.mode is None:
            return False
        if not _bounded_model_content_is_empty(artifact):
            return False
        if not _degraded_content_matches_artifact(
            call.name,
            content,
            artifact,
        ):
            return False
        if call.name in {
            "get_asset",
            "get_analysis",
            "get_baseline",
            "get_rms_series",
            "get_spectrum",
            "get_data_quality",
        }:
            if call.name == "get_asset":
                validate_degraded_asset_scope(
                    outcome.partial_data,
                    asset_id=request.asset_id,
                    company_id=request.identity.company_id,
                )
            if call.name == "get_analysis" and isinstance(
                outcome.partial_data,
                Mapping,
            ):
                expected_analysis_id = arguments.get("analysis_id")
                for field_name in ("id", "analysis_id"):
                    if field_name in outcome.partial_data and (
                        _validated_id(
                            outcome.partial_data[field_name],
                            _ANALYSIS_ID_ADAPTER,
                        )
                        != expected_analysis_id
                    ):
                        return False
            return (
                not artifact.truncated
                and artifact.omitted_items == 0
                and _recursive_scope_matches(
                    outcome.partial_data,
                    asset_id=request.asset_id,
                    point_id=arguments.get("point_id"),
                    company_id=request.identity.company_id,
                    allow_null_point=call.name in {"get_asset", "get_analysis"},
                )
            )
        if call.name == "list_asset_analyses" and isinstance(
            artifact,
            AnalysisListToolArtifact,
        ) and artifact.outcome.analyses is None:
            return (
                artifact.outcome.total_analyses is None
                and artifact.outcome.returned_analyses is None
                and artifact.outcome.omitted_analyses is None
                and not artifact.truncated
                and artifact.omitted_items == 0
                and _recursive_scope_matches(
                    outcome.partial_data,
                    asset_id=request.asset_id,
                    point_id=None,
                    company_id=request.identity.company_id,
                    allow_null_point=True,
                )
            )
        return _artifact_scope_matches(
            call,
            artifact,
            request,
            configured_model_id=configured_model_id,
        ) and _artifact_projection_metadata_matches(call.name, artifact)
    except (AttributeError, KeyError, TypeError, ValueError, ValidationError):
        return False


def _typed_ids(
    texts: Sequence[str],
    *,
    pattern: re.Pattern[str],
    adapter: TypeAdapter[str],
) -> frozenset[str]:
    validated: set[str] = set()
    for text in texts:
        for candidate in pattern.findall(text):
            try:
                validated.add(adapter.validate_python(candidate, strict=True))
            except ValidationError:
                continue
    return frozenset(validated)


@dataclass(frozen=True)
class _PlannerAuthorizedTargets:
    analysis_ids: frozenset[str]
    knowledge_document_ids: frozenset[str]
    model_ids: frozenset[str]
    point_ids: frozenset[str]


def _validated_id(value: object, adapter: TypeAdapter[str]) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return adapter.validate_python(value, strict=True)
    except ValidationError:
        return None


def _ids_from_rows(
    value: object,
    field_name: str,
    adapter: TypeAdapter[str],
) -> set[str]:
    if not isinstance(value, list):
        return set()
    validated: set[str] = set()
    for row in value:
        if not isinstance(row, Mapping):
            continue
        candidate = _validated_id(row.get(field_name), adapter)
        if candidate is not None:
            validated.add(candidate)
    return validated


def _structured_observation_ids(
    call: PersistedToolCall,
    observation: ToolObservation,
) -> _PlannerAuthorizedTargets:
    artifact = observation.artifact.validated_read_artifact()
    if artifact is None or artifact.outcome.error is not None:
        return _PlannerAuthorizedTargets(
            analysis_ids=frozenset(),
            knowledge_document_ids=frozenset(),
            model_ids=frozenset(),
            point_ids=frozenset(),
        )
    content = (
        observation.content.to_python()
        if observation.content is not None
        else None
    )
    if not isinstance(content, Mapping):
        return _PlannerAuthorizedTargets(
            analysis_ids=frozenset(),
            knowledge_document_ids=frozenset(),
            model_ids=frozenset(),
            point_ids=frozenset(),
        )

    analysis_ids: set[str] = set()
    knowledge_document_ids: set[str] = set()
    model_ids: set[str] = set()
    point_ids: set[str] = set()
    if call.name == "get_asset":
        point_ids.update(
            _ids_from_rows(content.get("points"), "id", _POINT_ID_ADAPTER)
        )
    elif call.name == "list_asset_analyses":
        analyses = content.get("analyses")
        analysis_ids.update(_ids_from_rows(analyses, "id", _ANALYSIS_ID_ADAPTER))
        point_ids.update(_ids_from_rows(analyses, "point_id", _POINT_ID_ADAPTER))
    elif call.name == "get_analysis":
        analysis_id = _validated_id(content.get("id"), _ANALYSIS_ID_ADAPTER)
        point_id = _validated_id(content.get("point_id"), _POINT_ID_ADAPTER)
        if analysis_id is not None:
            analysis_ids.add(analysis_id)
        if point_id is not None:
            point_ids.add(point_id)
    elif call.name == "search_knowledge":
        knowledge_document_ids.update(
            _ids_from_rows(
                content.get("results"),
                "id",
                _KNOWLEDGE_ID_ADAPTER,
            )
        )
    elif call.name == "get_knowledge_document":
        document = content.get("document")
        source = document if isinstance(document, Mapping) else content
        document_id = _validated_id(source.get("id"), _KNOWLEDGE_ID_ADAPTER)
        if document_id is not None:
            knowledge_document_ids.add(document_id)
    elif call.name == "get_model":
        model = content.get("model")
        source = model if isinstance(model, Mapping) else content
        model_id = _validated_id(source.get("id"), _MODEL_ID_ADAPTER)
        if model_id is not None:
            model_ids.add(model_id)
    elif call.name in _ASSET_ARGUMENT_TOOL_NAMES:
        point_id = _validated_id(content.get("point_id"), _POINT_ID_ADAPTER)
        if point_id is None and isinstance(content.get("partial_data"), Mapping):
            point_id = _validated_id(
                content["partial_data"].get("point_id"),
                _POINT_ID_ADAPTER,
            )
        if point_id is not None:
            point_ids.add(point_id)
    return _PlannerAuthorizedTargets(
        analysis_ids=frozenset(analysis_ids),
        knowledge_document_ids=frozenset(knowledge_document_ids),
        model_ids=frozenset(model_ids),
        point_ids=frozenset(point_ids),
    )


def _authorized_targets(
    request_message: str,
    interactions: Sequence[tuple[PersistedToolCall, ToolObservation]],
) -> _PlannerAuthorizedTargets:
    analysis_ids = set(
        _typed_ids(
            (request_message,),
            pattern=_ANALYSIS_ID_PATTERN,
            adapter=_ANALYSIS_ID_ADAPTER,
        )
    )
    knowledge_document_ids = set(
        _typed_ids(
            (request_message,),
            pattern=_KNOWLEDGE_ID_PATTERN,
            adapter=_KNOWLEDGE_ID_ADAPTER,
        )
    )
    model_ids = set(
        _typed_ids(
            (request_message,),
            pattern=_MODEL_ID_PATTERN,
            adapter=_MODEL_ID_ADAPTER,
        )
    )
    point_ids = set(
        _typed_ids(
            (request_message,),
            pattern=_POINT_ID_PATTERN,
            adapter=_POINT_ID_ADAPTER,
        )
    )
    for call, observation in interactions:
        observed = _structured_observation_ids(call, observation)
        analysis_ids.update(observed.analysis_ids)
        knowledge_document_ids.update(observed.knowledge_document_ids)
        model_ids.update(observed.model_ids)
        point_ids.update(observed.point_ids)
    return _PlannerAuthorizedTargets(
        analysis_ids=frozenset(analysis_ids),
        knowledge_document_ids=frozenset(knowledge_document_ids),
        model_ids=frozenset(model_ids),
        point_ids=frozenset(point_ids),
    )


def _selected_targets_are_authorized(
    *,
    tool_name: str,
    arguments: Mapping[str, object],
    request: SupportRequest | PersistedSupportRequest,
    authorized: _PlannerAuthorizedTargets,
) -> bool:
    if "model_id" in arguments or "configured_model_id" in arguments:
        return False
    if tool_name in _ANALYSIS_ARGUMENT_TOOL_NAMES:
        if arguments.get("analysis_id") not in authorized.analysis_ids:
            return False
    if tool_name == "get_knowledge_document":
        if arguments.get("document_id") not in authorized.knowledge_document_ids:
            return False
    if tool_name in _ASSET_ARGUMENT_TOOL_NAMES:
        if (
            request.asset_id is None
            or arguments.get("asset_id") != request.asset_id
        ):
            return False
    if tool_name in _POINT_ARGUMENT_TOOL_NAMES:
        point_id = arguments.get("point_id")
        if point_id is not None and point_id not in authorized.point_ids:
            return False
    return True


def _validated_current_interactions(
    request: SupportRequest | PersistedSupportRequest,
    request_id: str,
    tool_calls: Sequence[PersistedToolCall],
    tool_observations: Sequence[ToolObservation],
    *,
    usage: PlannerUsage | None = None,
    configured_model_id: str | None = None,
) -> tuple[tuple[PersistedToolCall, ToolObservation], ...]:
    calls = tuple(call for call in tool_calls if call.request_id == request_id)
    observations = tuple(
        observation
        for observation in tool_observations
        if observation.request_id == request_id
    )
    call_ids = tuple(call.call_id for call in calls)
    observation_ids = tuple(observation.call_id for observation in observations)
    if (
        len(calls) != len(observations)
        or len(call_ids) != len(set(call_ids))
        or len(observation_ids) != len(set(observation_ids))
    ):
        raise PlannerProtocolError(PlannerErrorCode.INVALID_HISTORY, usage=usage)
    interactions: list[tuple[PersistedToolCall, ToolObservation]] = []
    for call, observation in zip(calls, observations, strict=True):
        call_arguments = _canonical_catalog_arguments(
            call.name,
            call.arguments.to_python(),
        )
        artifact_arguments = _canonical_catalog_arguments(
            call.name,
            observation.artifact.arguments.to_python(),
        )
        if (
            observation.call_id != call.call_id
            or observation.content is None
            or observation.artifact.tool_name != call.name
            or call_arguments is None
            or artifact_arguments != call_arguments
        ):
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_HISTORY,
                usage=usage,
            )
        canonical_call = PersistedToolCall(
            request_id=call.request_id,
            call_id=call.call_id,
            name=call.name,
            arguments=call_arguments,
        )
        if not _read_observation_is_semantically_valid(
            canonical_call,
            observation,
            request,
            configured_model_id=configured_model_id,
        ):
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_HISTORY,
                usage=usage,
            )
        authorized = _authorized_targets(request.message, interactions)
        if not _selected_targets_are_authorized(
            tool_name=canonical_call.name,
            arguments=canonical_call.arguments.to_python(),
            request=request,
            authorized=authorized,
        ):
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_HISTORY,
                usage=usage,
            )
        interactions.append((canonical_call, observation))
    return tuple(interactions)


def select_planner_tools(
    state: AgentState,
    runtime: ReadToolRuntime,
) -> tuple[BaseTool, ...]:
    """Seleciona um subconjunto dos catálogos estáticos sem executar efeitos."""
    runtime_scope_matches = (
        runtime.identity == state.identity
        and runtime.permissions == state.permissions
        and (
            state.request.asset_id is None
            or runtime.central_asset_id == state.request.asset_id
        )
        and (
            not isinstance(runtime, WriteToolRuntime)
            or runtime.current_case_id == state.request.case_id
        )
    )
    if not runtime_scope_matches:
        raise PlannerProtocolError(PlannerErrorCode.RUNTIME_SCOPE_MISMATCH)
    authorized = _authorized_targets(
        state.request.message,
        _validated_current_interactions(
            state.request,
            state.request_id,
            state.tool_calls,
            state.tool_observations,
            configured_model_id=runtime.configured_model_id,
        ),
    )
    has_scoped_asset = (
        state.request.asset_id is not None
        and state.request.asset_id == runtime.central_asset_id
    )
    offered_reads: tuple[BaseTool, ...] = ()
    if "read" in state.permissions and "read" in runtime.permissions:
        offered_reads = tuple(
            tool
            for tool in READ_TOOLS
            if (tool.name not in _ASSET_ARGUMENT_TOOL_NAMES or has_scoped_asset)
            and (tool.name != "get_analysis" or authorized.analysis_ids)
            and (
                tool.name != "get_knowledge_document"
                or authorized.knowledge_document_ids
            )
        )
    if not isinstance(runtime, WriteToolRuntime):
        return offered_reads

    has_scoped_case = state.request.case_id == runtime.current_case_id
    proposal_requirements = {
        "propose_reprocess_analysis": (
            "action_low",
            bool(authorized.analysis_ids),
        ),
        "propose_request_specialist_analysis": (
            "action_low",
            bool(authorized.analysis_ids),
        ),
        "propose_update_asset_criticality": (
            "action_high",
            has_scoped_asset,
        ),
        "propose_request_model_retraining": (
            "action_high",
            runtime.configured_model_id in authorized.model_ids,
        ),
        "propose_escalate_case": (
            "escalate",
            has_scoped_case,
        ),
    }
    offered_proposals = tuple(
        tool
        for tool in WRITE_PROPOSAL_TOOLS
        if (
            (requirement := proposal_requirements[tool.name])[0]
            in state.permissions
            and requirement[0] in runtime.permissions
            and requirement[1]
        )
    )
    return (*offered_reads, *offered_proposals)


class PlannerContextStats(StrictModel):
    """Medição da representação realmente entregue à fronteira do modelo."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    characters: int = Field(ge=0)
    omitted_interactions: int = Field(ge=0)


class PlannerToolTurn(StrictModel):
    """Uma única chamada validada, pronta para entrar no estado."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["tool_call"] = "tool_call"
    tool_call: PersistedToolCall
    usage: PlannerUsage
    context: PlannerContextStats | None = None


class PlannerDecisionKind(str, Enum):
    GUIDE = "guide"
    REQUEST_INFORMATION = "request_information"
    REQUIRE_HUMAN_REVIEW = "require_human_review"


class PlannerStopReason(str, Enum):
    SUFFICIENT_EVIDENCE = "sufficient_evidence"
    MISSING_INFORMATION = "missing_information"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


class PlannerErrorCode(str, Enum):
    INVALID_SELECTION = "invalid_selection"
    INVALID_HISTORY = "invalid_history"
    DUPLICATE_TOOL_NAME = "duplicate_tool_name"
    UNKNOWN_TOOL = "unknown_tool"
    MULTIPLE_TOOL_CALLS = "multiple_tool_calls"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    INVALID_TERMINAL_OUTPUT = "invalid_terminal_output"
    INVALID_USAGE = "invalid_usage"
    SELECTION_LIMIT_EXCEEDED = "selection_limit_exceeded"
    TOOL_CALL_LIMIT_EXCEEDED = "tool_call_limit_exceeded"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    FINALIZATION_LIMIT_EXCEEDED = "finalization_limit_exceeded"
    CONTEXT_LIMIT_EXCEEDED = "context_limit_exceeded"
    RUNTIME_SCOPE_MISMATCH = "runtime_scope_mismatch"


class PlannerProtocolError(RuntimeError):
    """Falha fechada que não preserva a saída livre ou inválida do modelo."""

    def __init__(
        self,
        code: PlannerErrorCode,
        *,
        usage: PlannerUsage | None = None,
    ) -> None:
        self.code = code
        self.usage = usage
        super().__init__(f"planner protocol error: {code.value}")


def _context_character_count(
    messages: Sequence[BaseMessage],
    tool_wire: Sequence[dict[str, object]],
) -> int:
    return len(
        json.dumps(
            {
                "messages": convert_to_openai_messages(messages),
                "tools": tool_wire,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _build_planner_context(
    request_content: str,
    interactions: Sequence[tuple[AIMessage, ToolMessage, bool]],
    schemas: Sequence[BaseTool | type[StrictModel]],
    *,
    usage: PlannerUsage,
) -> tuple[list[BaseMessage], PlannerContextStats]:
    interaction_count = len(interactions)
    tool_wire = tuple(convert_to_openai_tool(schema) for schema in schemas)
    for omitted in range(interaction_count + 1):
        if any(interaction[2] for interaction in interactions[:omitted]):
            break
        marker = json.dumps(
            {"omitted_interactions": omitted},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        messages: list[BaseMessage] = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            SystemMessage(content=marker),
            HumanMessage(content=request_content),
        ]
        for assistant_message, tool_message, _ in interactions[omitted:]:
            messages.extend((assistant_message, tool_message))
        characters = _context_character_count(messages, tool_wire)
        if characters <= PLANNER_LIMITS.context_characters:
            return messages, PlannerContextStats(
                characters=characters,
                omitted_interactions=omitted,
            )
    raise PlannerProtocolError(
        PlannerErrorCode.CONTEXT_LIMIT_EXCEEDED,
        usage=usage,
    )


class PlannerTerminalDecision(StrictModel):
    """Decisão do planner; não contém texto final destinado ao cliente."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    decision: PlannerDecisionKind
    stop_reason: PlannerStopReason
    missing_information: str | None = Field(default=None, max_length=300)

    @field_validator("missing_information", mode="before")
    @classmethod
    def _normalize_missing_information(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("missing_information deve ser texto ou null")
        normalized = value.strip()
        if not normalized:
            raise ValueError("missing_information não pode ser vazio")
        return normalized

    @model_validator(mode="after")
    def _require_coherent_stop_contract(self) -> PlannerTerminalDecision:
        expected_reason = {
            PlannerDecisionKind.GUIDE: PlannerStopReason.SUFFICIENT_EVIDENCE,
            PlannerDecisionKind.REQUEST_INFORMATION: (
                PlannerStopReason.MISSING_INFORMATION
            ),
            PlannerDecisionKind.REQUIRE_HUMAN_REVIEW: (
                PlannerStopReason.HUMAN_REVIEW_REQUIRED
            ),
        }[self.decision]
        if self.stop_reason is not expected_reason:
            raise ValueError("stop_reason diverge da decisão terminal")
        requires_information = (
            self.decision is PlannerDecisionKind.REQUEST_INFORMATION
        )
        if requires_information != (self.missing_information is not None):
            raise ValueError("missing_information diverge da decisão terminal")
        return self


class PlannerDecisionTurn(StrictModel):
    """Encerramento validado da fatia do planner."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["decision"] = "decision"
    decision: PlannerTerminalDecision
    usage: PlannerUsage
    context: PlannerContextStats | None = None


class Planner:
    """Coordena as duas interfaces nativas do modelo sem executar tools."""

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    async def ainvoke(
        self,
        request: SupportRequest | PersistedSupportRequest,
        *,
        offered_tools: Sequence[BaseTool],
        request_id: str,
        usage: PlannerUsage,
        tool_calls: Sequence[PersistedToolCall] = (),
        tool_observations: Sequence[ToolObservation] = (),
    ) -> PlannerToolTurn | PlannerDecisionTurn:
        active_usage = usage
        if (
            not isinstance(request_id, str)
            or not request_id
            or not isinstance(active_usage, PlannerUsage)
        ):
            raise PlannerProtocolError(PlannerErrorCode.INVALID_USAGE)
        if active_usage.request_id != request_id:
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_USAGE,
                usage=active_usage,
            )
        if (
            active_usage.selection_count > PLANNER_LIMITS.selections
            or active_usage.finalization_count > PLANNER_LIMITS.finalizations
        ):
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_USAGE,
                usage=active_usage,
            )
        if active_usage.finalization_count == PLANNER_LIMITS.finalizations:
            raise PlannerProtocolError(
                PlannerErrorCode.FINALIZATION_LIMIT_EXCEEDED,
                usage=active_usage,
            )
        if active_usage.selection_count == PLANNER_LIMITS.selections:
            raise PlannerProtocolError(
                PlannerErrorCode.SELECTION_LIMIT_EXCEEDED,
                usage=active_usage,
            )
        tools = tuple(offered_tools)
        tool_names = tuple(tool.name for tool in tools)
        if len(tool_names) != len(set(tool_names)):
            raise PlannerProtocolError(
                PlannerErrorCode.DUPLICATE_TOOL_NAME,
                usage=active_usage,
            )
        tools_by_name = {tool.name: tool for tool in tools}
        request_content = json.dumps(
            {
                "case_id": request.case_id,
                "ticket_id": request.ticket_id,
                "asset_id": request.asset_id,
                "message": request.message,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        validated_interactions = _validated_current_interactions(
            request,
            request_id,
            tool_calls,
            tool_observations,
            usage=active_usage,
        )
        calls = tuple(call for call, _ in validated_interactions)
        observations = tuple(
            observation for _, observation in validated_interactions
        )
        call_ids = tuple(call.call_id for call in calls)
        call_fingerprints = tuple(_tool_call_fingerprint(call) for call in calls)
        if len(call_fingerprints) != len(set(call_fingerprints)):
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_HISTORY,
                usage=active_usage,
            )
        if len(calls) > PLANNER_LIMITS.tool_calls:
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_HISTORY,
                usage=active_usage,
            )
        if len(calls) == PLANNER_LIMITS.tool_calls:
            raise PlannerProtocolError(
                PlannerErrorCode.TOOL_CALL_LIMIT_EXCEEDED,
                usage=active_usage,
            )
        interactions: list[tuple[AIMessage, ToolMessage, bool]] = []
        authorized_interactions: list[
            tuple[PersistedToolCall, ToolObservation]
        ] = []
        for index, (call, observation) in enumerate(
            zip(calls, observations, strict=True)
        ):
            authorized_interactions.append((call, observation))
            interactions.append(
                (
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": call.name,
                                "args": call.arguments.to_python(),
                                "id": call.call_id,
                                "type": "tool_call",
                            }
                        ],
                    ),
                    ToolMessage(
                        content=observation.content.encoded,
                        tool_call_id=call.call_id,
                        name=call.name,
                    ),
                    (
                        index == len(calls) - 1
                        or observation.artifact.outcome.error is not None
                        or (
                            observation.artifact.outcome.mode is not None
                            and observation.artifact.outcome.mode.value != "complete"
                        )
                    ),
                )
            )
        authorized_targets = _authorized_targets(
            request.message,
            authorized_interactions,
        )
        messages, context_stats = _build_planner_context(
            request_content,
            interactions,
            tools,
            usage=active_usage,
        )
        selection = await self._model.bind_tools(tools).ainvoke(messages)
        selection_usage = PlannerUsage(
            request_id=active_usage.request_id,
            selection_count=active_usage.selection_count + 1,
            finalization_count=active_usage.finalization_count,
        )
        if not isinstance(selection, AIMessage):
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_SELECTION,
                usage=selection_usage,
            )
        if selection.invalid_tool_calls:
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_TOOL_ARGUMENTS,
                usage=selection_usage,
            )
        if not selection.tool_calls:
            terminal_messages, terminal_context_stats = _build_planner_context(
                request_content,
                interactions,
                (PlannerTerminalDecision,),
                usage=selection_usage,
            )
            final_usage = PlannerUsage(
                request_id=selection_usage.request_id,
                selection_count=selection_usage.selection_count,
                finalization_count=selection_usage.finalization_count + 1,
            )
            terminal_decision: PlannerTerminalDecision | None = None
            try:
                terminal_output = await self._model.with_structured_output(
                    PlannerTerminalDecision,
                    include_raw=False,
                ).ainvoke(terminal_messages)
                terminal_decision = PlannerTerminalDecision.model_validate(
                    terminal_output
                )
            except (TypeError, ValueError, ValidationError):
                pass
            if terminal_decision is None:
                raise PlannerProtocolError(
                    PlannerErrorCode.INVALID_TERMINAL_OUTPUT,
                    usage=final_usage,
                )
            return PlannerDecisionTurn(
                decision=terminal_decision,
                usage=final_usage,
                context=terminal_context_stats,
            )
        if len(selection.tool_calls) != 1:
            raise PlannerProtocolError(
                PlannerErrorCode.MULTIPLE_TOOL_CALLS,
                usage=selection_usage,
            )
        selected_call = selection.tool_calls[0]
        if selected_call["id"] in set(call_ids):
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_SELECTION,
                usage=selection_usage,
            )
        selected_tool = tools_by_name.get(selected_call["name"])
        if selected_tool is None:
            raise PlannerProtocolError(
                PlannerErrorCode.UNKNOWN_TOOL,
                usage=selection_usage,
            )
        tool_turn: PlannerToolTurn | None = None
        try:
            validated_arguments = selected_tool.tool_call_schema.model_validate(
                selected_call["args"]
            )
            arguments = validated_arguments.model_dump(mode="json")
            if _selected_targets_are_authorized(
                tool_name=selected_call["name"],
                arguments=arguments,
                request=request,
                authorized=authorized_targets,
            ):
                tool_turn = PlannerToolTurn(
                    tool_call=PersistedToolCall(
                        request_id=request_id,
                        call_id=selected_call["id"],
                        name=selected_call["name"],
                        arguments=arguments,
                    ),
                    usage=selection_usage,
                    context=context_stats,
                )
        except (TypeError, ValueError, ValidationError):
            pass
        if tool_turn is None:
            raise PlannerProtocolError(
                PlannerErrorCode.INVALID_TOOL_ARGUMENTS,
                usage=selection_usage,
            )
        selected_fingerprint = _tool_call_fingerprint(tool_turn.tool_call)
        if any(
            _tool_call_fingerprint(prior_call) == selected_fingerprint
            for prior_call in calls
        ):
            raise PlannerProtocolError(
                PlannerErrorCode.REPEATED_TOOL_CALL,
                usage=selection_usage,
            )
        return tool_turn


def _tool_call_fingerprint(call: PersistedToolCall) -> str:
    """Identifica intenção da tool sem depender do ID atribuído pelo provider."""
    return json.dumps(
        {
            "arguments": call.arguments.to_python(),
            "tool": call.name,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
