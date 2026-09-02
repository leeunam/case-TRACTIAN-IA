"""Ledger determinístico de evidências derivadas de observações tipadas.

O módulo recebe apenas ``ToolObservation`` persistida.  Em particular, ele não
usa ``content``: esse campo serve ao próximo turno do planner e nunca é uma
fonte de fatos para o ledger.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Literal

from pydantic import Field, JsonValue, field_validator

from tractian_agent.contracts import ResponseMode
from tractian_agent.state import FrozenStateModel, JsonSnapshot, ToolObservation


class EvidenceQuality(str, Enum):
    CLAIMABLE = "claimable"
    PARTIAL = "partial"
    OBSOLETE = "obsolete"


class EvidenceGapReason(str, Enum):
    ERROR = "error"
    MISSING_PROVENANCE = "missing_provenance"
    UNVALIDATED_ARTIFACT = "unvalidated_artifact"
    MISSING_RESPONSE_MODE = "missing_response_mode"
    PARTIAL = "partial"
    TRUNCATED = "truncated"
    UNAVAILABLE = "unavailable"
    INCONCLUSIVE = "inconclusive"
    CONFLICT = "conflict"
    OBSOLETE = "obsolete"
    NO_CLAIMABLE_FACT = "no_claimable_fact"


class EvidenceObsolescenceReason(str, Enum):
    ANALYSIS_STALE = "analysis_status_stale"
    BASELINE_INVALIDATED = "baseline_state_invalidated"
    DATA_QUALITY_STALE = "data_quality_staleness_flag"
    RECEIPT_OR_INTENT_EXPIRED = "receipt_or_intent_expired"


class EvidenceSufficiency(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class EvidenceItem(FrozenStateModel):
    """Um fato atômico com a proveniência necessária para auditá-lo."""

    evidence_id: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    request_id: str = Field(min_length=1, pattern=r"^\S+$")
    call_id: str = Field(min_length=1, pattern=r"^\S+$")
    tool: str = Field(min_length=1, pattern=r"^\S+$")
    resource: str = Field(min_length=1, pattern=r"^/")
    fact_path: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
    value: JsonSnapshot
    mode: ResponseMode
    source_at: datetime | None = None
    recorded_at: datetime
    limitations: tuple[str, ...] = ()
    quality: EvidenceQuality
    obsolescence: tuple[EvidenceObsolescenceReason, ...] = ()

    @field_validator("source_at", "recorded_at")
    @classmethod
    def _require_aware_time(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("instantes do ledger exigem timezone")
        return value

    @property
    def canonical_key(self) -> str:
        return f"{self.tool}:{self.resource}:{self.fact_path}"

    @property
    def claimable(self) -> bool:
        return self.quality is EvidenceQuality.CLAIMABLE


class EvidenceGap(FrozenStateModel):
    reason: EvidenceGapReason
    request_id: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    call_id: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    fact_path: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$",
    )
    blocking: Literal[True] = True


class EvidenceConflict(FrozenStateModel):
    canonical_key: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=2)
    blocking: Literal[True] = True


class EvidenceLedger(FrozenStateModel):
    items: tuple[EvidenceItem, ...] = ()
    gaps: tuple[EvidenceGap, ...] = ()
    conflicts: tuple[EvidenceConflict, ...] = ()


class EvidenceAssessment(FrozenStateModel):
    status: EvidenceSufficiency
    causes: tuple[EvidenceGapReason, ...] = ()

    @property
    def sufficient(self) -> bool:
        return self.status is EvidenceSufficiency.SUFFICIENT


_OUTCOME_METADATA = frozenset({"mode", "notes", "partial_data", "error"})
_SOURCE_TIME_KEYS = ("created_at", "established_at", "invalidated_at")


def _canonical_json(value: JsonValue) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _evidence_id(
    *, request_id: str, call_id: str, tool: str, resource: str, fact_path: str,
    value: JsonValue, mode: ResponseMode, source_at: datetime | None,
    limitations: tuple[str, ...], quality: EvidenceQuality,
    obsolescence: tuple[EvidenceObsolescenceReason, ...],
) -> str:
    payload: JsonValue = {
        "request_id": request_id, "call_id": call_id, "tool": tool,
        "resource": resource, "fact_path": fact_path, "value": value,
        "mode": mode.value, "source_at": source_at.isoformat() if source_at else None,
        "limitations": list(limitations), "quality": quality.value,
        "obsolescence": [reason.value for reason in obsolescence],
    }
    return "sha256:v1:" + hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _parse_source_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None


def _source_at(value: Mapping[str, JsonValue]) -> datetime | None:
    for key in _SOURCE_TIME_KEYS:
        parsed = _parse_source_time(value.get(key))
        if parsed is not None:
            return parsed
    for nested in value.values():
        if isinstance(nested, Mapping):
            parsed = _source_at(nested)
            if parsed is not None:
                return parsed
    return None


def _limitations(value: Mapping[str, JsonValue], notes: object, *, truncated: bool) -> tuple[str, ...]:
    found: list[str] = []
    def visit(node: JsonValue) -> None:
        if isinstance(node, Mapping):
            limitations = node.get("limitations")
            if isinstance(limitations, list):
                found.extend(item for item in limitations if isinstance(item, str) and item.strip())
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
    visit(value)
    if isinstance(notes, str) and notes.strip():
        found.append(notes)
    if truncated:
        found.append("resultado truncado")
    return tuple(dict.fromkeys(found))


def _obsolescence(value: Mapping[str, JsonValue], *, expired: bool) -> tuple[EvidenceObsolescenceReason, ...]:
    reasons: list[EvidenceObsolescenceReason] = []
    outcome = value.get("outcome")
    if not isinstance(outcome, Mapping):
        return ()
    analysis = outcome.get("analysis")
    baseline = outcome.get("baseline")
    data_quality = outcome.get("data_quality")
    if isinstance(analysis, Mapping) and analysis.get("status") == "stale":
        reasons.append(EvidenceObsolescenceReason.ANALYSIS_STALE)
    if isinstance(baseline, Mapping) and baseline.get("state") == "invalidated":
        reasons.append(EvidenceObsolescenceReason.BASELINE_INVALIDATED)
    if isinstance(data_quality, Mapping) and data_quality.get("staleness_flag") is True:
        reasons.append(EvidenceObsolescenceReason.DATA_QUALITY_STALE)
    if expired:
        reasons.append(EvidenceObsolescenceReason.RECEIPT_OR_INTENT_EXPIRED)
    return tuple(reasons)


def _leaves(value: JsonValue, prefix: str = "") -> Iterable[tuple[str, JsonValue]]:
    if isinstance(value, Mapping):
        for key in sorted(value):
            child = value[key]
            child_path = f"{prefix}.{key}" if prefix else key
            yield from _leaves(child, child_path)
    elif isinstance(value, list):
        # Lista é um fato íntegro; índices instáveis não viram caminhos de API.
        yield prefix, value
    elif value is not None:
        yield prefix, value


def _gap(reason: EvidenceGapReason, observation: ToolObservation, fact_path: str | None = None) -> EvidenceGap:
    return EvidenceGap(reason=reason, request_id=observation.request_id, call_id=observation.call_id, fact_path=fact_path)


def compile_observations(
    observations: Sequence[ToolObservation], *, recorded_at: datetime,
    expired_call_ids: frozenset[str] = frozenset(),
) -> EvidenceLedger:
    """Compila somente artefatos reidratados e deduplica fatos idênticos."""
    if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
        raise ValueError("recorded_at exige timezone")
    items: list[EvidenceItem] = []
    gaps: list[EvidenceGap] = []
    for observation in observations:
        if observation.request_id is None:
            gaps.append(_gap(EvidenceGapReason.MISSING_PROVENANCE, observation))
            continue
        artifact = observation.artifact.validated_read_artifact()
        if artifact is None:
            gaps.append(_gap(EvidenceGapReason.UNVALIDATED_ARTIFACT, observation))
            continue
        wire = artifact.model_dump(mode="json")
        outcome = wire["outcome"]
        if not isinstance(outcome, Mapping):
            gaps.append(_gap(EvidenceGapReason.UNVALIDATED_ARTIFACT, observation))
            continue
        if outcome.get("error") is not None:
            gaps.append(_gap(EvidenceGapReason.ERROR, observation))
            continue
        raw_mode = outcome.get("mode")
        if raw_mode is None:
            gaps.append(_gap(EvidenceGapReason.MISSING_RESPONSE_MODE, observation))
            continue
        mode = ResponseMode(raw_mode)
        if mode is ResponseMode.UNAVAILABLE:
            gaps.append(_gap(EvidenceGapReason.UNAVAILABLE, observation))
        elif mode is ResponseMode.INCONCLUSIVE:
            gaps.append(_gap(EvidenceGapReason.INCONCLUSIVE, observation))
        elif mode is ResponseMode.CONFLICT:
            gaps.append(_gap(EvidenceGapReason.CONFLICT, observation))
        elif mode is ResponseMode.PARTIAL:
            gaps.append(_gap(EvidenceGapReason.PARTIAL, observation))
        if artifact.truncated:
            gaps.append(_gap(EvidenceGapReason.TRUNCATED, observation))
        obsolete = _obsolescence(wire, expired=observation.call_id in expired_call_ids)
        if obsolete:
            gaps.append(_gap(EvidenceGapReason.OBSOLETE, observation))
        quality = (
            EvidenceQuality.OBSOLETE if obsolete else
            EvidenceQuality.CLAIMABLE if mode is ResponseMode.COMPLETE and not artifact.truncated else
            EvidenceQuality.PARTIAL
        )
        source_at = _source_at(outcome)
        limitations = _limitations(outcome, outcome.get("notes"), truncated=artifact.truncated)
        payloads: list[tuple[str, JsonValue]] = [
            (key, value) for key, value in outcome.items() if key not in _OUTCOME_METADATA and value is not None
        ]
        partial_data = outcome.get("partial_data")
        if isinstance(partial_data, (dict, list)):
            payloads.append(("partial_data", partial_data))
        for root, payload in payloads:
            for fact_path, value in _leaves(payload, root):
                evidence_id = _evidence_id(
                    request_id=observation.request_id, call_id=observation.call_id,
                    tool=artifact.tool_name, resource=artifact.source.resource,
                    fact_path=fact_path, value=value, mode=mode, source_at=source_at,
                    limitations=limitations, quality=quality, obsolescence=obsolete,
                )
                items.append(EvidenceItem(
                    evidence_id=evidence_id, request_id=observation.request_id,
                    call_id=observation.call_id, tool=artifact.tool_name,
                    resource=artifact.source.resource, fact_path=fact_path,
                    value=JsonSnapshot.capture(value, forbidden_names=frozenset()),
                    mode=mode, source_at=source_at, recorded_at=recorded_at,
                    limitations=limitations, quality=quality, obsolescence=obsolete,
                ))
    unique_items = tuple({item.evidence_id: item for item in items}.values())
    unique_gaps = tuple(
        {gap.model_dump_json(): gap for gap in gaps}.values()
    )
    groups: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in unique_items:
        groups[item.canonical_key].append(item)
    conflicts = tuple(
        EvidenceConflict(canonical_key=key, evidence_ids=tuple(item.evidence_id for item in group))
        for key, group in sorted(groups.items())
        if len({_canonical_json(item.value.to_python()) for item in group}) > 1
    )
    return EvidenceLedger(items=unique_items, gaps=unique_gaps, conflicts=conflicts)


def assess_evidence(ledger: EvidenceLedger) -> EvidenceAssessment:
    """Aplica a regra conservadora de suficiência sem carregar texto externo."""
    causes = {gap.reason for gap in ledger.gaps}
    if ledger.conflicts:
        causes.add(EvidenceGapReason.CONFLICT)
    if not any(item.claimable for item in ledger.items):
        causes.add(EvidenceGapReason.NO_CLAIMABLE_FACT)
    if causes:
        return EvidenceAssessment(status=EvidenceSufficiency.INSUFFICIENT, causes=tuple(sorted(causes, key=lambda cause: cause.value)))
    return EvidenceAssessment(status=EvidenceSufficiency.SUFFICIENT)


# Nomes explícitos para os consumidores do writer/gate que serão integrados
# nas próximas tarefas. Ambos mantêm o compilador puro acima como única fonte.
compile_evidence = compile_observations
compile_ledger = compile_observations
