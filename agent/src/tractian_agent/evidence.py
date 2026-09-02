"""Ledger determinístico de evidências derivadas de observações tipadas.

O módulo recebe apenas ``ToolObservation`` persistida.  Em particular, ele não
usa ``content``: esse campo serve ao próximo turno do planner e nunca é uma
fonte de fatos para o ledger.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
import hashlib
import json
import re
from pydantic import JsonValue

from tractian_agent.contracts import ResponseMode
from tractian_agent.state import (
    EvidenceAssessment,
    EvidenceConflict,
    EvidenceGap,
    EvidenceGapReason,
    EvidenceItem,
    EvidenceLedger,
    EvidenceObsolescenceReason,
    EvidenceQuality,
    EvidenceSourceKind,
    EvidenceSufficiency,
    JsonSnapshot,
    ToolObservation,
)
from tractian_agent.write_contracts import IntentStatus, WriteIntent


_OUTCOME_METADATA = frozenset({"mode", "notes", "partial_data", "error"})
_SOURCE_TIME_KEYS = (
    "created_at",
    "established_at",
    "invalidated_at",
    "ts",
    "collected_at",
    "last_run_at",
)


def _canonical_json(value: JsonValue) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _evidence_id(
    *, request_id: str, source_kind: EvidenceSourceKind, call_id: str | None = None,
    intent_id: str | None = None, tool: str | None = None, action: str | None = None,
    resource: str, fact_path: str,
    value: JsonValue, mode: ResponseMode, source_at: datetime | None,
    limitations: tuple[str, ...], quality: EvidenceQuality,
    obsolescence: tuple[EvidenceObsolescenceReason, ...],
) -> str:
    payload: JsonValue = {
        "request_id": request_id, "source_kind": source_kind.value,
        "call_id": call_id, "intent_id": intent_id, "tool": tool, "action": action,
        "resource": resource, "fact_path": fact_path, "value": value,
        "mode": mode.value, "source_at": source_at.isoformat() if source_at else None,
        "limitations": list(limitations), "quality": quality.value,
        "obsolescence": [reason.value for reason in obsolescence],
    }
    return "sha256:v1:" + hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def canonical_evidence_id(item: EvidenceItem) -> str:
    """Recompõe o ID público de um item sem confiar no ID persistido."""
    return _evidence_id(
        request_id=item.request_id,
        source_kind=item.source_kind,
        call_id=item.call_id,
        intent_id=item.intent_id,
        tool=item.tool,
        action=item.action,
        resource=item.resource,
        fact_path=item.fact_path,
        value=item.value.to_python(),
        mode=item.mode,
        source_at=item.source_at,
        limitations=item.limitations,
        quality=item.quality,
        obsolescence=item.obsolescence,
    )


def _parse_source_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None


def _source_at(value: JsonValue) -> datetime | None:
    if isinstance(value, Mapping):
        for key in _SOURCE_TIME_KEYS:
            parsed = _parse_source_time(value.get(key))
            if parsed is not None:
                return parsed
        for nested in value.values():
            parsed = _source_at(nested)
            if parsed is not None:
                return parsed
    elif isinstance(value, list):
        for nested in value:
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
    analyses = outcome.get("analyses")
    baseline = outcome.get("baseline")
    data_quality = outcome.get("data_quality")
    if isinstance(analysis, Mapping) and analysis.get("status") == "stale":
        reasons.append(EvidenceObsolescenceReason.ANALYSIS_STALE)
    if (
        isinstance(analyses, list)
        and any(
            isinstance(item, Mapping) and item.get("status") == "stale"
            for item in analyses
        )
    ):
        reasons.append(EvidenceObsolescenceReason.ANALYSIS_STALE)
    if isinstance(baseline, Mapping) and baseline.get("state") == "invalidated":
        reasons.append(EvidenceObsolescenceReason.BASELINE_INVALIDATED)
    if isinstance(data_quality, Mapping) and data_quality.get("staleness_flag") is True:
        reasons.append(EvidenceObsolescenceReason.DATA_QUALITY_STALE)
    if expired:
        reasons.append(EvidenceObsolescenceReason.RECEIPT_OR_INTENT_EXPIRED)
    return tuple(reasons)


def _canonical_segment(key: str) -> str:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key)
    segments = re.findall(r"[A-Za-z0-9]+", separated)
    normalized = "_".join(segment.casefold() for segment in segments)
    if not normalized or not normalized[0].isalpha():
        return "key_" + hashlib.sha256(key.encode()).hexdigest()[:12]
    return normalized


def _canonical_mapping_segments(value: Mapping[str, JsonValue]) -> dict[str, str]:
    """Normaliza chaves JSON sem perder irmãs que colidem após normalização."""
    bases = {key: _canonical_segment(key) for key in value}
    counts: dict[str, int] = defaultdict(int)
    for base in bases.values():
        counts[base] += 1
    return {
        key: base if counts[base] == 1 else f"{base}__key_{hashlib.sha256(key.encode()).hexdigest()[:12]}"
        for key, base in bases.items()
    }


def _leaves(value: JsonValue, prefix: str = "") -> Iterable[tuple[str, JsonValue]]:
    if isinstance(value, Mapping):
        segments = _canonical_mapping_segments(value)
        for key in sorted(value):
            child = value[key]
            segment = segments[key]
            child_path = f"{prefix}.{segment}" if prefix else segment
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
                    source_kind=EvidenceSourceKind.TOOL, tool=artifact.tool_name,
                    resource=artifact.source.resource,
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
        EvidenceConflict(
            canonical_key=key,
            evidence_ids=tuple(sorted({item.evidence_id for item in group})),
        )
        for key, group in sorted(groups.items())
        if len({_canonical_json(item.value.to_python()) for item in group}) > 1
    )
    request_ids = {
        value
        for value in (
            *(item.request_id for item in unique_items),
            *(gap.request_id for gap in unique_gaps),
        )
        if value is not None
    }
    return EvidenceLedger(
        request_id=next(iter(request_ids)) if len(request_ids) == 1 else None,
        items=unique_items,
        gaps=unique_gaps,
        conflicts=conflicts,
    )


def compile_action_intents(
    intents: Sequence[WriteIntent], *, recorded_at: datetime,
) -> EvidenceLedger:
    """Projeta somente o efeito terminal tipado; proposal e mensagem são excluídos."""
    if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
        raise ValueError("recorded_at exige timezone")
    items: list[EvidenceItem] = []
    gaps: list[EvidenceGap] = []
    for intent in intents:
        if intent.request_id is None or intent.status not in {
            IntentStatus.COMPLETED,
            IntentStatus.FAILED,
            IntentStatus.UNCERTAIN,
        }:
            continue
        if intent.status is IntentStatus.COMPLETED and intent.receipt is not None:
            action = intent.scope.action
            resource = f"/actions/{intent.receipt.action_id}"
            evidence_id = _evidence_id(
                request_id=intent.request_id,
                source_kind=EvidenceSourceKind.ACTION,
                intent_id=intent.intent_id,
                action=action,
                resource=resource,
                fact_path="accepted",
                value=True,
                mode=ResponseMode.COMPLETE,
                source_at=None,
                limitations=(),
                quality=EvidenceQuality.CLAIMABLE,
                obsolescence=(),
            )
            items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    request_id=intent.request_id,
                    source_kind=EvidenceSourceKind.ACTION,
                    intent_id=intent.intent_id,
                    action=action,
                    resource=resource,
                    fact_path="accepted",
                    value=JsonSnapshot.capture(True, forbidden_names=frozenset()),
                    mode=ResponseMode.COMPLETE,
                    recorded_at=recorded_at,
                    quality=EvidenceQuality.CLAIMABLE,
                )
            )
            continue
        gaps.append(
            EvidenceGap(
                reason=(
                    EvidenceGapReason.UNAVAILABLE
                    if intent.status is IntentStatus.UNCERTAIN
                    else EvidenceGapReason.ERROR
                ),
                request_id=intent.request_id,
                intent_id=intent.intent_id,
            )
        )
    request_ids = {
        value
        for value in (
            *(item.request_id for item in items),
            *(gap.request_id for gap in gaps),
        )
        if value is not None
    }
    return EvidenceLedger(
        request_id=next(iter(request_ids)) if len(request_ids) == 1 else None,
        items=tuple(items),
        gaps=tuple(gaps),
    )


def merge_ledgers(*ledgers: EvidenceLedger) -> EvidenceLedger:
    """Une deltas determinísticos, preservando o primeiro instante registrado."""
    items: dict[str, EvidenceItem] = {}
    gaps: dict[str, EvidenceGap] = {}
    for ledger in ledgers:
        for item in ledger.items:
            items.setdefault(item.evidence_id, item)
        for gap in ledger.gaps:
            gaps.setdefault(gap.model_dump_json(), gap)
    groups: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in items.values():
        groups[item.canonical_key].append(item)
    conflicts = tuple(
        EvidenceConflict(
            canonical_key=key,
            evidence_ids=tuple(sorted({item.evidence_id for item in group})),
        )
        for key, group in sorted(groups.items())
        if len({_canonical_json(item.value.to_python()) for item in group}) > 1
    )
    request_ids = {
        item.request_id for item in items.values()
    } | {gap.request_id for gap in gaps.values() if gap.request_id is not None}
    return EvidenceLedger(
        request_id=next(iter(request_ids)) if len(request_ids) == 1 else None,
        items=tuple(items.values()),
        gaps=tuple(gaps.values()),
        conflicts=conflicts,
    )


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
