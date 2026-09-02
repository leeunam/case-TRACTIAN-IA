from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from tractian_agent.contracts import ResponseMode
from tractian_agent.contracts import ApiError, ApiErrorCategory
from tractian_agent.evidence import (
    EvidenceGapReason,
    EvidenceQuality,
    EvidenceSufficiency,
    assess_evidence,
    compile_observations,
)
from tractian_agent.state import ToolObservation
from tractian_agent.tools.technical import (
    BaselineArtifact,
    BaselineToolArtifact,
    BaselineToolOutcome,
    DataQualityArtifact,
    DataQualityToolArtifact,
    DataQualityToolOutcome,
)
from tractian_agent.tools.analyses import (
    AnalysisArtifact,
    AnalysisDetailToolArtifact,
    AnalysisDetailToolOutcome,
)
from tractian_agent.tools.observations import ToolSource


RECORDED_AT = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)


def _quality_observation(
    *,
    call_id: str = "call_01",
    mode: ResponseMode = ResponseMode.COMPLETE,
    completeness: float = 0.98,
    stale: bool = False,
    truncated: bool = False,
    error: ApiError | None = None,
) -> ToolObservation:
    outcome = DataQualityToolOutcome(mode=mode, error=error)
    if error is None and mode is ResponseMode.COMPLETE:
        outcome = outcome.model_copy(
            update={
                "data_quality": DataQualityArtifact(
                    asset_id="asset_G501", point_id=None, completeness=completeness,
                    freshness_minutes=2, snr_db=24.5, staleness_flag=stale,
                )
            }
        )
    if error is None and mode is not ResponseMode.COMPLETE:
        outcome = outcome.model_copy(update={"partial_data": {"available": False}})
    return ToolObservation(
        request_id="req_01", call_id=call_id,
        artifact=DataQualityToolArtifact(
            tool_name="get_data_quality",
            arguments={"asset_id": "asset_G501", "point_id": None},
            source=ToolSource(kind="industrial_api", resource="/assets/asset_G501/data-quality"),
            outcome=outcome, truncated=truncated,
        ),
    )


def test_complete_typed_observation_compiles_claimable_facts_with_provenance():
    observation = ToolObservation(
        request_id="req_01",
        call_id="call_01",
        artifact=DataQualityToolArtifact(
            tool_name="get_data_quality",
            arguments={"asset_id": "asset_G501", "point_id": None},
            source=ToolSource(
                kind="industrial_api", resource="/assets/asset_G501/data-quality"
            ),
            outcome=DataQualityToolOutcome(
                mode=ResponseMode.COMPLETE,
                data_quality=DataQualityArtifact(
                    asset_id="asset_G501",
                    point_id=None,
                    completeness=0.98,
                    freshness_minutes=2,
                    snr_db=24.5,
                    staleness_flag=False,
                ),
            ),
        ),
    )

    ledger = compile_observations(
        (observation,),
        recorded_at=RECORDED_AT,
    )

    item = next(item for item in ledger.items if item.fact_path == "data_quality.completeness")
    assert item.request_id == "req_01"
    assert item.call_id == "call_01"
    assert item.resource == "/assets/asset_G501/data-quality"
    assert item.value.to_python() == 0.98
    assert item.quality is EvidenceQuality.CLAIMABLE
    assert item.evidence_id.startswith("sha256:v1:")


def test_error_produces_only_a_sanitized_blocking_gap():
    ledger = compile_observations((
        _quality_observation(error=ApiError(
            category=ApiErrorCategory.TIMEOUT, code="READ_TIMEOUT", message="segredo externo"
        )),
    ), recorded_at=RECORDED_AT)

    assert ledger.items == ()
    assert ledger.gaps[0].reason is EvidenceGapReason.ERROR
    assert "segredo externo" not in ledger.gaps[0].model_dump_json()


def test_partial_and_truncated_observations_are_preserved_but_not_claimable():
    ledger = compile_observations((
        _quality_observation(mode=ResponseMode.PARTIAL),
        _quality_observation(call_id="call_02", truncated=True),
    ), recorded_at=RECORDED_AT)

    assert {gap.reason for gap in ledger.gaps} == {
        EvidenceGapReason.PARTIAL, EvidenceGapReason.TRUNCATED
    }
    assert ledger.items
    assert all(item.quality is EvidenceQuality.PARTIAL for item in ledger.items)
    assert assess_evidence(ledger).status is EvidenceSufficiency.INSUFFICIENT


def test_unavailable_inconclusive_and_conflict_modes_create_blocking_gaps():
    ledger = compile_observations(tuple(
        _quality_observation(call_id=f"call_{mode.value}", mode=mode)
        for mode in (ResponseMode.UNAVAILABLE, ResponseMode.INCONCLUSIVE, ResponseMode.CONFLICT)
    ), recorded_at=RECORDED_AT)

    assert {gap.reason for gap in ledger.gaps} == {
        EvidenceGapReason.UNAVAILABLE, EvidenceGapReason.INCONCLUSIVE, EvidenceGapReason.CONFLICT,
    }


def test_data_quality_staleness_is_explicit_obsolescence_not_a_ttl():
    ledger = compile_observations((_quality_observation(stale=True),), recorded_at=RECORDED_AT)

    assert {gap.reason for gap in ledger.gaps} == {EvidenceGapReason.OBSOLETE}
    assert ledger.items
    assert all(item.quality is EvidenceQuality.OBSOLETE for item in ledger.items)


def test_only_explicit_persisted_expiration_marks_a_fact_obsolete():
    observation = _quality_observation()

    current = compile_observations((observation,), recorded_at=RECORDED_AT)
    expired = compile_observations(
        (observation,), recorded_at=RECORDED_AT, expired_call_ids=frozenset({"call_01"})
    )

    assert all(item.quality is EvidenceQuality.CLAIMABLE for item in current.items)
    assert all(item.quality is EvidenceQuality.OBSOLETE for item in expired.items)
    assert {gap.reason for gap in expired.gaps} == {EvidenceGapReason.OBSOLETE}


def test_analysis_stale_and_invalidated_baseline_are_explicitly_obsolete():
    analysis = ToolObservation(
        request_id="req_01", call_id="call_analysis",
        artifact=AnalysisDetailToolArtifact(
            tool_name="get_analysis", arguments={"analysis_id": "an_9906"},
            source=ToolSource(kind="industrial_api", resource="/analyses/an_9906"),
            outcome=AnalysisDetailToolOutcome(
                mode=ResponseMode.COMPLETE,
                analysis=AnalysisArtifact(
                    id="an_9906", asset_id="asset_G501", point_id="pt_001",
                    type="bearing_fault", detection_mode="symptom", severity="high",
                    confidence=0.9, baseline_state_at_detection="established", evidence=[],
                    limitations=[], model_version="v1", created_at="2026-09-01T10:00:00+00:00",
                    status="stale",
                ),
            ),
        ),
    )
    baseline = ToolObservation(
        request_id="req_01", call_id="call_baseline",
        artifact=BaselineToolArtifact(
            tool_name="get_baseline", arguments={"asset_id": "asset_G501", "point_id": None},
            source=ToolSource(kind="industrial_api", resource="/assets/asset_G501/baseline"),
            outcome=BaselineToolOutcome(
                mode=ResponseMode.COMPLETE,
                baseline=BaselineArtifact(
                    id="base_001", asset_id="asset_G501", point_id="pt_001",
                    state="invalidated", detection_mode="baseline", learnable=True,
                    established_at="2026-08-01T10:00:00+00:00",
                    invalidated_at="2026-09-01T10:00:00+00:00",
                    invalidation_reason="manutenção", features=[], alarm_threshold=None,
                ),
            ),
        ),
    )

    ledger = compile_observations((analysis, baseline), recorded_at=RECORDED_AT)

    assert {gap.reason for gap in ledger.gaps} == {EvidenceGapReason.OBSOLETE}
    assert all(item.quality is EvidenceQuality.OBSOLETE for item in ledger.items)
    assert next(item for item in ledger.items if item.fact_path == "analysis.status").source_at is not None


def test_divergent_sources_create_conflict_and_identical_repeat_is_deduplicated():
    first = _quality_observation(call_id="call_first", completeness=0.98)
    divergent = _quality_observation(call_id="call_second", completeness=0.61)

    conflict_ledger = compile_observations((first, divergent), recorded_at=RECORDED_AT)
    repeated_ledger = compile_observations((first, first), recorded_at=RECORDED_AT)

    assert len(conflict_ledger.conflicts) == 1
    assert conflict_ledger.conflicts[0].canonical_key.endswith("data_quality.completeness")
    assert assess_evidence(conflict_ledger).causes == (EvidenceGapReason.CONFLICT,)
    assert len(repeated_ledger.items) == len(compile_observations((first,), recorded_at=RECORDED_AT).items)


def test_assessment_requires_claimable_fact_and_ledger_round_trips_as_frozen_json():
    ledger = compile_observations((_quality_observation(),), recorded_at=RECORDED_AT)

    assessment = assess_evidence(ledger)
    restored = type(ledger).model_validate_json(ledger.model_dump_json())

    assert assessment.status is EvidenceSufficiency.SUFFICIENT
    assert assessment.causes == ()
    assert restored == ledger
    with pytest.raises(ValidationError):
        ledger.items[0].quality = EvidenceQuality.PARTIAL


def test_legacy_observation_without_request_provenance_never_becomes_claimable():
    legacy = _quality_observation().model_copy(update={"request_id": None})

    ledger = compile_observations((legacy,), recorded_at=RECORDED_AT)

    assert ledger.items == ()
    assert assess_evidence(ledger).causes == (
        EvidenceGapReason.MISSING_PROVENANCE,
        EvidenceGapReason.NO_CLAIMABLE_FACT,
    )
