import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import PrivateAttr, ValidationError
import pytest

from tractian_agent.state import (
    AgentDecision,
    EvidenceConflict,
    EvidenceItem,
    EvidenceGap,
    EvidenceGapReason,
    EvidenceLedger,
    EvidenceQuality,
    EvidenceSourceKind,
    JsonSnapshot,
    WriterFailureCode,
    WriterFailureRecord,
)
from tractian_agent.contracts import ResponseMode
from tractian_agent.checkpoint import open_checkpointer
from tractian_agent.graph import build_agent_graph
from tractian_agent.planner import Planner
from tractian_agent.writer import (
    WRITER_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT_VERSION,
    Writer,
    WriterDraft,
    WriterNextStep,
    WriterProtocolError,
    build_writer_context,
)


class _RecordingWriterModel(BaseChatModel):
    response: object
    _messages: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _schemas: list[object] = PrivateAttr(default_factory=list)
    _bind_calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "recording-writer-model"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise AssertionError("o writer deve usar saída estruturada")

    def bind_tools(self, *args: Any, **kwargs: Any) -> RunnableLambda:
        self._bind_calls += 1
        raise AssertionError("o writer não pode receber tools")

    def with_structured_output(
        self,
        schema: object,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> RunnableLambda:
        assert include_raw is False
        self._schemas.append(schema)

        async def invoke(messages: list[BaseMessage]) -> object:
            self._messages.append(list(messages))
            return self.response

        return RunnableLambda(invoke)


class _SequenceWriterModel(_RecordingWriterModel):
    responses: tuple[object, ...]
    _response_index: int = PrivateAttr(default=0)

    def with_structured_output(
        self,
        schema: object,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> RunnableLambda:
        assert include_raw is False
        self._schemas.append(schema)

        async def invoke(messages: list[BaseMessage]) -> object:
            self._messages.append(list(messages))
            response = self.responses[self._response_index]
            self._response_index += 1
            if isinstance(response, Exception):
                raise response
            return response

        return RunnableLambda(invoke)


def _claimable_ledger() -> EvidenceLedger:
    return EvidenceLedger(
        request_id="req_writer_01",
        items=(
            EvidenceItem(
                evidence_id="sha256:v1:" + "a" * 64,
                request_id="req_writer_01",
                source_kind=EvidenceSourceKind.TOOL,
                call_id="call_writer_01",
                tool="get_asset",
                resource="/assets/asset_G501",
                fact_path="asset.criticality",
                value=JsonSnapshot.capture("high", forbidden_names=frozenset()),
                mode=ResponseMode.COMPLETE,
                source_at=datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
                recorded_at=datetime(2026, 9, 2, 12, 1, tzinfo=timezone.utc),
                quality=EvidenceQuality.CLAIMABLE,
            ),
        ),
    )


def test_writer_receives_only_minimal_typed_context_without_tools() -> None:
    evidence_id = "sha256:v1:" + "a" * 64
    base = _claimable_ledger()
    secret_item = base.items[0].model_copy(
        update={
            "resource": "/assets/asset_SECRET_TARGET",
            "value": JsonSnapshot.capture(
                "SENSITIVE_VALUE",
                forbidden_names=frozenset(),
            ),
        }
    )
    ledger = EvidenceLedger(request_id=base.request_id, items=(secret_item,))
    model = _RecordingWriterModel(
        response=WriterDraft(
            decision=AgentDecision.GUIDE,
            evidence_ids=(evidence_id,),
            limitation_refs=(),
            next_step=WriterNextStep.MONITOR,
        )
    )

    draft = asyncio.run(
        Writer(model).ainvoke(
            decision=AgentDecision.GUIDE,
            ledger=ledger,
            missing_information=None,
        )
    )

    assert draft == WriterDraft(
        decision=AgentDecision.GUIDE,
        evidence_ids=(evidence_id,),
        limitation_refs=(),
        next_step=WriterNextStep.MONITOR,
    )
    assert WRITER_SYSTEM_PROMPT_VERSION == "writer-v1"
    assert model._bind_calls == 0
    assert model._schemas == [WriterDraft]
    assert len(model._messages) == 1
    assert isinstance(model._messages[0][0], SystemMessage)
    assert model._messages[0][0].content == WRITER_SYSTEM_PROMPT
    assert json.loads(str(model._messages[0][1].content)) == {
        "decision": "guide",
        "facts": [
            {
                "evidence_id": evidence_id,
                "kind": "technical_fact",
            }
        ],
        "limitations": [],
        "missing_information": None,
    }
    complete_wire = repr(model._messages)
    assert "/assets/asset_SECRET_TARGET" not in complete_wire
    assert "SENSITIVE_VALUE" not in complete_wire


def test_writer_draft_is_strict_frozen_and_contains_no_free_text_fields() -> None:
    evidence_id = "sha256:v1:" + "a" * 64

    with pytest.raises(ValidationError):
        WriterDraft.model_validate(
            {
                "decision": "guide",
                "evidence_ids": [evidence_id],
                "limitation_refs": [],
                "next_step": "monitor",
            }
        )

    with pytest.raises(ValidationError, match="limitação"):
        WriterDraft(
            decision=AgentDecision.GUIDE,
            evidence_ids=(evidence_id,),
            limitation_refs=("referencia-inventada",),
            next_step=WriterNextStep.MONITOR,
        )


def test_writer_repairs_format_once_with_the_same_minimal_context() -> None:
    evidence_id = "sha256:v1:" + "a" * 64
    invalid = {
        "decision": "guide",
        "evidence_ids": [evidence_id],
        "limitation_refs": [],
        "next_step": "texto_livre",
    }
    valid = WriterDraft(
        decision=AgentDecision.GUIDE,
        evidence_ids=(evidence_id,),
        limitation_refs=(),
        next_step=WriterNextStep.MONITOR,
    )
    model = _SequenceWriterModel(response=None, responses=(invalid, valid))

    result = asyncio.run(
        Writer(model).ainvoke(
            decision=AgentDecision.GUIDE,
            ledger=_claimable_ledger(),
            missing_information=None,
        )
    )

    assert result == valid
    assert model._schemas == [WriterDraft, WriterDraft]
    assert len(model._messages) == 2
    assert model._messages[0] == model._messages[1]


def test_writer_repairs_validation_error_raised_by_structured_wrapper_once() -> None:
    evidence_id = "sha256:v1:" + "a" * 64
    with pytest.raises(ValidationError) as invalid:
        WriterDraft.model_validate(
            {
                "decision": "guide",
                "evidence_ids": [evidence_id],
                "limitation_refs": [],
                "next_step": "texto_livre",
            }
        )
    valid = WriterDraft(
        decision=AgentDecision.GUIDE,
        evidence_ids=(evidence_id,),
        limitation_refs=(),
        next_step=WriterNextStep.MONITOR,
    )
    model = _SequenceWriterModel(
        response=None,
        responses=(invalid.value, valid),
    )

    result = asyncio.run(
        Writer(model).ainvoke(
            decision=AgentDecision.GUIDE,
            ledger=_claimable_ledger(),
            missing_information=None,
        )
    )

    assert result == valid
    assert len(model._messages) == 2
    assert model._messages[0] == model._messages[1]


def test_writer_single_attempt_drops_the_invalid_validation_error_graph() -> None:
    evidence_id = "sha256:v1:" + "a" * 64
    with pytest.raises(ValidationError) as invalid:
        WriterDraft.model_validate(
            {
                "decision": "guide",
                "evidence_ids": [evidence_id],
                "limitation_refs": [],
                "next_step": "RAW_SECRET_OUTPUT",
            }
        )
    model = _SequenceWriterModel(
        response=None,
        responses=(invalid.value,),
    )

    with pytest.raises(WriterProtocolError) as captured:
        asyncio.run(
            Writer(model).ainvoke_once(
                decision=AgentDecision.GUIDE,
                ledger=_claimable_ledger(),
                missing_information=None,
            )
        )

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "RAW_SECRET_OUTPUT" not in "".join(
        traceback.format_exception(captured.value)
    )
    current = captured.value.__traceback__
    while current is not None:
        if current.tb_frame.f_code.co_name == "ainvoke_once":
            retained = dict(current.tb_frame.f_locals)
            assert "RAW_SECRET_OUTPUT" not in repr(retained)
        current = current.tb_next


def test_writer_stops_after_two_invalid_formats_without_leaking_output() -> None:
    invalid = {"decision": "guide", "technical_text": "RAW_SECRET_OUTPUT"}
    model = _SequenceWriterModel(
        response=None,
        responses=(invalid, invalid),
    )

    with pytest.raises(WriterProtocolError) as captured:
        asyncio.run(
            Writer(model).ainvoke(
                decision=AgentDecision.GUIDE,
                ledger=_claimable_ledger(),
                missing_information=None,
            )
        )

    assert captured.value.attempts == 2
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert len(model._messages) == 2
    assert model._messages[0] == model._messages[1]
    exception_wire = "".join(traceback.format_exception(captured.value))
    chained_wire: list[str] = []
    current: BaseException | None = captured.value
    while current is not None:
        chained_wire.extend((str(current), repr(current)))
        current = current.__cause__ or current.__context__
    assert "RAW_SECRET_OUTPUT" not in exception_wire
    assert "RAW_SECRET_OUTPUT" not in "".join(chained_wire)
    assert "RAW_SECRET_OUTPUT" not in repr(model._messages)
    current_traceback = captured.value.__traceback__
    while current_traceback is not None:
        if current_traceback.tb_frame.f_globals.get("__name__") == (
            "tractian_agent.writer"
        ):
            assert "RAW_SECRET_OUTPUT" not in repr(
                dict(current_traceback.tb_frame.f_locals)
            )
        current_traceback = current_traceback.tb_next


def test_writer_does_not_retry_a_provider_failure_as_format_repair() -> None:
    model = _SequenceWriterModel(
        response=None,
        responses=(RuntimeError("RAW_SECRET_OUTPUT"),),
    )

    with pytest.raises(WriterProtocolError, match="model_failure") as captured:
        asyncio.run(
            Writer(model).ainvoke(
                decision=AgentDecision.GUIDE,
                ledger=_claimable_ledger(),
                missing_information=None,
            )
        )

    assert captured.value.attempts == 1
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "RAW_SECRET_OUTPUT" not in "".join(
        traceback.format_exception(captured.value)
    )
    current_traceback = captured.value.__traceback__
    while current_traceback is not None:
        if current_traceback.tb_frame.f_globals.get("__name__") == (
            "tractian_agent.writer"
        ):
            assert "RAW_SECRET_OUTPUT" not in repr(
                dict(current_traceback.tb_frame.f_locals)
            )
        current_traceback = current_traceback.tb_next
    assert len(model._messages) == 1


@pytest.mark.parametrize(
    ("code", "repairable"),
    [
        (WriterFailureCode.MODEL_FAILURE, True),
        (WriterFailureCode.INVALID_STRUCTURED_OUTPUT, False),
    ],
)
def test_writer_failure_record_cannot_change_retry_semantics(
    code: WriterFailureCode,
    repairable: bool,
) -> None:
    with pytest.raises(ValidationError, match="repairable"):
        WriterFailureRecord(
            code=code,
            attempts=1,
            repairable=repairable,
        )

    with pytest.raises(ValidationError):
        WriterFailureRecord(
            code=WriterFailureCode.INVALID_STRUCTURED_OUTPUT,
            attempts=1,
            repairable="true",
        )


def test_writer_context_assigns_stable_ids_to_current_limitations() -> None:
    base = _claimable_ledger()
    limited_item = base.items[0].model_copy(
        update={"limitations": ("janela de amostragem reduzida",)}
    )
    ledger = EvidenceLedger(
        request_id=base.request_id,
        items=(limited_item,),
        gaps=(
            EvidenceGap(
                reason=EvidenceGapReason.PARTIAL,
                request_id=base.request_id,
                call_id="call_writer_01",
                fact_path="asset.criticality",
            ),
        ),
    )

    first = build_writer_context(
        decision=AgentDecision.GUIDE,
        ledger=ledger,
        missing_information=None,
    )
    second = build_writer_context(
        decision=AgentDecision.GUIDE,
        ledger=ledger,
        missing_information=None,
    )

    assert first == second
    assert len(first.limitations) == 2
    refs = tuple(item.limitation_ref for item in first.limitations)
    assert refs == tuple(sorted(refs))
    assert len(set(refs)) == 2
    assert all(ref.startswith("limitation:v1:") for ref in refs)
    assert {item.kind for item in first.limitations} == {
        "evidence_limitation",
        "gap",
    }
    assert all(
        set(item.model_dump(mode="json")) == {"limitation_ref", "kind"}
        for item in first.limitations
    )
    assert "janela de amostragem reduzida" not in first.model_dump_json()
    assert "asset.criticality" not in first.model_dump_json()


def test_writer_projection_has_a_deterministic_reference_limit() -> None:
    base = _claimable_ledger()
    items = tuple(
        base.items[0].model_copy(
            update={
                "evidence_id": f"sha256:v1:{index:064x}",
                "call_id": f"call_writer_{index:03d}",
            }
        )
        for index in range(70)
    )

    context = build_writer_context(
        decision=AgentDecision.GUIDE,
        ledger=EvidenceLedger(request_id=base.request_id, items=items),
        missing_information=None,
    )

    assert len(context.facts) == 63
    assert len(context.facts) + len(context.limitations) == 64
    assert {item.kind for item in context.limitations} == {
        "projection_overflow"
    }


@pytest.mark.parametrize(
    (
        "fact_count",
        "limitation_count",
        "expected_facts",
        "expected_limitations",
        "overflow",
    ),
    [
        (63, 0, 63, 0, False),
        (64, 0, 64, 0, False),
        (65, 0, 63, 1, True),
        (63, 1, 63, 1, False),
        (63, 2, 63, 1, True),
        (64, 64, 63, 1, True),
        (32, 33, 32, 32, True),
    ],
)
def test_writer_projection_shares_one_total_budget_across_reference_kinds(
    fact_count,
    limitation_count,
    expected_facts,
    expected_limitations,
    overflow,
):
    base = _claimable_ledger()
    items = tuple(
        base.items[0].model_copy(
            update={
                "evidence_id": f"sha256:v1:{index:064x}",
                "call_id": f"call_budget_{index:03d}",
            }
        )
        for index in range(fact_count)
    )
    gaps = tuple(
        EvidenceGap(
            reason=EvidenceGapReason.PARTIAL,
            request_id=base.request_id,
            call_id=f"call_gap_{index:03d}",
            fact_path="asset.criticality",
        )
        for index in range(limitation_count)
    )

    context = build_writer_context(
        decision=AgentDecision.GUIDE,
        ledger=EvidenceLedger(
            request_id=base.request_id,
            items=items,
            gaps=gaps,
        ),
        missing_information=None,
    )

    assert len(context.facts) == expected_facts
    assert len(context.limitations) == expected_limitations
    assert len(context.facts) + len(context.limitations) <= 64
    assert (
        any(item.kind == "projection_overflow" for item in context.limitations)
        is overflow
    )


def test_conflict_limitation_id_covers_every_conflicting_evidence_reference() -> None:
    base = _claimable_ledger()
    first_item = base.items[0]
    second_item = first_item.model_copy(
        update={
            "evidence_id": "sha256:v1:" + "b" * 64,
            "value": JsonSnapshot.capture("low", forbidden_names=frozenset()),
        }
    )
    third_item = first_item.model_copy(
        update={
            "evidence_id": "sha256:v1:" + "c" * 64,
            "value": JsonSnapshot.capture("medium", forbidden_names=frozenset()),
        }
    )

    def conflict_ref(items: tuple[EvidenceItem, ...]) -> str:
        evidence_ids = tuple(sorted(item.evidence_id for item in items))
        ledger = EvidenceLedger(
            request_id=base.request_id,
            items=items,
            conflicts=(
                EvidenceConflict(
                    canonical_key=first_item.canonical_key,
                    evidence_ids=evidence_ids,
                ),
            ),
        )
        context = build_writer_context(
            decision=AgentDecision.GUIDE,
            ledger=ledger,
            missing_information=None,
        )
        limitation = next(
            item for item in context.limitations if item.kind == "conflict"
        )
        return limitation.limitation_ref

    pair_ref = conflict_ref((first_item, second_item))
    triple_ref = conflict_ref((first_item, second_item, third_item))

    assert pair_ref != triple_ref


def test_conflict_limitation_hash_defensively_sorts_and_deduplicates_ids() -> None:
    base = _claimable_ledger()
    first_id = "sha256:v1:" + "a" * 64
    second_id = "sha256:v1:" + "b" * 64

    def limitation_ref(evidence_ids: tuple[str, ...]) -> str:
        conflict = EvidenceConflict.model_construct(
            canonical_key=base.items[0].canonical_key,
            evidence_ids=evidence_ids,
            blocking=True,
        )
        ledger = EvidenceLedger.model_construct(
            request_id=base.request_id,
            items=base.items,
            gaps=(),
            conflicts=(conflict,),
        )
        context = build_writer_context(
            decision=AgentDecision.GUIDE,
            ledger=ledger,
            missing_information=None,
        )
        return next(
            item.limitation_ref
            for item in context.limitations
            if item.kind == "conflict"
        )

    assert limitation_ref((first_id, second_id)) == limitation_ref(
        (second_id, first_id, second_id)
    )


def test_planner_and_writer_are_required_as_separate_builder_dependencies(
    tmp_path: Path,
) -> None:
    model = _RecordingWriterModel(
        response=WriterDraft(
            decision=AgentDecision.GUIDE,
            evidence_ids=(),
            limitation_refs=(),
            next_step=WriterNextStep.MONITOR,
        )
    )

    async def scenario() -> None:
        async with open_checkpointer(tmp_path / "writer-builder.sqlite3") as saver:
            with pytest.raises(ValueError, match="planner.*writer"):
                build_agent_graph(saver, planner=Planner(model))
            with pytest.raises(ValueError, match="planner.*writer"):
                build_agent_graph(saver, writer=Writer(model))

    asyncio.run(scenario())
