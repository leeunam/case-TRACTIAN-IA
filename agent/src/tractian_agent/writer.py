"""Writer mínimo: seleciona referências, nunca cria afirmações técnicas."""

from __future__ import annotations

import hashlib
import json
from typing import Final, Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ConfigDict, Field, ValidationError

from tractian_agent.contracts import StrictModel
from tractian_agent.state import (
    AgentDecision,
    EvidenceItem,
    EvidenceLedger,
    WriterDraft,
    WriterNextStep,
)


WRITER_SYSTEM_PROMPT_VERSION: Final = "writer-v1"
WRITER_MAX_FACT_REFERENCES: Final = 64
WRITER_MAX_LIMITATION_REFERENCES: Final = 64
WRITER_SYSTEM_PROMPT: Final = """writer-v1
Você é o writer do atendimento industrial. Receba somente a decisão já tomada,
fatos canônicos e limitações selecionados em código. Não altere a decisão, não
crie fatos, valores, tools, permissões nem efeitos. Devolva apenas o contrato
estruturado: repita a decisão, os IDs ordenados recebidos, as referências de
limitação ordenadas e escolha um próximo passo enumerado. Não redija prosa.
""".strip()


class WriterProtocolError(RuntimeError):
    """Falha sanitizada; nunca carrega a saída inválida do modelo."""

    def __init__(self, code: str, *, attempts: int) -> None:
        self.code = code
        self.attempts = attempts
        super().__init__(f"writer protocol error: {code}")


class _WriterModel(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class WriterFact(_WriterModel):
    evidence_id: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    kind: Literal["technical_fact", "action_accepted"]


class WriterLimitation(_WriterModel):
    limitation_ref: str = Field(pattern=r"^limitation:v1:[0-9a-f]{64}$")
    kind: Literal[
        "evidence_limitation",
        "quality",
        "obsolescence",
        "gap",
        "conflict",
        "projection_overflow",
    ]


class _CanonicalLimitation(_WriterModel):
    limitation_ref: str = Field(pattern=r"^limitation:v1:[0-9a-f]{64}$")
    kind: Literal[
        "evidence_limitation",
        "quality",
        "obsolescence",
        "gap",
        "conflict",
        "projection_overflow",
    ]
    source_ref: str = Field(min_length=1, pattern=r"^\S+$")
    reason: str | None = None
    detail: str | None = None


class WriterContext(_WriterModel):
    decision: AgentDecision
    facts: tuple[WriterFact, ...] = ()
    limitations: tuple[WriterLimitation, ...] = ()
    missing_information: str | None = None


def _writer_fact(item: EvidenceItem) -> WriterFact:
    return WriterFact(
        evidence_id=item.evidence_id,
        kind=(
            "technical_fact"
            if item.source_kind.value == "tool"
            else "action_accepted"
        ),
    )


def _limitation_ref(
    *,
    request_id: str | None,
    kind: str,
    source_ref: str,
    reason: str | None,
    detail: str | None,
) -> str:
    payload = {
        "detail": detail,
        "kind": kind,
        "reason": reason,
        "request_id": request_id,
        "source_ref": source_ref,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"limitation:v1:{hashlib.sha256(encoded).hexdigest()}"


def _canonical_limitations(
    ledger: EvidenceLedger,
) -> tuple[_CanonicalLimitation, ...]:
    values: list[_CanonicalLimitation] = []

    def append(
        *,
        kind: Literal[
            "evidence_limitation",
            "quality",
            "obsolescence",
            "gap",
            "conflict",
        ],
        source_ref: str,
        reason: str | None = None,
        detail: str | None = None,
    ) -> None:
        values.append(
            _CanonicalLimitation(
                limitation_ref=_limitation_ref(
                    request_id=ledger.request_id,
                    kind=kind,
                    source_ref=source_ref,
                    reason=reason,
                    detail=detail,
                ),
                kind=kind,
                source_ref=source_ref,
                reason=reason,
                detail=detail,
            )
        )

    for item in ledger.items:
        for detail in item.limitations:
            append(
                kind="evidence_limitation",
                source_ref=item.evidence_id,
                detail=detail,
            )
        if not item.claimable:
            append(
                kind="quality",
                source_ref=item.evidence_id,
                reason=item.quality.value,
            )
        for reason in item.obsolescence:
            append(
                kind="obsolescence",
                source_ref=item.evidence_id,
                reason=reason.value,
            )
    for gap in ledger.gaps:
        source_ref = (
            f"call:{gap.call_id}"
            if gap.call_id is not None
            else f"intent:{gap.intent_id}"
            if gap.intent_id is not None
            else "request"
        )
        append(
            kind="gap",
            source_ref=source_ref,
            reason=gap.reason.value,
            detail=gap.fact_path,
        )
    for conflict in ledger.conflicts:
        canonical_ids = tuple(sorted(set(conflict.evidence_ids)))
        encoded_sources = json.dumps(
            canonical_ids,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        append(
            kind="conflict",
            source_ref=(
                "conflict:"
                f"{hashlib.sha256(encoded_sources).hexdigest()}"
            ),
            reason="conflict",
            detail=conflict.canonical_key,
        )
    by_ref = {value.limitation_ref: value for value in values}
    return tuple(by_ref[key] for key in sorted(by_ref))


def limitation_descriptions(ledger: EvidenceLedger) -> dict[str, str]:
    """Resolve conteúdo canônico em código; nada disso é enviado ao modelo."""
    return {
        limitation.limitation_ref: (
            limitation.detail or limitation.reason or limitation.kind
        )
        for limitation in _canonical_limitations(ledger)
    }


def build_writer_context(
    *,
    decision: AgentDecision,
    ledger: EvidenceLedger,
    missing_information: str | None,
) -> WriterContext:
    """Projeta a allowlist exata entregue ao modelo do writer."""
    all_facts = tuple(
        _writer_fact(item)
        for item in sorted(ledger.items, key=lambda candidate: candidate.evidence_id)
        if item.claimable
    )
    canonical_limitations = _canonical_limitations(ledger)
    projection_overflow = (
        len(all_facts) > WRITER_MAX_FACT_REFERENCES
        or len(canonical_limitations) > WRITER_MAX_LIMITATION_REFERENCES
    )
    limitation_budget = (
        WRITER_MAX_LIMITATION_REFERENCES - 1
        if projection_overflow
        else WRITER_MAX_LIMITATION_REFERENCES
    )
    limitations = [
        WriterLimitation(
            limitation_ref=limitation.limitation_ref,
            kind=limitation.kind,
        )
        for limitation in canonical_limitations[:limitation_budget]
    ]
    if projection_overflow:
        limitations.append(
            WriterLimitation(
                limitation_ref=_limitation_ref(
                    request_id=ledger.request_id,
                    kind="projection_overflow",
                    source_ref="writer_projection",
                    reason="reference_limit",
                    detail=(
                        f"facts:{len(all_facts)};"
                        f"limitations:{len(canonical_limitations)}"
                    ),
                ),
                kind="projection_overflow",
            )
        )
        limitations.sort(key=lambda value: value.limitation_ref)
    return WriterContext(
        decision=decision,
        facts=all_facts[:WRITER_MAX_FACT_REFERENCES],
        limitations=tuple(limitations),
        missing_information=missing_information,
    )


class Writer:
    """Invoca somente a interface estruturada do modelo, sem catálogo de tools."""

    def __init__(self, model: BaseChatModel) -> None:
        if not isinstance(model, BaseChatModel):
            raise TypeError("writer exige BaseChatModel")
        self._model = model

    async def ainvoke(
        self,
        *,
        decision: AgentDecision,
        ledger: EvidenceLedger,
        missing_information: str | None,
    ) -> WriterDraft:
        failure_code: str | None = None
        for attempt in (1, 2):
            try:
                return await self.ainvoke_once(
                    decision=decision,
                    ledger=ledger,
                    missing_information=missing_information,
                )
            except WriterProtocolError as error:
                failure_code = error.code
            if failure_code == "model_failure":
                raise WriterProtocolError(
                    failure_code,
                    attempts=attempt,
                ) from None
        assert failure_code is not None
        raise WriterProtocolError(
            failure_code,
            attempts=2,
        ) from None

    async def ainvoke_once(
        self,
        *,
        decision: AgentDecision,
        ledger: EvidenceLedger,
        missing_information: str | None,
    ) -> WriterDraft:
        """Executa uma tentativa; o grafo persiste cada reparo separadamente."""
        context = build_writer_context(
            decision=decision,
            ledger=ledger,
            missing_information=missing_information,
        )
        messages = [
            SystemMessage(content=WRITER_SYSTEM_PROMPT),
            HumanMessage(
                content=json.dumps(
                    context.model_dump(mode="json"),
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            ),
        ]
        output: object = None
        format_invalid = False
        model_failed = False
        try:
            output = await self._model.with_structured_output(
                WriterDraft,
                include_raw=False,
            ).ainvoke(messages)
        except (OutputParserException, ValidationError):
            format_invalid = True
        except Exception:
            model_failed = True
        if format_invalid:
            output = None
            raise WriterProtocolError(
                "invalid_structured_output",
                attempts=1,
            ) from None
        if model_failed:
            output = None
            raise WriterProtocolError(
                "model_failure",
                attempts=1,
            ) from None
        try:
            return WriterDraft.model_validate(output)
        except (TypeError, ValueError, ValidationError):
            format_invalid = True
        if format_invalid:
            output = None
            raise WriterProtocolError(
                "invalid_structured_output",
                attempts=1,
            ) from None
        raise AssertionError("validação do writer terminou sem resultado")


__all__ = [
    "WRITER_SYSTEM_PROMPT",
    "WRITER_SYSTEM_PROMPT_VERSION",
    "WRITER_MAX_FACT_REFERENCES",
    "WRITER_MAX_LIMITATION_REFERENCES",
    "Writer",
    "WriterContext",
    "WriterDraft",
    "WriterFact",
    "WriterLimitation",
    "WriterNextStep",
    "WriterProtocolError",
    "build_writer_context",
    "limitation_descriptions",
]
