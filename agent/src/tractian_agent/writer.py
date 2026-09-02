"""Writer mínimo: seleciona referências, nunca cria afirmações técnicas."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Final, Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ConfigDict, Field, JsonValue, ValidationError

from tractian_agent.contracts import StrictModel
from tractian_agent.state import (
    AgentDecision,
    EvidenceItem,
    EvidenceLedger,
    WriterDraft,
    WriterNextStep,
)


WRITER_SYSTEM_PROMPT_VERSION: Final = "writer-v1"
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
    fact_path: str
    resource: str
    source: str
    source_at: str | None
    source_kind: Literal["tool", "action"]
    value: JsonValue


class WriterLimitation(_WriterModel):
    limitation_ref: str = Field(pattern=r"^limitation:v1:[0-9a-f]{64}$")
    kind: Literal[
        "evidence_limitation",
        "quality",
        "obsolescence",
        "gap",
        "conflict",
    ]
    source_ref: str = Field(min_length=1, pattern=r"^\S+$")
    reason: str | None = None
    detail: str | None = None


class WriterContext(_WriterModel):
    decision: AgentDecision
    facts: tuple[WriterFact, ...] = ()
    limitations: tuple[WriterLimitation, ...] = ()
    missing_information: str | None = None


def _utc_wire(value: datetime | None) -> str | None:
    if value is None:
        return None
    normalized = value.astimezone(timezone.utc).isoformat()
    return normalized.replace("+00:00", "Z")


def _writer_fact(item: EvidenceItem) -> WriterFact:
    source = item.tool if item.tool is not None else item.action
    assert source is not None
    return WriterFact(
        evidence_id=item.evidence_id,
        fact_path=item.fact_path,
        resource=item.resource,
        source=source,
        source_at=_utc_wire(item.source_at),
        source_kind=item.source_kind.value,
        value=item.value.to_python(),
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


def _writer_limitations(ledger: EvidenceLedger) -> tuple[WriterLimitation, ...]:
    values: list[WriterLimitation] = []

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
            WriterLimitation(
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
        encoded_sources = json.dumps(
            conflict.evidence_ids,
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


def build_writer_context(
    *,
    decision: AgentDecision,
    ledger: EvidenceLedger,
    missing_information: str | None,
) -> WriterContext:
    """Projeta a allowlist exata entregue ao modelo do writer."""
    facts = tuple(
        _writer_fact(item)
        for item in sorted(ledger.items, key=lambda candidate: candidate.evidence_id)
        if item.claimable
    )
    return WriterContext(
        decision=decision,
        facts=facts,
        limitations=_writer_limitations(ledger),
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
        last_error: WriterProtocolError | None = None
        for attempt in (1, 2):
            try:
                return await self.ainvoke_once(
                    decision=decision,
                    ledger=ledger,
                    missing_information=missing_information,
                )
            except WriterProtocolError as error:
                last_error = error
                if attempt == 2:
                    break
        assert last_error is not None
        raise WriterProtocolError(
            last_error.code,
            attempts=2,
        ) from last_error

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
        try:
            output = await self._model.with_structured_output(
                WriterDraft,
                include_raw=False,
            ).ainvoke(messages)
        except (OutputParserException, ValidationError) as error:
            raise WriterProtocolError(
                "invalid_structured_output",
                attempts=1,
            ) from error
        try:
            return WriterDraft.model_validate(output)
        except (TypeError, ValueError, ValidationError) as error:
            raise WriterProtocolError(
                "invalid_structured_output",
                attempts=1,
            ) from error


__all__ = [
    "WRITER_SYSTEM_PROMPT",
    "WRITER_SYSTEM_PROMPT_VERSION",
    "Writer",
    "WriterContext",
    "WriterDraft",
    "WriterFact",
    "WriterLimitation",
    "WriterNextStep",
    "WriterProtocolError",
    "build_writer_context",
]
