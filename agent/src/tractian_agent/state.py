"""Contratos JSON-safe do estado persistível do agente."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import math
import re
from typing import Annotated, Final, Literal, TypeAlias, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

from tractian_agent.contracts import (
    ApiErrorCategory,
    ResponseMode,
    StrictModel,
    SupportRequest,
    ToolCall,
)
from tractian_agent.tools.analyses import (
    AnalysisDetailToolArtifact,
    AnalysisListToolArtifact,
)
from tractian_agent.tools.assets import AssetToolArtifact
from tractian_agent.tools.knowledge import (
    KnowledgeDocumentToolArtifact,
    KnowledgeSearchToolArtifact,
    ModelToolArtifact,
)
from tractian_agent.tools.observations import ToolArtifact, assert_safe_partial_json
from tractian_agent.tools.runtime import Permission, TrustedIdentity
from tractian_agent.tools.technical import (
    BaselineToolArtifact,
    DataQualityToolArtifact,
    RmsToolArtifact,
    SpectrumToolArtifact,
)
from tractian_agent.write_contracts import (
    IntentStatus,
    PersistedApiError,
    WriteIntent,
    WriteIntentScope,
    approval_matches_write_intent,
    proposal_matches_intent_scope,
)
from tractian_agent.write_policy import (
    PolicyDecision,
    TrustedActionApproval,
    TrustedWriteContext,
    WriteProposal,
)


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _proposal_matches_persisted_intent(
    proposal: WriteProposal,
    scope: WriteIntentScope,
    *,
    request: PersistedSupportRequest,
    payload_hash: str,
    trusted_context: TrustedWriteContext | None,
) -> bool:
    """Vincula o efeito terminal à proposal sem recuperar runtime no estado."""
    if trusted_context is None:
        return False
    return (
        trusted_context.current_case_id == request.case_id
        and (
            request.asset_id is None
            or trusted_context.central_asset_id == request.asset_id
        )
        and proposal_matches_intent_scope(
            proposal,
            scope,
            payload_hash=payload_hash,
            trusted_context=trusted_context,
        )
    )


def _key_segments(value: str) -> frozenset[str]:
    """Segmenta aliases snake/kebab/camel sem recorrer a substring livre."""
    separated_acronyms = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
    separated_camel = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        separated_acronyms,
    )
    return frozenset(
        segment.casefold()
        for segment in re.findall(r"[A-Za-z0-9]+", separated_camel)
    )


_TECHNICAL_FORBIDDEN_NAMES = frozenset(
    {
        "client",
        "transport",
        "runtime",
        "authorization",
        "apitoken",
        "token",
        "apikey",
        "credential",
        "credentials",
        "password",
        "secret",
        "cookie",
        "golden",
        "goldenset",
        "eval",
        "evaluation",
        "expectedpaths",
        "testscenarios",
        "evaluationseed",
        "seed",
        "rawhttpresponse",
        "reasoningtrace",
    }
)

_TECHNICAL_FORBIDDEN_SEGMENTS = frozenset(
    {
        "token",
        "password",
        "credential",
        "credentials",
        "authorization",
        "secret",
        "cookie",
        "evaluation",
        "eval",
    }
)

_TECHNICAL_FORBIDDEN_SEGMENT_PATTERNS = (
    frozenset({"access", "token"}),
    frozenset({"api", "token"}),
    frozenset({"api", "key"}),
    frozenset({"client", "secret"}),
    frozenset({"http", "response"}),
    frozenset({"response", "body"}),
    frozenset({"reasoning", "trace"}),
    frozenset({"golden", "set"}),
    frozenset({"expected", "path"}),
    frozenset({"expected", "paths"}),
    frozenset({"test", "scenario"}),
    frozenset({"test", "scenarios"}),
    frozenset({"evaluation", "seed"}),
)

_PUBLIC_ARGUMENT_FORBIDDEN_NAMES = _TECHNICAL_FORBIDDEN_NAMES | {
    "caseid",
    "companyid",
    "userid",
    "identity",
    "permissions",
    "approval",
    "threadid",
    "requestid",
    "executionid",
    "idempotencykey",
    "centralassetid",
    "configuredmodelid",
    "context",
    "url",
    "method",
    "header",
    "headers",
}

_PUBLIC_ARGUMENT_FORBIDDEN_SEGMENTS = _TECHNICAL_FORBIDDEN_SEGMENTS | {
    "identity",
    "permissions",
    "approval",
}

_PUBLIC_ARGUMENT_FORBIDDEN_SEGMENT_PATTERNS = (
    *_TECHNICAL_FORBIDDEN_SEGMENT_PATTERNS,
    frozenset({"trusted", "identity"}),
    frozenset({"action", "approval"}),
    frozenset({"agent", "thread", "id"}),
    frozenset({"thread", "id"}),
    frozenset({"request", "id"}),
    frozenset({"execution", "id"}),
    frozenset({"idempotency", "key"}),
    frozenset({"central", "asset", "id"}),
    frozenset({"configured", "model", "id"}),
    frozenset({"case", "id"}),
    frozenset({"company", "id"}),
    frozenset({"user", "id"}),
)


def _validate_json_boundary(
    value: JsonValue,
    *,
    forbidden_names: frozenset[str] | set[str],
    forbidden_segments: frozenset[str] | set[str] = frozenset(),
    forbidden_segment_patterns: tuple[frozenset[str], ...] = (),
) -> None:
    """Percorre um payload uma vez e aplica a política da sua fronteira."""
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            segments = _key_segments(key)
            if (
                _normalized_key(key) in forbidden_names
                or not segments.isdisjoint(forbidden_segments)
                or any(
                    pattern <= segments for pattern in forbidden_segment_patterns
                )
            ):
                raise ValueError("o estado contém um campo proibido")
            _validate_json_boundary(
                nested_value,
                forbidden_names=forbidden_names,
                forbidden_segments=forbidden_segments,
                forbidden_segment_patterns=forbidden_segment_patterns,
            )
    elif isinstance(value, list):
        for item in value:
            _validate_json_boundary(
                item,
                forbidden_names=forbidden_names,
                forbidden_segments=forbidden_segments,
                forbidden_segment_patterns=forbidden_segment_patterns,
            )
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("o estado contém número não finito")


_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue)
_ExactModelT = TypeVar("_ExactModelT", bound=BaseModel)
_READ_TOOL_ARTIFACT_MODELS: Final[dict[str, type[ToolArtifact]]] = {
    "get_asset": AssetToolArtifact,
    "list_asset_analyses": AnalysisListToolArtifact,
    "get_analysis": AnalysisDetailToolArtifact,
    "get_baseline": BaselineToolArtifact,
    "get_rms_series": RmsToolArtifact,
    "get_spectrum": SpectrumToolArtifact,
    "get_data_quality": DataQualityToolArtifact,
    "get_model": ModelToolArtifact,
    "search_knowledge": KnowledgeSearchToolArtifact,
    "get_knowledge_document": KnowledgeDocumentToolArtifact,
}


def _copy_exact_json_wire(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("o wire JSON exige chaves string")
        return {
            key: _copy_exact_json_wire(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_copy_exact_json_wire(item) for item in value]
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("o wire bruto não contém JSON estrito")


def validate_exact_json_model(
    model_type: type[_ExactModelT],
    value: object,
) -> _ExactModelT:
    """Exige o wire JSON completo, estrito e idêntico ao modelo canônico."""
    if not isinstance(value, Mapping):
        raise ValueError("o wire bruto deve ser um objeto JSON")
    raw_wire = _copy_exact_json_wire(value)
    raw_encoded = json.dumps(
        raw_wire,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    validated = model_type.model_validate_json(raw_encoded, strict=True)
    canonical = validated.model_dump(mode="json")
    if raw_encoded != json.dumps(
        canonical,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ):
        raise ValueError("o wire bruto diverge do modelo canônico")
    return validated


def validate_exact_read_artifact(
    tool_name: str,
    value: object,
) -> ToolArtifact:
    """Valida o artifact bruto pela classe exata da read tool selecionada."""
    artifact_model = _READ_TOOL_ARTIFACT_MODELS.get(tool_name)
    if artifact_model is None:
        raise ValueError("read tool não possui modelo de artifact")
    artifact = validate_exact_json_model(artifact_model, value)
    if artifact.tool_name != tool_name:
        raise ValueError("artifact bruto diverge da read tool")
    return artifact


class FrozenStateModel(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JsonSnapshot(FrozenStateModel):
    """JSON canônico imutável, desserializado sempre como uma nova cópia."""

    encoded: str

    @field_validator("encoded")
    @classmethod
    def _require_canonical_json(cls, value: str) -> str:
        try:
            validated = _JSON_VALUE_ADAPTER.validate_python(json.loads(value))
            return json.dumps(
                validated,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("encoded deve conter JSON canônico válido") from error

    @classmethod
    def capture(
        cls,
        value: object,
        *,
        forbidden_names: frozenset[str] | set[str],
        forbidden_segments: frozenset[str] | set[str] = frozenset(),
        forbidden_segment_patterns: tuple[frozenset[str], ...] = (),
    ) -> JsonSnapshot:
        if isinstance(value, cls):
            value = value.to_python()
        validated = _JSON_VALUE_ADAPTER.validate_python(value)
        _validate_json_boundary(
            validated,
            forbidden_names=forbidden_names,
            forbidden_segments=forbidden_segments,
            forbidden_segment_patterns=forbidden_segment_patterns,
        )
        return cls(
            encoded=json.dumps(
                validated,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )

    def to_python(self) -> JsonValue:
        return json.loads(self.encoded)

    def __getitem__(self, key: str | int) -> JsonValue:
        return self.to_python()[key]


def _snapshot_domain_value(value: object, validation_mode: str) -> object:
    if isinstance(value, JsonSnapshot):
        return value.to_python()
    if (
        isinstance(value, Mapping)
        and set(value) == {"encoded"}
        and isinstance(value["encoded"], str)
    ):
        return JsonSnapshot.model_validate(value).to_python()
    if validation_mode == "json":
        return JsonSnapshot.model_validate(value).to_python()
    return value


def _capture_public_argument_object(
    value: object,
    validation_mode: str,
) -> JsonSnapshot:
    domain_value = _snapshot_domain_value(value, validation_mode)
    if not isinstance(domain_value, Mapping) or any(
        not isinstance(key, str) for key in domain_value
    ):
        raise ValueError("arguments deve ser um objeto JSON com chaves string")
    return JsonSnapshot.capture(
        domain_value,
        forbidden_names=_PUBLIC_ARGUMENT_FORBIDDEN_NAMES,
        forbidden_segments=_PUBLIC_ARGUMENT_FORBIDDEN_SEGMENTS,
        forbidden_segment_patterns=_PUBLIC_ARGUMENT_FORBIDDEN_SEGMENT_PATTERNS,
    )


class PersistedRequestIdentity(FrozenStateModel):
    user_id: str = Field(min_length=1, pattern=r"^\S+$")
    company_id: str = Field(min_length=1, pattern=r"^\S+$")


class PersistedSupportRequest(FrozenStateModel):
    case_id: str = Field(min_length=1, pattern=r"^\S+$")
    ticket_id: str = Field(min_length=1, pattern=r"^\S+$")
    asset_id: str | None = Field(min_length=1, pattern=r"^\S+$")
    message: str = Field(min_length=1, pattern=r"\S")
    identity: PersistedRequestIdentity

    @model_validator(mode="before")
    @classmethod
    def _copy_shared_request(cls, value: object) -> object:
        if isinstance(value, SupportRequest):
            return value.model_dump(mode="python")
        return value


class PersistedToolCall(FrozenStateModel):
    request_id: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    call_id: str = Field(min_length=1, pattern=r"^\S+$")
    name: str = Field(min_length=1, pattern=r"^\S+$")
    arguments: JsonSnapshot

    @model_validator(mode="before")
    @classmethod
    def _copy_shared_call(cls, value: object) -> object:
        if isinstance(value, ToolCall):
            return value.model_dump(mode="python")
        return value

    @field_validator("arguments", mode="before")
    @classmethod
    def _snapshot_public_arguments(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> JsonSnapshot:
        return _capture_public_argument_object(value, info.mode)


class PersistedToolSource(FrozenStateModel):
    kind: Literal["industrial_api"]
    resource: str = Field(min_length=1, pattern=r"^/")


class PersistedToolOutcome(FrozenStateModel):
    mode: ResponseMode | None = None
    notes: str | None = None
    partial_data: JsonSnapshot | None = None
    error: PersistedApiError | None = None

    @field_validator("partial_data", mode="before")
    @classmethod
    def _snapshot_technical_result(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> JsonSnapshot | None:
        if value is None:
            return None
        return JsonSnapshot.capture(
            _snapshot_domain_value(value, info.mode),
            forbidden_names=_TECHNICAL_FORBIDDEN_NAMES,
            forbidden_segments=_TECHNICAL_FORBIDDEN_SEGMENTS,
            forbidden_segment_patterns=_TECHNICAL_FORBIDDEN_SEGMENT_PATTERNS,
        )

    @model_validator(mode="after")
    def _reject_unsafe_degraded_partial_data(self) -> PersistedToolOutcome:
        if (
            self.mode is not None
            and self.mode is not ResponseMode.COMPLETE
            and self.partial_data is not None
        ):
            assert_safe_partial_json(self.partial_data.to_python())
        return self


class PersistedToolArtifact(FrozenStateModel):
    tool_name: str = Field(min_length=1, pattern=r"^\S+$")
    arguments: JsonSnapshot
    source: PersistedToolSource
    outcome: PersistedToolOutcome
    truncated: bool = False
    omitted_items: int = Field(default=0, ge=0)
    typed_artifact: JsonSnapshot | None = None

    @model_validator(mode="before")
    @classmethod
    def _project_shared_artifact(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        if isinstance(value, cls):
            return value
        trusted_tool_object = isinstance(value, ToolArtifact)
        if trusted_tool_object:
            artifact_wire = value.model_dump(mode="python")
        elif isinstance(value, Mapping):
            artifact_wire = dict(value)
        else:
            return value
        tool_name = artifact_wire.get("tool_name")
        artifact_model = (
            _READ_TOOL_ARTIFACT_MODELS.get(tool_name)
            if isinstance(tool_name, str)
            else None
        )
        if artifact_model is None:
            return artifact_wire
        _capture_public_argument_object(
            artifact_wire.get("arguments"),
            info.mode,
        )
        typed_value = artifact_wire.get("typed_artifact")
        if typed_value is None:
            candidate = {
                key: nested_value
                for key, nested_value in artifact_wire.items()
                if key != "typed_artifact"
            }
        else:
            candidate = _snapshot_domain_value(typed_value, info.mode)
        typed_artifact = (
            artifact_model.model_validate(candidate)
            if trusted_tool_object and typed_value is None
            else validate_exact_read_artifact(tool_name, candidate)
        )
        canonical = typed_artifact.model_dump(mode="json")
        outcome = canonical["outcome"]
        projected_outcome = PersistedToolOutcome.model_validate(
            {
                field_name: outcome[field_name]
                for field_name in PersistedToolOutcome.model_fields
            }
        )
        return {
            "tool_name": canonical["tool_name"],
            "arguments": JsonSnapshot.capture(
                canonical["arguments"],
                forbidden_names=_PUBLIC_ARGUMENT_FORBIDDEN_NAMES,
                forbidden_segments=_PUBLIC_ARGUMENT_FORBIDDEN_SEGMENTS,
                forbidden_segment_patterns=_PUBLIC_ARGUMENT_FORBIDDEN_SEGMENT_PATTERNS,
            ),
            "source": canonical["source"],
            "outcome": projected_outcome,
            "truncated": canonical["truncated"],
            "omitted_items": canonical["omitted_items"],
            "typed_artifact": JsonSnapshot.capture(
                canonical,
                forbidden_names=_TECHNICAL_FORBIDDEN_NAMES,
                forbidden_segments=_TECHNICAL_FORBIDDEN_SEGMENTS,
                forbidden_segment_patterns=_TECHNICAL_FORBIDDEN_SEGMENT_PATTERNS,
            ),
        }

    @field_validator("arguments", mode="before")
    @classmethod
    def _snapshot_public_arguments(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> JsonSnapshot:
        return _capture_public_argument_object(value, info.mode)

    @field_validator("typed_artifact", mode="before")
    @classmethod
    def _snapshot_typed_artifact(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> JsonSnapshot | None:
        if value is None:
            return None
        return JsonSnapshot.capture(
            _snapshot_domain_value(value, info.mode),
            forbidden_names=_TECHNICAL_FORBIDDEN_NAMES,
            forbidden_segments=_TECHNICAL_FORBIDDEN_SEGMENTS,
            forbidden_segment_patterns=_TECHNICAL_FORBIDDEN_SEGMENT_PATTERNS,
        )

    def validated_read_artifact(self) -> ToolArtifact | None:
        """Reidrata o artifact pela classe concreta vinculada à read tool."""
        artifact_model = _READ_TOOL_ARTIFACT_MODELS.get(self.tool_name)
        if artifact_model is None or self.typed_artifact is None:
            return None
        return artifact_model.model_validate(self.typed_artifact.to_python())


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class AgentDecision(str, Enum):
    GUIDE = "guide"
    ACT = "act"
    ESCALATE = "escalate"
    REQUEST_INFORMATION = "request_information"
    REQUEST_CONFIRMATION = "request_confirmation"
    REQUIRE_HUMAN_REVIEW = "require_human_review"


class WriterNextStep(str, Enum):
    """Próximo passo fechado; o modelo não produz instruções livres."""

    MONITOR = "monitor"
    VERIFY_ACTION = "verify_action"
    AWAIT_ESCALATION = "await_escalation"
    PROVIDE_INFORMATION = "provide_information"
    CONFIRM_ACTION = "confirm_action"
    AWAIT_HUMAN_REVIEW = "await_human_review"
    REQUEST_HUMAN_DISPOSITION = "request_human_disposition"


class WriterDraft(FrozenStateModel):
    """Seleção estruturada do writer, sem fatos, valores ou prosa técnica."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    decision: AgentDecision
    evidence_ids: tuple[str, ...] = ()
    limitation_refs: tuple[str, ...] = ()
    next_step: WriterNextStep

    @field_validator("evidence_ids")
    @classmethod
    def _require_ordered_evidence_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if tuple(sorted(value)) != value or len(value) != len(set(value)):
            raise ValueError("IDs de evidência devem ser únicos e ordenados")
        if any(not re.fullmatch(r"sha256:v1:[0-9a-f]{64}", item) for item in value):
            raise ValueError("ID de evidência inválido")
        return value

    @field_validator("limitation_refs")
    @classmethod
    def _require_ordered_limitation_refs(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if tuple(sorted(value)) != value or len(value) != len(set(value)):
            raise ValueError("referências de limitação devem ser únicas e ordenadas")
        if any(
            not re.fullmatch(r"limitation:v1:[0-9a-f]{64}", item)
            for item in value
        ):
            raise ValueError("referência de limitação inválida")
        return value


class WriterFailureCode(str, Enum):
    INVALID_STRUCTURED_OUTPUT = "invalid_structured_output"
    MODEL_FAILURE = "model_failure"


class WriterFailureRecord(FrozenStateModel):
    code: WriterFailureCode
    attempts: int = Field(ge=1, le=2, strict=True)
    repairable: bool = Field(strict=True)

    @model_validator(mode="after")
    def _require_closed_retry_semantics(self) -> WriterFailureRecord:
        expected = self.code is WriterFailureCode.INVALID_STRUCTURED_OUTPUT
        if self.repairable is not expected:
            raise ValueError("repairable diverge do código da falha do writer")
        return self


class ReleaseGateOutcome(str, Enum):
    RELEASE = "release"
    REQUEST_INFORMATION = "request_information"
    REQUEST_CONFIRMATION = "request_confirmation"
    REQUIRE_HUMAN_REVIEW = "require_human_review"


class ReleaseGateReason(str, Enum):
    PASSED = "passed"
    INFORMATION_REQUIRED = "information_required"
    CONFIRMATION_REQUIRED = "confirmation_required"
    HUMAN_REVIEW_REQUESTED = "human_review_requested"
    HUMAN_DISPOSITION_REQUIRED = "human_disposition_required"
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"
    WRITER_FAILURE = "writer_failure"
    REQUEST_MISMATCH = "request_mismatch"
    DECISION_MISMATCH = "decision_mismatch"
    EVIDENCE_REFERENCE_MISMATCH = "evidence_reference_mismatch"
    LIMITATION_REFERENCE_MISMATCH = "limitation_reference_mismatch"
    NEXT_STEP_MISMATCH = "next_step_mismatch"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PERMISSION_INCOMPATIBLE = "permission_incompatible"
    INTENT_MISSING = "intent_missing"
    INTENT_UNCERTAIN = "intent_uncertain"
    INTENT_NOT_COMPLETED = "intent_not_completed"
    INTENT_POLICY_MISMATCH = "intent_policy_mismatch"
    APPROVAL_MISMATCH = "approval_mismatch"
    ACTION_EVIDENCE_MISSING = "action_evidence_missing"
    MISSING_INFORMATION_INVALID = "missing_information_invalid"


class ReleaseGateRecord(FrozenStateModel):
    subject_decision: AgentDecision
    outcome: ReleaseGateOutcome
    reason: ReleaseGateReason
    draft_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    ledger_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    context_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    review_digest: str | None = Field(default=None, pattern=r"^sha256:v1:[0-9a-f]{64}$")
    review_audit_digest: str | None = Field(default=None, pattern=r"^sha256:v1:[0-9a-f]{64}$")


class ReviewStateModel(FrozenStateModel):
    """Base estrita no Python e compatível com round-trip JSON explícito."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ReviewOperation(str, Enum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


class ReviewQuestion(str, Enum):
    CONFIRM_HUMAN_DISPOSITION = "confirm_human_disposition"
    REBUILD_STRUCTURED_DRAFT = "rebuild_structured_draft"
    ASSESS_BLOCKING_SAFETY = "assess_blocking_safety"


class ReviewerIdentity(ReviewStateModel):
    reviewer_id: str = Field(min_length=1, pattern=r"^\S+$")
    company_id: str = Field(min_length=1, pattern=r"^\S+$")
    permission: Literal["review"]


def _review_contract_digest(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"sha256:v1:{hashlib.sha256(encoded).hexdigest()}"


class ReviewRequest(ReviewStateModel):
    review_id: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    request_id: str = Field(min_length=1, pattern=r"^\S+$")
    request_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    thread_scope_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    basis_permissions: tuple[Permission, ...]
    gate_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    gate_basis: ReleaseGateRecord
    draft_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    reason: ReleaseGateReason
    question: ReviewQuestion
    subject_decision: AgentDecision
    eligible_evidence_ids: tuple[str, ...]
    draft: WriterDraft | None = None
    created_at: datetime
    expires_at: datetime
    allowed_operations: tuple[ReviewOperation, ...]

    @field_validator("eligible_evidence_ids")
    @classmethod
    def _validate_eligible_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("evidências elegíveis devem ser únicas e ordenadas")
        if any(
            not re.fullmatch(r"sha256:v1:[0-9a-f]{64}", item)
            for item in value
        ):
            raise ValueError("ID de evidência elegível inválido")
        return value

    @field_validator("basis_permissions")
    @classmethod
    def _validate_basis_permissions(
        cls, value: tuple[Permission, ...]
    ) -> tuple[Permission, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("permissões da base devem ser únicas e ordenadas")
        return value

    @field_validator("created_at", "expires_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("instantes de revisão exigem UTC aware")
        return value

    @model_validator(mode="after")
    def _require_exact_ttl(self) -> ReviewRequest:
        if self.expires_at - self.created_at != timedelta(hours=24):
            raise ValueError("a revisão deve expirar exatamente após 24 horas")
        if (
            self.gate_digest != _review_contract_digest(self.gate_basis)
            or self.draft_digest != _review_contract_digest(self.draft)
            or self.gate_basis.reason is not self.reason
            or self.gate_basis.subject_decision is not self.subject_decision
            or self.gate_basis.outcome is not ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
        ):
            raise ValueError("base da revisão diverge dos digests tipados")
        expected_question = (
            ReviewQuestion.CONFIRM_HUMAN_DISPOSITION
            if self.reason is ReleaseGateReason.HUMAN_DISPOSITION_REQUIRED
            else ReviewQuestion.REBUILD_STRUCTURED_DRAFT
            if self.reason is ReleaseGateReason.WRITER_FAILURE
            else ReviewQuestion.ASSESS_BLOCKING_SAFETY
        )
        expected_operations = (
            (
                ReviewOperation.APPROVE,
                ReviewOperation.EDIT,
                ReviewOperation.REJECT,
            )
            if self.reason is ReleaseGateReason.HUMAN_DISPOSITION_REQUIRED
            and self.draft is not None
            else (ReviewOperation.EDIT, ReviewOperation.REJECT)
        )
        if (
            self.question is not expected_question
            or self.allowed_operations != expected_operations
        ):
            raise ValueError("pergunta ou operações divergem do motivo")
        identity_fields = {
            "request_id": self.request_id,
            "request_digest": self.request_digest,
            "thread_scope_digest": self.thread_scope_digest,
            "basis_permissions": list(self.basis_permissions),
            "gate_digest": self.gate_digest,
            "draft_digest": self.draft_digest,
            "reason": self.reason.value,
            "question": self.question.value,
            "subject_decision": self.subject_decision.value,
            "eligible_evidence_ids": list(self.eligible_evidence_ids),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "allowed_operations": [item.value for item in self.allowed_operations],
        }
        if self.review_id != _review_contract_digest(
            {"review": identity_fields, "version": "human-review-v1"}
        ):
            raise ValueError("review_id diverge da solicitação canônica")
        return self


class ReviewInterruptPayload(ReviewStateModel):
    review_id: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    reason: ReleaseGateReason
    question: ReviewQuestion
    eligible_evidence_ids: tuple[str, ...]
    draft_present: bool = Field(strict=True)
    allowed_operations: tuple[ReviewOperation, ...]
    created_at: datetime
    expires_at: datetime


class ReviewApproveReply(ReviewStateModel):
    review_id: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    operation: Literal["approve"]


class ReviewEditReply(ReviewStateModel):
    review_id: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    operation: Literal["edit"]
    evidence_ids: tuple[str, ...]
    next_step: WriterNextStep

    @field_validator("evidence_ids")
    @classmethod
    def _require_unique_human_order(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("edição não aceita evidência duplicada")
        if any(
            not re.fullmatch(r"sha256:v1:[0-9a-f]{64}", item)
            for item in value
        ):
            raise ValueError("edição contém ID de evidência inválido")
        return value


class ReviewRejectReply(ReviewStateModel):
    review_id: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    operation: Literal["reject"]


ReviewReply: TypeAlias = Annotated[
    ReviewApproveReply | ReviewEditReply | ReviewRejectReply,
    Field(discriminator="operation"),
]


class ReviewResolution(ReviewStateModel):
    """Envelope confiável persistido; nunca integra o payload do interrupt."""

    review_id: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    reply: ReviewReply
    reviewer: ReviewerIdentity
    received_at: datetime

    @field_validator("received_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("resolução exige instante UTC aware")
        return value

    @model_validator(mode="after")
    def _bind_reply(self) -> ReviewResolution:
        if self.reply.review_id != self.review_id:
            raise ValueError("resolução pertence a outra revisão")
        return self


class ReviewedDraft(ReviewStateModel):
    decision: AgentDecision
    evidence_ids: tuple[str, ...]
    limitation_refs: tuple[str, ...]
    next_step: WriterNextStep

    @field_validator("evidence_ids")
    @classmethod
    def _require_unique_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("reviewed draft não aceita evidência duplicada")
        if any(
            not re.fullmatch(r"sha256:v1:[0-9a-f]{64}", item)
            for item in value
        ):
            raise ValueError("reviewed draft contém ID de evidência inválido")
        return value

    @field_validator("limitation_refs")
    @classmethod
    def _require_derived_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("limitações derivadas devem ser únicas e ordenadas")
        return value


class ReviewAudit(ReviewStateModel):
    review_id: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    review_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    reviewer_id: str = Field(min_length=1, pattern=r"^\S+$")
    company_id: str = Field(min_length=1, pattern=r"^\S+$")
    reviewer_permission: Literal["review"]
    operation: ReviewOperation
    received_at: datetime
    reply_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    resolution_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    before_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    after_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    structural_change: bool = Field(strict=True)

    @field_validator("received_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("auditoria exige instante UTC aware")
        return value


class ReviewExpiry(ReviewStateModel):
    review_id: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    review_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    resolution_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    expired_at: datetime

    @field_validator("expired_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("expiração exige instante UTC aware")
        return value


class ReviewStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    APPROVED = "approved"
    REJECTED = "rejected"


class ResumeAnchor(str, Enum):
    """Último nó concluído, inclusive o pseudo-nó inicial."""

    START = "__start__"
    INGEST = "ingest"
    ROUTE = "route"
    FINISH = "finish"
    PLANNER_SELECT = "planner_select"
    PLANNER_TOOL = "planner_tool"
    PLANNER_FINALIZE = "planner_finalize"
    WRITER = "writer"
    RELEASE_GATE = "release_gate"
    AWAIT_HUMAN_REVIEW = "await_human_review"
    WRITE_POLICY = "write_policy"
    CONFIRMATION_GATE = "confirmation_gate"
    PREPARE_INTENT = "prepare_intent"
    EXECUTE_ACTION = "execute_action"


class ThreadScope(FrozenStateModel):
    thread_id: str = Field(min_length=1, pattern=r"^\S+$")
    case_id: str = Field(min_length=1, pattern=r"^\S+$")
    company_id: str = Field(min_length=1, pattern=r"^\S+$")
    user_id: str = Field(min_length=1, pattern=r"^\S+$")


class PersistedMessage(FrozenStateModel):
    role: MessageRole
    content: str


class ToolObservation(FrozenStateModel):
    request_id: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    call_id: str = Field(min_length=1, pattern=r"^\S+$")
    content: JsonSnapshot | None = None
    artifact: PersistedToolArtifact

    @field_validator("content", mode="before")
    @classmethod
    def _snapshot_next_turn_content(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> JsonSnapshot | None:
        if value is None:
            return None
        return JsonSnapshot.capture(
            _snapshot_domain_value(value, info.mode),
            forbidden_names=_TECHNICAL_FORBIDDEN_NAMES,
            forbidden_segments=_TECHNICAL_FORBIDDEN_SEGMENTS,
            forbidden_segment_patterns=_TECHNICAL_FORBIDDEN_SEGMENT_PATTERNS,
        )


class StateEvidence(FrozenStateModel):
    """Snapshot legado anterior ao ledger compilado.

    Ele continua desserializável para checkpoints históricos, mas não traz a
    proveniência completa exigida por ``EvidenceItem`` e, portanto, nunca deve
    ser usado como fato claimable. O compilador da Fase 7 recebe as observações
    tipadas e não promove este snapshot.
    """

    evidence_id: str = Field(min_length=1, pattern=r"^\S+$")
    request_id: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    call_id: str = Field(min_length=1, pattern=r"^\S+$")
    value: JsonSnapshot

    @field_validator("value", mode="before")
    @classmethod
    def _snapshot_value(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> JsonSnapshot:
        return JsonSnapshot.capture(
            _snapshot_domain_value(value, info.mode),
            forbidden_names=_TECHNICAL_FORBIDDEN_NAMES,
            forbidden_segments=_TECHNICAL_FORBIDDEN_SEGMENTS,
            forbidden_segment_patterns=_TECHNICAL_FORBIDDEN_SEGMENT_PATTERNS,
        )


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


class EvidenceSourceKind(str, Enum):
    TOOL = "tool"
    ACTION = "action"


class EvidenceItem(FrozenStateModel):
    """Fato persistível, ligado exclusivamente a uma tool ou intenção terminal."""

    evidence_id: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    request_id: str = Field(min_length=1, pattern=r"^\S+$")
    source_kind: EvidenceSourceKind = EvidenceSourceKind.TOOL
    call_id: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    intent_id: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    tool: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    action: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    resource: str = Field(min_length=1, pattern=r"^/")
    fact_path: str = Field(
        min_length=1,
        pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$",
    )
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

    @model_validator(mode="after")
    def _require_exclusive_provenance(self) -> EvidenceItem:
        if self.source_kind is EvidenceSourceKind.TOOL:
            if self.call_id is None or self.intent_id is not None or self.tool is None or self.action is not None:
                raise ValueError("evidência de tool exige somente call_id e tool")
        elif self.intent_id is None or self.call_id is not None or self.action is None or self.tool is not None:
            raise ValueError("evidência de ação exige somente intent_id e action")
        return self

    @property
    def canonical_key(self) -> str:
        source = self.tool if self.source_kind is EvidenceSourceKind.TOOL else self.action
        assert source is not None
        return f"{self.source_kind.value}:{source}:{self.resource}:{self.fact_path}"

    @property
    def claimable(self) -> bool:
        return self.quality is EvidenceQuality.CLAIMABLE


class EvidenceGap(FrozenStateModel):
    reason: EvidenceGapReason
    request_id: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    call_id: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    intent_id: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    fact_path: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$",
    )
    blocking: Literal[True] = True

    @model_validator(mode="after")
    def _require_exclusive_provenance(self) -> EvidenceGap:
        if self.call_id is not None and self.intent_id is not None:
            raise ValueError("lacuna não pode referenciar tool e intenção")
        return self


class EvidenceConflict(FrozenStateModel):
    canonical_key: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=2)
    blocking: Literal[True] = True

    @field_validator("evidence_ids")
    @classmethod
    def _require_canonical_evidence_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("IDs do conflito devem ser únicos")
        if tuple(sorted(value)) != value:
            raise ValueError("IDs do conflito devem ser ordenados")
        if any(
            not re.fullmatch(r"sha256:v1:[0-9a-f]{64}", item)
            for item in value
        ):
            raise ValueError("ID de evidência inválido no conflito")
        return value


class EvidenceLedger(FrozenStateModel):
    """Ledger atual de uma request; histórico permanece em ``ledger_history``."""

    request_id: str | None = Field(default=None, min_length=1, pattern=r"^\S+$")
    items: tuple[EvidenceItem, ...] = ()
    gaps: tuple[EvidenceGap, ...] = ()
    conflicts: tuple[EvidenceConflict, ...] = ()


class EvidenceAssessment(FrozenStateModel):
    status: EvidenceSufficiency
    causes: tuple[EvidenceGapReason, ...] = ()

    @property
    def sufficient(self) -> bool:
        return self.status is EvidenceSufficiency.SUFFICIENT


class FinalResult(FrozenStateModel):
    decision: AgentDecision
    message: str = Field(min_length=1, pattern=r"\S")
    evidence_ids: tuple[str, ...] = ()
    limitation_refs: tuple[str, ...] = ()
    next_step: WriterNextStep | None = None


@dataclass(frozen=True)
class ConservativeNonIdempotentResumeTerminal:
    """Contrato único do terminal local que proíbe um segundo despacho."""

    error: PersistedApiError
    final_result: FinalResult

    def matches_state(self, state: AgentState) -> bool | None:
        """Retorna ``None`` fora do caso; ``False`` denuncia shape adulterado."""

        current_intents = tuple(
            intent
            for intent in state.intents
            if intent.request_id == state.request_id
        )
        if len(current_intents) != 1:
            return None
        intent = current_intents[0]
        structural_case = (
            state.resume_anchor is ResumeAnchor.EXECUTE_ACTION
            and intent.scope.action != "reprocess_analysis"
            and intent.status is IntentStatus.UNCERTAIN
            and intent.attempts == 0
            and intent.receipt is None
            and intent.prepared_execution_id is not None
            and intent.prepared_execution_id != state.execution_id
        )
        claims_canonical_error = (
            intent.error is not None and intent.error.code == self.error.code
        )
        if not structural_case and not claims_canonical_error:
            return None
        return (
            structural_case
            and intent.error == self.error
            and state.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
            and state.final_result == self.final_result
            and state.review is None
            and state.planner_terminal is None
            and state.planner_failure is None
            and state.writer_draft is None
            and state.writer_failure is None
            and state.writer_attempts == 0
            and state.release_gate is None
        )


_CONSERVATIVE_NON_IDEMPOTENT_RESUME_TERMINAL = (
    ConservativeNonIdempotentResumeTerminal(
        error=PersistedApiError(
            category=ApiErrorCategory.API,
            code="NON_IDEMPOTENT_OUTCOME_UNKNOWN_AFTER_RESUME",
            message=(
                "A execução preparadora terminou sem resultado terminal observável."
            ),
            status_code=None,
        ),
        final_result=FinalResult(
            decision=AgentDecision.REQUIRE_HUMAN_REVIEW,
            message=(
                "O resultado remoto da ação é desconhecido e ela não será "
                "reenviada automaticamente."
            ),
            evidence_ids=(),
            limitation_refs=(),
            next_step=None,
        ),
    )
)


def canonical_non_idempotent_resume_terminal(
) -> ConservativeNonIdempotentResumeTerminal:
    """Fornece erro e resposta fixos, além do predicado estrutural compartilhado."""

    return _CONSERVATIVE_NON_IDEMPOTENT_RESUME_TERMINAL


class ReviewRecord(FrozenStateModel):
    status: ReviewStatus
    reason: str | None = Field(default=None, min_length=1, pattern=r"\S")


class PlannerUsage(FrozenStateModel):
    """Contadores do planner vinculados a uma única solicitação."""

    request_id: str = Field(min_length=1, pattern=r"^\S+$")
    selection_count: int = Field(default=0, ge=0, le=8, strict=True)
    finalization_count: int = Field(default=0, ge=0, le=1, strict=True)


class PlannerTerminalRecord(FrozenStateModel):
    """Decisão terminal observável, ainda sem texto do futuro writer."""

    decision: Literal[
        "guide",
        "request_information",
        "require_human_review",
    ]
    stop_reason: Literal[
        "sufficient_evidence",
        "missing_information",
        "human_review_required",
    ]
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
    def _require_coherent_stop_contract(self) -> PlannerTerminalRecord:
        expected_reason = {
            "guide": "sufficient_evidence",
            "request_information": "missing_information",
            "require_human_review": "human_review_required",
        }[self.decision]
        if self.stop_reason != expected_reason:
            raise ValueError("stop_reason diverge da decisão terminal")
        requires_information = self.decision == "request_information"
        if requires_information != (self.missing_information is not None):
            raise ValueError("missing_information diverge da decisão terminal")
        return self


class PlannerFailureRecord(FrozenStateModel):
    """Falha sanitizada do ciclo; nunca guarda exceção ou saída livre."""

    stage: Literal["planner_select", "planner_tool", "planner_finalize"]
    code: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")


class AgentState(FrozenStateModel):
    request: PersistedSupportRequest
    identity: TrustedIdentity
    permissions: frozenset[Permission]
    request_id: str = Field(min_length=1, pattern=r"^\S+$")
    thread_id: str = Field(min_length=1, pattern=r"^\S+$")
    execution_id: str = Field(min_length=1, pattern=r"^\S+$")
    thread_scope: ThreadScope
    trusted_write_context: TrustedWriteContext | None = None
    messages: tuple[PersistedMessage, ...] = ()
    tool_calls: tuple[PersistedToolCall, ...] = ()
    tool_observations: tuple[ToolObservation, ...] = ()
    evidence: tuple[StateEvidence, ...] = ()
    ledger: EvidenceLedger = Field(default_factory=EvidenceLedger)
    ledger_history: tuple[EvidenceLedger, ...] = ()
    decision: AgentDecision | None = None
    step_count: int = Field(default=0, ge=0)
    step_limit: int = Field(gt=0)
    pending_proposal: WriteProposal | None = None
    approval: TrustedActionApproval | None = None
    intents: tuple[WriteIntent, ...] = ()
    final_result: FinalResult | None = None
    review: ReviewRecord | None = None
    planner_usage: PlannerUsage
    resume_anchor: ResumeAnchor = ResumeAnchor.START
    planner_terminal: PlannerTerminalRecord | None = None
    planner_failure: PlannerFailureRecord | None = None
    writer_draft: WriterDraft | None = None
    writer_failure: WriterFailureRecord | None = None
    writer_attempts: int = Field(default=0, ge=0, le=2, strict=True)
    release_gate: ReleaseGateRecord | None = None
    review_request: ReviewRequest | None = None
    review_resolution: ReviewResolution | None = None
    reviewed_draft: ReviewedDraft | None = None
    review_audit: ReviewAudit | None = None
    review_expiry: ReviewExpiry | None = None

    def has_coherent_terminal_result(self) -> bool:
        """Confere a matriz fechada dos resultados terminais persistidos."""
        conservative_match = (
            canonical_non_idempotent_resume_terminal().matches_state(self)
        )
        if conservative_match is not None:
            return conservative_match
        if self.final_result is None:
            return True
        if self.resume_anchor is ResumeAnchor.AWAIT_HUMAN_REVIEW:
            return (
                self.final_result.decision
                is AgentDecision.REQUIRE_HUMAN_REVIEW
                and self.review_request is not None
                and self.release_gate is not None
                and (
                    (
                        self.review_expiry is not None
                        and self.review_audit is None
                        and self.reviewed_draft is None
                    )
                    or (
                        self.review_audit is not None
                        and self.review_audit.operation is ReviewOperation.REJECT
                        and self.reviewed_draft is None
                    )
                )
            )
        if self.decision is None or self.final_result.decision is not self.decision:
            return False

        current_intents = tuple(
            intent
            for intent in self.intents
            if intent.request_id == self.request_id
        )
        if self.planner_failure is not None:
            expected_anchor = {
                "planner_select": ResumeAnchor.PLANNER_SELECT,
                "planner_tool": ResumeAnchor.PLANNER_TOOL,
                "planner_finalize": ResumeAnchor.PLANNER_FINALIZE,
            }[self.planner_failure.stage]
            return (
                self.resume_anchor is expected_anchor
                and self.planner_terminal is None
                and not current_intents
            )

        if self.resume_anchor is ResumeAnchor.FINISH:
            return (
                self.decision is AgentDecision.GUIDE
                and self.planner_terminal is None
                and self.pending_proposal is None
                and self.approval is None
                and not current_intents
            )

        if self.resume_anchor is ResumeAnchor.PLANNER_FINALIZE:
            terminal = self.planner_terminal
            if terminal is None or self.pending_proposal is not None or current_intents:
                return False
            expected_decision = AgentDecision(terminal.decision)
            if self.decision is not expected_decision:
                return False
            if expected_decision is AgentDecision.REQUIRE_HUMAN_REVIEW:
                return (
                    self.review is not None
                    and self.review.status is ReviewStatus.REQUIRED
                )
            return self.review is None

        if self.resume_anchor is ResumeAnchor.RELEASE_GATE:
            gate = self.release_gate
            if gate is None:
                return False
            if gate.outcome is ReleaseGateOutcome.RELEASE:
                coherent_release = (
                    (self.writer_draft is not None or self.reviewed_draft is not None)
                    and (self.writer_failure is None or self.reviewed_draft is not None)
                    and self.decision
                    is (
                        self.reviewed_draft.decision
                        if self.reviewed_draft is not None
                        else self.writer_draft.decision
                    )
                    and (
                        self.review is None
                        or self.review.status is ReviewStatus.APPROVED
                    )
                )
                if not coherent_release:
                    return False
                if self.decision not in {AgentDecision.ACT, AgentDecision.ESCALATE}:
                    return True
                return (
                    self.pending_proposal is not None
                    and len(current_intents) == 1
                    and current_intents[0].status is IntentStatus.COMPLETED
                    and approval_matches_write_intent(
                        self.pending_proposal,
                        current_intents[0],
                        approval=self.approval,
                        trusted_context=self.trusted_write_context,
                    )
                )
            if gate.outcome is ReleaseGateOutcome.REQUEST_INFORMATION:
                return (
                    self.decision is AgentDecision.REQUEST_INFORMATION
                    and self.review is None
                )
            if gate.outcome is ReleaseGateOutcome.REQUEST_CONFIRMATION:
                return (
                    self.decision is AgentDecision.REQUEST_CONFIRMATION
                    and self.review is None
                )
            return (
                self.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
                and self.review is not None
                and self.review.status is ReviewStatus.REQUIRED
            )

        if self.resume_anchor in {
            ResumeAnchor.WRITE_POLICY,
            ResumeAnchor.CONFIRMATION_GATE,
        }:
            return (
                self.decision is AgentDecision.GUIDE
                and self.planner_terminal is None
                and self.pending_proposal is not None
                and len(current_intents) == 1
                and current_intents[0].status is IntentStatus.DENIED
                and current_intents[0].decision.decision is PolicyDecision.DENY
            )

        if self.resume_anchor is ResumeAnchor.EXECUTE_ACTION:
            if (
                self.planner_terminal is not None
                or self.pending_proposal is None
                or len(current_intents) != 1
            ):
                return False
            intent = current_intents[0]
            if intent.decision.decision is not PolicyDecision.ALLOW:
                return False
            if intent.status is IntentStatus.COMPLETED:
                if not _proposal_matches_persisted_intent(
                    self.pending_proposal,
                    intent.scope,
                    request=self.request,
                    payload_hash=intent.payload_hash,
                    trusted_context=self.trusted_write_context,
                ):
                    return False
                if not approval_matches_write_intent(
                    self.pending_proposal,
                    intent,
                    approval=self.approval,
                    trusted_context=self.trusted_write_context,
                ):
                    return False
                expected_decision = (
                    AgentDecision.ESCALATE
                    if intent.scope.action == "escalate_case"
                    else AgentDecision.ACT
                )
                return self.decision is expected_decision
            if intent.status is IntentStatus.UNCERTAIN:
                return self.decision is AgentDecision.REQUIRE_HUMAN_REVIEW
            if intent.status is IntentStatus.FAILED:
                return self.decision in {
                    AgentDecision.GUIDE,
                    AgentDecision.REQUIRE_HUMAN_REVIEW,
                }
            return False

        return False

    @model_validator(mode="before")
    @classmethod
    def _restore_request_bound_planner_state(cls, value: object) -> object:
        """Inicializa uso legado sem inventar proveniência para o histórico."""
        if isinstance(value, Mapping) and isinstance(value.get("request_id"), str):
            request_id = value["request_id"]
            restored = dict(value)
            if value.get("planner_usage") is None:
                restored["planner_usage"] = {
                    "request_id": request_id,
                    "selection_count": 0,
                    "finalization_count": 0,
                }
            return restored
        return value

    @field_validator("pending_proposal", mode="before")
    @classmethod
    def _restore_legacy_reprocess_proposal(cls, value: object) -> object:
        """Adiciona o discriminador somente ao shape inequívoco anterior."""
        if isinstance(value, Mapping) and set(value) == {
            "analysis_id",
            "justification",
        }:
            return {
                "action": "reprocess_analysis",
                "analysis_id": value["analysis_id"],
                "justification": value["justification"],
            }
        return value

    @field_validator("writer_draft", mode="before")
    @classmethod
    def _restore_strict_writer_draft(cls, value: object) -> object:
        """Restaura o wire JSON sem afrouxar o contrato Python do draft."""
        if isinstance(value, Mapping):
            return WriterDraft.model_validate_json(
                json.dumps(
                    value,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        return value

    @field_validator(
        "review_request",
        "review_resolution",
        "reviewed_draft",
        "review_audit",
        "review_expiry",
        mode="before",
    )
    @classmethod
    def _restore_strict_review_models(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        if not isinstance(value, Mapping):
            return value
        model = {
            "review_request": ReviewRequest,
            "review_resolution": ReviewResolution,
            "reviewed_draft": ReviewedDraft,
            "review_audit": ReviewAudit,
            "review_expiry": ReviewExpiry,
        }[info.field_name]
        return model.model_validate_json(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )

    @model_validator(mode="after")
    def _validate_scope_and_budget(self) -> AgentState:
        expected_scope = (
            self.thread_id,
            self.request.case_id,
            self.identity.company_id,
            self.identity.user_id,
        )
        persisted_scope = (
            self.thread_scope.thread_id,
            self.thread_scope.case_id,
            self.thread_scope.company_id,
            self.thread_scope.user_id,
        )
        if expected_scope != persisted_scope:
            raise ValueError("thread reutilizado fora do escopo confiável")
        if (
            self.request.identity.company_id != self.identity.company_id
            or self.request.identity.user_id != self.identity.user_id
        ):
            raise ValueError("identidade da solicitação diverge da fronteira confiável")
        if self.trusted_write_context is not None and (
            self.trusted_write_context.current_case_id != self.request.case_id
            or (
                self.request.asset_id is not None
                and self.trusted_write_context.central_asset_id
                != self.request.asset_id
            )
        ):
            raise ValueError("escopo confiável diverge da solicitação")
        write_anchors = {
            ResumeAnchor.WRITE_POLICY,
            ResumeAnchor.CONFIRMATION_GATE,
            ResumeAnchor.PREPARE_INTENT,
            ResumeAnchor.EXECUTE_ACTION,
        }
        if self.trusted_write_context is None and (
            self.pending_proposal is not None
            or self.approval is not None
            or bool(self.intents)
            or self.resume_anchor in write_anchors
        ):
            raise ValueError(
                "o ciclo de escrita exige contexto confiável persistido"
            )
        thread_intent_scope = (
            self.thread_scope.case_id,
            self.thread_scope.company_id,
            self.thread_scope.user_id,
        )
        if any(
            (
                intent.scope.case_id,
                intent.scope.company_id,
                intent.scope.user_id,
            )
            != thread_intent_scope
            for intent in self.intents
        ):
            raise ValueError("intenção persistida fora do escopo do thread")
        intent_ids = tuple(intent.intent_id for intent in self.intents)
        if len(intent_ids) != len(set(intent_ids)):
            raise ValueError("IDs de intenção devem ser únicos no estado")
        active_statuses = {
            IntentStatus.PROPOSED,
            IntentStatus.AWAITING_CONFIRMATION,
            IntentStatus.PREPARED,
        }
        active_request_ids = [
            intent.request_id
            for intent in self.intents
            if intent.request_id is not None
            and intent.status in active_statuses
        ]
        if len(active_request_ids) != len(set(active_request_ids)):
            raise ValueError("cada request_id aceita no máximo uma intenção ativa")
        current_intents = tuple(
            intent
            for intent in self.intents
            if intent.request_id == self.request_id
        )
        if self.pending_proposal is not None and current_intents:
            if (
                len(current_intents) != 1
                or not _proposal_matches_persisted_intent(
                    self.pending_proposal,
                    current_intents[0].scope,
                    request=self.request,
                    payload_hash=current_intents[0].payload_hash,
                    trusted_context=self.trusted_write_context,
                )
            ):
                raise ValueError(
                    "ciclo de escrita diverge da assinatura canônica persistida"
                )
        if self.step_count > self.step_limit:
            raise ValueError("contador de passos excede o orçamento")
        if self.planner_usage.request_id != self.request_id:
            raise ValueError("uso do planner pertence a outra request_id")
        current_call_ids = tuple(
            call.call_id
            for call in self.tool_calls
            if call.request_id == self.request_id
        )
        current_observation_ids = tuple(
            observation.call_id
            for observation in self.tool_observations
            if observation.request_id == self.request_id
        )
        if len(current_call_ids) > 7:
            raise ValueError("histórico do planner excede sete tool calls")
        if (
            len(current_call_ids) != len(set(current_call_ids))
            or len(current_observation_ids) != len(set(current_observation_ids))
        ):
            raise ValueError("IDs de tool call devem ser únicos por request_id")
        if self.planner_terminal is not None and self.planner_failure is not None:
            raise ValueError(
                "decisão terminal e falha do planner são mutuamente exclusivos"
            )
        if self.planner_failure is not None and (
            self.decision is not AgentDecision.REQUIRE_HUMAN_REVIEW
            or self.final_result is None
            or self.final_result.decision
            is not AgentDecision.REQUIRE_HUMAN_REVIEW
            or self.review is None
            or self.review.status is not ReviewStatus.REQUIRED
        ):
            raise ValueError(
                "falha do planner exige encerramento seguro e revisão humana"
            )
        if self.writer_draft is not None and self.writer_failure is not None:
            raise ValueError("draft e falha do writer são mutuamente exclusivos")
        if self.writer_attempts == 0 and (
            self.writer_draft is not None or self.writer_failure is not None
        ):
            raise ValueError("resultado do writer exige tentativa persistida")
        if self.writer_attempts > 0 and (
            self.writer_draft is None and self.writer_failure is None
        ):
            raise ValueError("tentativa do writer exige draft ou falha")
        if self.writer_failure is not None and (
            self.writer_failure.attempts != self.writer_attempts
        ):
            raise ValueError("falha do writer diverge do contador persistido")
        if self.resume_anchor is ResumeAnchor.WRITER and self.writer_attempts == 0:
            raise ValueError("âncora do writer exige tentativa persistida")
        if self.writer_attempts > 0 and self.resume_anchor not in {
            ResumeAnchor.WRITER,
            ResumeAnchor.RELEASE_GATE,
            ResumeAnchor.AWAIT_HUMAN_REVIEW,
        }:
            raise ValueError("resultado do writer diverge da âncora persistida")
        if self.release_gate is not None and self.resume_anchor not in {
            ResumeAnchor.RELEASE_GATE,
            ResumeAnchor.AWAIT_HUMAN_REVIEW,
        }:
            raise ValueError("atestado do gate exige sua âncora persistida")
        terminal_anchors = {
            ResumeAnchor.FINISH,
            ResumeAnchor.PLANNER_FINALIZE,
            ResumeAnchor.WRITE_POLICY,
            ResumeAnchor.CONFIRMATION_GATE,
            ResumeAnchor.EXECUTE_ACTION,
            ResumeAnchor.RELEASE_GATE,
            ResumeAnchor.AWAIT_HUMAN_REVIEW,
        }
        if (
            self.resume_anchor in terminal_anchors
            and not self.has_coherent_terminal_result()
        ):
            raise ValueError("resultado terminal diverge da âncora persistida")
        self._validate_current_ledger()
        self._validate_human_review()
        self._validate_release_attestation()
        return self

    def _validate_human_review(self) -> None:
        if self.review_request is None:
            if any(
                value is not None
                for value in (
                    self.reviewed_draft,
                    self.review_resolution,
                    self.review_audit,
                    self.review_expiry,
                )
            ):
                raise ValueError("resultado de revisão exige solicitação persistida")
            return
        from tractian_agent.human_review import (
            build_review_request,
            canonical_digest,
            render_review_expired_result,
            render_review_rejected_result,
            review_audit_is_canonical,
        )
        from tractian_agent.release_gate import ReleaseGateContext, evaluate_release

        request = self.review_request
        basis_context = ReleaseGateContext(
            request_id=self.request_id,
            decision=request.subject_decision,
            ledger=self.ledger,
            draft=self.writer_draft,
            permissions=frozenset(request.basis_permissions),
            intents=tuple(
                intent
                for intent in self.intents
                if intent.request_id == self.request_id
            ),
            proposal=self.pending_proposal,
            trusted_write_context=self.trusted_write_context,
            planner_terminal=self.planner_terminal,
            approval=self.approval,
            missing_information=(
                self.planner_terminal.missing_information
                if self.planner_terminal is not None
                else None
            ),
            writer_failure=self.writer_failure,
        )
        canonical_gate = evaluate_release(basis_context)
        if canonical_gate != request.gate_basis:
            raise ValueError("base do gate diverge do estado que originou a revisão")
        expected = build_review_request(
            request_id=self.request_id,
            request=self.request,
            thread_scope=self.thread_scope,
            permissions=frozenset(request.basis_permissions),
            gate=canonical_gate,
            ledger=self.ledger,
            draft=self.writer_draft,
            created_at=request.created_at,
        )
        if request != expected:
            raise ValueError("solicitação de revisão diverge da base canônica")
        if self.review_audit is not None and self.review_expiry is not None:
            raise ValueError("auditoria e expiração são mutuamente exclusivas")
        resolution = self.review_resolution
        if self.review_expiry is not None:
            if (
                resolution is not None
                or self.review_expiry.review_id != request.review_id
                or self.review_expiry.review_digest != canonical_digest(request)
                or self.review_expiry.expired_at < request.expires_at
                or self.reviewed_draft is not None
                or self.final_result != render_review_expired_result()
            ):
                raise ValueError("expiração diverge da revisão persistida")
            return
        if resolution is None:
            if self.review_audit is not None:
                raise ValueError("resultado da revisão exige envelope confiável")
        elif (
            resolution.review_id != request.review_id
            or resolution.reviewer.company_id != self.identity.company_id
            or resolution.received_at < request.created_at
            or resolution.reply.operation
            not in {operation.value for operation in request.allowed_operations}
        ):
            raise ValueError("envelope confiável diverge da revisão")
        audit = self.review_audit
        if audit is None:
            if self.reviewed_draft is not None or resolution is not None:
                raise ValueError("draft revisado exige auditoria")
            return
        assert resolution is not None
        if audit.company_id != self.identity.company_id:
            raise ValueError("auditoria pertence a outra empresa")
        if not review_audit_is_canonical(
            request, audit, self.reviewed_draft, self.ledger, resolution
        ):
            raise ValueError("auditoria de revisão foi adulterada")
        if audit.operation is ReviewOperation.REJECT:
            if (
                self.reviewed_draft is not None
                or self.final_result != render_review_rejected_result()
            ):
                raise ValueError("rejeição não aceita draft revisado")
        elif self.reviewed_draft is None:
            raise ValueError("aprovação ou edição exige draft revisado")

    def _validate_release_attestation(self) -> None:
        """Recalcula o gate e a resposta; checkpoint não pode inventar prosa."""
        if self.release_gate is None:
            return
        if self.resume_anchor is ResumeAnchor.AWAIT_HUMAN_REVIEW:
            return

        from tractian_agent.release_gate import (
            ReleaseGateContext,
            build_budget_exhausted_gate,
            evaluate_release,
            render_budget_exhausted_result,
            render_non_release_result,
            render_released_result,
        )

        context = ReleaseGateContext(
            request_id=self.request_id,
            decision=self.release_gate.subject_decision,
            ledger=self.ledger,
            draft=(
                self.reviewed_draft
                if self.reviewed_draft is not None
                else self.writer_draft
            ),
            permissions=self.permissions,
            intents=tuple(
                intent
                for intent in self.intents
                if intent.request_id == self.request_id
            ),
            proposal=self.pending_proposal,
            trusted_write_context=self.trusted_write_context,
            planner_terminal=self.planner_terminal,
            approval=self.approval,
            missing_information=(
                self.planner_terminal.missing_information
                if self.planner_terminal is not None
                else None
            ),
            writer_failure=self.writer_failure,
            review_request=(
                self.review_request if self.review_audit is not None else None
            ),
            review_audit=self.review_audit,
            review_resolution=self.review_resolution,
        )
        budget_exhausted = (
            self.release_gate.reason is ReleaseGateReason.STEP_BUDGET_EXHAUSTED
        )
        if budget_exhausted and self.step_count != self.step_limit:
            raise ValueError("esgotamento do gate exige orçamento consumido")
        expected_gate = (
            build_budget_exhausted_gate(context)
            if budget_exhausted
            else evaluate_release(context)
        )
        if self.release_gate != expected_gate:
            raise ValueError("atestado do gate diverge da recomputação")
        if self.final_result is None:
            if (
                expected_gate.outcome is not ReleaseGateOutcome.REQUIRE_HUMAN_REVIEW
                or self.review_request is None
                or self.review_audit is not None
            ):
                raise ValueError("atestado pendente não corresponde à revisão")
            return
        expected_result = (
            render_budget_exhausted_result(context, expected_gate)
            if budget_exhausted
            else render_released_result(context, expected_gate)
            if expected_gate.outcome is ReleaseGateOutcome.RELEASE
            else render_non_release_result(context, expected_gate)
        )
        if self.final_result != expected_result:
            raise ValueError("resultado final diverge do renderer determinístico")

    def _validate_current_ledger(self) -> None:
        """Recompila integralmente o ledger atual e o histórico, sem subconjuntos."""
        observations_by_request, intents_by_request = self._ledger_sources_by_request()
        history_request_ids = tuple(
            historic.request_id for historic in self.ledger_history
        )
        if any(request_id is None for request_id in history_request_ids):
            raise ValueError("histórico do ledger exige request_id")
        if len(history_request_ids) != len(set(history_request_ids)):
            raise ValueError("histórico do ledger não aceita request_id duplicado")
        for historic in self.ledger_history:
            assert historic.request_id is not None
            self._validate_ledger_for_request(
                historic,
                request_id=historic.request_id,
                label="histórico do ledger",
                observations=observations_by_request.get(historic.request_id, ()),
                intents=intents_by_request.get(historic.request_id, ()),
            )
        self._validate_ledger_for_request(
            self.ledger,
            request_id=self.request_id,
            label="ledger atual",
            observations=observations_by_request.get(self.request_id, ()),
            intents=intents_by_request.get(self.request_id, ()),
        )

    def _ledger_sources_by_request(
        self,
    ) -> tuple[
        dict[str, tuple[ToolObservation, ...]],
        dict[str, tuple[WriteIntent, ...]],
    ]:
        """Indexa uma vez as fontes claimable para toda a validação do estado."""
        observation_lists: dict[str, list[ToolObservation]] = {}
        intent_lists: dict[str, list[WriteIntent]] = {}
        for observation in self.tool_observations:
            if (
                observation.request_id is not None
                and observation.artifact.validated_read_artifact() is not None
            ):
                observation_lists.setdefault(observation.request_id, []).append(
                    observation
                )
        for intent in self.intents:
            if intent.request_id is not None:
                intent_lists.setdefault(intent.request_id, []).append(intent)
        return (
            {request_id: tuple(values) for request_id, values in observation_lists.items()},
            {request_id: tuple(values) for request_id, values in intent_lists.items()},
        )

    def _validate_ledger_for_request(
        self,
        ledger: EvidenceLedger,
        *,
        request_id: str,
        label: str,
        observations: tuple[ToolObservation, ...],
        intents: tuple[WriteIntent, ...],
    ) -> None:
        """Compara uma projeção semântica completa com todas as fontes tipadas."""
        if ledger.request_id not in {request_id, None}:
            raise ValueError(f"{label} pertence a outra request_id")
        if ledger.request_id is None and (
            ledger.items or ledger.gaps or ledger.conflicts
        ):
            raise ValueError(f"{label} com conteúdo exige request_id")

        # Import tardio preserva state -> evidence como fronteira sem ciclo.
        from tractian_agent.evidence import (
            compile_action_intents,
            compile_observations,
            merge_ledgers,
        )

        canonical_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
        expected = merge_ledgers(
            compile_observations(observations, recorded_at=canonical_time),
            compile_action_intents(intents, recorded_at=canonical_time),
        )
        if self._ledger_semantics(ledger, canonical_time=canonical_time) != self._ledger_semantics(
            expected,
            canonical_time=canonical_time,
        ):
            raise ValueError(f"{label} diverge da recompilação canônica")

    @staticmethod
    def _ledger_semantics(
        ledger: EvidenceLedger,
        *,
        canonical_time: datetime,
    ) -> tuple[object, ...]:
        """Normaliza somente o instante de gravação, que não vem da fonte."""
        items = tuple(
            sorted(
                (
                    item.model_copy(update={"recorded_at": canonical_time})
                    for item in ledger.items
                ),
                key=lambda item: item.evidence_id,
            )
        )
        gaps = tuple(sorted(ledger.gaps, key=lambda gap: gap.model_dump_json()))
        conflicts = tuple(
            sorted(
                ledger.conflicts,
                key=lambda conflict: (conflict.canonical_key, conflict.evidence_ids),
            )
        )
        return (ledger.request_id, items, gaps, conflicts)


    def continue_with(
        self,
        *,
        request: SupportRequest,
        identity: TrustedIdentity,
        permissions: frozenset[Permission],
        request_id: str,
        execution_id: str,
        step_limit: int | None = None,
        trusted_write_context: TrustedWriteContext | None = None,
    ) -> AgentState:
        """Cria uma invocação no mesmo thread, validando novamente seu escopo."""
        if execution_id == self.execution_id:
            raise ValueError("cada continuação exige um novo execution_id")
        persisted_request = PersistedSupportRequest.model_validate(request)
        same_request = request_id == self.request_id
        if same_request and persisted_request != self.request:
            raise ValueError("retomada da mesma request_id exige solicitação idêntica")
        data = {
            field_name: getattr(self, field_name)
            for field_name in type(self).model_fields
        }
        data.update(
            request=persisted_request,
            identity=identity,
            permissions=permissions,
            request_id=request_id,
            execution_id=execution_id,
            trusted_write_context=(
                self.trusted_write_context
                if same_request
                else trusted_write_context
            ),
        )
        if (
            same_request
            and self.review_request is not None
            and self.final_result is None
            and permissions != self.permissions
        ):
            # O atestado anterior é apenas a base imutável da revisão. A
            # permissão atual será atestada novamente depois do julgamento.
            data["release_gate"] = None
        if not same_request:
            history = self.ledger_history
            if self.ledger.request_id is not None:
                history = (*history, self.ledger)
            data.update(
                decision=None,
                step_count=0,
                step_limit=self.step_limit if step_limit is None else step_limit,
                pending_proposal=None,
                approval=None,
                final_result=None,
                review=None,
                planner_usage=PlannerUsage(request_id=request_id),
                resume_anchor=ResumeAnchor.START,
                planner_terminal=None,
                planner_failure=None,
                writer_draft=None,
                writer_failure=None,
                writer_attempts=0,
                release_gate=None,
                review_request=None,
                review_resolution=None,
                reviewed_draft=None,
                review_audit=None,
                review_expiry=None,
                ledger=EvidenceLedger(),
                ledger_history=history,
            )
        return type(self).model_validate(data)

    def advance_step(self) -> AgentState:
        """Avança uma unidade sem permitir ultrapassar o orçamento."""
        if self.step_count >= self.step_limit:
            raise ValueError("orçamento de passos esgotado")
        data = {
            field_name: getattr(self, field_name)
            for field_name in type(self).model_fields
        }
        data["step_count"] = self.step_count + 1
        return type(self).model_validate(data)
