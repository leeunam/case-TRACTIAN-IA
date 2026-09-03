"""Telemetria manual, sanitizada e opcional do agente.

Este módulo nunca importa o SDK Logfire no import da aplicação. A superfície
pública aceita somente contratos fechados; valores brutos de correlação são
pseudonimizados antes de chegar a qualquer backend.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import importlib
import os
from time import perf_counter
import secrets
from typing import Any, Literal, Protocol

from langgraph.errors import GraphInterrupt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TraceId(_StrictFrozenModel):
    """Identificador técnico opaco, independente do trace ID OpenTelemetry."""

    value: StrictStr = Field(pattern=r"^trc_[0-9a-f]{32}$")

    @classmethod
    def new(cls) -> TraceId:
        return cls(value=f"trc_{secrets.token_hex(16)}")


class CorrelationKind(str, Enum):
    REQUEST = "request"
    THREAD = "thread"
    EXECUTION = "execution"
    CASE = "case"
    COMPANY = "company"
    USER = "user"
    EXPERIMENT = "experiment"
    BENCHMARK_CASE = "benchmark_case"


class CorrelationReference(_StrictFrozenModel):
    value: StrictStr = Field(pattern=r"^hmac:v1:[0-9a-f]{32}$")


class Outcome(str, Enum):
    OK = "ok"
    ERROR = "error"
    SUSPENDED = "suspended"
    REPLAYED = "replayed"
    DENIED = "denied"
    UNCERTAIN = "uncertain"
    CANCELLED = "cancelled"


class ErrorCode(str, Enum):
    NONE = "none"
    PROTOCOL = "protocol_error"
    RUNTIME = "runtime_error"
    MODEL = "model_error"
    TOOL = "tool_error"
    ACTION = "action_error"
    POLICY = "policy_blocked"
    GATE = "gate_blocked"
    REVIEW = "review_blocked"
    CANCELLED = "cancelled"


class ResponseDecision(str, Enum):
    GUIDE = "guide"
    ACT = "act"
    ESCALATE = "escalate"
    REQUEST_INFORMATION = "request_information"
    REQUEST_CONFIRMATION = "request_confirmation"
    REQUIRE_HUMAN_REVIEW = "require_human_review"


class SpanName(str, Enum):
    REQUEST = "tractian.agent.request"
    NODE = "tractian.agent.node"
    PLANNER = "tractian.agent.planner"
    WRITER = "tractian.agent.writer"
    TOOL = "tractian.agent.tool"
    POLICY = "tractian.agent.policy"
    ACTION = "tractian.agent.action"
    GATE = "tractian.agent.gate"
    REVIEW = "tractian.agent.review"
    RESPONSE = "tractian.agent.response"
    EVALUATION = "tractian.agent.evaluation"


class MetricName(str, Enum):
    EXECUTIONS = "tractian.agent.executions"
    ERRORS = "tractian.agent.errors"
    STAGE_DURATION = "tractian.agent.stage.duration"


class GraphNodeName(str, Enum):
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


class PlannerOperation(str, Enum):
    SELECT = "select"
    FINALIZE = "finalize"


class PolicyOperation(str, Enum):
    EVALUATE = "evaluate"
    CONFIRM = "confirm"


class ReviewOperation(str, Enum):
    WAIT = "wait"
    RESUME = "resume"


class ActionName(str, Enum):
    REPROCESS_ANALYSIS = "reprocess_analysis"
    REQUEST_SPECIALIST_ANALYSIS = "request_specialist_analysis"
    UPDATE_ASSET_CRITICALITY = "update_asset_criticality"
    REQUEST_MODEL_RETRAINING = "request_model_retraining"
    ESCALATE_CASE = "escalate_case"


class ToolName(str, Enum):
    GET_ASSET = "get_asset"
    LIST_ASSET_ANALYSES = "list_asset_analyses"
    GET_ANALYSIS = "get_analysis"
    GET_BASELINE = "get_baseline"
    GET_RMS_SERIES = "get_rms_series"
    GET_SPECTRUM = "get_spectrum"
    GET_DATA_QUALITY = "get_data_quality"
    GET_MODEL = "get_model"
    SEARCH_KNOWLEDGE = "search_knowledge"
    GET_KNOWLEDGE_DOCUMENT = "get_knowledge_document"
    PROPOSE_REPROCESS_ANALYSIS = "propose_reprocess_analysis"
    PROPOSE_REQUEST_SPECIALIST_ANALYSIS = "propose_request_specialist_analysis"
    PROPOSE_UPDATE_ASSET_CRITICALITY = "propose_update_asset_criticality"
    PROPOSE_REQUEST_MODEL_RETRAINING = "propose_request_model_retraining"
    PROPOSE_ESCALATE_CASE = "propose_escalate_case"


class _SpanAttributes(_StrictFrozenModel):
    schema_version: Literal["1"] = "1"
    trace_id: StrictStr | None = Field(default=None, pattern=r"^trc_[0-9a-f]{32}$")
    outcome: Outcome = Outcome.OK
    error_code: ErrorCode = ErrorCode.NONE


class RequestSpanAttributes(_SpanAttributes):
    request_ref: StrictStr = Field(pattern=r"^hmac:v1:[0-9a-f]{32}$")
    thread_ref: StrictStr = Field(pattern=r"^hmac:v1:[0-9a-f]{32}$")
    execution_ref: StrictStr = Field(pattern=r"^hmac:v1:[0-9a-f]{32}$")
    case_ref: StrictStr = Field(pattern=r"^hmac:v1:[0-9a-f]{32}$")
    company_ref: StrictStr = Field(pattern=r"^hmac:v1:[0-9a-f]{32}$")
    user_ref: StrictStr = Field(pattern=r"^hmac:v1:[0-9a-f]{32}$")
    experiment_ref: StrictStr | None = Field(
        default=None, pattern=r"^hmac:v1:[0-9a-f]{32}$"
    )
    benchmark_case_ref: StrictStr | None = Field(
        default=None, pattern=r"^hmac:v1:[0-9a-f]{32}$"
    )
    planner_enabled: StrictBool


class NodeSpanAttributes(_SpanAttributes):
    node: GraphNodeName


class PlannerSpanAttributes(_SpanAttributes):
    operation: PlannerOperation


class WriterSpanAttributes(_SpanAttributes):
    attempt: StrictInt = Field(ge=1, le=2)


class ToolSpanAttributes(_SpanAttributes):
    tool: ToolName


class PolicySpanAttributes(_SpanAttributes):
    operation: PolicyOperation


class ActionSpanAttributes(_SpanAttributes):
    action: ActionName
    attempt: StrictInt = Field(ge=1, le=2)


class GateSpanAttributes(_SpanAttributes):
    pass


class ReviewSpanAttributes(_SpanAttributes):
    operation: ReviewOperation


class ResponseSpanAttributes(_SpanAttributes):
    planner_enabled: StrictBool
    replayed: StrictBool
    decision: ResponseDecision | None = None


class EvaluationSpanAttributes(_SpanAttributes):
    experiment_ref: StrictStr = Field(pattern=r"^hmac:v1:[0-9a-f]{32}$")
    benchmark_case_ref: StrictStr = Field(pattern=r"^hmac:v1:[0-9a-f]{32}$")


SpanAttributes = (
    RequestSpanAttributes
    | NodeSpanAttributes
    | PlannerSpanAttributes
    | WriterSpanAttributes
    | ToolSpanAttributes
    | PolicySpanAttributes
    | ActionSpanAttributes
    | GateSpanAttributes
    | ReviewSpanAttributes
    | ResponseSpanAttributes
    | EvaluationSpanAttributes
)


_ATTRIBUTE_TYPE_BY_NAME: Mapping[SpanName, type[_SpanAttributes]] = {
    SpanName.REQUEST: RequestSpanAttributes,
    SpanName.NODE: NodeSpanAttributes,
    SpanName.PLANNER: PlannerSpanAttributes,
    SpanName.WRITER: WriterSpanAttributes,
    SpanName.TOOL: ToolSpanAttributes,
    SpanName.POLICY: PolicySpanAttributes,
    SpanName.ACTION: ActionSpanAttributes,
    SpanName.GATE: GateSpanAttributes,
    SpanName.REVIEW: ReviewSpanAttributes,
    SpanName.RESPONSE: ResponseSpanAttributes,
    SpanName.EVALUATION: EvaluationSpanAttributes,
}


class ExecutionCorrelations(_StrictFrozenModel):
    request_id: StrictStr = Field(min_length=1, max_length=4096)
    thread_id: StrictStr = Field(min_length=1, max_length=4096)
    execution_id: StrictStr = Field(min_length=1, max_length=4096)
    case_id: StrictStr = Field(min_length=1, max_length=4096)
    company_id: StrictStr = Field(min_length=1, max_length=4096)
    user_id: StrictStr = Field(min_length=1, max_length=4096)
    planner_enabled: StrictBool
    review_resumed: StrictBool = False
    experiment_id: StrictStr | None = Field(default=None, min_length=1, max_length=4096)
    benchmark_case_id: StrictStr | None = Field(
        default=None, min_length=1, max_length=4096
    )

    @model_validator(mode="after")
    def _paired_evaluation_correlations(self) -> ExecutionCorrelations:
        if (self.experiment_id is None) != (self.benchmark_case_id is None):
            raise ValueError("correlações de avaliação devem ser fornecidas juntas")
        return self


PrimitiveAttribute = str | bool | int
AttributeItems = tuple[tuple[str, PrimitiveAttribute], ...]


@dataclass(frozen=True, slots=True)
class RecordedSpan:
    name: SpanName
    attributes: AttributeItems
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class RecordedMetric:
    name: MetricName
    value: float | int
    labels: AttributeItems
    outcome: Outcome
    error_code: ErrorCode


class _BackendSpan(Protocol):
    def finish(self, outcome: Outcome, error_code: ErrorCode) -> None: ...

    def close(self, duration_seconds: float) -> None: ...


class _TelemetryBackend(Protocol):
    def start_span(
        self, name: SpanName, attributes: Mapping[str, PrimitiveAttribute]
    ) -> _BackendSpan: ...

    def record_metric(
        self,
        name: MetricName,
        value: float | int,
        labels: Mapping[str, PrimitiveAttribute],
    ) -> None: ...


class _NullBackendSpan:
    def finish(self, outcome: Outcome, error_code: ErrorCode) -> None:
        return None

    def close(self, duration_seconds: float) -> None:
        return None


class _NullBackend:
    def start_span(
        self, name: SpanName, attributes: Mapping[str, PrimitiveAttribute]
    ) -> _BackendSpan:
        return _NullBackendSpan()

    def record_metric(
        self,
        name: MetricName,
        value: float | int,
        labels: Mapping[str, PrimitiveAttribute],
    ) -> None:
        return None


class _RecordingBackendSpan:
    def __init__(
        self,
        backend: _RecordingBackend,
        name: SpanName,
        attributes: Mapping[str, PrimitiveAttribute],
    ) -> None:
        self._backend = backend
        self._name = name
        self._attributes = dict(attributes)

    def finish(self, outcome: Outcome, error_code: ErrorCode) -> None:
        self._attributes["outcome"] = outcome.value
        self._attributes["error_code"] = error_code.value

    def close(self, duration_seconds: float) -> None:
        self._backend.spans.append(
            RecordedSpan(
                name=self._name,
                attributes=tuple(sorted(self._attributes.items())),
                duration_seconds=duration_seconds,
            )
        )


class _RecordingBackend:
    def __init__(self) -> None:
        self.spans: list[RecordedSpan] = []
        self.metrics: list[RecordedMetric] = []

    def start_span(
        self, name: SpanName, attributes: Mapping[str, PrimitiveAttribute]
    ) -> _BackendSpan:
        return _RecordingBackendSpan(self, name, attributes)

    def record_metric(
        self,
        name: MetricName,
        value: float | int,
        labels: Mapping[str, PrimitiveAttribute],
    ) -> None:
        self.metrics.append(
            RecordedMetric(
                name=name,
                value=value,
                labels=tuple(sorted(labels.items())),
                outcome=Outcome(str(labels["outcome"])),
                error_code=ErrorCode(str(labels["error_code"])),
            )
        )


_CURRENT_TRACE: ContextVar[ExecutionTrace | None] = ContextVar(
    "tractian_agent_execution_trace", default=None
)
_CURRENT_ACTION_ATTEMPT: ContextVar[tuple[ActionName, int] | None] = ContextVar(
    "tractian_agent_action_attempt", default=None
)


class _SpanScope:
    def __init__(
        self,
        trace: ExecutionTrace,
        name: SpanName,
        attributes: _SpanAttributes,
    ) -> None:
        self._trace = trace
        self._name = name
        self._attributes = attributes
        self._handle: _BackendSpan | None = None
        self._started_at = 0.0
        self._outcome = attributes.outcome
        self._error_code = attributes.error_code

    def __enter__(self) -> _SpanScope:
        self._started_at = perf_counter()
        try:
            self._handle = self._trace._backend.start_span(
                self._name,
                self._attributes.model_dump(mode="json", exclude_none=True),
            )
        except BaseException:
            self._handle = None
        return self

    def finish(
        self,
        outcome: Outcome,
        error_code: ErrorCode = ErrorCode.NONE,
    ) -> None:
        if not isinstance(outcome, Outcome) or not isinstance(error_code, ErrorCode):
            raise TypeError("outcome e error_code devem pertencer aos catálogos")
        self._outcome = outcome
        self._error_code = error_code

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if exc_type is not None and self._outcome is Outcome.OK:
            if isinstance(exc, GraphInterrupt):
                self._outcome = Outcome.SUSPENDED
                self._error_code = ErrorCode.NONE
            else:
                is_cancelled = isinstance(exc, asyncio.CancelledError)
                self._outcome = Outcome.CANCELLED if is_cancelled else Outcome.ERROR
                self._error_code = (
                    ErrorCode.CANCELLED if is_cancelled else ErrorCode.RUNTIME
                )
        duration = max(0.0, perf_counter() - self._started_at)
        if self._handle is not None:
            try:
                self._handle.finish(self._outcome, self._error_code)
            except BaseException:
                pass
            try:
                self._handle.close(duration)
            except BaseException:
                pass
        try:
            self._trace._record_metrics(
                self._name,
                duration,
                self._outcome,
                self._error_code,
                replayed=(
                    self._outcome is Outcome.REPLAYED or self._attributes.replayed
                    if isinstance(self._attributes, ResponseSpanAttributes)
                    else self._outcome is Outcome.REPLAYED
                ),
            )
        except BaseException:
            pass
        # Nunca suprime a exceção do negócio.
        return False


class ExecutionTrace:
    def __init__(
        self,
        *,
        trace_id: TraceId,
        request_attributes: RequestSpanAttributes,
        backend: _TelemetryBackend,
        planner_enabled: bool,
        review_resumed: bool,
    ) -> None:
        self.trace_id = trace_id
        self.request_attributes = request_attributes
        self._backend = backend
        self._planner_enabled = planner_enabled
        self.review_resumed = review_resumed

    @contextmanager
    def activate(self) -> Iterator[ExecutionTrace]:
        token = _CURRENT_TRACE.set(self)
        try:
            yield self
        finally:
            _CURRENT_TRACE.reset(token)

    def span(self, name: SpanName, attributes: SpanAttributes) -> _SpanScope:
        if not isinstance(name, SpanName):
            raise TypeError("o nome do span deve pertencer à allowlist")
        expected = _ATTRIBUTE_TYPE_BY_NAME[name]
        if type(attributes) is not expected:
            raise TypeError("os atributos não correspondem ao tipo do span")
        correlated = attributes.model_copy(update={"trace_id": self.trace_id.value})
        return _SpanScope(self, name, correlated)

    def evaluation_span(self, attributes: EvaluationSpanAttributes) -> _SpanScope:
        return self.span(SpanName.EVALUATION, attributes)

    def _record_metrics(
        self,
        name: SpanName,
        duration: float,
        outcome: Outcome,
        error_code: ErrorCode,
        *,
        replayed: bool,
    ) -> None:
        labels: dict[str, PrimitiveAttribute] = {
            "stage": name.value.removeprefix("tractian.agent."),
            "outcome": outcome.value,
            "error_code": error_code.value,
            "planner_enabled": self._planner_enabled,
            "replayed": replayed,
        }
        operations: list[tuple[MetricName, float | int]] = [
            (MetricName.STAGE_DURATION, duration)
        ]
        if name is SpanName.REQUEST:
            operations.append((MetricName.EXECUTIONS, 1))
        if outcome in {Outcome.ERROR, Outcome.CANCELLED}:
            operations.append((MetricName.ERRORS, 1))
        for metric_name, value in operations:
            try:
                self._backend.record_metric(metric_name, value, labels)
            except BaseException:
                pass


def current_execution_trace() -> ExecutionTrace | None:
    return _CURRENT_TRACE.get()


class _FailOpenSpanHandle:
    """Suprime somente falhas da fachada, nunca exceções do negócio."""

    def __init__(self, handle: object | None) -> None:
        self._handle = handle

    def finish(
        self,
        outcome: Outcome,
        error_code: ErrorCode = ErrorCode.NONE,
    ) -> None:
        if self._handle is None:
            return
        try:
            finish = getattr(self._handle, "finish")
            finish(outcome, error_code)
        except BaseException:
            pass


@contextmanager
def activate_trace_fail_open(trace: ExecutionTrace) -> Iterator[ExecutionTrace]:
    """Ativa o contexto conhecido mesmo se a implementação injetada falhar."""

    outer_token = _CURRENT_TRACE.set(trace)
    activation = None
    entered = False
    try:
        try:
            activation = trace.activate()
            activation.__enter__()
            entered = True
        except BaseException:
            activation = None
        active_token = _CURRENT_TRACE.set(trace)
        try:
            yield trace
        finally:
            _CURRENT_TRACE.reset(active_token)
            if entered and activation is not None:
                try:
                    activation.__exit__(None, None, None)
                except BaseException:
                    pass
    finally:
        _CURRENT_TRACE.reset(outer_token)


@contextmanager
def span_fail_open(
    trace: ExecutionTrace,
    name: SpanName,
    attributes: SpanAttributes,
) -> Iterator[_FailOpenSpanHandle]:
    """Executa todo o protocolo do span sem entregar exceções do negócio."""

    context_manager = None
    entered = False
    handle = None
    try:
        try:
            context_manager = trace.span(name, attributes)
            handle = context_manager.__enter__()
            entered = True
        except BaseException:
            context_manager = None
        yield _FailOpenSpanHandle(handle)
    finally:
        if entered and context_manager is not None:
            try:
                context_manager.__exit__(None, None, None)
            except BaseException:
                pass


@contextmanager
def bind_action_attempt(action: ActionName, attempt: int) -> Iterator[None]:
    """Propaga somente catálogo/ordinal até a fronteira HTTP modificadora."""

    if (
        not isinstance(action, ActionName)
        or type(attempt) is not int
        or not 1 <= attempt <= 2
    ):
        raise TypeError("tentativa de ação deve usar catálogo e ordinal válidos")
    token = _CURRENT_ACTION_ATTEMPT.set((action, attempt))
    try:
        yield
    finally:
        _CURRENT_ACTION_ATTEMPT.reset(token)


def current_action_attempt() -> tuple[ActionName, int] | None:
    return _CURRENT_ACTION_ATTEMPT.get()


class AgentTelemetry:
    def __init__(self, *, pseudonym_key: bytes, backend: _TelemetryBackend) -> None:
        self._pseudonymizer = Pseudonymizer(pseudonym_key)
        self._backend = backend

    def start_execution(self, correlations: ExecutionCorrelations) -> ExecutionTrace:
        if not isinstance(correlations, ExecutionCorrelations):
            raise TypeError("correlações devem usar o contrato estrito")
        trace_id = TraceId.new()

        def ref(kind: CorrelationKind, raw: str) -> str:
            return self._pseudonymizer.reference(kind, raw).value

        attributes = RequestSpanAttributes(
            trace_id=trace_id.value,
            request_ref=ref(CorrelationKind.REQUEST, correlations.request_id),
            thread_ref=ref(CorrelationKind.THREAD, correlations.thread_id),
            execution_ref=ref(CorrelationKind.EXECUTION, correlations.execution_id),
            case_ref=ref(CorrelationKind.CASE, correlations.case_id),
            company_ref=ref(CorrelationKind.COMPANY, correlations.company_id),
            user_ref=ref(CorrelationKind.USER, correlations.user_id),
            experiment_ref=(
                ref(CorrelationKind.EXPERIMENT, correlations.experiment_id)
                if correlations.experiment_id is not None
                else None
            ),
            benchmark_case_ref=(
                ref(CorrelationKind.BENCHMARK_CASE, correlations.benchmark_case_id)
                if correlations.benchmark_case_id is not None
                else None
            ),
            planner_enabled=correlations.planner_enabled,
        )
        return ExecutionTrace(
            trace_id=trace_id,
            request_attributes=attributes,
            backend=self._backend,
            planner_enabled=correlations.planner_enabled,
            review_resumed=correlations.review_resumed,
        )


class Pseudonymizer:
    """Deriva referências não reversíveis com separação explícita de domínio."""

    _DOMAIN = b"tractian-observability:hmac:v1\x00"

    def __init__(self, key: bytes) -> None:
        if not isinstance(key, bytes) or not 32 <= len(key) <= 4096:
            raise ValueError("a chave de pseudonimização deve ter entre 32 B e 4 KiB")
        self._key = key

    def reference(
        self,
        kind: CorrelationKind,
        value: str,
    ) -> CorrelationReference:
        if not isinstance(kind, CorrelationKind):
            raise TypeError("o tipo de correlação deve pertencer ao catálogo")
        if not isinstance(value, str) or not value or len(value) > 4096:
            raise ValueError("o identificador bruto deve ser uma string limitada")
        material = (
            self._DOMAIN + kind.value.encode("ascii") + b"\x00" + value.encode("utf-8")
        )
        digest = hmac.new(self._key, material, hashlib.sha256).digest()[:16]
        return CorrelationReference(value=f"hmac:v1:{digest.hex()}")


class NullTelemetry(AgentTelemetry):
    """Fachada padrão: produz correlação local, mas não exporta nada."""

    def __init__(self) -> None:
        super().__init__(pseudonym_key=secrets.token_bytes(32), backend=_NullBackend())


def start_execution_fail_open(
    telemetry: AgentTelemetry,
    correlations: ExecutionCorrelations,
) -> ExecutionTrace:
    """Isola inclusive ``BaseException`` originada pela telemetria injetada."""

    try:
        trace = telemetry.start_execution(correlations)
        if not isinstance(trace, ExecutionTrace):
            raise TypeError("telemetria devolveu trace incompatível")
        if (
            type(trace.trace_id) is not TraceId
            or type(trace.request_attributes) is not RequestSpanAttributes
            or trace.request_attributes.trace_id != trace.trace_id.value
            or type(trace.review_resumed) is not bool
        ):
            raise TypeError("telemetria devolveu metadados de trace incompatíveis")
        return trace
    except BaseException:
        return NullTelemetry().start_execution(correlations)


class RecordingTelemetry(AgentTelemetry):
    """Recorder em memória que conserva somente o contrato já sanitizado."""

    def __init__(self, *, pseudonym_key: bytes) -> None:
        backend = _RecordingBackend()
        super().__init__(pseudonym_key=pseudonym_key, backend=backend)
        self._recording_backend = backend

    @property
    def spans(self) -> tuple[RecordedSpan, ...]:
        return tuple(self._recording_backend.spans)

    @property
    def metrics(self) -> tuple[RecordedMetric, ...]:
        return tuple(self._recording_backend.metrics)


class _LogfireBackendSpan:
    def __init__(
        self,
        context_manager: Any,
        span: Any,
    ) -> None:
        self._context_manager = context_manager
        self._span = span

    def finish(self, outcome: Outcome, error_code: ErrorCode) -> None:
        setter = getattr(self._span, "set_attribute", None)
        if callable(setter):
            setter("outcome", outcome.value)
            setter("error_code", error_code.value)

    def close(self, duration_seconds: float) -> None:
        # Nunca encaminhar a exceção do negócio ao SDK.
        self._context_manager.__exit__(None, None, None)


class _LogfireBackend:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._instruments: dict[MetricName, Any] = {}

    def start_span(
        self, name: SpanName, attributes: Mapping[str, PrimitiveAttribute]
    ) -> _BackendSpan:
        context_manager = self._client.span(name.value, **dict(attributes))
        span = context_manager.__enter__()
        return _LogfireBackendSpan(context_manager, span)

    def record_metric(
        self,
        name: MetricName,
        value: float | int,
        labels: Mapping[str, PrimitiveAttribute],
    ) -> None:
        instrument = self._instruments.get(name)
        if instrument is None:
            factory_name = (
                "metric_histogram"
                if name is MetricName.STAGE_DURATION
                else "metric_counter"
            )
            instrument = getattr(self._client, factory_name)(name.value)
            self._instruments[name] = instrument
        operation = "record" if name is MetricName.STAGE_DURATION else "add"
        getattr(instrument, operation)(value, dict(labels))


class LogfireTelemetry(AgentTelemetry):
    """Adaptador estreito sobre uma instância Logfire configurada manualmente."""

    def __init__(self, *, pseudonym_key: bytes, client: Any) -> None:
        super().__init__(
            pseudonym_key=pseudonym_key,
            backend=_LogfireBackend(client),
        )


def _default_sdk_loader() -> Any:
    return importlib.import_module("logfire")


def _valid_token(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 4096
        and value == value.strip()
        and "," not in value
        and not any(character.isspace() for character in value)
    )


def _pseudonym_key(environ: Mapping[str, str]) -> bytes | None:
    raw = environ.get("TRACTIAN_LOGFIRE_PSEUDONYM_KEY")
    if not isinstance(raw, str):
        return None
    try:
        encoded = raw.encode("utf-8")
    except UnicodeEncodeError:
        return None
    return encoded if 32 <= len(encoded) <= 4096 else None


def build_agent_telemetry(
    environ: Mapping[str, str] | None = None,
    *,
    sdk_loader: Callable[[], Any] | None = None,
) -> AgentTelemetry:
    """Habilita exportação apenas após validar todo o opt-in local."""

    environment = os.environ if environ is None else environ
    source: dict[str, str] = {}
    try:
        for name in (
            "TRACTIAN_LOGFIRE_ENABLED",
            "LOGFIRE_TOKEN",
            "TRACTIAN_LOGFIRE_PSEUDONYM_KEY",
        ):
            try:
                value = environment[name]
            except KeyError:
                pass
            else:
                if type(value) is not str:
                    return NullTelemetry()
                source[name] = value
        if source.get("TRACTIAN_LOGFIRE_ENABLED") != "true":
            return NullTelemetry()
        if not _valid_token(source.get("LOGFIRE_TOKEN")):
            return NullTelemetry()
        pseudonym_key = _pseudonym_key(source)
        if pseudonym_key is None:
            return NullTelemetry()
        token = source["LOGFIRE_TOKEN"]
    except BaseException:
        return NullTelemetry()
    assert isinstance(token, str) and pseudonym_key is not None
    try:
        sdk = (sdk_loader or _default_sdk_loader)()
        configured = sdk.configure(
            local=True,
            send_to_logfire="if-token-present",
            token=token,
            service_name="tractian-agent",
            service_version="0.1.0",
            environment="runtime",
            console=False,
            inspect_arguments=False,
            add_baggage_to_attributes=False,
            distributed_tracing=False,
            advanced=sdk.AdvancedOptions(
                emit_configuration_span=False,
                resource_detectors=(),
            ),
        )
    except BaseException:
        return NullTelemetry()
    return LogfireTelemetry(pseudonym_key=pseudonym_key, client=configured)
