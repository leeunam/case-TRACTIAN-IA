from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager
import json
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
import httpx
from pydantic import ValidationError

from tractian_agent.client import IndustrialApiClient
from tractian_agent.checkpoint import open_checkpointer
from tractian_agent.contracts import Identity, SupportRequest
from tractian_agent.entrypoint import (
    AgentInvocationResult,
    invoke_agent,
    invoke_agent_observed,
)
from tractian_agent.graph import build_agent_graph

from tractian_agent.observability import (
    ActionName,
    CorrelationKind,
    ErrorCode,
    EvaluationSpanAttributes,
    ExecutionCorrelations,
    GraphNodeName,
    LogfireTelemetry,
    NodeSpanAttributes,
    NullTelemetry,
    Outcome,
    Pseudonymizer,
    RecordingTelemetry,
    RequestSpanAttributes,
    ResponseDecision,
    ResponseSpanAttributes,
    SpanName,
    TraceId,
    bind_action_attempt,
    build_agent_telemetry,
    current_execution_trace,
)
from tractian_agent.tools.runtime import ReadToolRuntime
from tractian_agent.tools.runtime import WriteToolRuntime
from tractian_agent.state import AgentState
from tractian_agent.write_policy import (
    ApprovalSource,
    ReprocessProposal,
    TrustedActionApproval,
    UpdateAssetCriticalityProposal,
)
from tractian_agent.write_operations import execute_update_asset_criticality


def test_trace_id_and_hmac_references_are_opaque_and_domain_separated():
    first_trace = TraceId.new()
    second_trace = TraceId.new()
    assert re.fullmatch(r"trc_[0-9a-f]{32}", first_trace.value)
    assert first_trace != second_trace

    pseudonymizer = Pseudonymizer(b"k" * 32)
    request_ref = pseudonymizer.reference(CorrelationKind.REQUEST, "literal-id")
    same_request_ref = pseudonymizer.reference(CorrelationKind.REQUEST, "literal-id")
    thread_ref = pseudonymizer.reference(CorrelationKind.THREAD, "literal-id")

    assert request_ref == same_request_ref
    assert re.fullmatch(r"hmac:v1:[0-9a-f]{32}", request_ref.value)
    assert request_ref != thread_ref
    assert "literal-id" not in request_ref.value


@pytest.mark.parametrize(
    "environ",
    [
        {},
        {"TRACTIAN_LOGFIRE_ENABLED": "TRUE"},
        {
            "TRACTIAN_LOGFIRE_ENABLED": "true",
            "LOGFIRE_TOKEN": " token",
            "TRACTIAN_LOGFIRE_PSEUDONYM_KEY": "k" * 32,
        },
        {
            "TRACTIAN_LOGFIRE_ENABLED": "true",
            "LOGFIRE_TOKEN": "two,tokens",
            "TRACTIAN_LOGFIRE_PSEUDONYM_KEY": "k" * 32,
        },
        {
            "TRACTIAN_LOGFIRE_ENABLED": "true",
            "LOGFIRE_TOKEN": "token",
            "TRACTIAN_LOGFIRE_PSEUDONYM_KEY": "é" * 15,
        },
        {
            "TRACTIAN_LOGFIRE_ENABLED": "true",
            "LOGFIRE_TOKEN": "token",
            "TRACTIAN_LOGFIRE_PSEUDONYM_KEY": "k" * 4097,
        },
    ],
)
def test_incomplete_opt_in_never_loads_the_sdk(environ):
    loads = 0

    def loader():
        nonlocal loads
        loads += 1
        raise AssertionError("the loader must not be called")

    telemetry = build_agent_telemetry(environ, sdk_loader=loader)

    assert isinstance(telemetry, NullTelemetry)
    assert loads == 0


def test_recording_telemetry_accepts_only_typed_spans_and_safe_metric_labels():
    telemetry = RecordingTelemetry(pseudonym_key=b"p" * 32)
    trace = telemetry.start_execution(
        ExecutionCorrelations(
            request_id="raw-request",
            thread_id="raw-thread",
            execution_id="raw-execution",
            case_id="raw-case",
            company_id="raw-company",
            user_id="raw-user",
            planner_enabled=True,
        )
    )

    with trace.activate():
        with trace.span(SpanName.REQUEST, trace.request_attributes):
            with trace.span(
                SpanName.NODE,
                NodeSpanAttributes(node=GraphNodeName.INGEST),
            ):
                pass

    assert [span.name for span in telemetry.spans] == [
        SpanName.NODE,
        SpanName.REQUEST,
    ]
    root_attributes = dict(telemetry.spans[-1].attributes)
    assert RequestSpanAttributes.model_validate_json(json.dumps(root_attributes))
    serialized = repr(telemetry.spans) + repr(telemetry.metrics)
    for raw in (
        "raw-request",
        "raw-thread",
        "raw-execution",
        "raw-case",
        "raw-company",
        "raw-user",
    ):
        assert raw not in serialized
    assert all(
        set(dict(metric.labels))
        == {"stage", "outcome", "error_code", "planner_enabled", "replayed"}
        for metric in telemetry.metrics
    )
    assert {metric.name.value for metric in telemetry.metrics} >= {
        "tractian.agent.executions",
        "tractian.agent.stage.duration",
    }
    assert all(metric.outcome is Outcome.OK for metric in telemetry.metrics)
    assert all(metric.error_code is ErrorCode.NONE for metric in telemetry.metrics)

    with pytest.raises(TypeError):
        trace.span(SpanName.NODE, trace.request_attributes)

    with pytest.raises(ValidationError):
        ResponseSpanAttributes(
            planner_enabled=True,
            replayed=False,
            decision="texto livre",
        )

    assert (
        ResponseSpanAttributes(
            planner_enabled=True,
            replayed=False,
            decision=ResponseDecision.GUIDE,
        ).decision
        is ResponseDecision.GUIDE
    )


def test_complete_opt_in_configures_only_the_manual_logfire_instance():
    calls = []
    instrument_calls = []

    class AdvancedOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    configured = SimpleNamespace()

    def configure(**kwargs):
        calls.append(kwargs)
        return configured

    sdk = SimpleNamespace(
        AdvancedOptions=AdvancedOptions,
        configure=configure,
        instrument_langchain=lambda: instrument_calls.append("langchain"),
        instrument_httpx=lambda: instrument_calls.append("httpx"),
        instrument_fastapi=lambda: instrument_calls.append("fastapi"),
        instrument_pydantic=lambda: instrument_calls.append("pydantic"),
    )
    telemetry = build_agent_telemetry(
        {
            "TRACTIAN_LOGFIRE_ENABLED": "true",
            "LOGFIRE_TOKEN": "pylf_v1_local-token",
            "TRACTIAN_LOGFIRE_PSEUDONYM_KEY": "p" * 32,
        },
        sdk_loader=lambda: sdk,
    )

    assert isinstance(telemetry, LogfireTelemetry)
    assert len(calls) == 1
    options = calls[0].pop("advanced")
    assert options.kwargs == {
        "emit_configuration_span": False,
        "resource_detectors": (),
    }
    assert calls[0] == {
        "local": True,
        "send_to_logfire": "if-token-present",
        "token": "pylf_v1_local-token",
        "service_name": "tractian-agent",
        "service_version": "0.1.0",
        "environment": "runtime",
        "console": False,
        "inspect_arguments": False,
        "add_baggage_to_attributes": False,
        "distributed_tracing": False,
    }
    assert instrument_calls == []


@pytest.mark.parametrize("failure_stage", ["import", "configure"])
def test_sdk_import_or_configuration_failure_returns_null(failure_stage):
    class AdvancedOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def configure(**kwargs):
        raise RuntimeError("configuration-secret-sentinel")

    def loader():
        if failure_stage == "import":
            raise ImportError("import-secret-sentinel")
        return SimpleNamespace(AdvancedOptions=AdvancedOptions, configure=configure)

    telemetry = build_agent_telemetry(
        {
            "TRACTIAN_LOGFIRE_ENABLED": "true",
            "LOGFIRE_TOKEN": "token",
            "TRACTIAN_LOGFIRE_PSEUDONYM_KEY": "p" * 32,
        },
        sdk_loader=loader,
    )

    assert isinstance(telemetry, NullTelemetry)


def test_opt_in_uses_one_protected_environment_snapshot():
    original = {
        "TRACTIAN_LOGFIRE_ENABLED": "true",
        "LOGFIRE_TOKEN": "original-token",
        "TRACTIAN_LOGFIRE_PSEUDONYM_KEY": "p" * 32,
    }
    replacement = {
        "TRACTIAN_LOGFIRE_ENABLED": "false",
        "LOGFIRE_TOKEN": "replacement-token",
        "TRACTIAN_LOGFIRE_PSEUDONYM_KEY": "q" * 32,
    }

    class MutatingEnvironment(Mapping):
        def __init__(self):
            self.reads = {key: 0 for key in original}

        def __iter__(self):
            return iter(original)

        def __len__(self):
            return len(original)

        def __getitem__(self, key):
            value = original[key] if self.reads[key] == 0 else replacement[key]
            self.reads[key] += 1
            return value

    calls = []

    class AdvancedOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    configured = SimpleNamespace()
    sdk = SimpleNamespace(
        AdvancedOptions=AdvancedOptions,
        configure=lambda **kwargs: calls.append(kwargs) or configured,
    )
    environment = MutatingEnvironment()

    telemetry = build_agent_telemetry(environment, sdk_loader=lambda: sdk)

    assert isinstance(telemetry, LogfireTelemetry)
    assert calls[0]["token"] == "original-token"
    assert environment.reads == {key: 1 for key in original}


def test_environment_snapshot_failure_is_null_and_inert():
    class BrokenEnvironment(Mapping):
        def __iter__(self):
            raise _TelemetryBaseError

        def __len__(self):
            raise _TelemetryBaseError

        def __getitem__(self, key):
            raise _TelemetryBaseError

    assert isinstance(build_agent_telemetry(BrokenEnvironment()), NullTelemetry)


@pytest.mark.parametrize("broken_field", ["token", "pseudonym_key"])
def test_environment_value_failure_is_null_and_never_loads_sdk(broken_field):
    class BrokenToken(str):
        def strip(self, *args, **kwargs):
            raise _TelemetryBaseError

    class BrokenPseudonymKey(str):
        def encode(self, *args, **kwargs):
            raise _TelemetryBaseError

    loads = 0

    def loader():
        nonlocal loads
        loads += 1
        raise AssertionError("the loader must not be called")

    environment = {
        "TRACTIAN_LOGFIRE_ENABLED": "true",
        "LOGFIRE_TOKEN": (BrokenToken("token") if broken_field == "token" else "token"),
        "TRACTIAN_LOGFIRE_PSEUDONYM_KEY": (
            BrokenPseudonymKey("p" * 32)
            if broken_field == "pseudonym_key"
            else "p" * 32
        ),
    }

    telemetry = build_agent_telemetry(environment, sdk_loader=loader)

    assert isinstance(telemetry, NullTelemetry)
    assert loads == 0


def test_build_telemetry_without_argument_reads_real_environment_snapshot(
    monkeypatch,
):
    monkeypatch.setenv("TRACTIAN_LOGFIRE_ENABLED", "true")
    monkeypatch.setenv("LOGFIRE_TOKEN", "environment-token")
    monkeypatch.setenv("TRACTIAN_LOGFIRE_PSEUDONYM_KEY", "p" * 32)
    calls = []

    class AdvancedOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    configured = SimpleNamespace()
    sdk = SimpleNamespace(
        AdvancedOptions=AdvancedOptions,
        configure=lambda **kwargs: calls.append(kwargs) or configured,
    )

    telemetry = build_agent_telemetry(sdk_loader=lambda: sdk)

    assert isinstance(telemetry, LogfireTelemetry)
    assert calls[0]["token"] == "environment-token"


@pytest.mark.parametrize(
    ("existing", "expected"),
    [
        (None, "logfire-plugin"),
        ("another-plugin", "another-plugin,logfire-plugin"),
        ("true", "true"),
        ("TRUE", "TRUE,logfire-plugin"),
        ("1", "1"),
        ("__all__", "__all__"),
    ],
)
def test_fresh_default_import_disables_logfire_pydantic_plugin_before_models(
    existing,
    expected,
):
    environment = os.environ.copy()
    environment.update(
        {
            "TRACTIAN_LOGFIRE_ENABLED": "true",
            "LOGFIRE_TOKEN": "token-sentinel",
            "TRACTIAN_LOGFIRE_PSEUDONYM_KEY": "key-sentinel-" + "k" * 32,
            "LOGFIRE_PYDANTIC_PLUGIN_RECORD": "all",
        }
    )
    if existing is None:
        environment.pop("PYDANTIC_DISABLE_PLUGINS", None)
    else:
        environment["PYDANTIC_DISABLE_PLUGINS"] = existing
    code = """
import json
import os
import sys
from tractian_agent.contracts import Identity, SupportRequest
import tractian_agent.state
import tractian_agent.entrypoint
SupportRequest(
    case_id="case_plugin_guard",
    ticket_id="TKT-PLUGIN-GUARD",
    asset_id="asset_plugin_guard",
    message="request-message-sentinel",
    identity=Identity(user_id="user-plugin-guard", company_id="company-plugin-guard"),
)
print(json.dumps({
    "disabled": os.environ.get("PYDANTIC_DISABLE_PLUGINS"),
    "logfire_modules": sorted(name for name in sys.modules if name == "logfire" or name.startswith("logfire.")),
}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result = json.loads(completed.stdout)

    assert result["disabled"] == expected
    assert result["logfire_modules"] == []


def test_logfire_backend_failure_is_inert_and_never_receives_business_exception():
    exits = []
    attributes = []

    class Span:
        def set_attribute(self, key, value):
            attributes.append((key, value))

    class SpanContext:
        def __enter__(self):
            return Span()

        def __exit__(self, *args):
            exits.append(args)
            raise RuntimeError("backend-close-sentinel")

    class BrokenMetric:
        def add(self, *args):
            raise RuntimeError("backend-counter-sentinel")

        def record(self, *args):
            raise RuntimeError("backend-histogram-sentinel")

    client = SimpleNamespace(
        span=lambda *args, **kwargs: SpanContext(),
        metric_counter=lambda *args: BrokenMetric(),
        metric_histogram=lambda *args: BrokenMetric(),
    )
    telemetry = LogfireTelemetry(pseudonym_key=b"p" * 32, client=client)
    trace = telemetry.start_execution(
        ExecutionCorrelations(
            request_id="req",
            thread_id="thread",
            execution_id="exec",
            case_id="case",
            company_id="company",
            user_id="user",
            planner_enabled=False,
        )
    )
    sentinel = LookupError("business-secret-sentinel")
    caught = None

    with pytest.raises(LookupError) as raised:
        with trace.activate():
            assert current_execution_trace() is trace
            with trace.span(
                SpanName.NODE,
                NodeSpanAttributes(node=GraphNodeName.INGEST),
            ):
                raise sentinel
    caught = raised.value

    assert caught is sentinel
    assert caught.__cause__ is None
    assert exits == [(None, None, None)]
    assert attributes == [
        ("outcome", Outcome.ERROR.value),
        ("error_code", ErrorCode.RUNTIME.value),
    ]
    assert current_execution_trace() is None


def test_logfire_span_start_failure_is_inert():
    def fail_start(*args, **kwargs):
        raise RuntimeError("backend-start-secret-sentinel")

    telemetry = LogfireTelemetry(
        pseudonym_key=b"p" * 32,
        client=SimpleNamespace(span=fail_start),
    )
    trace = telemetry.start_execution(
        ExecutionCorrelations(
            request_id="req",
            thread_id="thread",
            execution_id="exec",
            case_id="case",
            company_id="company",
            user_id="user",
            planner_enabled=False,
        )
    )

    with trace.activate():
        with trace.span(
            SpanName.NODE,
            NodeSpanAttributes(node=GraphNodeName.INGEST),
        ):
            assert current_execution_trace() is trace

    assert current_execution_trace() is None


def test_logfire_span_attribute_failure_still_closes_neutrally():
    exits = []

    class BrokenSpan:
        def set_attribute(self, key, value):
            raise RuntimeError("backend-attribute-secret-sentinel")

    class SpanContext:
        def __enter__(self):
            return BrokenSpan()

        def __exit__(self, *args):
            exits.append(args)

    class Metric:
        def record(self, value, labels):
            pass

    telemetry = LogfireTelemetry(
        pseudonym_key=b"p" * 32,
        client=SimpleNamespace(
            span=lambda *args, **kwargs: SpanContext(),
            metric_histogram=lambda *args: Metric(),
        ),
    )
    trace = telemetry.start_execution(
        ExecutionCorrelations(
            request_id="req",
            thread_id="thread",
            execution_id="exec",
            case_id="case",
            company_id="company",
            user_id="user",
            planner_enabled=False,
        )
    )

    with trace.span(
        SpanName.NODE,
        NodeSpanAttributes(node=GraphNodeName.INGEST),
    ):
        pass

    assert exits == [(None, None, None)]


def test_evaluation_span_is_explicit_and_pseudonymized():
    telemetry = RecordingTelemetry(pseudonym_key=b"p" * 32)
    trace = telemetry.start_execution(
        ExecutionCorrelations(
            request_id="request",
            thread_id="thread",
            execution_id="execution",
            case_id="case",
            company_id="company",
            user_id="user",
            planner_enabled=True,
            experiment_id="raw-experiment-sentinel",
            benchmark_case_id="raw-benchmark-sentinel",
        )
    )
    assert trace.request_attributes.experiment_ref is not None
    assert trace.request_attributes.benchmark_case_ref is not None

    with trace.activate():
        with trace.evaluation_span(
            EvaluationSpanAttributes(
                experiment_ref=trace.request_attributes.experiment_ref,
                benchmark_case_ref=trace.request_attributes.benchmark_case_ref,
            )
        ):
            pass

    assert [span.name for span in telemetry.spans] == [SpanName.EVALUATION]
    serialized = repr(telemetry.spans)
    assert "raw-experiment-sentinel" not in serialized
    assert "raw-benchmark-sentinel" not in serialized


def test_real_sdk_in_memory_export_is_recursively_sanitized_and_omits_none():
    import logfire
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    client = logfire.configure(
        local=True,
        send_to_logfire=False,
        console=False,
        inspect_arguments=False,
        add_baggage_to_attributes=False,
        distributed_tracing=False,
        additional_span_processors=[SimpleSpanProcessor(exporter)],
        advanced=logfire.AdvancedOptions(
            emit_configuration_span=False,
            resource_detectors=(),
        ),
    )
    telemetry = LogfireTelemetry(pseudonym_key=b"p" * 32, client=client)
    sentinels = (
        "request-message-sentinel",
        "authorization-header-sentinel",
        "evidence-value-sentinel",
        "artifact-content-sentinel",
        "provider-error-sentinel",
        "golden-gabarito-sentinel",
        "token-secret-sentinel",
    )
    trace = telemetry.start_execution(
        ExecutionCorrelations(
            request_id="|".join(sentinels),
            thread_id=sentinels[1],
            execution_id=sentinels[2],
            case_id=sentinels[3],
            company_id=sentinels[4],
            user_id=sentinels[5],
            planner_enabled=False,
        )
    )

    with trace.activate():
        with trace.span(SpanName.REQUEST, trace.request_attributes):
            with trace.span(
                SpanName.NODE,
                NodeSpanAttributes(node=GraphNodeName.INGEST),
            ):
                pass
    client.force_flush()

    exported = exporter.get_finished_spans()
    assert exported
    root = next(span for span in exported if span.name == SpanName.REQUEST.value)
    assert "experiment_ref" not in root.attributes
    assert "benchmark_case_ref" not in root.attributes
    recursive_export = [
        {
            "name": span.name,
            "attributes": dict(span.attributes or {}),
            "events": [
                {"name": event.name, "attributes": dict(event.attributes or {})}
                for event in span.events
            ],
            "resource": dict(span.resource.attributes),
            "scope": {
                "name": span.instrumentation_scope.name,
                "version": span.instrumentation_scope.version,
                "schema_url": span.instrumentation_scope.schema_url,
                "attributes": dict(span.instrumentation_scope.attributes or {}),
            },
            "links": [
                {
                    "trace_id": link.context.trace_id,
                    "span_id": link.context.span_id,
                    "attributes": dict(link.attributes or {}),
                }
                for link in span.links
            ],
            "status": {
                "code": span.status.status_code.name,
                "description": span.status.description,
            },
            "sdk_json": span.to_json(),
        }
        for span in exported
    ]
    serialized = json.dumps(recursive_export, default=str, sort_keys=True).casefold()
    for sentinel in sentinels:
        assert sentinel.casefold() not in serialized


class _MinimalGraph:
    planner_enabled = False

    def __init__(self, telemetry):
        self.telemetry = telemetry
        self.values = {}

    @asynccontextmanager
    async def thread_lock(self, thread_id):
        yield

    async def aget_state(self, config):
        return SimpleNamespace(
            values=self.values,
            config=config,
            next=(),
            interrupts=(),
        )

    async def ainvoke(self, input, config, *, context, durability):
        if input is not None:
            self.values = input
        return self.values

    async def aupdate_state(self, config, values, *, as_node):
        self.values = values
        return config


class _TelemetryBaseError(BaseException):
    pass


class _FaultContext:
    def __init__(self, *, fail_enter=False, fail_exit=False, value=None, error_factory):
        self._fail_enter = fail_enter
        self._fail_exit = fail_exit
        self._value = value
        self._error_factory = error_factory

    def __enter__(self):
        if self._fail_enter:
            raise self._error_factory()
        return self._value

    def __exit__(self, *args):
        if self._fail_exit:
            raise self._error_factory()
        return False


class _FaultSpanHandle:
    def __init__(self, error_factory):
        self._error_factory = error_factory

    def finish(self, *args):
        raise self._error_factory()


class _FaultTelemetry(RecordingTelemetry):
    def __init__(self, stage, error_factory):
        super().__init__(pseudonym_key=b"p" * 32)
        self._stage = stage
        self._error_factory = error_factory

    def start_execution(self, correlations):
        if self._stage == "start_execution":
            raise self._error_factory()
        trace = super().start_execution(correlations)
        if self._stage == "activate_call":
            trace.activate = lambda: (_ for _ in ()).throw(self._error_factory())
        elif self._stage == "activate_enter":
            trace.activate = lambda: _FaultContext(
                fail_enter=True,
                error_factory=self._error_factory,
            )
        elif self._stage == "activate_exit":
            trace.activate = lambda: _FaultContext(
                fail_exit=True,
                error_factory=self._error_factory,
            )
        elif self._stage == "span_call":
            trace.span = lambda *args: (_ for _ in ()).throw(self._error_factory())
        elif self._stage == "span_enter":
            trace.span = lambda *args: _FaultContext(
                fail_enter=True,
                error_factory=self._error_factory,
            )
        elif self._stage == "span_exit":
            trace.span = lambda *args: _FaultContext(
                fail_exit=True,
                value=_FaultSpanHandle(self._error_factory),
                error_factory=self._error_factory,
            )
        elif self._stage == "span_finish":
            trace.span = lambda *args: _FaultContext(
                value=_FaultSpanHandle(self._error_factory),
                error_factory=self._error_factory,
            )
        elif self._stage == "metrics":
            trace._record_metrics = lambda *args, **kwargs: (_ for _ in ()).throw(
                self._error_factory()
            )
        return trace


def test_observed_envelope_does_not_change_or_persist_the_public_state():
    async def scenario():
        request = SupportRequest(
            case_id="case_tkt_inv_04",
            ticket_id="TKT-INV-04",
            asset_id="asset_G501",
            message="Mensagem literal que não deve aparecer na telemetria.",
            identity=Identity(
                user_id="usr_pedro",
                company_id="comp_mineracao_andes",
            ),
        )
        async with IndustrialApiClient("https://industrial.invalid") as client:
            runtime = ReadToolRuntime.create(
                user_id="usr_pedro",
                company_id="comp_mineracao_andes",
                permissions=frozenset({"read"}),
                central_asset_id="asset_G501",
                client=client,
            )
            first_telemetry = NullTelemetry()
            plain = await invoke_agent(
                _MinimalGraph(first_telemetry),
                request=request,
                runtime=runtime,
                thread_id="thread_01",
                request_id="request_01",
                execution_id="execution_01",
            )
            second_telemetry = RecordingTelemetry(pseudonym_key=b"p" * 32)
            observed = await invoke_agent_observed(
                _MinimalGraph(second_telemetry),
                request=request,
                runtime=runtime,
                thread_id="thread_01",
                request_id="request_01",
                execution_id="execution_01",
            )
        return plain, observed, second_telemetry

    plain, observed, telemetry = asyncio.run(scenario())

    assert isinstance(observed, AgentInvocationResult)
    assert observed.state.model_dump_json() == plain.model_dump_json()
    assert re.fullmatch(r"trc_[0-9a-f]{32}", observed.trace_id.value)
    assert observed.trace_id.value not in observed.state.model_dump_json()
    assert [span.name for span in telemetry.spans] == [
        SpanName.RESPONSE,
        SpanName.REQUEST,
    ]


@pytest.mark.parametrize(
    "stage",
    [
        "start_execution",
        "activate_call",
        "activate_enter",
        "activate_exit",
        "span_call",
        "span_enter",
        "span_exit",
        "span_finish",
        "metrics",
    ],
)
@pytest.mark.parametrize("error_factory", [_TelemetryBaseError, asyncio.CancelledError])
def test_public_invocation_is_fail_open_for_every_telemetry_operation(
    stage,
    error_factory,
):
    async def scenario():
        request = SupportRequest(
            case_id="case_tkt_inv_04",
            ticket_id="TKT-INV-04",
            asset_id="asset_G501",
            message="Mensagem não exportável.",
            identity=Identity(
                user_id="usr_pedro",
                company_id="comp_mineracao_andes",
            ),
        )
        async with IndustrialApiClient("https://industrial.invalid") as client:
            runtime = ReadToolRuntime.create(
                user_id="usr_pedro",
                company_id="comp_mineracao_andes",
                permissions=frozenset({"read"}),
                central_asset_id="asset_G501",
                client=client,
            )
            baseline = await invoke_agent(
                _MinimalGraph(NullTelemetry()),
                request=request,
                runtime=runtime,
                thread_id="thread_fault_matrix",
                request_id="request_fault_matrix",
                execution_id="execution_fault_matrix",
            )
            observed = await invoke_agent_observed(
                _MinimalGraph(_FaultTelemetry(stage, error_factory)),
                request=request,
                runtime=runtime,
                thread_id="thread_fault_matrix",
                request_id="request_fault_matrix",
                execution_id="execution_fault_matrix",
            )
        return baseline, observed

    baseline, observed = asyncio.run(scenario())

    assert observed.state.model_dump_json() == baseline.model_dump_json()
    assert re.fullmatch(r"trc_[0-9a-f]{32}", observed.trace_id.value)


@pytest.mark.parametrize("corruption_stage", ["start", "activate"])
def test_public_invocation_uses_a_safe_trace_snapshot(corruption_stage):
    class CorruptingTelemetry(RecordingTelemetry):
        def start_execution(self, correlations):
            trace = super().start_execution(correlations)
            if corruption_stage == "start":
                trace.trace_id = "invalid-trace-id"
                return trace

            original_activate = trace.activate

            class MutatingActivation:
                def __enter__(self):
                    entered = original_activate()
                    entered.__enter__()
                    self.entered = entered
                    trace.trace_id = "invalid-trace-id"
                    return trace

                def __exit__(self, *args):
                    return self.entered.__exit__(*args)

            trace.activate = MutatingActivation
            return trace

    async def scenario():
        request = SupportRequest(
            case_id="case_tkt_inv_04",
            ticket_id="TKT-INV-04",
            asset_id="asset_G501",
            message="Mensagem não exportável.",
            identity=Identity(
                user_id="usr_pedro",
                company_id="comp_mineracao_andes",
            ),
        )
        async with IndustrialApiClient("https://industrial.invalid") as client:
            runtime = ReadToolRuntime.create(
                user_id="usr_pedro",
                company_id="comp_mineracao_andes",
                permissions=frozenset({"read"}),
                central_asset_id="asset_G501",
                client=client,
            )
            baseline = await invoke_agent(
                _MinimalGraph(NullTelemetry()),
                request=request,
                runtime=runtime,
                thread_id="thread_corrupt_trace",
                request_id="request_corrupt_trace",
                execution_id="execution_corrupt_trace",
            )
            observed = await invoke_agent_observed(
                _MinimalGraph(CorruptingTelemetry(pseudonym_key=b"p" * 32)),
                request=request,
                runtime=runtime,
                thread_id="thread_corrupt_trace",
                request_id="request_corrupt_trace",
                execution_id="execution_corrupt_trace",
            )
            return baseline, observed

    baseline, observed = asyncio.run(scenario())

    assert observed.state.model_dump_json() == baseline.model_dump_json()
    assert re.fullmatch(r"trc_[0-9a-f]{32}", observed.trace_id.value)


@pytest.mark.parametrize(
    "business_error", [LookupError("business"), asyncio.CancelledError()]
)
@pytest.mark.parametrize(
    "telemetry_stage", ["activate_exit", "span_exit", "span_finish"]
)
def test_telemetry_failure_never_replaces_business_exception(
    business_error,
    telemetry_stage,
):
    class FailingGraph(_MinimalGraph):
        async def ainvoke(self, input, config, *, context, durability):
            self.values = input
            raise business_error

    async def scenario():
        request = SupportRequest(
            case_id="case_tkt_inv_04",
            ticket_id="TKT-INV-04",
            asset_id="asset_G501",
            message="Mensagem não exportável.",
            identity=Identity(user_id="usr_pedro", company_id="comp_mineracao_andes"),
        )
        async with IndustrialApiClient("https://industrial.invalid") as client:
            runtime = ReadToolRuntime.create(
                user_id="usr_pedro",
                company_id="comp_mineracao_andes",
                permissions=frozenset({"read"}),
                central_asset_id="asset_G501",
                client=client,
            )
            await invoke_agent_observed(
                FailingGraph(_FaultTelemetry(telemetry_stage, _TelemetryBaseError)),
                request=request,
                runtime=runtime,
                thread_id="thread_business_error",
                request_id="request_business_error",
                execution_id="execution_business_error",
            )

    with pytest.raises(type(business_error)) as raised:
        asyncio.run(scenario())

    assert raised.value is business_error
    assert raised.value.__cause__ is None
    traceback_names = []
    traceback = raised.value.__traceback__
    while traceback is not None:
        traceback_names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert "ainvoke" in traceback_names


@pytest.mark.parametrize("telemetry_kind", ["null", "recording"])
def test_real_fallback_pairs_null_and_recording_without_business_drift(
    tmp_path,
    telemetry_kind,
):
    async def scenario():
        telemetry = (
            NullTelemetry()
            if telemetry_kind == "null"
            else RecordingTelemetry(pseudonym_key=b"p" * 32)
        )
        request = SupportRequest(
            case_id="case_tkt_inv_04",
            ticket_id="TKT-INV-04",
            asset_id="asset_G501",
            message="Consulte o ativo.",
            identity=Identity(
                user_id="usr_pedro",
                company_id="comp_mineracao_andes",
            ),
        )
        async with open_checkpointer(tmp_path / "observability.sqlite3") as saver:
            graph = build_agent_graph(saver, telemetry=telemetry)
            async with IndustrialApiClient("https://industrial.invalid") as client:
                runtime = ReadToolRuntime.create(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                    permissions=frozenset({"read"}),
                    central_asset_id="asset_G501",
                    client=client,
                )
                result = await invoke_agent_observed(
                    graph,
                    request=request,
                    runtime=runtime,
                    thread_id="thread_real_fallback",
                    request_id="request_real_fallback",
                    execution_id="execution_real_fallback",
                )
                snapshot = await graph.aget_state(
                    {"configurable": {"thread_id": "thread_real_fallback"}}
                )
        return result, snapshot, telemetry

    result, snapshot, telemetry = asyncio.run(scenario())

    if isinstance(telemetry, RecordingTelemetry):
        node_names = [
            dict(span.attributes)["node"]
            for span in telemetry.spans
            if span.name is SpanName.NODE
        ]
        assert node_names == ["ingest", "route", "finish"]
    assert result.state == AgentState.model_validate(snapshot.values)
    assert result.trace_id.value not in json.dumps(snapshot.values, default=str)
    if isinstance(telemetry, RecordingTelemetry):
        assert not any(span.name is SpanName.EVALUATION for span in telemetry.spans)


@pytest.mark.parametrize("second_preflight_blocks", [False, True])
@pytest.mark.parametrize("telemetry_kind", ["null", "recording"])
def test_reprocess_retry_pairs_null_and_recording_and_records_real_attempts(
    tmp_path,
    second_preflight_blocks,
    telemetry_kind,
):
    posts = []
    gets = []

    def handler(request):
        if request.method == "GET":
            gets.append(request)
            if second_preflight_blocks and len(gets) == 2:
                return httpx.Response(
                    503,
                    json={"code": "SERVER_ERROR", "message": "preflight blocked"},
                )
            return httpx.Response(
                200,
                json={
                    "mode": "complete",
                    "notes": None,
                    "data": {
                        "id": "an_9901",
                        "asset_id": "asset_M101",
                        "point_id": "pt_M101_de",
                        "type": "bearing_fault",
                        "detection_mode": "baseline",
                        "severity": "high",
                        "confidence": 0.78,
                        "baseline_state_at_detection": "established",
                        "evidence": [],
                        "limitations": [],
                        "model_version": "3.2.1",
                        "created_at": "2026-01-02T03:04:05+00:00",
                        "status": "current",
                    },
                },
            )
        posts.append(request)
        if len(posts) == 1:
            raise httpx.ReadTimeout("secret transport detail", request=request)
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "action_id": "act_reprocess_01",
                "message": "Reprocesso aceito.",
            },
        )

    async def scenario():
        telemetry = (
            NullTelemetry()
            if telemetry_kind == "null"
            else RecordingTelemetry(pseudonym_key=b"p" * 32)
        )
        request = SupportRequest(
            case_id="case_tkt_exe_12",
            ticket_id="TKT-EXE-12",
            asset_id="asset_M101",
            message="Reprocesse a análise.",
            identity=Identity(user_id="usr_ana", company_id="comp_forja_br"),
        )
        proposal = ReprocessProposal(
            analysis_id="an_9901",
            justification="O rolamento foi trocado e a análise precisa ser refeita.",
        )
        approval = TrustedActionApproval(
            action="reprocess_analysis",
            target_id="an_9901",
            source=ApprovalSource.ORIGINAL_REQUEST,
        )
        client = IndustrialApiClient(
            "https://simulator.test",
            transport=httpx.MockTransport(handler),
        )
        runtime = WriteToolRuntime.create(
            user_id="usr_ana",
            company_id="comp_forja_br",
            permissions=frozenset({"read", "action_low"}),
            central_asset_id="asset_M101",
            current_case_id="case_tkt_exe_12",
            client=client,
        )
        try:
            async with open_checkpointer(tmp_path / "action-spans.sqlite3") as saver:
                state = await invoke_agent(
                    build_agent_graph(saver, telemetry=telemetry),
                    request=request,
                    runtime=runtime,
                    thread_id="thread_action_spans",
                    request_id="request_action_spans",
                    execution_id="execution_action_spans",
                    proposal=proposal,
                    original_approval=approval,
                )
        finally:
            await client.aclose()
        return state, telemetry

    state, telemetry = asyncio.run(scenario())

    if isinstance(telemetry, RecordingTelemetry):
        action_attributes = [
            dict(span.attributes)
            for span in telemetry.spans
            if span.name is SpanName.ACTION
        ]
        assert [attributes["attempt"] for attributes in action_attributes] == (
            [1] if second_preflight_blocks else [1, 2]
        )
        assert all(
            attributes["action"] == "reprocess_analysis"
            for attributes in action_attributes
        )
        assert any(span.name is SpanName.POLICY for span in telemetry.spans)
    assert state.intents[0].attempts == 2
    assert len(posts) == (1 if second_preflight_blocks else 2)
    assert len(gets) == 2


def test_concurrent_traces_and_cancellation_restore_context_without_sentinels():
    telemetry = RecordingTelemetry(pseudonym_key=b"p" * 32)

    def correlations(marker):
        return ExecutionCorrelations(
            request_id=f"raw-request-{marker}",
            thread_id=f"raw-thread-{marker}",
            execution_id=f"raw-execution-{marker}",
            case_id=f"raw-case-{marker}",
            company_id=f"raw-company-{marker}",
            user_id=f"raw-user-{marker}",
            planner_enabled=False,
        )

    async def scenario():
        first = telemetry.start_execution(correlations("first-secret"))
        second = telemetry.start_execution(correlations("second-secret"))
        entered = asyncio.Event()
        release = asyncio.Event()

        async def worker(trace):
            with trace.activate():
                with trace.span(SpanName.REQUEST, trace.request_attributes):
                    entered.set()
                    await release.wait()
                    assert current_execution_trace() is trace

        first_task = asyncio.create_task(worker(first))
        second_task = asyncio.create_task(worker(second))
        await entered.wait()
        release.set()
        await asyncio.gather(first_task, second_task)

        cancelled = telemetry.start_execution(correlations("cancel-secret"))

        async def cancelled_worker():
            with cancelled.activate():
                with cancelled.span(
                    SpanName.RESPONSE,
                    ResponseSpanAttributes(
                        planner_enabled=False,
                        replayed=False,
                    ),
                ):
                    await asyncio.sleep(60)

        task = asyncio.create_task(cancelled_worker())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return first, second, cancelled

    first, second, cancelled = asyncio.run(scenario())

    assert current_execution_trace() is None
    trace_ids = {dict(span.attributes)["trace_id"] for span in telemetry.spans}
    assert trace_ids == {
        first.trace_id.value,
        second.trace_id.value,
        cancelled.trace_id.value,
    }
    cancelled_spans = [
        dict(span.attributes)
        for span in telemetry.spans
        if dict(span.attributes)["trace_id"] == cancelled.trace_id.value
    ]
    assert cancelled_spans[0]["outcome"] == "cancelled"
    assert cancelled_spans[0]["error_code"] == "cancelled"
    serialized = repr(telemetry.spans) + repr(telemetry.metrics)
    for sentinel in (
        "first-secret",
        "second-secret",
        "cancel-secret",
        "authorization",
        "golden",
        "score",
    ):
        assert sentinel not in serialized.casefold()

    with pytest.raises(ValidationError):
        NodeSpanAttributes.model_validate({"node": "ingest", "payload": "forbidden"})


def test_cancelled_action_dispatch_preserves_cancellation_category():
    telemetry = RecordingTelemetry(pseudonym_key=b"p" * 32)
    trace = telemetry.start_execution(
        ExecutionCorrelations(
            request_id="request",
            thread_id="thread",
            execution_id="execution",
            case_id="case",
            company_id="company",
            user_id="user",
            planner_enabled=False,
        )
    )

    async def cancelled_transport(_request):
        raise asyncio.CancelledError

    client = IndustrialApiClient(
        "https://simulator.test",
        transport=httpx.MockTransport(cancelled_transport),
    )
    runtime = WriteToolRuntime.create(
        user_id="user",
        company_id="company",
        permissions=frozenset({"read", "action_high"}),
        central_asset_id="asset_1",
        current_case_id="case_1",
        client=client,
    )
    proposal = UpdateAssetCriticalityProposal(
        criticality="critical",
        justification="Mudança autorizada.",
    )

    async def scenario():
        try:
            with (
                trace.activate(),
                bind_action_attempt(
                    ActionName.UPDATE_ASSET_CRITICALITY,
                    1,
                ),
            ):
                with pytest.raises(asyncio.CancelledError):
                    await execute_update_asset_criticality(proposal, runtime)
        finally:
            await client.aclose()

    asyncio.run(scenario())

    attributes = dict(telemetry.spans[0].attributes)
    assert attributes["outcome"] == "cancelled"
    assert attributes["error_code"] == "cancelled"
