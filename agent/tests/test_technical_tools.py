from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import math

import httpx
import pytest
from langgraph.prebuilt import ToolRuntime
from pydantic import ValidationError

from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import ApiError, ApiErrorCategory
from tractian_agent.tools.technical import (
    execute_get_baseline,
    execute_get_data_quality,
    execute_get_rms_series,
    execute_get_spectrum,
    get_baseline,
    get_data_quality,
    get_rms_series,
    get_spectrum,
)
from tractian_agent.tools.observations import assert_safe_partial_json
from tractian_agent.tools.runtime import ReadToolRuntime


def _runtime(
    handler: object, *, permissions: frozenset[str] = frozenset({"read"})
) -> ReadToolRuntime:
    return ReadToolRuntime.create(
        user_id="usr_ana",
        company_id="comp_forja_br",
        permissions=permissions,
        central_asset_id="asset_M101",
        client=IndustrialApiClient(
            "https://simulator.test",
            transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        ),
        seed="fixed-technical",
    )


def _baseline_payload() -> dict[str, object]:
    return {
        "id": "bs_M101_de",
        "asset_id": "asset_M101",
        "point_id": "pt_M101_de",
        "state": "established",
        "detection_mode": "baseline",
        "learnable": True,
        "established_at": "2026-01-01T00:00:00+00:00",
        "invalidated_at": None,
        "invalidation_reason": None,
        "features": [
            {"feature": "bpfo_amplitude", "reference": 1.0, "tolerance": 0.2},
            {"feature": "rms_mm_s", "reference": 2.5, "tolerance": 0.4},
        ],
    }


def test_get_baseline_uses_fixed_endpoint_and_derives_only_rms_threshold():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json={"mode": "complete", "notes": None, "data": _baseline_payload()}
        )

    async def invoke():
        runtime = _runtime(handler)
        try:
            return await execute_get_baseline("asset_M101", "pt_M101_de", runtime)
        finally:
            await runtime.client.aclose()

    result = asyncio.run(invoke())

    assert get_baseline.tool_call_schema.model_json_schema()["properties"] == {
        "asset_id": {
            "pattern": r"^asset_[A-Za-z0-9_-]{1,64}$",
            "title": "Asset Id",
            "type": "string",
        },
        "point_id": {
            "anyOf": [
                {"pattern": r"^pt_[A-Za-z0-9_-]{1,64}$", "type": "string"},
                {"type": "null"},
            ],
            "default": None,
            "title": "Point Id",
        },
    }
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/assets/asset_M101/baseline"
    assert dict(requests[0].url.params) == {
        "point_id": "pt_M101_de",
        "seed": "fixed-technical",
    }
    assert result.content.alarm_threshold == 2.9
    assert result.artifact.outcome.baseline.features[0].feature == "bpfo_amplitude"


def test_get_rms_series_normalizes_chronology_and_declares_two_projections():
    requests: list[httpx.Request] = []
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    samples = [
        {
            "ts": (origin + timedelta(hours=index)).isoformat(),
            "value": float(index),
        }
        for index in range(1001)
    ]
    samples.append(samples[500].copy())
    samples.reverse()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "mode": "complete",
                "notes": None,
                "data": {
                    "asset_id": "asset_M101",
                    "point_id": "pt_M101_de",
                    "unit": "mm/s",
                    "baseline_reference": 2.5,
                    "baseline_state": "established",
                    "alarm_threshold": 2.9,
                    "samples": samples,
                },
            },
        )

    async def invoke():
        runtime = _runtime(handler)
        try:
            return await execute_get_rms_series("asset_M101", "pt_M101_de", runtime)
        finally:
            await runtime.client.aclose()

    result = asyncio.run(invoke())

    assert get_rms_series.name == "get_rms_series"
    assert requests[0].url.path == "/assets/asset_M101/rms"
    assert dict(requests[0].url.params) == {
        "point_id": "pt_M101_de",
        "seed": "fixed-technical",
    }
    assert result.content.total_samples == 1002
    assert result.content.omitted_samples == 902
    assert len(result.content.samples) == 100
    assert result.content.samples[0].value == 0.0
    assert result.content.samples[-1].value == 1000.0
    assert len(result.artifact.outcome.rms.samples) == 1000
    assert result.artifact.outcome.rms.samples[-1].value == 1000.0
    assert result.artifact.model_content == result.content
    assert "model_content" not in result.content.model_dump()
    assert result.artifact.truncated is True
    assert result.artifact.omitted_items == 2


def test_get_spectrum_stably_orders_projects_peaks_and_preserves_missing_bands():
    requests: list[httpx.Request] = []
    peaks = [
        {"freq_hz": float(index), "amplitude_mm_s": float(index) / 10, "note": None}
        for index in range(201, 0, -1)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "mode": "complete",
                "notes": None,
                "data": {
                    "asset_id": "asset_M101",
                    "point_id": "pt_M101_de",
                    "peaks": peaks,
                    "bands_missing": ["2x_line", "bpfo"],
                    "collected_at": "2026-01-03T00:00:00+00:00",
                },
            },
        )

    async def invoke():
        runtime = _runtime(handler)
        try:
            return await execute_get_spectrum("asset_M101", None, runtime)
        finally:
            await runtime.client.aclose()

    result = asyncio.run(invoke())

    assert get_spectrum.name == "get_spectrum"
    assert requests[0].url.path == "/assets/asset_M101/spectrum"
    assert dict(requests[0].url.params) == {"seed": "fixed-technical"}
    assert result.content.bands_missing == ["2x_line", "bpfo"]
    assert "frequency_resolution_hz" not in result.content.model_dump()
    assert result.content.total_peaks == 201
    assert result.content.omitted_peaks == 181
    assert len(result.content.peaks) == 20
    assert result.content.peaks[0].freq_hz == 1.0
    assert result.content.peaks[-1].freq_hz == 201.0
    assert len(result.artifact.outcome.spectrum.peaks) == 200
    assert result.artifact.model_content == result.content
    assert "model_content" not in result.content.model_dump()
    assert result.artifact.truncated is True
    assert result.artifact.omitted_items == 1


def test_get_data_quality_preserves_all_quality_fields():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "mode": "complete",
                "notes": "Dados atuais.",
                "data": {
                    "asset_id": "asset_M101",
                    "point_id": "pt_M101_de",
                    "completeness": 0.82,
                    "freshness_minutes": 12,
                    "snr_db": 21.5,
                    "staleness_flag": False,
                },
            },
        )

    async def invoke():
        runtime = _runtime(handler)
        try:
            return await execute_get_data_quality("asset_M101", "pt_M101_de", runtime)
        finally:
            await runtime.client.aclose()

    result = asyncio.run(invoke())

    assert get_data_quality.name == "get_data_quality"
    assert requests[0].url.path == "/assets/asset_M101/data-quality"
    assert dict(requests[0].url.params) == {
        "point_id": "pt_M101_de",
        "seed": "fixed-technical",
    }
    assert result.content.model_dump() == {
        "asset_id": "asset_M101",
        "point_id": "pt_M101_de",
        "completeness": 0.82,
        "freshness_minutes": 12,
        "snr_db": 21.5,
        "staleness_flag": False,
    }
    assert result.artifact.outcome.mode.value == "complete"


TECHNICAL_TOOLS = [
    ("baseline", get_baseline, execute_get_baseline),
    ("rms", get_rms_series, execute_get_rms_series),
    ("spectrum", get_spectrum, execute_get_spectrum),
    ("data_quality", get_data_quality, execute_get_data_quality),
]


def _complete_payload(
    name: str, *, asset_id: str = "asset_M101", point_id: str = "pt_M101_de"
) -> dict[str, object]:
    if name == "baseline":
        payload = _baseline_payload()
        payload["asset_id"] = asset_id
        payload["point_id"] = point_id
        return payload
    if name == "rms":
        return {
            "asset_id": asset_id,
            "point_id": point_id,
            "unit": "mm/s",
            "baseline_reference": 2.5,
            "baseline_state": "established",
            "alarm_threshold": 2.9,
            "samples": [{"ts": "2026-01-01T00:00:00+00:00", "value": 2.5}],
        }
    if name == "spectrum":
        return {
            "asset_id": asset_id,
            "point_id": point_id,
            "peaks": [{"freq_hz": 30.0, "amplitude_mm_s": 0.4}],
            "bands_missing": ["2x_line"],
            "collected_at": "2026-01-01T00:00:00+00:00",
        }
    return {
        "asset_id": asset_id,
        "point_id": point_id,
        "completeness": 0.82,
        "freshness_minutes": 12,
        "snr_db": 21.5,
        "staleness_flag": False,
    }


async def _invoke_adapter(
    tool: object, runtime: ReadToolRuntime, arguments: dict[str, object]
):
    tool_runtime = ToolRuntime(
        state={},
        context=runtime,
        config={},
        stream_writer=lambda _: None,
        tool_call_id=None,
        store=None,
    )
    try:
        return await tool.ainvoke({**arguments, "runtime": tool_runtime})  # type: ignore[union-attr]
    finally:
        await runtime.client.aclose()


@pytest.mark.parametrize("name,tool,_", TECHNICAL_TOOLS)
def test_technical_tools_expose_only_strict_asset_and_optional_point_contract(
    name, tool, _
):
    schema = tool.tool_call_schema.model_json_schema()

    expected_name = "get_rms_series" if name == "rms" else f"get_{name}"
    assert tool.name == expected_name
    assert set(schema["properties"]) == {"asset_id", "point_id"}
    assert schema["required"] == ["asset_id"]
    assert schema["properties"]["asset_id"]["pattern"] == r"^asset_[A-Za-z0-9_-]{1,64}$"
    assert (
        schema["properties"]["point_id"]["anyOf"][0]["pattern"]
        == r"^pt_[A-Za-z0-9_-]{1,64}$"
    )
    serialized = str(schema).lower()
    for hidden_name in (
        "runtime",
        "identity",
        "permissions",
        "client",
        "seed",
        "url",
        "method",
    ):
        assert hidden_name not in serialized


@pytest.mark.parametrize("_,tool,__", TECHNICAL_TOOLS)
def test_invalid_public_point_or_extra_argument_stops_before_http(_, tool, __):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    with pytest.raises(ValidationError):
        asyncio.run(
            _invoke_adapter(
                tool, _runtime(handler), {"asset_id": "asset_M101", "point_id": "wrong"}
            )
        )
    with pytest.raises(ValidationError):
        asyncio.run(
            _invoke_adapter(
                tool, _runtime(handler), {"asset_id": "asset_M101", "unexpected": True}
            )
        )
    assert calls == 0


@pytest.mark.parametrize("name,tool,_", TECHNICAL_TOOLS)
def test_technical_tools_run_through_the_langchain_adapter(name, tool, _):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"mode": "complete", "notes": None, "data": _complete_payload(name)},
        )

    content = asyncio.run(
        _invoke_adapter(
            tool,
            _runtime(handler),
            {"asset_id": "asset_M101", "point_id": "pt_M101_de"},
        )
    )

    assert content["asset_id"] == "asset_M101"
    assert calls == 1


@pytest.mark.parametrize("_,__,execute", TECHNICAL_TOOLS)
def test_permission_and_asset_scope_stop_before_http(_, __, execute):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async def invoke():
        denied = _runtime(handler, permissions=frozenset())
        outside = _runtime(handler)
        try:
            with pytest.raises(PermissionError):
                await execute("asset_M101", None, denied)
            with pytest.raises(ValueError, match="ativo central"):
                await execute("asset_OTHER", None, outside)
        finally:
            await denied.client.aclose()
            await outside.client.aclose()

    asyncio.run(invoke())
    assert calls == 0


@pytest.mark.parametrize("name,_,execute", TECHNICAL_TOOLS)
def test_complete_responses_reject_asset_or_requested_point_outside_scope(
    name, _, execute
):
    payloads = [
        _complete_payload(name, asset_id="asset_OTHER"),
        _complete_payload(name, point_id="pt_OTHER"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"mode": "complete", "notes": None, "data": payloads.pop(0)}
        )

    async def invoke():
        runtime = _runtime(handler)
        try:
            with pytest.raises(ValueError, match="fora do escopo"):
                await execute("asset_M101", "pt_M101_de", runtime)
            with pytest.raises(ValueError, match="ponto diferente"):
                await execute("asset_M101", "pt_M101_de", runtime)
        finally:
            await runtime.client.aclose()

    asyncio.run(invoke())


@pytest.mark.parametrize("_,__,execute", TECHNICAL_TOOLS)
def test_degraded_observations_preserve_safe_data_and_reject_known_contradictions(
    _, __, execute
):
    responses = [
        {
            "mode": "partial",
            "notes": "Campos ausentes.",
            "data": {"asset_id": "asset_M101", "point_id": "pt_M101_de"},
        },
        {
            "mode": "partial",
            "notes": "Campos ausentes.",
            "data": {"asset_id": "asset_M101", "point_id": "pt_OTHER"},
        },
        {
            "mode": "partial",
            "notes": "Campos ausentes.",
            "data": {"asset_id": "asset_M101", "client_secret": "blocked"},
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses.pop(0))

    async def invoke():
        runtime = _runtime(handler)
        try:
            safe = await execute("asset_M101", "pt_M101_de", runtime)
            assert safe.content is None
            assert safe.artifact.outcome.mode.value == "partial"
            assert safe.artifact.outcome.notes == "Campos ausentes."
            assert safe.artifact.outcome.partial_data == {
                "asset_id": "asset_M101",
                "point_id": "pt_M101_de",
            }
            with pytest.raises(ValueError, match="ponto diferente"):
                await execute("asset_M101", "pt_M101_de", runtime)
            with pytest.raises(ValueError, match="campo proibido"):
                await execute("asset_M101", "pt_M101_de", runtime)
        finally:
            await runtime.client.aclose()

    asyncio.run(invoke())


@pytest.mark.parametrize("_,__,execute", TECHNICAL_TOOLS)
def test_tools_preserve_exact_api_error(_, __, execute):
    expected = ApiError(
        category=ApiErrorCategory.API,
        code="NOT_FOUND",
        message="Recurso não encontrado.",
        status_code=404,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, json={"code": "NOT_FOUND", "message": "Recurso não encontrado."}
        )

    async def invoke():
        runtime = _runtime(handler)
        try:
            return await execute("asset_M101", None, runtime)
        finally:
            await runtime.client.aclose()

    result = asyncio.run(invoke())
    assert result.error == expected
    assert result.artifact.outcome.error == expected


@pytest.mark.parametrize("name,_,execute", TECHNICAL_TOOLS)
def test_complete_wire_models_reject_extra_fields(name, _, execute):
    payload = _complete_payload(name)
    payload["unexpected"] = True

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"mode": "complete", "notes": None, "data": payload}
        )

    async def invoke():
        runtime = _runtime(handler)
        try:
            return await execute("asset_M101", None, runtime)
        finally:
            await runtime.client.aclose()

    result = asyncio.run(invoke())
    assert result.error == ApiError(
        category=ApiErrorCategory.INVALID_RESPONSE,
        code="INVALID_SCHEMA_RESPONSE",
        message="A resposta da API não corresponde ao contrato esperado.",
        status_code=200,
    )


def test_baseline_keeps_threshold_none_without_the_exact_rms_feature():
    payload = _baseline_payload()
    payload["features"] = [
        {"feature": "overall_rms", "reference": 2.5, "tolerance": 0.4}
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"mode": "complete", "notes": None, "data": payload}
        )

    async def invoke():
        runtime = _runtime(handler)
        try:
            return await execute_get_baseline("asset_M101", None, runtime)
        finally:
            await runtime.client.aclose()

    result = asyncio.run(invoke())
    assert result.content.state == "established"
    assert result.content.alarm_threshold is None


@pytest.mark.parametrize("_,__,execute", TECHNICAL_TOOLS)
@pytest.mark.parametrize("invalid_point", ["wrong", b"pt_M101_de", "../pt_M101_de"])
def test_execute_rejects_invalid_point_before_http(_, __, execute, invalid_point):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async def invoke():
        runtime = _runtime(handler)
        try:
            with pytest.raises(ValidationError):
                await execute("asset_M101", invalid_point, runtime)
        finally:
            await runtime.client.aclose()

    asyncio.run(invoke())
    assert calls == 0


@pytest.mark.parametrize("_,__,execute", TECHNICAL_TOOLS)
def test_degraded_scope_checks_nested_exact_asset_and_point_keys(_, __, execute):
    responses = [
        {
            "mode": "partial",
            "notes": None,
            "data": {
                "parent_asset_id": "asset_OTHER",
                "nested": {"asset_id": "asset_OTHER"},
            },
        },
        {"mode": "partial", "notes": None, "data": {"items": [{"point_id": None}]}},
        {
            "mode": "partial",
            "notes": None,
            "data": {"items": [{"point_id": "pt_OTHER"}]},
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses.pop(0))

    async def invoke():
        runtime = _runtime(handler)
        try:
            with pytest.raises(ValueError, match="ativo fora do escopo"):
                await execute("asset_M101", "pt_M101_de", runtime)
            with pytest.raises(ValueError, match="ponto inválido"):
                await execute("asset_M101", "pt_M101_de", runtime)
            with pytest.raises(ValueError, match="ponto diferente"):
                await execute("asset_M101", "pt_M101_de", runtime)
        finally:
            await runtime.client.aclose()

    asyncio.run(invoke())


@pytest.mark.parametrize("name,_,execute", TECHNICAL_TOOLS)
def test_complete_payloads_require_a_verifiable_point(name, _, execute):
    payload = _complete_payload(name, point_id=None)  # type: ignore[arg-type]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"mode": "complete", "notes": None, "data": payload}
        )

    async def invoke():
        runtime = _runtime(handler)
        try:
            return await execute("asset_M101", None, runtime)
        finally:
            await runtime.client.aclose()

    result = asyncio.run(invoke())
    assert result.error.code == "INVALID_SCHEMA_RESPONSE"


@pytest.mark.parametrize(
    "name,mutate",
    [
        (
            "baseline",
            lambda payload: payload.__setitem__("established_at", "2026-01-01"),
        ),
        (
            "rms",
            lambda payload: payload["samples"][0].__setitem__(
                "ts", "2026-01-01T00:00:00"
            ),
        ),
        ("spectrum", lambda payload: payload.__setitem__("collected_at", "2026-01-01")),
    ],
)
def test_complete_timestamps_require_time_and_timezone(name, mutate):
    payload = _complete_payload(name)
    mutate(payload)
    execute = next(
        execute for tool_name, _, execute in TECHNICAL_TOOLS if tool_name == name
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"mode": "complete", "notes": None, "data": payload}
        )

    async def invoke():
        runtime = _runtime(handler)
        try:
            return await execute("asset_M101", None, runtime)
        finally:
            await runtime.client.aclose()

    assert asyncio.run(invoke()).error.code == "INVALID_SCHEMA_RESPONSE"


def test_spectrum_requires_bands_missing_and_rejects_the_removed_resolution_field():
    missing_bands = _complete_payload("spectrum")
    missing_bands.pop("bands_missing")
    removed_field = _complete_payload("spectrum")
    removed_field["frequency_resolution_hz"] = 0.25
    responses = [missing_bands, removed_field]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"mode": "complete", "notes": None, "data": responses.pop(0)}
        )

    async def invoke():
        runtime = _runtime(handler)
        try:
            first = await execute_get_spectrum("asset_M101", None, runtime)
            second = await execute_get_spectrum("asset_M101", None, runtime)
            return first, second
        finally:
            await runtime.client.aclose()

    first, second = asyncio.run(invoke())
    assert first.error.code == "INVALID_SCHEMA_RESPONSE"
    assert second.error.code == "INVALID_SCHEMA_RESPONSE"


@pytest.mark.parametrize(
    "name,mutate",
    [
        (
            "baseline",
            lambda payload: payload["features"][0].__setitem__("reference", math.nan),
        ),
        ("rms", lambda payload: payload["samples"][0].__setitem__("value", math.inf)),
        (
            "spectrum",
            lambda payload: payload["peaks"][0].__setitem__("freq_hz", -math.inf),
        ),
        ("data_quality", lambda payload: payload.__setitem__("snr_db", math.nan)),
    ],
)
def test_complete_nonfinite_numbers_are_invalid_and_artifacts_remain_strict_json(
    name, mutate
):
    invalid_payload = _complete_payload(name)
    mutate(invalid_payload)
    valid_payload = _complete_payload(name)
    execute = next(
        execute for tool_name, _, execute in TECHNICAL_TOOLS if tool_name == name
    )
    responses = [invalid_payload, valid_payload]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"mode": "complete", "notes": None, "data": responses.pop(0)}
        )

    async def invoke():
        runtime = _runtime(handler)
        try:
            invalid = await execute("asset_M101", None, runtime)
            valid = await execute("asset_M101", None, runtime)
            return invalid, valid
        finally:
            await runtime.client.aclose()

    invalid, valid = asyncio.run(invoke())
    assert invalid.error.code == "INVALID_SCHEMA_RESPONSE"
    json.dumps(valid.artifact.model_dump(mode="json"), allow_nan=False)


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_assert_safe_partial_json_rejects_nonfinite_values_without_rejecting_json_numbers(
    nonfinite,
):
    safe = {"measurements": [0, 1.25, {"count": 2}], "complete": True}

    assert_safe_partial_json(safe)
    json.dumps(safe, allow_nan=False)
    with pytest.raises(ValueError, match="não finito"):
        assert_safe_partial_json({"measurements": [{"value": nonfinite}]})


@pytest.mark.parametrize("_,__,execute", TECHNICAL_TOOLS)
def test_degraded_technical_observation_rejects_nonfinite_json_before_artifact(
    _, __, execute
):
    responses = [
        {
            "mode": "partial",
            "notes": None,
            "data": {"asset_id": "asset_M101", "values": [1.0, 2]},
        },
        {
            "mode": "partial",
            "notes": None,
            "data": {"asset_id": "asset_M101", "values": [math.nan]},
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses.pop(0))

    async def invoke():
        runtime = _runtime(handler)
        try:
            safe = await execute("asset_M101", None, runtime)
            json.dumps(safe.artifact.model_dump(mode="json"), allow_nan=False)
            with pytest.raises(ValueError, match="não finito"):
                await execute("asset_M101", None, runtime)
        finally:
            await runtime.client.aclose()

    asyncio.run(invoke())


@pytest.mark.parametrize(
    "name,field",
    [
        ("baseline", "established_at"),
        ("baseline", "invalidated_at"),
        ("baseline", "invalidation_reason"),
        ("rms", "baseline_reference"),
        ("rms", "alarm_threshold"),
    ],
)
def test_complete_nullable_fields_are_required_on_the_wire(name, field):
    payload = _complete_payload(name)
    payload.pop(field)
    execute = next(
        execute for tool_name, _, execute in TECHNICAL_TOOLS if tool_name == name
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"mode": "complete", "notes": None, "data": payload}
        )

    async def invoke():
        runtime = _runtime(handler)
        try:
            return await execute("asset_M101", None, runtime)
        finally:
            await runtime.client.aclose()

    assert asyncio.run(invoke()).error.code == "INVALID_SCHEMA_RESPONSE"


@pytest.mark.parametrize(
    "name,field",
    [
        ("baseline", "established_at"),
        ("baseline", "invalidated_at"),
        ("baseline", "invalidation_reason"),
        ("rms", "baseline_reference"),
        ("rms", "alarm_threshold"),
    ],
)
def test_complete_nullable_fields_accept_explicit_null(name, field):
    payload = _complete_payload(name)
    payload[field] = None
    execute = next(
        execute for tool_name, _, execute in TECHNICAL_TOOLS if tool_name == name
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"mode": "complete", "notes": None, "data": payload}
        )

    async def invoke():
        runtime = _runtime(handler)
        try:
            return await execute("asset_M101", None, runtime)
        finally:
            await runtime.client.aclose()

    result = asyncio.run(invoke())
    assert result.error is None
    assert getattr(result.content, field) is None
