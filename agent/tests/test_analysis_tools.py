from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import math
from typing import Any

import httpx
import pytest
from langgraph.prebuilt import ToolRuntime
from pydantic import ValidationError

from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import ApiError, ApiErrorCategory
from tractian_agent.tools.analyses import (
    execute_get_analysis,
    execute_list_asset_analyses,
    get_analysis,
    list_asset_analyses,
)
from tractian_agent.tools.runtime import ReadToolRuntime


def _runtime(
    handler: object,
    *,
    permissions: frozenset[str] = frozenset({"read"}),
    central_asset_id: str = "asset_M101",
    seed: str | None = "fixed-analysis",
) -> ReadToolRuntime:
    return ReadToolRuntime.create(
        user_id="usr_ana",
        company_id="comp_forja_br",
        permissions=permissions,
        central_asset_id=central_asset_id,
        client=IndustrialApiClient(
            "https://simulator.test",
            transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        ),
        seed=seed,
    )


def _analysis_payload(
    *,
    analysis_id: str = "an_9901",
    asset_id: str = "asset_M101",
    point_id: str = "pt_M101_de",
    status: str = "current",
    created_at: str = "2026-01-02T03:04:05+00:00",
) -> dict[str, object]:
    return {
        "id": analysis_id,
        "asset_id": asset_id,
        "point_id": point_id,
        "type": "bearing_fault",
        "detection_mode": "baseline",
        "severity": "high",
        "confidence": 0.78,
        "baseline_state_at_detection": "established",
        "evidence": [
            {
                "metric": "bpfo_amplitude",
                "value": 1.4,
                "reference": 0.6,
                "note": "BPFO acima do baseline",
            }
        ],
        "limitations": ["processing_delayed"],
        "model_version": "3.2.1",
        "created_at": created_at,
        "status": status,
    }


def _run(coro: Any):
    return asyncio.run(coro)


async def _list(
    runtime: ReadToolRuntime, asset_id: object = "asset_M101", status: object = None
):
    try:
        return await execute_list_asset_analyses(asset_id, status, runtime)
    finally:
        await runtime.client.aclose()


async def _detail(runtime: ReadToolRuntime, analysis_id: object = "an_9901"):
    try:
        return await execute_get_analysis(analysis_id, runtime)
    finally:
        await runtime.client.aclose()


def test_analysis_tools_expose_only_safe_strict_public_schemas():
    list_schema = list_asset_analyses.tool_call_schema.model_json_schema()
    detail_schema = get_analysis.tool_call_schema.model_json_schema()

    assert list_asset_analyses.name == "list_asset_analyses"
    assert set(list_schema["properties"]) == {"asset_id", "status"}
    assert list_schema["required"] == ["asset_id"]
    assert (
        list_schema["properties"]["asset_id"]["pattern"]
        == r"^asset_[A-Za-z0-9_-]{1,64}$"
    )
    assert list_schema["properties"]["status"]["anyOf"][0]["enum"] == [
        "current",
        "stale",
        "pending",
        "inconclusive",
    ]
    assert set(detail_schema["properties"]) == {"analysis_id"}
    assert detail_schema["required"] == ["analysis_id"]
    assert (
        detail_schema["properties"]["analysis_id"]["pattern"]
        == r"^an_[A-Za-z0-9_-]{1,64}$"
    )

    serialized = json.dumps({"list": list_schema, "detail": detail_schema}).lower()
    for hidden in (
        "runtime",
        "identity",
        "permissions",
        "client",
        "seed",
        "url",
        "method",
    ):
        assert hidden not in serialized


def test_list_calls_exact_fixed_endpoint_with_only_status_and_seed():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "mode": "complete",
                "notes": None,
                "data": {"analyses": [_analysis_payload()]},
            },
        )

    result = _run(_list(_runtime(handler), status="current"))

    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/assets/asset_M101/analyses"
    assert dict(requests[0].url.params) == {
        "status": "current",
        "seed": "fixed-analysis",
    }
    assert requests[0].headers["x-user-id"] == "usr_ana"
    assert result.content.model_dump() == {
        "analyses": [
            {
                "id": "an_9901",
                "asset_id": "asset_M101",
                "point_id": "pt_M101_de",
                "type": "bearing_fault",
                "severity": "high",
                "confidence": 0.78,
                "status": "current",
                "created_at": "2026-01-02T03:04:05+00:00",
                "limitations": ["processing_delayed"],
            }
        ],
        "total_analyses": 1,
        "returned_analyses": 1,
        "omitted_analyses": 0,
        "truncated": False,
    }
    normalized = result.artifact.outcome.analyses[0]
    assert normalized.evidence[0].reference == 0.6
    assert normalized.detection_mode == "baseline"
    assert normalized.baseline_state_at_detection == "established"
    assert normalized.model_version == "3.2.1"


def test_list_omits_status_query_parameter_when_no_filter_is_requested():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"mode": "complete", "notes": None, "data": {"analyses": []}},
        )

    _run(_list(_runtime(handler)))

    assert len(requests) == 1
    assert dict(requests[0].url.params) == {"seed": "fixed-analysis"}


def test_detail_calls_exact_fixed_endpoint_and_keeps_complete_analysis():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json={"mode": "complete", "notes": None, "data": _analysis_payload()}
        )

    result = _run(_detail(_runtime(handler)))

    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/analyses/an_9901"
    assert dict(requests[0].url.params) == {"seed": "fixed-analysis"}
    assert result.content.model_dump() == result.artifact.outcome.analysis.model_dump()
    assert result.content.evidence[0].note == "BPFO acima do baseline"


def test_detail_accepts_nullable_evidence_reference_for_symptomatic_detection():
    payload = _analysis_payload()
    payload.update(
        {
            "type": "lubrication",
            "detection_mode": "symptom",
            "baseline_state_at_detection": "not_applicable",
        }
    )
    payload["evidence"][0]["reference"] = None

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"mode": "complete", "notes": None, "data": payload}
        )

    result = _run(_detail(_runtime(handler)))

    assert result.content.evidence[0].reference is None


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        ("list", ("bad", None)),
        ("list", ("asset_M101", "wrong")),
        ("detail", ("analysis_bad",)),
        ("detail", ("an_" + "x" * 65,)),
    ],
)
def test_core_rejects_invalid_ids_and_status_before_http(
    operation: str, arguments: tuple[object, ...]
):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    runtime = _runtime(handler)
    with pytest.raises(ValidationError):
        if operation == "list":
            _run(_list(runtime, *arguments))
        else:
            _run(_detail(runtime, *arguments))
    assert calls == 0


@pytest.mark.parametrize("operation", ["list", "detail"])
def test_permission_is_rejected_before_http(operation: str):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    runtime = _runtime(handler, permissions=frozenset())
    with pytest.raises(PermissionError, match="read"):
        if operation == "list":
            _run(_list(runtime))
        else:
            _run(_detail(runtime))
    assert calls == 0


def test_list_rejects_outside_central_asset_before_http():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    with pytest.raises(ValueError, match="ativo central"):
        _run(_list(_runtime(handler), "asset_other"))
    assert calls == 0


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"analyses": [_analysis_payload(asset_id="asset_other")]}, "fora do escopo"),
        (
            {
                "analyses": [
                    _analysis_payload(analysis_id="an_9901"),
                    _analysis_payload(analysis_id="an_9901"),
                ]
            },
            "duplicado",
        ),
        ({"analyses": [_analysis_payload(status="stale")]}, "filtro"),
    ],
)
def test_list_rejects_parent_duplicate_and_filter_mismatches(
    payload: dict[str, object], match: str
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"mode": "complete", "notes": None, "data": payload}
        )

    with pytest.raises(ValueError, match=match):
        _run(_list(_runtime(handler), status="current"))


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (_analysis_payload(analysis_id="an_9902"), "identificador"),
        (_analysis_payload(asset_id="asset_other"), "fora do escopo"),
    ],
)
def test_detail_rejects_returned_id_and_parent_mismatch(
    payload: dict[str, object], match: str
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"mode": "complete", "notes": None, "data": payload}
        )

    with pytest.raises(ValueError, match=match):
        _run(_detail(_runtime(handler)))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda data: data.pop("evidence"), "INVALID_SCHEMA_RESPONSE"),
        (lambda data: data["evidence"][0].pop("reference"), "INVALID_SCHEMA_RESPONSE"),
        (
            lambda data: data.update({"created_at": "2026-01-02"}),
            "INVALID_SCHEMA_RESPONSE",
        ),
        (lambda data: data.update({"confidence": math.nan}), "INVALID_SCHEMA_RESPONSE"),
        (lambda data: data.update({"unexpected": True}), "INVALID_SCHEMA_RESPONSE"),
    ],
)
def test_complete_wire_rejects_missing_nullable_timestamp_nonfinite_and_extra_fields(
    mutate: Any, match: str
):
    payload = _analysis_payload()
    mutate(payload)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"mode": "complete", "notes": None, "data": payload}
        )

    result = _run(_detail(_runtime(handler)))
    assert result.error is not None
    assert result.error.code == match


def test_list_orders_newest_first_stably_and_declares_prompt_and_artifact_cuts():
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    analyses = [
        _analysis_payload(
            analysis_id=f"an_{index}",
            created_at=(origin + timedelta(seconds=index // 2)).isoformat(),
        )
        for index in range(201)
    ]
    analyses.reverse()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"mode": "complete", "notes": None, "data": {"analyses": analyses}},
        )

    result = _run(_list(_runtime(handler)))

    assert result.content.total_analyses == 201
    assert result.content.returned_analyses == 20
    assert result.content.omitted_analyses == 181
    assert result.content.truncated is True
    assert len(result.content.analyses) == 20
    assert result.content.analyses[0].id == "an_200"
    assert result.content.analyses[1].id == "an_199"
    assert len(result.artifact.outcome.analyses) == 200
    assert result.artifact.outcome.total_analyses == 201
    assert result.artifact.outcome.returned_analyses == 200
    assert result.artifact.outcome.omitted_analyses == 1
    assert result.artifact.truncated is True
    assert result.artifact.omitted_items == 1


def test_list_declares_the_prompt_cut_at_twenty_one_items():
    analyses = [
        _analysis_payload(
            analysis_id=f"an_{index}",
            created_at=f"2026-01-02T03:04:{index:02d}+00:00",
        )
        for index in range(21)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"mode": "complete", "notes": None, "data": {"analyses": analyses}},
        )

    result = _run(_list(_runtime(handler)))

    assert result.content.returned_analyses == 20
    assert result.content.omitted_analyses == 1
    assert result.content.truncated is True
    assert result.artifact.outcome.returned_analyses == 21
    assert result.artifact.truncated is False


def test_degraded_list_projects_real_simulator_rows_without_raw_evidence_or_model_fields():
    first = _analysis_payload(
        analysis_id="an_9901", created_at="2026-01-01T00:00:00+00:00"
    )
    second = _analysis_payload(
        analysis_id="an_9902", created_at="2026-01-02T00:00:00+00:00"
    )
    second.pop("evidence")
    second.pop("model_version")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mode": "conflict",
                "notes": "Análises em conflito.",
                "data": {"analyses": [first, second], "conflict": True},
            },
        )

    result = _run(_list(_runtime(handler)))

    assert result.content.model_dump() == {
        "mode": "conflict",
        "notes": "Análises em conflito.",
        "analyses": [
            {
                "id": "an_9902",
                "asset_id": "asset_M101",
                "point_id": "pt_M101_de",
                "type": "bearing_fault",
                "severity": "high",
                "confidence": 0.78,
                "status": "current",
                "created_at": "2026-01-02T00:00:00+00:00",
                "limitations": ["processing_delayed"],
            },
            {
                "id": "an_9901",
                "asset_id": "asset_M101",
                "point_id": "pt_M101_de",
                "type": "bearing_fault",
                "severity": "high",
                "confidence": 0.78,
                "status": "current",
                "created_at": "2026-01-01T00:00:00+00:00",
                "limitations": ["processing_delayed"],
            },
        ],
        "total_analyses": 2,
        "returned_analyses": 2,
        "omitted_analyses": 0,
        "truncated": False,
        "partial_data": {"conflict": True},
    }
    assert result.artifact.outcome.partial_data == {"conflict": True}
    assert len(result.artifact.outcome.analyses) == 2
    assert "evidence" not in result.artifact.outcome.analyses[0]
    assert "model_version" not in result.artifact.outcome.analyses[0]


def test_degraded_list_validates_all_rows_before_21_and_201_cuts_and_declares_counts():
    rows = [
        _analysis_payload(
            analysis_id=f"an_{index}",
            created_at=(
                datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index)
            ).isoformat(),
        )
        for index in range(201)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mode": "partial",
                "notes": "Dados parciais.",
                "data": {"analyses": rows, "inconclusive": True},
            },
        )

    result = _run(_list(_runtime(handler)))

    assert result.content.returned_analyses == 20
    assert result.content.omitted_analyses == 181
    assert result.content.truncated is True
    assert result.content.analyses[0]["id"] == "an_200"
    assert result.content.analyses[-1]["id"] == "an_181"
    assert result.artifact.outcome.returned_analyses == 200
    assert result.artifact.outcome.omitted_analyses == 1
    assert result.artifact.truncated is True
    assert result.artifact.omitted_items == 1
    assert result.artifact.outcome.analyses[0]["id"] == "an_200"
    assert result.artifact.outcome.analyses[-1]["id"] == "an_1"


def test_degraded_list_rejects_unknown_top_level_data_instead_of_bypassing_limits():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mode": "partial",
                "notes": "Dados parciais.",
                "data": {"analyses": [], "raw_rows": ["unexpected"]},
            },
        )

    with pytest.raises(ValueError, match="topo inesperado"):
        _run(_list(_runtime(handler)))


@pytest.mark.parametrize(
    ("rows", "status", "match"),
    [
        (
            [
                _analysis_payload(analysis_id="an_1"),
                _analysis_payload(analysis_id="an_1"),
            ],
            None,
            "duplicado",
        ),
        ([_analysis_payload(asset_id="asset_other")], None, "fora do escopo"),
        ([_analysis_payload(status="stale")], "current", "filtro"),
        ([{**_analysis_payload(), "status": None}], "current", "status inválido"),
    ],
)
def test_degraded_list_rejects_duplicate_scope_and_filter_violations_outside_windows(
    rows: list[dict[str, object]], status: str | None, match: str
):
    rows = rows * 201

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"mode": "partial", "notes": None, "data": {"analyses": rows}}
        )

    with pytest.raises(ValueError, match=match):
        _run(_list(_runtime(handler), status=status))


@pytest.mark.parametrize(
    ("bad_row", "status", "match"),
    [
        (_analysis_payload(analysis_id="an_0"), None, "duplicado"),
        (_analysis_payload(asset_id="asset_other"), None, "fora do escopo"),
        (_analysis_payload(status="stale"), "current", "filtro"),
    ],
)
def test_degraded_list_checks_a_bad_row_after_the_artifact_window(
    bad_row: dict[str, object], status: str | None, match: str
):
    rows = [
        _analysis_payload(
            analysis_id=f"an_{index}",
            created_at=(
                datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index)
            ).isoformat(),
        )
        for index in range(201)
    ]
    rows.append(bad_row)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"mode": "partial", "notes": None, "data": {"analyses": rows}}
        )

    with pytest.raises(ValueError, match=match):
        _run(_list(_runtime(handler), status=status))


def test_degraded_list_keeps_missing_summary_fields_absent_without_inventing_nulls():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mode": "partial",
                "notes": "Linha parcial.",
                "data": {"analyses": [{"id": "an_9901", "asset_id": "asset_M101"}]},
            },
        )

    result = _run(_list(_runtime(handler)))

    assert result.content.analyses == [{"id": "an_9901", "asset_id": "asset_M101"}]
    assert result.artifact.outcome.analyses == [
        {"id": "an_9901", "asset_id": "asset_M101"}
    ]


@pytest.mark.parametrize(
    ("operation", "data", "match"),
    [
        ("list", {"asset_id": None}, "ativo fora do escopo"),
        ("list", {"nested": [{"asset_id": "asset_other"}]}, "ativo fora do escopo"),
        ("detail", {"id": None}, "identificador nulo"),
        ("detail", {"id": "an_other"}, "identificador"),
        ("detail", {"asset_id": "asset_other"}, "ativo fora do escopo"),
    ],
)
def test_degraded_data_is_safe_and_rejects_contradictory_known_scope(
    operation: str, data: dict[str, object], match: str
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"mode": "partial", "notes": "Dados incompletos.", "data": data}
        )

    with pytest.raises(ValueError, match=match):
        if operation == "list":
            _run(_list(_runtime(handler)))
        else:
            _run(_detail(_runtime(handler)))


def test_degraded_data_and_api_errors_are_preserved_without_retry():
    requests: list[httpx.Request] = []

    def partial_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "mode": "partial",
                "notes": "Dados parciais.",
                "data": {"observed": True},
            },
        )

    partial = _run(_list(_runtime(partial_handler)))
    assert len(requests) == 1
    assert partial.content is None
    assert partial.artifact.outcome.mode.value == "partial"
    assert partial.artifact.outcome.partial_data == {"observed": True}

    def error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404, json={"code": "NOT_FOUND", "message": "Análise não encontrada."}
        )

    error = _run(_detail(_runtime(error_handler)))
    assert error.error == ApiError(
        category=ApiErrorCategory.API,
        code="NOT_FOUND",
        message="Análise não encontrada.",
        status_code=404,
    )


def test_degraded_detail_ignores_nested_model_id_but_checks_top_level_id():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mode": "partial",
                "notes": "Detalhe incompleto.",
                "data": {"id": "an_9901", "model": {"id": "mdl_vib_v3"}},
            },
        )

    result = _run(_detail(_runtime(handler)))

    assert result.artifact.outcome.partial_data == {
        "id": "an_9901",
        "model": {"id": "mdl_vib_v3"},
    }


def test_adapters_return_model_content_and_reject_unknown_arguments():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "mode": "complete",
                "notes": None,
                "data": {"analyses": [_analysis_payload()]},
            },
        )

    async def invoke(tool: object, arguments: dict[str, object]):
        runtime = _runtime(handler)
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

    content = _run(invoke(list_asset_analyses, {"asset_id": "asset_M101"}))
    assert content["total_analyses"] == 1
    with pytest.raises(ValidationError):
        _run(
            invoke(list_asset_analyses, {"asset_id": "asset_M101", "unexpected": True})
        )
    assert calls == 1


def test_detail_adapter_keeps_evidence_and_rejects_unknown_arguments():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={"mode": "complete", "notes": None, "data": _analysis_payload()}
        )

    async def invoke(arguments: dict[str, object]):
        runtime = _runtime(handler)
        tool_runtime = ToolRuntime(
            state={},
            context=runtime,
            config={},
            stream_writer=lambda _: None,
            tool_call_id=None,
            store=None,
        )
        try:
            return await get_analysis.ainvoke({**arguments, "runtime": tool_runtime})
        finally:
            await runtime.client.aclose()

    content = _run(invoke({"analysis_id": "an_9901"}))
    assert content["evidence"][0]["reference"] == 0.6
    with pytest.raises(ValidationError):
        _run(invoke({"analysis_id": "an_9901", "unexpected": True}))
    assert calls == 1
