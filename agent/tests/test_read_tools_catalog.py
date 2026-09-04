from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, ToolRuntime

from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import ApiErrorCategory, ResponseMode
from tractian_agent.tools import (
    READ_TOOLS,
    get_analysis,
    get_asset,
    get_baseline,
    get_data_quality,
    get_knowledge_document,
    get_model,
    get_rms_series,
    get_spectrum,
    list_asset_analyses,
    search_knowledge,
)
from tractian_agent.tools.runtime import ReadToolRuntime


HttpHandler = Callable[[httpx.Request], httpx.Response]


def _asset_payload() -> dict[str, object]:
    return {
        "id": "asset_M101",
        "name": "Motor principal da forja",
        "company_id": "comp_forja_br",
        "criticality": "critical",
        "plant": "Planta 1",
        "line": "Forjamento",
        "parent_asset_id": None,
        "machine_type": "motor_induction",
        "rotation_rpm": 1780,
        "bearing_pn": "NU 310",
        "bpfo_hz": 142.3,
        "bpfi_hz": 218.1,
        "bsf_hz": 58.7,
        "ftf_hz": 11.9,
        "line_frequency_hz": 60,
        "sensor_status": "online",
        "points": [
            {
                "id": "pt_M101_de",
                "asset_id": "asset_M101",
                "location": "DE",
                "sensor_status": "online",
            }
        ],
    }


def _analysis_payload() -> dict[str, object]:
    return {
        "id": "an_A1",
        "asset_id": "asset_M101",
        "point_id": "pt_M101_de",
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
        "created_at": "2026-01-02T03:04:05+00:00",
        "status": "current",
    }


def _knowledge_document_payload() -> dict[str, object]:
    return {
        "id": "kb_doc_1",
        "type": "procedure",
        "title": "Procedimento de troca",
        "body": "Passos seguros para troca de rolamento.",
        "tags": ["rolamento", "baseline"],
    }


def _complete_payload_for_path(path: str) -> dict[str, object]:
    payloads = {
        "/assets/asset_M101": _asset_payload(),
        "/assets/asset_M101/analyses": {"analyses": [_analysis_payload()]},
        "/analyses/an_A1": _analysis_payload(),
        "/assets/asset_M101/baseline": {
            "id": "bs_M101_de",
            "asset_id": "asset_M101",
            "point_id": "pt_M101_de",
            "state": "established",
            "detection_mode": "baseline",
            "learnable": True,
            "established_at": "2026-01-01T00:00:00+00:00",
            "invalidated_at": None,
            "invalidation_reason": None,
            "features": [{"feature": "rms_mm_s", "reference": 2.5, "tolerance": 0.4}],
        },
        "/assets/asset_M101/rms": {
            "asset_id": "asset_M101",
            "point_id": "pt_M101_de",
            "unit": "mm/s",
            "baseline_reference": 2.5,
            "baseline_state": "established",
            "alarm_threshold": 2.9,
            "samples": [{"ts": "2026-01-01T00:00:00+00:00", "value": 2.5}],
        },
        "/assets/asset_M101/spectrum": {
            "asset_id": "asset_M101",
            "point_id": "pt_M101_de",
            "peaks": [{"freq_hz": 30.0, "amplitude_mm_s": 0.4}],
            "bands_missing": ["2x_line"],
            "collected_at": "2026-01-01T00:00:00+00:00",
        },
        "/assets/asset_M101/data-quality": {
            "asset_id": "asset_M101",
            "point_id": "pt_M101_de",
            "completeness": 0.82,
            "freshness_minutes": 12,
            "snr_db": 21.5,
            "staleness_flag": False,
        },
        "/models/mdl_vib_v3": {
            "id": "mdl_vib_v3",
            "version": "3.2.1",
            "coverage": [
                {
                    "machine_type": "motor_induction",
                    "supported": True,
                    "can_learn_baseline": True,
                }
            ],
            "requirements": {
                "min_completeness": 0.8,
                "min_snr_db": 12.0,
                "min_rotation_rpm": None,
            },
            "processing_state": "delayed",
            "last_run_at": "2026-01-02T03:04:05+00:00",
        },
        "/knowledge/search": {"results": [_knowledge_document_payload()]},
        "/knowledge/kb_doc_1": _knowledge_document_payload(),
    }
    return payloads[path]


def _runtime(
    handler: HttpHandler,
    *,
    user_id: str = "usr_ana",
    company_id: str = "comp_forja_br",
    seed: str | None = "catalog-seed",
) -> ReadToolRuntime:
    return ReadToolRuntime.create(
        user_id=user_id,
        company_id=company_id,
        permissions=frozenset({"read"}),
        central_asset_id="asset_M101",
        client=IndustrialApiClient(
            "https://simulator.test",
            transport=httpx.MockTransport(handler),
        ),
        seed=seed,
    )


def _catalog_graph():
    graph = StateGraph(MessagesState, context_schema=ReadToolRuntime)
    graph.add_node("tools", ToolNode(READ_TOOLS))
    graph.add_edge(START, "tools")
    return graph.compile()


async def _invoke_catalog(
    handler: HttpHandler,
    tool_calls: list[dict[str, Any]],
    *,
    user_id: str = "usr_ana",
    company_id: str = "comp_forja_br",
    seed: str | None = "catalog-seed",
) -> dict[str, Any]:
    runtime = _runtime(
        handler,
        user_id=user_id,
        company_id=company_id,
        seed=seed,
    )
    try:
        return await _catalog_graph().ainvoke(
            {"messages": [AIMessage(content="", tool_calls=tool_calls)]},
            context=runtime,
        )
    finally:
        await runtime.client.aclose()


async def _invoke_tool_directly(
    tool: BaseTool,
    arguments: dict[str, object],
    handler: HttpHandler,
) -> object:
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
        return await tool.ainvoke({**arguments, "runtime": tool_runtime})
    finally:
        await runtime.client.aclose()


def _single_call(name: str, arguments: dict[str, object]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "args": arguments,
            "id": f"call_{name}",
            "type": "tool_call",
        }
    ]


def _valid_tool_arguments() -> dict[str, dict[str, object]]:
    return {
        "get_asset": {"asset_id": "asset_M101"},
        "list_asset_analyses": {"asset_id": "asset_M101", "status": "current"},
        "get_analysis": {"analysis_id": "an_A1"},
        "get_baseline": {"asset_id": "asset_M101", "point_id": "pt_M101_de"},
        "get_rms_series": {"asset_id": "asset_M101", "point_id": "pt_M101_de"},
        "get_spectrum": {"asset_id": "asset_M101", "point_id": "pt_M101_de"},
        "get_data_quality": {
            "asset_id": "asset_M101",
            "point_id": "pt_M101_de",
        },
        "get_model": {},
        "search_knowledge": {"query": "rolamento", "document_type": "procedure"},
        "get_knowledge_document": {"document_id": "kb_doc_1"},
    }


def _all_tool_calls() -> list[dict[str, Any]]:
    arguments = _valid_tool_arguments()
    return [
        {
            "name": tool.name,
            "args": arguments[tool.name],
            "id": f"call_{index}",
            "type": "tool_call",
        }
        for index, tool in enumerate(READ_TOOLS)
    ]


def test_read_tools_catalog_is_static_complete_and_unique():
    expected_names = (
        "get_asset",
        "list_asset_analyses",
        "get_analysis",
        "get_baseline",
        "get_rms_series",
        "get_spectrum",
        "get_data_quality",
        "get_model",
        "search_knowledge",
        "get_knowledge_document",
    )

    assert isinstance(READ_TOOLS, tuple)
    assert tuple(tool.name for tool in READ_TOOLS) == expected_names
    assert len({tool.name for tool in READ_TOOLS}) == len(READ_TOOLS) == 10


def test_every_tool_schema_exposes_exactly_its_approved_public_fields():
    approved_fields = {
        "get_asset": {"asset_id"},
        "list_asset_analyses": {"asset_id", "status"},
        "get_analysis": {"analysis_id"},
        "get_baseline": {"asset_id", "point_id"},
        "get_rms_series": {"asset_id", "point_id"},
        "get_spectrum": {"asset_id", "point_id"},
        "get_data_quality": {"asset_id", "point_id"},
        "get_model": set(),
        "search_knowledge": {"query", "document_type"},
        "get_knowledge_document": {"document_id"},
    }
    forbidden_fields = {
        "runtime",
        "context",
        "identity",
        "user_id",
        "company_id",
        "permissions",
        "central_asset_id",
        "client",
        "url",
        "method",
        "headers",
        "seed",
        "configured_model_id",
        "model_id",
        "golden_set",
        "expected_paths",
        "evaluation",
        "eval",
    }

    for tool in READ_TOOLS:
        schema = tool.tool_call_schema.model_json_schema()
        public_fields = set(schema.get("properties", {}))

        assert public_fields == approved_fields[tool.name]
        assert public_fields.isdisjoint(forbidden_fields)


def test_catalog_runs_through_real_tool_node_with_hidden_runtime_and_json_artifact():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"mode": "complete", "notes": None, "data": _asset_payload()},
        )

    result = asyncio.run(
        _invoke_catalog(
            handler,
            _single_call("get_asset", {"asset_id": "asset_M101"}),
            user_id="usr_hidden_runtime",
            seed="hidden_runtime_seed",
        )
    )

    assert len(requests) == 1
    assert requests[0].headers["x-user-id"] == "usr_hidden_runtime"
    assert dict(requests[0].url.params) == {"seed": "hidden_runtime_seed"}
    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    assert message.name == "get_asset"
    assert message.tool_call_id == "call_get_asset"
    assert json.loads(message.content)["id"] == "asset_M101"
    assert message.artifact["tool_name"] == "get_asset"
    json.dumps(message.artifact, allow_nan=False)
    exposed = json.dumps(
        {"content": message.content, "artifact": message.artifact},
        ensure_ascii=False,
    ).casefold()
    assert "usr_hidden_runtime" not in exposed
    assert "hidden_runtime_seed" not in exposed
    assert "x-user-id" not in exposed
    assert "industrialapiclient" not in exposed
    assert "raw_response" not in exposed
    assert "golden" not in exposed


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (get_asset, {"asset_id": "asset_M101/analyses"}),
        (get_asset, {"asset_id": "asset_other"}),
        (
            list_asset_analyses,
            {"asset_id": "asset_M101", "status": "deleted"},
        ),
        (get_analysis, {"analysis_id": "analysis_A1"}),
        (
            get_baseline,
            {"asset_id": "asset_M101", "point_id": "sensor_M101"},
        ),
        (
            get_rms_series,
            {"asset_id": "asset_M101", "point_id": "pt_M101?seed=other"},
        ),
        (get_spectrum, {"asset_id": "asset_M101 ", "point_id": None}),
        (
            get_data_quality,
            {"asset_id": "asset_M101", "point_id": "pt_M101 de"},
        ),
        (get_model, {"model_id": "mdl_other"}),
        (search_knowledge, {"query": " procedimento"}),
        (
            search_knowledge,
            {"query": "procedimento", "document_type": "manual"},
        ),
        (get_knowledge_document, {"document_id": "kb_doc/../../secret"}),
    ],
    ids=lambda value: getattr(value, "name", None),
)
def test_invalid_ids_filters_and_hidden_arguments_never_reach_http(
    tool: BaseTool,
    arguments: dict[str, object],
):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    with pytest.raises((ValueError, TypeError)):
        asyncio.run(_invoke_tool_directly(tool, arguments, handler))

    assert calls == 0


@pytest.mark.parametrize(
    ("category", "expected_error"),
    [
        (
            ApiErrorCategory.API,
            {
                "ok": False,
                "category": "api",
                "code": "ASSET_NOT_FOUND",
                "message": "Ativo não encontrado.",
                "status_code": 404,
            },
        ),
        (
            ApiErrorCategory.SERVER,
            {
                "ok": False,
                "category": "server",
                "code": "SERVICE_UNAVAILABLE",
                "message": "Serviço indisponível.",
                "status_code": 503,
            },
        ),
        (
            ApiErrorCategory.TIMEOUT,
            {
                "ok": False,
                "category": "timeout",
                "code": "READ_TIMEOUT",
                "message": "A API não respondeu dentro do tempo limite.",
                "status_code": None,
            },
        ),
        (
            ApiErrorCategory.TRANSPORT,
            {
                "ok": False,
                "category": "transport",
                "code": "TRANSPORT_ERROR",
                "message": "Não foi possível comunicar com a API.",
                "status_code": None,
            },
        ),
        (
            ApiErrorCategory.INVALID_RESPONSE,
            {
                "ok": False,
                "category": "invalid_response",
                "code": "INVALID_JSON_RESPONSE",
                "message": "A API retornou um corpo que não é JSON válido.",
                "status_code": 200,
            },
        ),
    ],
)
def test_every_api_error_category_remains_visible_in_content_and_artifact(
    category: ApiErrorCategory,
    expected_error: dict[str, object],
):
    def handler(request: httpx.Request) -> httpx.Response:
        if category is ApiErrorCategory.API:
            return httpx.Response(
                404,
                json={
                    "code": "ASSET_NOT_FOUND",
                    "message": "Ativo não encontrado.",
                },
            )
        if category is ApiErrorCategory.SERVER:
            return httpx.Response(
                503,
                json={
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "Serviço indisponível.",
                },
            )
        if category is ApiErrorCategory.TIMEOUT:
            raise httpx.ReadTimeout("timeout", request=request)
        if category is ApiErrorCategory.TRANSPORT:
            raise httpx.ConnectError("connection failed", request=request)
        return httpx.Response(200, content=b"not-json")

    result = asyncio.run(
        _invoke_catalog(
            handler,
            _single_call("get_asset", {"asset_id": "asset_M101"}),
        )
    )
    message = result["messages"][-1]

    assert json.loads(message.content) == {"error": expected_error}
    assert message.artifact["outcome"]["error"] == expected_error


@pytest.mark.parametrize("mode", list(ResponseMode))
def test_every_response_mode_remains_visible_in_the_tool_artifact(mode: ResponseMode):
    notes = f"Resposta em modo {mode.value}."
    data = (
        _asset_payload()
        if mode is ResponseMode.COMPLETE
        else {"id": "asset_M101", "sensor_status": "offline"}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"mode": mode.value, "notes": notes, "data": data},
        )

    result = asyncio.run(
        _invoke_catalog(
            handler,
            _single_call("get_asset", {"asset_id": "asset_M101"}),
        )
    )
    message = result["messages"][-1]
    content = json.loads(message.content)

    assert message.artifact["outcome"]["mode"] == mode.value
    assert message.artifact["outcome"]["notes"] == notes
    if mode is ResponseMode.COMPLETE:
        assert content["id"] == "asset_M101"
    else:
        assert content == {
            "mode": mode.value,
            "notes": notes,
            "partial_data": data,
        }


def test_all_catalog_tools_return_complete_json_without_hidden_runtime_data():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "mode": "complete",
                "notes": None,
                "data": _complete_payload_for_path(request.url.path),
            },
        )

    result = asyncio.run(
        _invoke_catalog(
            handler,
            _all_tool_calls(),
            user_id="usr_success_secret",
            company_id="comp_forja_br",
            seed="seed_success_secret",
        )
    )
    messages = result["messages"][-len(READ_TOOLS) :]
    messages_by_id = {message.tool_call_id: message for message in messages}
    expected_paths = Counter(
        {
            "/assets/asset_M101": 1,
            "/assets/asset_M101/analyses": 1,
            "/analyses/an_A1": 1,
            "/assets/asset_M101/baseline": 1,
            "/assets/asset_M101/rms": 1,
            "/assets/asset_M101/spectrum": 1,
            "/assets/asset_M101/data-quality": 1,
            "/models/mdl_vib_v3": 1,
            "/knowledge/search": 1,
            "/knowledge/kb_doc_1": 1,
        }
    )

    assert len(requests) == len(messages) == len(READ_TOOLS) == 10
    assert Counter(request.url.path for request in requests) == expected_paths
    assert all(request.method == "GET" for request in requests)
    assert all(
        request.headers["x-user-id"] == "usr_success_secret" for request in requests
    )
    assert all(
        dict(request.url.params)["seed"] == "seed_success_secret"
        for request in requests
    )
    for index, tool in enumerate(READ_TOOLS):
        message = messages_by_id[f"call_{index}"]
        assert isinstance(message, ToolMessage)
        assert message.name == tool.name
        assert message.tool_call_id == f"call_{index}"
        content = json.loads(message.content)
        assert "error" not in content
        assert message.artifact["tool_name"] == tool.name
        assert message.artifact["outcome"]["mode"] == "complete"
        assert message.artifact["outcome"]["error"] is None
        json.dumps(content, allow_nan=False)
        json.dumps(message.artifact, allow_nan=False)

        exposed = json.dumps(
            {"content": content, "artifact": message.artifact},
            ensure_ascii=False,
        ).casefold()
        for forbidden in (
            "usr_success_secret",
            "seed_success_secret",
            "x-user-id",
            "authorization",
            '"headers"',
            '"identity"',
            '"permissions"',
            '"runtime"',
            '"client"',
            "industrialapiclient",
            "mocktransport",
            "httpx.response",
            "raw_response",
            "rawresponse",
            "expected-paths.json",
            "test-scenarios.md",
            "cases.parquet",
            "golden set",
            "golden_set",
            '"evaluation',
        ):
            assert forbidden not in exposed


def test_all_catalog_error_outputs_exclude_trusted_context_and_transport_data():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            404,
            json={"code": "NOT_FOUND", "message": "Recurso não encontrado."},
        )

    result = asyncio.run(
        _invoke_catalog(
            handler,
            _all_tool_calls(),
            user_id="usr_runtime_secret",
            company_id="comp_runtime_secret",
            seed="seed_runtime_secret",
        )
    )
    messages = result["messages"][-len(READ_TOOLS) :]

    assert len(requests) == len(messages) == 10
    assert {message.name for message in messages} == {tool.name for tool in READ_TOOLS}
    assert all(request.method == "GET" for request in requests)
    assert all(
        request.headers["x-user-id"] == "usr_runtime_secret" for request in requests
    )
    for message in messages:
        assert isinstance(message, ToolMessage)
        json.loads(message.content)
        json.dumps(message.artifact, allow_nan=False)
        exposed = json.dumps(
            {"content": message.content, "artifact": message.artifact},
            ensure_ascii=False,
        ).casefold()
        for forbidden in (
            "usr_runtime_secret",
            "comp_runtime_secret",
            "seed_runtime_secret",
            "x-user-id",
            "authorization",
            '"headers"',
            '"identity"',
            '"permissions"',
            '"runtime"',
            '"client"',
            "industrialapiclient",
            "mocktransport",
            "httpx.response",
            "raw_response",
            "rawresponse",
            "expected-paths.json",
            "test-scenarios.md",
            "cases.parquet",
            "golden set",
            "golden_set",
        ):
            assert forbidden not in exposed
