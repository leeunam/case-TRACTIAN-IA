from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, ToolRuntime
from pydantic import ValidationError

from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import ApiError, ApiErrorCategory, ResponseMode
from tractian_agent.tools.assets import execute_get_asset, get_asset
from tractian_agent.tools.runtime import ReadToolRuntime


def _asset_payload(*, asset_id: str = "asset_M101", company_id: str = "comp_forja_br"):
    return {
        "id": asset_id,
        "name": "Motor principal da forja",
        "company_id": company_id,
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
                "asset_id": asset_id,
                "location": "DE",
                "sensor_status": "online",
            }
        ],
    }


def _runtime(
    handler: httpx.AsyncByteStream | object,
    *,
    permissions: frozenset[str] = frozenset({"read"}),
    central_asset_id: str = "asset_M101",
    seed: str | None = "fixed-asset",
) -> ReadToolRuntime:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return ReadToolRuntime.create(
        user_id="usr_ana",
        company_id="comp_forja_br",
        permissions=permissions,
        central_asset_id=central_asset_id,
        client=IndustrialApiClient("https://simulator.test", transport=transport),
        seed=seed,
    )


async def _invoke_async(runtime: ReadToolRuntime, asset_id: str = "asset_M101"):
    try:
        return await execute_get_asset(asset_id, runtime)
    finally:
        await runtime.client.aclose()


def _invoke(runtime: ReadToolRuntime, asset_id: str = "asset_M101"):
    return asyncio.run(_invoke_async(runtime, asset_id))


async def _invoke_adapter_async(runtime: ReadToolRuntime, arguments: dict[str, object]):
    try:
        tool_runtime = ToolRuntime(
            state={},
            context=runtime,
            config={},
            stream_writer=lambda _: None,
            tool_call_id=None,
            store=None,
        )
        return await get_asset.ainvoke({**arguments, "runtime": tool_runtime})
    finally:
        await runtime.client.aclose()


async def _invoke_through_tool_node(runtime: ReadToolRuntime) -> dict[str, Any]:
    graph = StateGraph(MessagesState, context_schema=ReadToolRuntime)
    graph.add_node("tools", ToolNode([get_asset]))
    graph.add_edge(START, "tools")
    app = graph.compile()
    try:
        return await app.ainvoke(
            {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "get_asset",
                                "args": {"asset_id": "asset_M101"},
                                "id": "call_get_asset_1",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            },
            context=runtime,
        )
    finally:
        await runtime.client.aclose()


def test_get_asset_exposes_only_the_safe_public_contract():
    schema = get_asset.tool_call_schema.model_json_schema()

    assert get_asset.name == "get_asset"
    assert "cadastro técnico" in get_asset.description.lower()

    assert set(schema["properties"]) == {"asset_id"}
    assert schema["required"] == ["asset_id"]
    assert schema["properties"]["asset_id"]["pattern"] == (
        r"^asset_[A-Za-z0-9_-]{1,64}$"
    )

    serialized_schema = str(schema).lower()

    for hidden_name in (
        "runtime",
        "identity",
        "user_id",
        "company_id",
        "permissions",
        "client",
        "seed",
        "url",
        "method",
    ):
        assert hidden_name not in serialized_schema


def test_get_asset_calls_the_fixed_endpoint_and_normalizes_complete_asset():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"mode": "complete", "notes": None, "data": _asset_payload()},
        )

    result = _invoke(_runtime(handler))

    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/assets/asset_M101"
    assert dict(requests[0].url.params) == {"seed": "fixed-asset"}
    assert requests[0].headers["x-user-id"] == "usr_ana"
    assert result.content.model_dump() == {
        "id": "asset_M101",
        "name": "Motor principal da forja",
        "criticality": "critical",
        "machine_type": "motor_induction",
        "rotation_rpm": 1780,
        "sensor_status": "online",
        "points": [
            {"id": "pt_M101_de", "location": "DE", "sensor_status": "online"}
        ],
    }
    assert result.artifact.model_dump(mode="json") == {
        "tool_name": "get_asset",
        "arguments": {"asset_id": "asset_M101"},
        "source": {"kind": "industrial_api", "resource": "/assets/asset_M101"},
        "outcome": {
            "mode": "complete",
            "notes": None,
            "asset": {
                "id": "asset_M101",
                "name": "Motor principal da forja",
                "company_id": "comp_forja_br",
                "criticality": "critical",
                "hierarchy": {
                    "plant": "Planta 1",
                    "line": "Forjamento",
                    "parent_asset_id": None,
                },
                "points": [
                    {"id": "pt_M101_de", "location": "DE", "sensor_status": "online"}
                ],
                "technical_configuration": {
                    "machine_type": "motor_induction",
                    "rotation_rpm": 1780.0,
                    "bearing_specs": {
                        "part_number": "NU 310",
                        "bpfo_hz": 142.3,
                        "bpfi_hz": 218.1,
                        "bsf_hz": 58.7,
                        "ftf_hz": 11.9,
                    },
                    "line_frequency_hz": 60.0,
                },
                "sensor_status": "online",
            },
            "partial_data": None,
            "error": None,
        },
        "truncated": False,
        "omitted_items": 0,
    }
    assert "usr_ana" not in json.dumps(result.artifact.model_dump(mode="json"))


def test_get_asset_adapter_returns_only_model_content_and_rejects_unknown_arguments():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"mode": "complete", "notes": None, "data": _asset_payload()},
        )

    content = asyncio.run(
        _invoke_adapter_async(_runtime(handler), {"asset_id": "asset_M101"})
    )

    assert content == {
        "id": "asset_M101",
        "name": "Motor principal da forja",
        "criticality": "critical",
        "machine_type": "motor_induction",
        "rotation_rpm": 1780.0,
        "sensor_status": "online",
        "points": [
            {"id": "pt_M101_de", "location": "DE", "sensor_status": "online"}
        ],
    }
    assert calls == 1

    with pytest.raises(ValidationError):
        asyncio.run(
            _invoke_adapter_async(
                _runtime(handler),
                {"asset_id": "asset_M101", "unexpected": True},
            )
        )
    assert calls == 1


def test_get_asset_runs_through_tool_node_with_injected_langgraph_runtime():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"mode": "complete", "notes": None, "data": _asset_payload()},
        )

    result = asyncio.run(_invoke_through_tool_node(_runtime(handler)))

    assert len(requests) == 1
    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    assert message.name == "get_asset"
    assert message.tool_call_id == "call_get_asset_1"
    assert json.loads(message.content)["id"] == "asset_M101"
    assert message.artifact["tool_name"] == "get_asset"
    assert message.artifact["outcome"]["asset"]["technical_configuration"][
        "machine_type"
    ] == "motor_induction"


def test_read_tool_runtime_does_not_allow_trusted_identity_mutation():
    runtime = _runtime(lambda request: httpx.Response(200, json={}))
    try:
        with pytest.raises(ValidationError):
            runtime.identity.user_id = "usr_other"
        with pytest.raises(ValidationError):
            runtime.identity.company_id = "comp_other"
    finally:
        asyncio.run(runtime.client.aclose())


def test_get_asset_rejects_missing_read_permission_before_http():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    with pytest.raises(PermissionError, match="read"):
        _invoke(_runtime(handler, permissions=frozenset()))

    assert calls == 0


def test_get_asset_rejects_an_asset_outside_the_central_scope_before_http():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    with pytest.raises(ValueError, match="ativo central"):
        _invoke(_runtime(handler), "asset_other")

    assert calls == 0


@pytest.mark.parametrize(
    ("asset_id", "company_id", "match"),
    [
        ("asset_other", "comp_forja_br", "identificador"),
        ("asset_M101", "comp_other", "empresa"),
    ],
)
def test_get_asset_rejects_complete_response_with_mismatched_scope(
    asset_id: str, company_id: str, match: str
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mode": "complete",
                "notes": None,
                "data": _asset_payload(asset_id=asset_id, company_id=company_id),
            },
        )

    with pytest.raises(ValueError, match=match):
        _invoke(_runtime(handler))


def test_get_asset_rejects_complete_response_with_a_point_from_another_asset():
    payload = _asset_payload()
    payload["points"][0]["asset_id"] = "asset_other"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"mode": "complete", "notes": None, "data": payload},
        )

    with pytest.raises(ValueError, match="ponto"):
        _invoke(_runtime(handler))


def test_get_asset_preserves_degraded_mode_notes_and_partial_json():
    partial_data = {"id": "asset_M101", "sensor_status": "offline"}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"mode": "partial", "notes": "Pontos indisponíveis.", "data": partial_data},
        )

    result = _invoke(_runtime(handler, seed=None))

    assert result.content is None
    assert result.artifact.outcome.mode is ResponseMode.PARTIAL
    assert result.artifact.outcome.notes == "Pontos indisponíveis."
    assert result.artifact.outcome.partial_data == partial_data
    assert result.artifact.outcome.asset is None


def test_get_asset_rejects_a_contradictory_id_in_degraded_data():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mode": "partial",
                "notes": "Resposta inconsistente.",
                "data": {"id": "asset_other"},
            },
        )

    with pytest.raises(ValueError, match="identificador"):
        _invoke(_runtime(handler))


@pytest.mark.parametrize(
    ("partial_data", "match"),
    [
        ({"id": None}, "identificador"),
        ({"company_id": None}, "empresa"),
        (
            {
                "id": "asset_M101",
                "company_id": "comp_forja_br",
                "parent_asset_id": "asset_parent",
                "points": [{"id": "pt_any", "asset_id": "asset_other"}],
            },
            "ponto",
        ),
    ],
)
def test_get_asset_rejects_inconsistent_or_null_scope_fields_in_degraded_data(
    partial_data: dict[str, object], match: str
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"mode": "partial", "notes": "Resposta inconsistente.", "data": partial_data},
        )

    with pytest.raises(ValueError, match=match):
        _invoke(_runtime(handler))


def test_get_asset_rejects_forbidden_nested_partial_context_before_exposure():
    partial_data = {
        "id": "asset_M101",
        "details": {"headers": {"Authorization": "Bearer secret"}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"mode": "partial", "notes": "Resposta degradada.", "data": partial_data},
        )

    with pytest.raises(ValueError, match="proibido"):
        _invoke(_runtime(handler))


@pytest.mark.parametrize(
    "partial_data",
    [
        {"id": "asset_M101", "nested": {"X-User-Id": "usr_secret"}},
        {"id": "asset_M101", "nested": {"api key": "secret"}},
        {"id": "asset_M101", "nested": {"Refresh-Token": "secret"}},
        {"id": "asset_M101", "nested": {"RAW response": {"body": "..."}}},
        {"id": "asset_M101", "nested": {"Central Asset Id": "asset_M101"}},
    ],
)
def test_get_asset_rejects_normalized_sensitive_partial_keys(
    partial_data: dict[str, object],
):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"mode": "partial", "notes": "Resposta degradada.", "data": partial_data},
        )

    with pytest.raises(ValueError, match="proibido"):
        _invoke(_runtime(handler))


def test_get_asset_preserves_legitimate_partial_domain_and_status_fields():
    partial_data = {
        "id": "asset_M101",
        "company_id": "comp_forja_br",
        "status_code": 206,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"mode": "partial", "notes": "Resposta degradada.", "data": partial_data},
        )

    result = _invoke(_runtime(handler))

    assert result.artifact.outcome.partial_data == partial_data


@pytest.mark.parametrize(
    "points",
    [
        {"asset_id": "asset_other"},
        ["pt_M101_de"],
        [{"asset_id": None}],
        [{"asset_id": "asset_other"}],
    ],
)
def test_get_asset_rejects_invalid_degraded_points_structure(points: object):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mode": "partial",
                "notes": "Resposta degradada.",
                "data": {"id": "asset_M101", "points": points},
            },
        )

    with pytest.raises(ValueError, match="pontos|ponto"):
        _invoke(_runtime(handler))


def test_get_asset_preserves_api_error_exactly():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"code": "ASSET_NOT_FOUND", "message": "Ativo não encontrado."},
        )

    result = _invoke(_runtime(handler))

    expected = ApiError(
        category=ApiErrorCategory.API,
        code="ASSET_NOT_FOUND",
        message="Ativo não encontrado.",
        status_code=404,
    )
    assert result.error == expected
    assert result.artifact.outcome.error == expected
