from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from langgraph.prebuilt import ToolRuntime
from pydantic import ValidationError

from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import ApiError, ApiErrorCategory
from tractian_agent.tools.knowledge import (
    execute_get_knowledge_document,
    execute_get_model,
    execute_search_knowledge,
    get_knowledge_document,
    get_model,
    search_knowledge,
)
from tractian_agent.tools.runtime import ReadToolRuntime


def _runtime(
    handler: object,
    *,
    permissions: frozenset[str] = frozenset({"read"}),
    configured_model_id: str = "mdl_vib_v3",
    seed: str | None = "fixed-knowledge",
) -> ReadToolRuntime:
    return ReadToolRuntime.create(
        user_id="usr_ana",
        company_id="comp_forja_br",
        permissions=permissions,
        central_asset_id="asset_M101",
        client=IndustrialApiClient(
            "https://simulator.test", transport=httpx.MockTransport(handler)  # type: ignore[arg-type]
        ),
        seed=seed,
        configured_model_id=configured_model_id,
    )


def _model_payload(*, model_id: str = "mdl_vib_v3") -> dict[str, object]:
    return {
        "id": model_id,
        "version": "3.2.1",
        "coverage": [
            {
                "machine_type": "motor_induction",
                "supported": True,
                "can_learn_baseline": True,
            },
            {
                "machine_type": "motor_dc",
                "supported": True,
                "can_learn_baseline": False,
                "note": "detecção apenas sintomática",
            },
        ],
        "requirements": {
            "min_completeness": 0.8,
            "min_snr_db": 12.0,
            "min_rotation_rpm": None,
        },
        "processing_state": "delayed",
        "last_run_at": "2026-01-02T03:04:05+00:00",
    }


def _document_payload(
    *, document_id: str = "kb_proc_001", document_type: str = "procedure", body: str = "Passos seguros."
) -> dict[str, object]:
    return {
        "id": document_id,
        "type": document_type,
        "title": "Procedimento de troca",
        "body": body,
        "tags": ["rolamento", "baseline"],
    }


def _run(coro: Any):
    return asyncio.run(coro)


async def _model(runtime: ReadToolRuntime):
    try:
        return await execute_get_model(runtime)
    finally:
        await runtime.client.aclose()


async def _search(runtime: ReadToolRuntime, query: object = "rolamento", document_type: object = None):
    try:
        return await execute_search_knowledge(query, document_type, runtime)
    finally:
        await runtime.client.aclose()


async def _document(runtime: ReadToolRuntime, document_id: object = "kb_proc_001"):
    try:
        return await execute_get_knowledge_document(document_id, runtime)
    finally:
        await runtime.client.aclose()


def test_public_schemas_hide_runtime_seed_and_configured_model_id():
    model_schema = get_model.tool_call_schema.model_json_schema()
    search_schema = search_knowledge.tool_call_schema.model_json_schema()
    document_schema = get_knowledge_document.tool_call_schema.model_json_schema()

    assert model_schema["properties"] == {}
    assert model_schema.get("required", []) == []
    assert set(search_schema["properties"]) == {"query", "document_type"}
    assert search_schema["required"] == ["query"]
    assert set(document_schema["properties"]) == {"document_id"}
    assert document_schema["properties"]["document_id"]["pattern"] == r"^kb_[A-Za-z0-9_-]{1,64}$"
    serialized = json.dumps([model_schema, search_schema, document_schema]).lower()
    for hidden in ("runtime", "identity", "permissions", "client", "seed", "configured_model_id"):
        assert hidden not in serialized


def test_runtime_default_is_strict_frozen_and_configurable():
    runtime = _runtime(lambda request: httpx.Response(200, json={}))
    assert runtime.configured_model_id == "mdl_vib_v3"
    assert runtime.model_dump()["configured_model_id"] == "mdl_vib_v3"
    assert _runtime(lambda request: httpx.Response(200, json={}), configured_model_id="mdl_other").configured_model_id == "mdl_other"
    with pytest.raises(ValidationError):
        _runtime(lambda request: httpx.Response(200, json={}), configured_model_id="model bad")
    with pytest.raises(ValidationError):
        runtime.configured_model_id = "mdl_other"  # type: ignore[misc]


def test_model_uses_only_configured_fixed_endpoint_and_real_payload():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"mode": "complete", "notes": None, "data": _model_payload()})

    result = _run(_model(_runtime(handler)))

    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/models/mdl_vib_v3"
    assert dict(requests[0].url.params) == {"seed": "fixed-knowledge"}
    assert result.content.id == "mdl_vib_v3"
    assert result.content.coverage[1].can_learn_baseline is False
    assert result.content.requirements.min_rotation_rpm is None


def test_search_and_document_use_fixed_paths_params_and_normalized_payloads():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/knowledge/search":
            return httpx.Response(200, json={"mode": "complete", "notes": None, "data": {"results": [_document_payload()]}})
        return httpx.Response(200, json={"mode": "complete", "notes": None, "data": _document_payload()})

    search_result = _run(_search(_runtime(handler), "rolamento", "procedure"))
    document_result = _run(_document(_runtime(handler)))

    assert requests[0].url.path == "/knowledge/search"
    assert dict(requests[0].url.params) == {"q": "rolamento", "type": "procedure", "seed": "fixed-knowledge"}
    assert search_result.content.results[0].snippet == "Passos seguros."
    assert "body" not in search_result.content.results[0].model_dump()
    assert requests[1].url.path == "/knowledge/kb_proc_001"
    assert dict(requests[1].url.params) == {"seed": "fixed-knowledge"}
    assert document_result.content.body == "Passos seguros."


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        ("search", (" x", None)),
        ("search", ("x ", None)),
        ("search", ("x", None)),
        ("search", ("x" * 201, None)),
        ("search", ("ok", "other")),
        ("document", ("kb_",)),
        ("document", ("kb_" + "x" * 65,)),
    ],
)
def test_core_rejects_invalid_arguments_before_http(operation: str, arguments: tuple[object, ...]):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    runtime = _runtime(handler)
    with pytest.raises(ValidationError):
        if operation == "search":
            _run(_search(runtime, *arguments))
        else:
            _run(_document(runtime, *arguments))
    assert calls == 0


@pytest.mark.parametrize("operation", ["model", "search", "document"])
def test_all_tools_require_read_before_http(operation: str):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    runtime = _runtime(handler, permissions=frozenset())
    with pytest.raises(PermissionError, match="read"):
        if operation == "model":
            _run(_model(runtime))
        elif operation == "search":
            _run(_search(runtime))
        else:
            _run(_document(runtime))
    assert calls == 0


def test_model_rejects_returned_id_wrong_schema_duplicates_and_naive_timestamp():
    payload = _model_payload(model_id="mdl_other")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"mode": "complete", "notes": None, "data": payload})

    with pytest.raises(ValueError, match="modelo diferente"):
        _run(_model(_runtime(handler)))

    for mutation in (
        lambda value: value["coverage"].append(value["coverage"][0].copy()),  # type: ignore[index]
        lambda value: value.update({"last_run_at": "2026-01-02T03:04:05"}),
        lambda value: value["requirements"].pop("min_rotation_rpm"),  # type: ignore[index]
    ):
        bad = _model_payload()
        mutation(bad)
        result = _run(_model(_runtime(lambda request, bad=bad: httpx.Response(200, json={"mode": "complete", "notes": None, "data": bad}))))
        assert result.error is not None
        assert result.error.code == "INVALID_SCHEMA_RESPONSE"


def test_search_validates_all_rows_before_limit_filter_ids_and_snippets():
    documents = [_document_payload(document_id=f"kb_proc_{index}", body=" a\n  b " + "z" * 300) for index in range(11)]

    result = _run(_search(_runtime(lambda request: httpx.Response(200, json={"mode": "complete", "notes": None, "data": {"results": documents}})), "rolamento", "procedure"))

    assert result.content.total_results == 11
    assert result.content.returned_results == 10
    assert result.content.omitted_results == 1
    assert result.content.truncated is True
    assert len(result.content.results[0].snippet) == 240
    assert result.artifact.outcome.returned_results == 10

    duplicate = [_document_payload(), _document_payload()]
    with pytest.raises(ValueError, match="duplicado"):
        _run(_search(_runtime(lambda request: httpx.Response(200, json={"mode": "complete", "notes": None, "data": {"results": duplicate}})), "rolamento"))
    wrong_type = [_document_payload(document_type="guidance")]
    with pytest.raises(ValueError, match="filtro"):
        _run(_search(_runtime(lambda request: httpx.Response(200, json={"mode": "complete", "notes": None, "data": {"results": wrong_type}})), "rolamento", "procedure"))


@pytest.mark.parametrize("mode", ["partial", "inconclusive", "conflict", "unavailable"])
def test_degraded_stable_search_projects_full_rows_without_bodies(mode: str):
    data: dict[str, object] = {"results": [_document_payload()]}
    if mode == "conflict":
        data["conflict"] = True
    result = _run(_search(_runtime(lambda request: httpx.Response(200, json={"mode": mode, "notes": "degradado", "data": data}))))

    assert result.content.mode.value == mode
    assert result.content.results[0]["id"] == "kb_proc_001"
    assert "body" not in result.content.results[0]
    assert result.artifact.outcome.partial_data == ({"conflict": True} if mode == "conflict" else {})


def test_document_applies_8k_and_32k_body_limits_in_complete_and_degraded_modes():
    body = "x" * 40_001
    payload = _document_payload(body=body)
    result = _run(_document(_runtime(lambda request: httpx.Response(200, json={"mode": "complete", "notes": None, "data": payload}))))

    assert len(result.content.body) == 8_000
    assert result.content.returned_body_characters == 8_000
    assert result.content.omitted_body_characters == 32_001
    assert result.content.truncated is True
    artifact_document = result.artifact.outcome.document
    assert len(artifact_document.body) == 32_000
    assert artifact_document.omitted_body_characters == 8_001

    partial = {"id": "kb_proc_001", "body": body, "conflict": True}
    degraded = _run(_document(_runtime(lambda request: httpx.Response(200, json={"mode": "conflict", "notes": "conflito", "data": partial}))))
    assert len(degraded.content.document["body"]) == 8_000
    assert len(degraded.artifact.outcome.document["body"]) == 32_000
    assert degraded.artifact.outcome.partial_data == {"conflict": True}
    assert json.dumps(degraded.content.model_dump(mode="json"), allow_nan=False)
    assert json.dumps(degraded.artifact.model_dump(mode="json"), allow_nan=False)


def test_degraded_model_projects_only_known_fields_and_api_errors_are_exact():
    result = _run(_model(_runtime(lambda request: httpx.Response(200, json={"mode": "partial", "notes": "parcial", "data": {"id": "mdl_vib_v3", "processing_state": "delayed"}}))))
    assert result.content.mode.value == "partial"
    assert result.content.model == {"id": "mdl_vib_v3", "processing_state": "delayed"}
    with pytest.raises(ValueError, match="campo de topo"):
        _run(_model(_runtime(lambda request: httpx.Response(200, json={"mode": "partial", "notes": None, "data": {"id": "mdl_vib_v3", "raw": "no"}}))))

    error = ApiError(category=ApiErrorCategory.API, code="NOT_FOUND", message="Não encontrado.", status_code=404)
    result = _run(_document(_runtime(lambda request: httpx.Response(404, json={"code": "NOT_FOUND", "message": "Não encontrado."}))))
    assert result.error == error
    assert result.artifact.outcome.error == error


def test_adapters_return_json_safe_content_and_artifact():
    runtime = _runtime(lambda request: httpx.Response(200, json={"mode": "complete", "notes": None, "data": _model_payload()}))
    tool_runtime = ToolRuntime(
        state={}, context=runtime, config={}, stream_writer=lambda _: None, tool_call_id=None, store=None
    )

    async def invoke():
        try:
            return await get_model.ainvoke({"runtime": tool_runtime})
        finally:
            await runtime.client.aclose()

    content = _run(invoke())
    assert content["id"] == "mdl_vib_v3"
    assert get_model.response_format == "content_and_artifact"
    assert search_knowledge.response_format == "content_and_artifact"
    assert get_knowledge_document.response_format == "content_and_artifact"
    assert json.dumps(get_model.tool_call_schema.model_json_schema(), allow_nan=False)
