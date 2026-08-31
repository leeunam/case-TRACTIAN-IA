from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, get_type_hints

import httpx
import pytest
from pydantic import ValidationError

import tractian_agent.write_operations as write_operations
from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import ActionReceipt, ApiError, ApiErrorCategory
from tractian_agent.tools.runtime import WriteToolRuntime
from tractian_agent.write_operations import (
    execute_escalate_case,
    execute_reprocess_analysis,
    execute_request_model_retraining,
    execute_request_specialist_analysis,
    execute_update_asset_criticality,
)
from tractian_agent.write_policy import (
    EscalateCaseProposal,
    ReprocessProposal,
    RequestModelRetrainingProposal,
    RequestSpecialistAnalysisProposal,
    UpdateAssetCriticalityProposal,
)


def _analysis_payload(
    *,
    analysis_id: str = "an_9901",
    asset_id: str = "asset_M101",
) -> dict[str, object]:
    return {
        "id": analysis_id,
        "asset_id": asset_id,
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


def _runtime(
    handler: Any,
    *,
    central_asset_id: str = "asset_M101",
    current_case_id: str = "case_tkt_exe_12",
    configured_model_id: str = "mdl_vib_v3",
) -> WriteToolRuntime:
    return WriteToolRuntime.create(
        user_id="usr_ana",
        company_id="comp_forja_br",
        permissions=frozenset({"read", "action_low", "action_high", "escalate"}),
        central_asset_id=central_asset_id,
        current_case_id=current_case_id,
        configured_model_id=configured_model_id,
        client=IndustrialApiClient(
            "https://simulator.test",
            transport=httpx.MockTransport(handler),
        ),
    )


async def _execute_and_close(operation: Any, runtime: WriteToolRuntime):
    try:
        return await operation
    finally:
        await runtime.client.aclose()


def _operation_for_error_case(name: str, runtime: WriteToolRuntime):
    justification = "A evidência registrada sustenta esta ação industrial."
    if name == "reprocess":
        return execute_reprocess_analysis(
            ReprocessProposal(
                analysis_id="an_9901",
                justification=justification,
            ),
            runtime,
            idempotency_key="tractian-agent:intent-01",
        )
    if name == "specialist":
        return execute_request_specialist_analysis(
            RequestSpecialistAnalysisProposal(
                analysis_id="an_9901",
                justification=justification,
            ),
            runtime,
        )
    if name == "criticality":
        return execute_update_asset_criticality(
            UpdateAssetCriticalityProposal(
                criticality="high",
                justification=justification,
            ),
            runtime,
        )
    if name == "retraining":
        return execute_request_model_retraining(
            RequestModelRetrainingProposal(justification=justification),
            runtime,
        )
    return execute_escalate_case(
        EscalateCaseProposal(justification=justification),
        runtime,
    )


def test_write_runtime_factory_requires_a_valid_current_case_and_is_frozen():
    client = IndustrialApiClient(
        "https://simulator.test",
        transport=httpx.MockTransport(lambda _: httpx.Response(500)),
    )

    runtime = WriteToolRuntime.create(
        user_id="usr_ana",
        company_id="comp_forja_br",
        permissions=frozenset({"read", "action_low"}),
        central_asset_id="asset_M101",
        current_case_id="case_tkt_exe_12",
        configured_model_id="mdl_vib_v3",
        client=client,
    )

    assert runtime.current_case_id == "case_tkt_exe_12"
    assert runtime.identity.user_id == "usr_ana"
    with pytest.raises(ValidationError):
        runtime.current_case_id = "case_other"
    with pytest.raises(ValidationError):
        WriteToolRuntime.create(
            user_id="usr_ana",
            company_id="comp_forja_br",
            permissions=frozenset({"read", "action_low"}),
            central_asset_id="asset_M101",
            current_case_id="ticket_without_case_prefix",
            client=client,
        )
    asyncio.run(client.aclose())


def test_reprocess_preflights_scope_then_posts_exact_request_with_persisted_key():
    requests: list[httpx.Request] = []
    justification = "O rolamento foi trocado e a análise precisa ser refeita."

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "mode": "complete",
                    "notes": None,
                    "data": _analysis_payload(),
                },
            )
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "action_id": "act_reprocess_01",
                "message": "Reprocesso aceito.",
            },
        )

    runtime = _runtime(handler)
    proposal = ReprocessProposal(
        analysis_id="an_9901",
        justification=justification,
    )

    result = asyncio.run(
        _execute_and_close(
            execute_reprocess_analysis(
                proposal,
                runtime,
                idempotency_key="tractian-agent:intent-01",
            ),
            runtime,
        )
    )

    assert result == ActionReceipt(
        accepted=True,
        action_id="act_reprocess_01",
        message="Reprocesso aceito.",
    )
    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/analyses/an_9901"),
        ("POST", "/analyses/an_9901/reprocess"),
    ]
    assert dict(requests[0].url.params) == {}
    assert requests[0].headers["x-user-id"] == "usr_ana"
    assert "idempotency-key" not in requests[0].headers
    assert json.loads(requests[1].content) == {"justification": justification}
    assert requests[1].headers["x-user-id"] == "usr_ana"
    assert requests[1].headers["idempotency-key"] == "tractian-agent:intent-01"


def test_specialist_request_preflights_scope_then_posts_without_idempotency_key():
    requests: list[httpx.Request] = []
    justification = "Os dados conflitantes exigem uma análise especializada."

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"mode": "complete", "notes": None, "data": _analysis_payload()},
            )
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "action_id": "act_specialist_01",
                "message": "Análise especializada solicitada.",
            },
        )

    runtime = _runtime(handler)
    proposal = RequestSpecialistAnalysisProposal(
        analysis_id="an_9901",
        justification=justification,
    )

    result = asyncio.run(
        _execute_and_close(
            execute_request_specialist_analysis(proposal, runtime),
            runtime,
        )
    )

    assert result == ActionReceipt(
        accepted=True,
        action_id="act_specialist_01",
        message="Análise especializada solicitada.",
    )
    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/analyses/an_9901"),
        ("POST", "/analyses/an_9901/request-specialist"),
    ]
    assert json.loads(requests[1].content) == {"justification": justification}
    assert requests[1].headers["x-user-id"] == "usr_ana"
    assert all("idempotency-key" not in request.headers for request in requests)


def test_criticality_update_patches_only_the_central_asset_with_closed_body():
    requests: list[httpx.Request] = []
    justification = "O impacto operacional agora exige criticidade crítica."

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "action_id": "act_criticality_01",
                "message": "Criticidade atualizada.",
            },
        )

    runtime = _runtime(handler, central_asset_id="asset_G501")
    proposal = UpdateAssetCriticalityProposal(
        criticality="critical",
        justification=justification,
    )

    result = asyncio.run(
        _execute_and_close(
            execute_update_asset_criticality(proposal, runtime),
            runtime,
        )
    )

    assert result == ActionReceipt(
        accepted=True,
        action_id="act_criticality_01",
        message="Criticidade atualizada.",
    )
    assert [(request.method, request.url.path) for request in requests] == [
        ("PATCH", "/assets/asset_G501")
    ]
    assert json.loads(requests[0].content) == {
        "justification": justification,
        "changes": {"criticality": "critical"},
    }
    assert requests[0].headers["x-user-id"] == "usr_ana"
    assert "idempotency-key" not in requests[0].headers


def test_retraining_request_posts_only_to_the_configured_model_without_key():
    requests: list[httpx.Request] = []
    justification = "Erros sistemáticos sustentam solicitar um novo treinamento."

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "action_id": "act_retraining_01",
                "message": "Retreinamento solicitado.",
            },
        )

    runtime = _runtime(handler, configured_model_id="mdl_custom_v4")
    proposal = RequestModelRetrainingProposal(justification=justification)

    result = asyncio.run(
        _execute_and_close(
            execute_request_model_retraining(proposal, runtime),
            runtime,
        )
    )

    assert result == ActionReceipt(
        accepted=True,
        action_id="act_retraining_01",
        message="Retreinamento solicitado.",
    )
    assert [(request.method, request.url.path) for request in requests] == [
        ("POST", "/models/mdl_custom_v4/request-retraining")
    ]
    assert json.loads(requests[0].content) == {"justification": justification}
    assert requests[0].headers["x-user-id"] == "usr_ana"
    assert "idempotency-key" not in requests[0].headers


def test_escalation_posts_only_to_the_current_case_without_key():
    requests: list[httpx.Request] = []
    justification = "O caso ultrapassa o atendimento remoto e exige campo."

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "action_id": "act_escalate_01",
                "message": "Caso escalado.",
            },
        )

    runtime = _runtime(handler, current_case_id="case_tkt_exe_16")
    proposal = EscalateCaseProposal(justification=justification)

    result = asyncio.run(
        _execute_and_close(execute_escalate_case(proposal, runtime), runtime)
    )

    assert result == ActionReceipt(
        accepted=True,
        action_id="act_escalate_01",
        message="Caso escalado.",
    )
    assert [(request.method, request.url.path) for request in requests] == [
        ("POST", "/cases/case_tkt_exe_16/escalate")
    ]
    assert json.loads(requests[0].content) == {"justification": justification}
    assert requests[0].headers["x-user-id"] == "usr_ana"
    assert "idempotency-key" not in requests[0].headers


def test_module_publishes_only_the_five_typed_fixed_operations():
    expected_parameters = {
        "execute_reprocess_analysis": [
            "proposal",
            "runtime",
            "idempotency_key",
        ],
        "execute_request_specialist_analysis": ["proposal", "runtime"],
        "execute_update_asset_criticality": ["proposal", "runtime"],
        "execute_request_model_retraining": ["proposal", "runtime"],
        "execute_escalate_case": ["proposal", "runtime"],
    }

    assert write_operations.__all__ == list(expected_parameters)
    for name, parameters in expected_parameters.items():
        operation = getattr(write_operations, name)
        assert list(inspect.signature(operation).parameters) == parameters
        assert get_type_hints(operation)["return"] == ActionReceipt | ApiError
    assert (
        inspect.signature(execute_reprocess_analysis)
        .parameters["idempotency_key"]
        .kind
        is inspect.Parameter.KEYWORD_ONLY
    )


def test_reprocess_returns_a_stable_error_for_invalid_key_before_any_http():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    runtime = _runtime(handler)
    proposal = ReprocessProposal(
        analysis_id="an_9901",
        justification="O rolamento foi trocado e a análise precisa ser refeita.",
    )

    result = asyncio.run(
        _execute_and_close(
            execute_reprocess_analysis(
                proposal,
                runtime,
                idempotency_key="invalid key with spaces",
            ),
            runtime,
        )
    )

    assert result == ApiError(
        category=ApiErrorCategory.API,
        code="INVALID_IDEMPOTENCY_KEY",
        message="A chave idempotente persistida é inválida.",
    )
    assert requests == []


@pytest.mark.parametrize("mode", ["partial", "inconclusive", "conflict", "unavailable"])
def test_degraded_analysis_never_authorizes_a_write(mode: str):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "mode": mode,
                "notes": "A análise não está completa.",
                "data": {"id": "an_9901", "asset_id": "asset_M101"},
            },
        )

    runtime = _runtime(handler)
    proposal = ReprocessProposal(
        analysis_id="an_9901",
        justification="O rolamento foi trocado e a análise precisa ser refeita.",
    )

    result = asyncio.run(
        _execute_and_close(
            execute_reprocess_analysis(
                proposal,
                runtime,
                idempotency_key="tractian-agent:intent-01",
            ),
            runtime,
        )
    )

    assert result == ApiError(
        category=ApiErrorCategory.INVALID_RESPONSE,
        code="ANALYSIS_SCOPE_UNCONFIRMED",
        message="Não foi possível confirmar que a análise pertence ao ativo central.",
    )
    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/analyses/an_9901")
    ]


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        (
            _analysis_payload(asset_id="asset_other"),
            ApiError(
                category=ApiErrorCategory.INVALID_RESPONSE,
                code="ANALYSIS_SCOPE_UNCONFIRMED",
                message=(
                    "Não foi possível confirmar que a análise pertence ao ativo central."
                ),
            ),
        ),
        (
            {"id": "an_9901", "asset_id": "asset_M101"},
            ApiError(
                category=ApiErrorCategory.INVALID_RESPONSE,
                code="INVALID_SCHEMA_RESPONSE",
                message="A resposta da API não corresponde ao contrato esperado.",
                status_code=200,
            ),
        ),
    ],
)
def test_divergent_asset_and_invalid_complete_response_block_the_write(
    payload: dict[str, object],
    expected_error: ApiError,
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"mode": "complete", "notes": None, "data": payload},
        )

    runtime = _runtime(handler)
    proposal = RequestSpecialistAnalysisProposal(
        analysis_id="an_9901",
        justification="Os dados conflitantes exigem uma análise especializada.",
    )

    result = asyncio.run(
        _execute_and_close(
            execute_request_specialist_analysis(proposal, runtime),
            runtime,
        )
    )

    assert result == expected_error
    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/analyses/an_9901")
    ]


def test_preflight_preserves_the_api_error_and_never_posts():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            404,
            json={"code": "NOT_FOUND", "message": "Análise não encontrada."},
        )

    runtime = _runtime(handler)
    proposal = ReprocessProposal(
        analysis_id="an_9901",
        justification="O rolamento foi trocado e a análise precisa ser refeita.",
    )

    result = asyncio.run(
        _execute_and_close(
            execute_reprocess_analysis(
                proposal,
                runtime,
                idempotency_key="tractian-agent:intent-01",
            ),
            runtime,
        )
    )

    assert result == ApiError(
        category=ApiErrorCategory.API,
        code="NOT_FOUND",
        message="Análise não encontrada.",
        status_code=404,
    )
    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/analyses/an_9901")
    ]


@pytest.mark.parametrize(
    ("operation_name", "failure", "expected_error", "expected_requests"),
    [
        (
            "reprocess",
            "api",
            ApiError(
                category=ApiErrorCategory.API,
                code="FORBIDDEN",
                message="A ação foi rejeitada pela API.",
                status_code=403,
            ),
            2,
        ),
        (
            "specialist",
            "server",
            ApiError(
                category=ApiErrorCategory.SERVER,
                code="INTERNAL_ERROR",
                message="Erro interno durante o processamento.",
                status_code=500,
            ),
            2,
        ),
        (
            "criticality",
            "timeout",
            ApiError(
                category=ApiErrorCategory.TIMEOUT,
                code="READ_TIMEOUT",
                message="A API não respondeu dentro do tempo limite.",
            ),
            1,
        ),
        (
            "retraining",
            "transport",
            ApiError(
                category=ApiErrorCategory.TRANSPORT,
                code="TRANSPORT_ERROR",
                message="Não foi possível comunicar com a API.",
            ),
            1,
        ),
        (
            "escalate",
            "invalid_response",
            ApiError(
                category=ApiErrorCategory.INVALID_RESPONSE,
                code="INVALID_SCHEMA_RESPONSE",
                message="A resposta da API não corresponde ao contrato esperado.",
                status_code=200,
            ),
            1,
        ),
    ],
)
def test_write_errors_are_preserved_without_internal_retry(
    operation_name: str,
    failure: str,
    expected_error: ApiError,
    expected_requests: int,
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"mode": "complete", "notes": None, "data": _analysis_payload()},
            )
        if failure == "api":
            return httpx.Response(
                403,
                json={"code": "FORBIDDEN", "message": "A ação foi rejeitada pela API."},
            )
        if failure == "server":
            return httpx.Response(
                500,
                json={
                    "code": "INTERNAL_ERROR",
                    "message": "Erro interno durante o processamento.",
                },
            )
        if failure == "timeout":
            raise httpx.ReadTimeout("tempo esgotado", request=request)
        if failure == "transport":
            raise httpx.ConnectError("conexão recusada", request=request)
        return httpx.Response(200, json={"accepted": True, "action_id": "act_only"})

    runtime = _runtime(handler)
    result = asyncio.run(
        _execute_and_close(
            _operation_for_error_case(operation_name, runtime),
            runtime,
        )
    )

    assert result == expected_error
    assert len(requests) == expected_requests
    assert sum(request.method in {"POST", "PATCH"} for request in requests) == 1
