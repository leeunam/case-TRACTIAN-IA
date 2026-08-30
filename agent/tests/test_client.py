import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import (
    ActionReceipt,
    ApiError,
    ApiErrorCategory,
    ApiResult,
    Identity,
    ResponseMode,
)


class AssetPayload(BaseModel):
    id: str
    name: str


def test_query_returns_typed_complete_result_and_propagates_identity():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL(
            "https://industrial.test/assets/asset_G501?seed=complete"
        )
        assert request.headers["x-user-id"] == "usr_pedro"
        assert "x-company-id" not in request.headers
        assert "idempotency-key" not in request.headers
        assert request.extensions["timeout"] == {
            "connect": 2.0,
            "read": 10.0,
            "write": 5.0,
            "pool": 2.0,
        }
        return httpx.Response(
            200,
            json={
                "mode": "complete",
                "notes": None,
                "data": {"id": "asset_G501", "name": "Redutor da correia"},
            },
        )

    async def scenario():
        async with IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await client.query(
                "/assets/asset_G501",
                response_model=AssetPayload,
                identity=Identity(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                ),
                params={"seed": "complete"},
            )

    result = asyncio.run(scenario())

    assert isinstance(result, ApiResult)
    assert result.mode is ResponseMode.COMPLETE
    assert result.data == AssetPayload(
        id="asset_G501",
        name="Redutor da correia",
    )


@pytest.mark.parametrize(
    ("mode", "data"),
    [
        (ResponseMode.PARTIAL, {"id": "asset_G501"}),
        (ResponseMode.INCONCLUSIVE, {"inconclusive": True}),
        (ResponseMode.CONFLICT, {"id": "asset_G501", "conflict": True}),
        (ResponseMode.UNAVAILABLE, {}),
    ],
)
def test_query_preserves_degraded_result_without_complete_schema(mode, data):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"mode": mode.value, "notes": "Resposta degradada.", "data": data},
        )

    async def scenario():
        async with IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await client.query(
                "/assets/asset_G501",
                response_model=AssetPayload,
                identity=Identity(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                ),
            )

    result = asyncio.run(scenario())

    assert isinstance(result, ApiResult)
    assert result.mode is mode
    assert result.data == data


def test_request_json_returns_typed_action_and_propagates_protocol_headers():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.headers["x-user-id"] == "usr_pedro"
        assert request.headers["idempotency-key"] == "tractian-agent:intention-01"
        assert json.loads(request.content) == {
            "justification": "Rolamento substituído; solicitar novo processamento."
        }
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "action_id": "act_1234abcd",
                "message": "Reprocesso aceito.",
            },
        )

    async def scenario():
        async with IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await client.request_json(
                "POST",
                "/analyses/an_9906/reprocess",
                response_model=ActionReceipt,
                identity=Identity(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                ),
                body={
                    "justification": (
                        "Rolamento substituído; solicitar novo processamento."
                    )
                },
                idempotency_key="tractian-agent:intention-01",
            )

    result = asyncio.run(scenario())

    assert isinstance(result, ApiResult)
    assert result.mode is None
    assert result.data.action_id == "act_1234abcd"


def test_query_normalizes_api_rejection():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={"code": "NOT_FOUND", "message": "Ativo não encontrado."},
        )

    async def scenario():
        async with IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await client.query(
                "/assets/inexistente",
                response_model=AssetPayload,
                identity=Identity(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                ),
            )

    result = asyncio.run(scenario())

    assert result == ApiError(
        category=ApiErrorCategory.API,
        code="NOT_FOUND",
        message="Ativo não encontrado.",
        status_code=404,
    )


def test_request_json_preserves_idempotency_rejection_code():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "code": "IDEMPOTENCY_OUTCOME_UNKNOWN",
                "message": "O resultado da tentativa anterior é incerto.",
            },
        )

    async def scenario():
        async with IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await client.request_json(
                "POST",
                "/analyses/an_9906/reprocess",
                response_model=ActionReceipt,
                identity=Identity(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                ),
                body={"justification": "Justificativa suficientemente detalhada."},
                idempotency_key="tractian-agent:intention-01",
            )

    result = asyncio.run(scenario())

    assert isinstance(result, ApiError)
    assert result.category is ApiErrorCategory.API
    assert result.code == "IDEMPOTENCY_OUTCOME_UNKNOWN"
    assert result.status_code == 409


def test_query_normalizes_server_error():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "code": "INTERNAL_ERROR",
                "message": "Erro interno durante o processamento.",
            },
        )

    async def scenario():
        async with IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await client.query(
                "/assets/asset_G501",
                response_model=AssetPayload,
                identity=Identity(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                ),
            )

    result = asyncio.run(scenario())

    assert isinstance(result, ApiError)
    assert result.category is ApiErrorCategory.SERVER
    assert result.code == "INTERNAL_ERROR"
    assert result.status_code == 500


def test_request_json_normalizes_fastapi_validation_error():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "type": "missing",
                        "loc": ["body", "justification"],
                        "msg": "Field required",
                        "input": {},
                    }
                ]
            },
        )

    async def scenario():
        async with IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await client.request_json(
                "POST",
                "/analyses/an_9906/reprocess",
                response_model=ActionReceipt,
                identity=Identity(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                ),
                body={},
                idempotency_key="tractian-agent:intention-01",
            )

    result = asyncio.run(scenario())

    assert result == ApiError(
        category=ApiErrorCategory.API,
        code="VALIDATION_ERROR",
        message="Requisição inválida.",
        status_code=422,
    )


def test_query_normalizes_invalid_json_response():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"isto nao e json",
            headers={"content-type": "application/json"},
        )

    async def scenario():
        async with IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await client.query(
                "/assets/asset_G501",
                response_model=AssetPayload,
                identity=Identity(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                ),
            )

    result = asyncio.run(scenario())

    assert result == ApiError(
        category=ApiErrorCategory.INVALID_RESPONSE,
        code="INVALID_JSON_RESPONSE",
        message="A API retornou um corpo que não é JSON válido.",
        status_code=200,
    )


def test_query_normalizes_complete_payload_schema_mismatch():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mode": "complete",
                "notes": None,
                "data": {"id": "asset_G501"},
            },
        )

    async def scenario():
        async with IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await client.query(
                "/assets/asset_G501",
                response_model=AssetPayload,
                identity=Identity(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                ),
            )

    result = asyncio.run(scenario())

    assert result == ApiError(
        category=ApiErrorCategory.INVALID_RESPONSE,
        code="INVALID_SCHEMA_RESPONSE",
        message="A resposta da API não corresponde ao contrato esperado.",
        status_code=200,
    )


@pytest.mark.parametrize(
    ("exception_type", "expected_code"),
    [
        (httpx.ConnectTimeout, "CONNECT_TIMEOUT"),
        (httpx.ReadTimeout, "READ_TIMEOUT"),
        (httpx.WriteTimeout, "WRITE_TIMEOUT"),
        (httpx.PoolTimeout, "POOL_TIMEOUT"),
    ],
)
def test_query_normalizes_timeout_without_retry(exception_type, expected_code):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise exception_type("tempo esgotado", request=request)

    async def scenario():
        async with IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await client.query(
                "/assets/asset_G501",
                response_model=AssetPayload,
                identity=Identity(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                ),
            )

    result = asyncio.run(scenario())

    assert result == ApiError(
        category=ApiErrorCategory.TIMEOUT,
        code=expected_code,
        message="A API não respondeu dentro do tempo limite.",
        status_code=None,
    )
    assert calls == 1


def test_query_normalizes_transport_error_without_retry():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("conexão recusada", request=request)

    async def scenario():
        async with IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await client.query(
                "/assets/asset_G501",
                response_model=AssetPayload,
                identity=Identity(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                ),
            )

    result = asyncio.run(scenario())

    assert result == ApiError(
        category=ApiErrorCategory.TRANSPORT,
        code="TRANSPORT_ERROR",
        message="Não foi possível comunicar com a API.",
        status_code=None,
    )
    assert calls == 1


def test_query_rejects_redirect_without_following_it():
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            302,
            headers={"location": "https://untrusted.test/resource"},
        )

    async def scenario():
        async with IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await client.query(
                "/assets/asset_G501",
                response_model=AssetPayload,
                identity=Identity(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                ),
            )

    result = asyncio.run(scenario())

    assert result == ApiError(
        category=ApiErrorCategory.INVALID_RESPONSE,
        code="UNEXPECTED_STATUS",
        message="A API retornou um status HTTP inesperado.",
        status_code=302,
    )
    assert calls == 1


def test_query_normalizes_invalid_error_payload_schema():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "estrutura inesperada"})

    async def scenario():
        async with IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await client.query(
                "/assets/asset_G501",
                response_model=AssetPayload,
                identity=Identity(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                ),
            )

    result = asyncio.run(scenario())

    assert result == ApiError(
        category=ApiErrorCategory.INVALID_RESPONSE,
        code="INVALID_SCHEMA_RESPONSE",
        message="A resposta da API não corresponde ao contrato esperado.",
        status_code=500,
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"code": "", "message": "Erro interno."},
        {"code": "INTERNAL_ERROR", "message": "   "},
    ],
)
def test_query_normalizes_blank_error_fields_as_invalid_schema(payload):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json=payload)

    async def scenario():
        async with IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await client.query(
                "/assets/asset_G501",
                response_model=AssetPayload,
                identity=Identity(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                ),
            )

    result = asyncio.run(scenario())

    assert result == ApiError(
        category=ApiErrorCategory.INVALID_RESPONSE,
        code="INVALID_SCHEMA_RESPONSE",
        message="A resposta da API não corresponde ao contrato esperado.",
        status_code=500,
    )


@pytest.mark.parametrize(
    "path",
    [
        "https://untrusted.test/assets/asset_G501",
        "//untrusted.test/assets/asset_G501",
    ],
)
def test_query_rejects_external_url_before_sending_identity(path):
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    async def scenario():
        async with IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            await client.query(
                path,
                response_model=AssetPayload,
                identity=Identity(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                ),
            )

    with pytest.raises(ValueError, match="caminho relativo"):
        asyncio.run(scenario())

    assert calls == 0


@pytest.mark.parametrize(
    "idempotency_key",
    ["", "chave com espaço", "x" * 256],
)
def test_request_json_rejects_invalid_idempotency_key_before_request(
    idempotency_key,
):
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "accepted": True,
                "action_id": "act_1234abcd",
                "message": "Reprocesso aceito.",
            },
        )

    async def scenario():
        async with IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            await client.request_json(
                "POST",
                "/analyses/an_9906/reprocess",
                response_model=ActionReceipt,
                identity=Identity(
                    user_id="usr_pedro",
                    company_id="comp_mineracao_andes",
                ),
                body={"justification": "Justificativa suficientemente detalhada."},
                idempotency_key=idempotency_key,
            )

    with pytest.raises(ValidationError):
        asyncio.run(scenario())

    assert calls == 0
