from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import Literal, TypeVar

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
)

from .contracts import (
    ApiError,
    ApiErrorCategory,
    ApiResult,
    IdempotencyKey,
    Identity,
    ResponseMode,
)

PayloadT = TypeVar("PayloadT")
QueryValue = str | int | float | bool | None
HttpMethod = Literal["GET", "POST", "PATCH"]

DEFAULT_TIMEOUT = httpx.Timeout(
    connect=2.0,
    read=10.0,
    write=5.0,
    pool=2.0,
)
_IDEMPOTENCY_KEY_ADAPTER = TypeAdapter(IdempotencyKey)


class _QueryEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: ResponseMode
    notes: str | None = None
    data: JsonValue


class _ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, pattern=r"^\S+$")
    message: str = Field(min_length=1, pattern=r"\S")


def _invalid_response(
    response: httpx.Response,
    *,
    code: str,
    message: str,
) -> ApiError:
    return ApiError(
        category=ApiErrorCategory.INVALID_RESPONSE,
        code=code,
        message=message,
        status_code=response.status_code,
    )


def _decode_json(response: httpx.Response) -> object | ApiError:
    try:
        return response.json()
    except ValueError:
        return _invalid_response(
            response,
            code="INVALID_JSON_RESPONSE",
            message="A API retornou um corpo que não é JSON válido.",
        )


def _invalid_schema(response: httpx.Response) -> ApiError:
    return _invalid_response(
        response,
        code="INVALID_SCHEMA_RESPONSE",
        message="A resposta da API não corresponde ao contrato esperado.",
    )


def _unexpected_status(response: httpx.Response) -> ApiError:
    return _invalid_response(
        response,
        code="UNEXPECTED_STATUS",
        message="A API retornou um status HTTP inesperado.",
    )


def _normalize_http_error(response: httpx.Response) -> ApiError:
    payload = _decode_json(response)
    if isinstance(payload, ApiError):
        return payload
    try:
        body = _ErrorBody.model_validate(payload)
    except ValidationError:
        if isinstance(payload, dict) and isinstance(payload.get("detail"), list):
            body = _ErrorBody(
                code="VALIDATION_ERROR",
                message="Requisição inválida.",
            )
        else:
            return _invalid_schema(response)
    category = (
        ApiErrorCategory.API if response.status_code < 500 else ApiErrorCategory.SERVER
    )
    return ApiError(
        category=category,
        code=body.code,
        message=body.message,
        status_code=response.status_code,
    )


class IndustrialApiClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            follow_redirects=False,
        )

    async def __aenter__(self) -> IndustrialApiClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _send(
        self,
        method: HttpMethod,
        path: str,
        *,
        identity: Identity,
        params: Mapping[str, QueryValue] | None = None,
        json_body: object | None = None,
        idempotency_key: str | None = None,
    ) -> httpx.Response | ApiError:
        target = httpx.URL(path)
        if target.is_absolute_url or target.host or not path.startswith("/"):
            raise ValueError("Informe um caminho relativo iniciado por '/'.")
        headers = {"x-user-id": identity.user_id}
        if idempotency_key is not None:
            headers["Idempotency-Key"] = _IDEMPOTENCY_KEY_ADAPTER.validate_python(
                idempotency_key
            )
        try:
            return await self._client.request(
                method,
                path,
                params=params,
                headers=headers,
                json=json_body,
            )
        except httpx.TimeoutException as exc:
            timeout_codes = {
                httpx.ConnectTimeout: "CONNECT_TIMEOUT",
                httpx.ReadTimeout: "READ_TIMEOUT",
                httpx.WriteTimeout: "WRITE_TIMEOUT",
                httpx.PoolTimeout: "POOL_TIMEOUT",
            }
            code = next(
                (
                    value
                    for exception_type, value in timeout_codes.items()
                    if isinstance(exc, exception_type)
                ),
                "TIMEOUT",
            )
            return ApiError(
                category=ApiErrorCategory.TIMEOUT,
                code=code,
                message="A API não respondeu dentro do tempo limite.",
            )
        except httpx.RequestError:
            return ApiError(
                category=ApiErrorCategory.TRANSPORT,
                code="TRANSPORT_ERROR",
                message="Não foi possível comunicar com a API.",
            )

    async def query(
        self,
        path: str,
        *,
        response_model: type[PayloadT],
        identity: Identity,
        params: Mapping[str, QueryValue] | None = None,
    ) -> ApiResult[PayloadT] | ApiResult[JsonValue] | ApiError:
        response = await self._send(
            "GET",
            path,
            identity=identity,
            params=params,
        )
        if isinstance(response, ApiError):
            return response
        if response.is_error:
            return _normalize_http_error(response)
        if not response.is_success:
            return _unexpected_status(response)
        payload = _decode_json(response)
        if isinstance(payload, ApiError):
            return payload
        try:
            envelope = _QueryEnvelope.model_validate(payload)
        except ValidationError:
            return _invalid_schema(response)
        if envelope.mode is ResponseMode.COMPLETE:
            try:
                data = TypeAdapter(response_model).validate_python(envelope.data)
            except ValidationError:
                return _invalid_schema(response)
            return ApiResult[PayloadT](
                status_code=response.status_code,
                data=data,
                mode=envelope.mode,
                notes=envelope.notes,
            )
        return ApiResult[JsonValue](
            status_code=response.status_code,
            data=envelope.data,
            mode=envelope.mode,
            notes=envelope.notes,
        )

    async def request_json(
        self,
        method: HttpMethod,
        path: str,
        *,
        response_model: type[PayloadT],
        identity: Identity,
        params: Mapping[str, QueryValue] | None = None,
        body: BaseModel | Mapping[str, JsonValue] | None = None,
        idempotency_key: str | None = None,
    ) -> ApiResult[PayloadT] | ApiError:
        json_body = (
            body.model_dump(mode="json") if isinstance(body, BaseModel) else body
        )
        response = await self._send(
            method,
            path,
            identity=identity,
            params=params,
            json_body=json_body,
            idempotency_key=idempotency_key,
        )
        if isinstance(response, ApiError):
            return response
        if response.is_error:
            return _normalize_http_error(response)
        if not response.is_success:
            return _unexpected_status(response)
        payload = _decode_json(response)
        if isinstance(payload, ApiError):
            return payload
        try:
            data = TypeAdapter(response_model).validate_python(payload)
        except ValidationError:
            return _invalid_schema(response)
        return ApiResult[PayloadT](
            status_code=response.status_code,
            data=data,
        )
