from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
import asyncio
import json
import os
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from tractian_demo.contracts import (
    CaseDetail,
    CreateCaseRequest,
    DemoCase,
    DemoConfig,
    EnqueueMessageRequest,
    EnqueueMessageResponse,
    Persona,
    DecisionRequest,
    ResolveDecisionRequest,
    RetryNotificationRequest,
    OutboxEvent,
)
from tractian_demo.repository import DemoRepository
from tractian_demo.settings import DemoSettings


PersonaLoader = Callable[[], Awaitable[tuple[Persona, ...]]]


def _load_cases(settings: DemoSettings) -> list[dict[str, object]]:
    value = json.loads(settings.public_cases_path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("public case file must contain a list")
    return value


def _detail(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


async def _default_personas(settings: DemoSettings) -> tuple[Persona, ...]:
    public = _load_cases(settings)
    requester_ids = sorted({str(item["user_id"]) for item in public})
    values: list[Persona] = []
    async with httpx.AsyncClient(
        base_url=settings.industrial_api_url, timeout=5
    ) as client:
        for user_id in requester_ids:
            response = await client.get("/users/me", headers={"x-user-id": user_id})
            response.raise_for_status()
            item = response.json()
            permissions = frozenset(str(value) for value in item.get("permissions", ()))
            values.append(
                Persona(
                    id=item["id"],
                    name=item["name"],
                    profile="authority"
                    if "action_high" in permissions
                    else "requester",
                    company_id=item["company_id"],
                    permissions=permissions,
                )
            )
    values.append(
        Persona(
            id="tractian_reviewer",
            name="Equipe TRACTIAN",
            profile="tractian",
            company_id=None,
            permissions=frozenset(
                {"technical_review", "specialist", "retraining", "escalate"}
            ),
        )
    )
    return tuple(values)


def create_app(
    settings: DemoSettings | None = None,
    *,
    persona_loader: PersonaLoader | None = None,
) -> FastAPI:
    active_settings = settings or DemoSettings.from_env(os.environ)
    repository = DemoRepository(active_settings.database_path)
    loader = persona_loader or (lambda: _default_personas(active_settings))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        repository.open(public_cases=_load_cases(active_settings))
        try:
            yield
        finally:
            repository.close()

    app = FastAPI(title="TRACTIAN Demo", version="0.1.0", lifespan=lifespan)
    app.state.repository = repository
    app.state.settings = active_settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(active_settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )

    @app.get("/v1/demo/config", response_model=DemoConfig)
    async def config() -> DemoConfig:
        return DemoConfig(
            primary_provider=active_settings.primary_provider,
            fallback_provider=active_settings.fallback_provider,
            slack_configured=bool(
                active_settings.slack_tractian_channel
                and active_settings.slack_authority_channel
                and active_settings.slack_access_token_configured
            ),
        )

    @app.get("/v1/personas", response_model=tuple[Persona, ...])
    async def personas() -> tuple[Persona, ...]:
        try:
            return await loader()
        except (httpx.HTTPError, ValueError, KeyError) as error:
            raise HTTPException(
                503,
                _detail(
                    "PERSONA_DIRECTORY_UNAVAILABLE",
                    "Não foi possível carregar as personas.",
                ),
            ) from error

    @app.get("/v1/cases", response_model=tuple[DemoCase, ...])
    async def cases() -> tuple[DemoCase, ...]:
        return repository.list_cases()

    @app.get("/v1/cases/{case_id}", response_model=CaseDetail)
    async def case_detail(case_id: str) -> CaseDetail:
        try:
            case = repository.get_case(case_id)
        except KeyError as error:
            raise HTTPException(
                404, _detail("CASE_NOT_FOUND", "Caso não encontrado.")
            ) from error
        return CaseDetail(
            case=case,
            messages=repository.list_messages(case_id),
            executions=repository.list_executions(case_id),
        )

    @app.post("/v1/cases", response_model=DemoCase, status_code=status.HTTP_201_CREATED)
    async def create_case(body: CreateCaseRequest) -> DemoCase:
        try:
            if body.requester_id is not None:
                known = {item.id: item for item in await loader()}
                persona = known.get(body.requester_id)
                if (
                    persona is None
                    or persona.profile == "tractian"
                    or persona.company_id != body.company_id
                ):
                    raise HTTPException(
                        403,
                        _detail(
                            "CUSTOM_CASE_SCOPE_MISMATCH",
                            "Pessoa e empresa não formam um escopo válido.",
                        ),
                    )
            return repository.create_case(body)
        except KeyError as error:
            raise HTTPException(
                404, _detail("PUBLIC_CASE_NOT_FOUND", "Caso público não encontrado.")
            ) from error

    @app.post(
        "/v1/cases/{case_id}/messages",
        response_model=EnqueueMessageResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def enqueue(
        case_id: str, body: EnqueueMessageRequest
    ) -> EnqueueMessageResponse:
        try:
            case = repository.get_case(case_id)
            if case.immutable:
                raise HTTPException(
                    409,
                    _detail(
                        "IMMUTABLE_PUBLIC_CASE",
                        "Duplique o caso público antes de conversar.",
                    ),
                )
            known_personas = {item.id: item for item in await loader()}
            persona = known_personas.get(body.persona_id)
            if persona is None or persona.id != case.requester_id:
                raise HTTPException(
                    403,
                    _detail(
                        "PERSONA_OUT_OF_SCOPE",
                        "A persona não pode enviar mensagens neste caso.",
                    ),
                )
            message, execution = repository.enqueue_message(
                case_id=case_id,
                persona_id=body.persona_id,
                content=body.content,
                idempotency_key=body.idempotency_key,
            )
            return EnqueueMessageResponse(message=message, execution=execution)
        except KeyError as error:
            raise HTTPException(
                404, _detail("CASE_NOT_FOUND", "Caso não encontrado.")
            ) from error
        except ValueError as error:
            raise HTTPException(
                409, _detail(str(error), "A chave já foi usada com outro conteúdo.")
            ) from error

    @app.get("/v1/cases/{case_id}/events")
    async def events(
        case_id: str,
        request: Request,
        last_event_id: int | None = Header(default=None, alias="Last-Event-ID"),
        after: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        try:
            repository.get_case(case_id)
        except KeyError as error:
            raise HTTPException(
                404, _detail("CASE_NOT_FOUND", "Caso não encontrado.")
            ) from error
        cursor = last_event_id if last_event_id is not None else after

        async def stream() -> AsyncIterator[str]:
            current = cursor
            while not await request.is_disconnected():
                pending = repository.list_events(case_id, after_id=current)
                if not pending:
                    yield ": keepalive\n\n"
                    await asyncio.sleep(1)
                    continue
                for event in pending:
                    current = event.id
                    payload = json.dumps(
                        event.model_dump(mode="json"), ensure_ascii=False
                    )
                    yield f"id: {event.id}\nevent: {event.kind}\ndata: {payload}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    async def resolved_persona(persona_id: str) -> Persona:
        try:
            persona = next(
                (item for item in await loader() if item.id == persona_id), None
            )
        except (httpx.HTTPError, ValueError, KeyError) as error:
            raise HTTPException(
                503,
                _detail(
                    "PERSONA_DIRECTORY_UNAVAILABLE",
                    "Não foi possível validar a persona.",
                ),
            ) from error
        if persona is None:
            raise HTTPException(
                403, _detail("PERSONA_UNKNOWN", "Persona desconhecida.")
            )
        return persona

    @app.get("/v1/decisions", response_model=tuple[DecisionRequest, ...])
    async def decisions(
        persona_id: str = Query(min_length=1),
    ) -> tuple[DecisionRequest, ...]:
        return repository.list_decisions(await resolved_persona(persona_id))

    @app.post("/v1/decisions/{decision_id}/resolve", response_model=DecisionRequest)
    async def resolve_decision(
        decision_id: str, body: ResolveDecisionRequest
    ) -> DecisionRequest:
        try:
            return repository.resolve_decision(
                decision_id,
                persona=await resolved_persona(body.persona_id),
                resolution=body.resolution,
            )
        except KeyError as error:
            raise HTTPException(
                404, _detail("DECISION_NOT_FOUND", "Decisão não encontrada.")
            ) from error
        except PermissionError as error:
            raise HTTPException(
                403,
                _detail(
                    "DECISION_FORBIDDEN", "Esta persona não pode resolver a decisão."
                ),
            ) from error
        except ValueError as error:
            raise HTTPException(
                409, _detail(str(error), "A decisão não está mais disponível.")
            ) from error

    @app.post("/v1/notifications/{notification_id}/retry", response_model=OutboxEvent)
    async def retry_notification(
        notification_id: str, body: RetryNotificationRequest
    ) -> OutboxEvent:
        persona = await resolved_persona(body.persona_id)
        if persona.profile not in {"tractian", "authority"}:
            raise HTTPException(
                403,
                _detail(
                    "NOTIFICATION_RETRY_FORBIDDEN",
                    "A persona não pode reenviar notificações.",
                ),
            )
        try:
            notification = repository.get_outbox(notification_id)
            decision = repository.get_decision(notification.decision_id)
            if not repository.list_decisions(persona) and not (
                persona.profile == "tractian" and decision.audience == "tractian"
            ):
                raise PermissionError
            return repository.retry_outbox(notification_id)
        except KeyError as error:
            raise HTTPException(
                404, _detail("NOTIFICATION_NOT_FOUND", "Notificação não encontrada.")
            ) from error
        except PermissionError as error:
            raise HTTPException(
                403,
                _detail(
                    "NOTIFICATION_RETRY_FORBIDDEN",
                    "A persona não pode reenviar esta notificação.",
                ),
            ) from error
        except ValueError as error:
            raise HTTPException(
                409, _detail(str(error), "A notificação não admite reenvio.")
            ) from error

    return app


app = create_app()
