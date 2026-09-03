"""Operações HTTP fixas, isoladas das proposal tools e do grafo."""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from tractian_agent.contracts import (
    ActionReceipt,
    ApiError,
    ApiErrorCategory,
    IdempotencyKey,
    StrictModel,
)
from tractian_agent.observability import (
    ActionSpanAttributes,
    ErrorCode,
    Outcome,
    SpanName,
    current_action_attempt,
    current_execution_trace,
    span_fail_open,
)
from tractian_agent.tools.analyses import (
    AnalysisScopeValidationError,
    execute_get_analysis,
)
from tractian_agent.tools.runtime import WriteToolRuntime
from tractian_agent.write_policy import (
    AssetCriticality,
    EscalateCaseProposal,
    ReprocessProposal,
    RequestModelRetrainingProposal,
    RequestSpecialistAnalysisProposal,
    UpdateAssetCriticalityProposal,
)

__all__ = [
    "execute_reprocess_analysis",
    "execute_request_specialist_analysis",
    "execute_update_asset_criticality",
    "execute_request_model_retraining",
    "execute_escalate_case",
]

_IDEMPOTENCY_KEY_ADAPTER = TypeAdapter(IdempotencyKey)


class _JustificationBody(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    justification: str


class _AssetCriticalityChanges(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    criticality: AssetCriticality


class _UpdateAssetCriticalityBody(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    justification: str
    changes: _AssetCriticalityChanges


def _unconfirmed_analysis_scope() -> ApiError:
    return ApiError(
        category=ApiErrorCategory.INVALID_RESPONSE,
        code="ANALYSIS_SCOPE_UNCONFIRMED",
        message="Não foi possível confirmar que a análise pertence ao ativo central.",
    )


async def _preflight_analysis_scope(
    analysis_id: str,
    runtime: WriteToolRuntime,
) -> ApiError | None:
    if "read" not in runtime.permissions:
        return _unconfirmed_analysis_scope()
    try:
        result = await execute_get_analysis(analysis_id, runtime)
    except AnalysisScopeValidationError:
        return _unconfirmed_analysis_scope()
    if result.error is not None:
        return result.error
    if result.content is None:
        return _unconfirmed_analysis_scope()
    return None


async def _request_action(
    method: Literal["POST", "PATCH"],
    path: str,
    *,
    body: BaseModel,
    runtime: WriteToolRuntime,
    idempotency_key: IdempotencyKey | None = None,
) -> ActionReceipt | ApiError:
    trace = current_execution_trace()
    attempt = current_action_attempt()

    async def dispatch() -> ActionReceipt | ApiError:
        result = await runtime.client.request_json(
            method,
            path,
            response_model=ActionReceipt,
            identity=runtime.identity,
            body=body,
            idempotency_key=idempotency_key,
        )
        if isinstance(result, ApiError):
            return result
        if isinstance(result.data, ActionReceipt):
            return result.data
        return ApiError(
            category=ApiErrorCategory.INVALID_RESPONSE,
            code="INVALID_SCHEMA_RESPONSE",
            message="A resposta da API não corresponde ao contrato esperado.",
            status_code=result.status_code,
        )

    if trace is None or attempt is None:
        return await dispatch()
    action, ordinal = attempt
    with span_fail_open(
        trace,
        SpanName.ACTION,
        ActionSpanAttributes(action=action, attempt=ordinal),
    ) as action_span:
        try:
            outcome = await dispatch()
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError):
                action_span.finish(Outcome.CANCELLED, ErrorCode.CANCELLED)
            else:
                action_span.finish(Outcome.ERROR, ErrorCode.ACTION)
            raise
        if isinstance(outcome, ApiError):
            action_span.finish(Outcome.ERROR, ErrorCode.ACTION)
        elif not outcome.accepted:
            action_span.finish(Outcome.DENIED, ErrorCode.ACTION)
        return outcome


async def execute_reprocess_analysis(
    proposal: ReprocessProposal,
    runtime: WriteToolRuntime,
    *,
    idempotency_key: IdempotencyKey,
) -> ActionReceipt | ApiError:
    try:
        validated_key = _IDEMPOTENCY_KEY_ADAPTER.validate_python(idempotency_key)
    except ValidationError:
        return ApiError(
            category=ApiErrorCategory.API,
            code="INVALID_IDEMPOTENCY_KEY",
            message="A chave idempotente persistida é inválida.",
        )
    preflight_error = await _preflight_analysis_scope(proposal.analysis_id, runtime)
    if preflight_error is not None:
        return preflight_error
    return await _request_action(
        "POST",
        f"/analyses/{proposal.analysis_id}/reprocess",
        body=_JustificationBody(justification=proposal.justification),
        runtime=runtime,
        idempotency_key=validated_key,
    )


async def execute_request_specialist_analysis(
    proposal: RequestSpecialistAnalysisProposal,
    runtime: WriteToolRuntime,
) -> ActionReceipt | ApiError:
    preflight_error = await _preflight_analysis_scope(proposal.analysis_id, runtime)
    if preflight_error is not None:
        return preflight_error
    return await _request_action(
        "POST",
        f"/analyses/{proposal.analysis_id}/request-specialist",
        body=_JustificationBody(justification=proposal.justification),
        runtime=runtime,
    )


async def execute_update_asset_criticality(
    proposal: UpdateAssetCriticalityProposal,
    runtime: WriteToolRuntime,
) -> ActionReceipt | ApiError:
    return await _request_action(
        "PATCH",
        f"/assets/{runtime.central_asset_id}",
        body=_UpdateAssetCriticalityBody(
            justification=proposal.justification,
            changes=_AssetCriticalityChanges(criticality=proposal.criticality),
        ),
        runtime=runtime,
    )


async def execute_request_model_retraining(
    proposal: RequestModelRetrainingProposal,
    runtime: WriteToolRuntime,
) -> ActionReceipt | ApiError:
    return await _request_action(
        "POST",
        f"/models/{runtime.configured_model_id}/request-retraining",
        body=_JustificationBody(justification=proposal.justification),
        runtime=runtime,
    )


async def execute_escalate_case(
    proposal: EscalateCaseProposal,
    runtime: WriteToolRuntime,
) -> ActionReceipt | ApiError:
    return await _request_action(
        "POST",
        f"/cases/{runtime.current_case_id}/escalate",
        body=_JustificationBody(justification=proposal.justification),
        runtime=runtime,
    )
