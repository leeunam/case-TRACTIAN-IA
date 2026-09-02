import asyncio
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

from langgraph.checkpoint.base import LATEST_VERSION
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
import httpx
import pytest
from pydantic import PrivateAttr, ValidationError

from tractian_agent.checkpoint import open_checkpointer
from tractian_agent.client import IndustrialApiClient
from tractian_agent.contracts import ApiError, ApiErrorCategory, Identity, ResponseMode, SupportRequest
from tractian_agent.evidence import compile_observations
from tractian_agent.planner import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT_VERSION,
    Planner,
    PlannerDecisionKind,
    PlannerDecisionTurn,
    PlannerErrorCode,
    PlannerLimits,
    PlannerProtocolError,
    PlannerUsage,
    PlannerStopReason,
    PlannerTerminalDecision,
    PlannerToolTurn,
    select_planner_tools,
)
from tractian_agent.state import (
    AgentState,
    JsonSnapshot,
    PersistedToolCall,
    ThreadScope,
    ToolObservation,
)
from tractian_agent.tools.analyses import (
    AnalysisArtifact,
    AnalysisDetailToolArtifact,
    AnalysisDetailToolOutcome,
    AnalysisListToolArtifact,
    AnalysisListToolOutcome,
    DegradedAnalysisListModelContent,
    execute_get_analysis,
    execute_list_asset_analyses,
    get_analysis,
    list_asset_analyses,
)
from tractian_agent.tools.assets import (
    AssetArtifact,
    AssetHierarchy,
    AssetModelContent,
    AssetToolArtifact,
    AssetToolOutcome,
    BearingSpecifications,
    TechnicalConfiguration,
    execute_get_asset,
    get_asset,
)
from tractian_agent.tools.knowledge import (
    DegradedKnowledgeDocumentContent,
    DegradedKnowledgeSearchModelContent,
    DegradedModelContent,
    KnowledgeDocumentToolArtifact,
    KnowledgeDocumentToolOutcome,
    KnowledgeDocumentContent,
    KnowledgeSearchModelContent,
    KnowledgeSearchItem,
    KnowledgeSearchToolArtifact,
    KnowledgeSearchToolOutcome,
    ModelArtifact,
    ModelToolArtifact,
    ModelToolOutcome,
    execute_get_knowledge_document,
    get_knowledge_document,
    get_model,
    search_knowledge,
)
from tractian_agent.tools.observations import ToolArtifact, ToolOutcome, ToolSource
from tractian_agent.tools import READ_TOOLS, WRITE_PROPOSAL_TOOLS
from tractian_agent.tools.technical import (
    BaselineArtifact,
    BaselineToolArtifact,
    BaselineToolOutcome,
    DataQualityArtifact,
    DataQualityToolArtifact,
    DataQualityToolOutcome,
    RmsToolArtifact,
    SpectrumToolArtifact,
    execute_get_baseline,
    execute_get_data_quality,
    execute_get_rms_series,
    execute_get_spectrum,
    get_baseline,
    get_data_quality,
    get_rms_series,
    get_spectrum,
)
from tractian_agent.tools.runtime import (
    ReadToolRuntime,
    TrustedIdentity,
    WriteToolRuntime,
)


class _RecordingPlannerModel(BaseChatModel):
    selector_response: AIMessage
    terminal_response: object | None = None
    _events: list[str] = PrivateAttr(default_factory=list)
    _bound_tool_names: tuple[str, ...] = PrivateAttr(default=())
    _selection_messages: list[BaseMessage] = PrivateAttr(default_factory=list)
    _terminal_messages: list[BaseMessage] = PrivateAttr(default_factory=list)
    _structured_schema: type | dict[str, Any] | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "recording-planner-model"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise AssertionError("o planner deve usar somente os wrappers públicos")

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> RunnableLambda:
        self._events.append("bind_tools")
        self._bound_tool_names = tuple(tool.name for tool in tools)

        async def select(messages: list[BaseMessage]) -> AIMessage:
            self._events.append("selection_request")
            self._selection_messages = list(messages)
            return self.selector_response

        return RunnableLambda(select)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> RunnableLambda:
        self._events.append("with_structured_output")
        self._structured_schema = schema

        async def finalize(messages: list[BaseMessage]) -> object:
            self._events.append("terminal_request")
            self._terminal_messages = list(messages)
            return self.terminal_response

        return RunnableLambda(finalize)


def _request() -> SupportRequest:
    return SupportRequest(
        case_id="case_tkt_inv_04",
        ticket_id="TKT-INV-04",
        asset_id="asset_G501",
        message="Consulte o cadastro técnico deste ativo.",
        identity=Identity(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
        ),
    )


def _state(
    *,
    permissions: frozenset[str] = frozenset({"read"}),
    request: SupportRequest | None = None,
    tool_calls: tuple[PersistedToolCall, ...] = (),
    tool_observations: tuple[ToolObservation, ...] = (),
    with_trusted_target_artifacts: bool = False,
) -> AgentState:
    request = _request() if request is None else request
    if with_trusted_target_artifacts:
        tool_calls, tool_observations = _history_with_trusted_target_artifacts(
            tool_calls,
            tool_observations,
        )
    return AgentState(
        request=request,
        identity=TrustedIdentity(
            user_id=request.identity.user_id,
            company_id=request.identity.company_id,
        ),
        permissions=permissions,
        request_id="req_planner_01",
        thread_id="thread_planner_01",
        execution_id="exec_planner_01",
        thread_scope=ThreadScope(
            thread_id="thread_planner_01",
            case_id=request.case_id,
            company_id=request.identity.company_id,
            user_id=request.identity.user_id,
        ),
        tool_calls=tool_calls,
        tool_observations=tool_observations,
        ledger=compile_observations(
            tuple(
                observation
                for observation in tool_observations
                if observation.request_id == "req_planner_01"
                and observation.artifact.validated_read_artifact() is not None
            ),
            recorded_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
        ),
        step_limit=3,
    )


def _read_runtime(
    *,
    permissions: frozenset[str] = frozenset({"read"}),
    user_id: str = "usr_pedro",
    company_id: str = "comp_mineracao_andes",
    central_asset_id: str = "asset_G501",
) -> ReadToolRuntime:
    return ReadToolRuntime.create(
        user_id=user_id,
        company_id=company_id,
        permissions=permissions,
        central_asset_id=central_asset_id,
        client=IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(
                lambda request: pytest.fail(f"HTTP inesperado: {request.url}")
            ),
        ),
    )


def _write_runtime(
    *,
    permissions: frozenset[str] = frozenset(
        {"read", "action_low", "action_high", "escalate"}
    ),
    current_case_id: str = "case_tkt_inv_04",
    central_asset_id: str = "asset_G501",
) -> WriteToolRuntime:
    return WriteToolRuntime.create(
        user_id="usr_pedro",
        company_id="comp_mineracao_andes",
        permissions=permissions,
        central_asset_id=central_asset_id,
        current_case_id=current_case_id,
        configured_model_id="mdl_vib_v3",
        client=IndustrialApiClient(
            "https://industrial.test",
            transport=httpx.MockTransport(
                lambda request: pytest.fail(f"HTTP inesperado: {request.url}")
            ),
        ),
    )


def _planner_history(
    count: int,
    *,
    request_id: str = "req_planner_01",
) -> tuple[tuple[PersistedToolCall, ...], tuple[ToolObservation, ...]]:
    calls: list[PersistedToolCall] = []
    observations: list[ToolObservation] = []
    for index in range(count):
        if request_id == "req_planner_01":
            tool_name = "search_knowledge"
            arguments = {
                "query": f"historico autorizado {index}",
                "document_type": None,
            }
            content = KnowledgeSearchModelContent(
                results=[],
                total_results=0,
                returned_results=0,
                omitted_results=0,
                truncated=False,
            ).model_dump(mode="json")
            resource = "/knowledge/search"
        else:
            asset_id = f"asset_G{500 + index}"
            tool_name = "get_asset"
            arguments = {"asset_id": asset_id}
            content = {"id": asset_id, "mode": "complete"}
            resource = f"/assets/{asset_id}"
        call = PersistedToolCall(
            request_id=request_id,
            call_id=f"call_{request_id}_{index}",
            name=tool_name,
            arguments=arguments,
        )
        calls.append(call)
        observations.append(
            ToolObservation(
                request_id=request_id,
                call_id=call.call_id,
                content=content,
                artifact=(
                    KnowledgeSearchToolArtifact(
                        tool_name=call.name,
                        arguments=arguments,
                        source=ToolSource(
                            kind="industrial_api",
                            resource=resource,
                        ),
                        outcome=KnowledgeSearchToolOutcome(
                            mode=ResponseMode.COMPLETE,
                            partial_data={},
                            results=[],
                            total_results=0,
                            returned_results=0,
                            omitted_results=0,
                        ),
                    )
                    if request_id == "req_planner_01"
                    else ToolArtifact(
                        tool_name=call.name,
                        arguments=arguments,
                        source=ToolSource(
                            kind="industrial_api",
                            resource=resource,
                        ),
                        outcome=ToolOutcome(partial_data=content),
                    )
                ),
            )
        )
    return tuple(calls), tuple(observations)


def _analysis_list_observation(
    call: PersistedToolCall,
    *,
    analysis_ids: tuple[str, ...] = (),
    notes: str | None = None,
    partial_data: object = None,
) -> ToolObservation:
    rows = [
        {"id": analysis_id, "asset_id": "asset_G501"}
        for analysis_id in analysis_ids
    ]
    flags = {} if partial_data is None else partial_data
    content = (
        DegradedAnalysisListModelContent(
            mode=ResponseMode.PARTIAL,
            notes=notes,
            analyses=rows,
            total_analyses=len(rows),
            returned_analyses=len(rows),
            omitted_analyses=0,
            truncated=False,
            partial_data=flags,
        ).model_dump(mode="json")
        if analysis_ids
        else {
            "mode": ResponseMode.PARTIAL.value,
            "notes": notes,
            "partial_data": flags,
        }
    )
    return ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content=content,
        artifact=AnalysisListToolArtifact(
            tool_name=call.name,
            arguments=call.arguments.to_python(),
            source=ToolSource(
                kind="industrial_api",
                resource="/assets/asset_G501/analyses",
            ),
            outcome=AnalysisListToolOutcome(
                mode=ResponseMode.PARTIAL,
                notes=notes,
                partial_data=flags,
                analyses=rows if analysis_ids else None,
                total_analyses=len(rows) if analysis_ids else None,
                returned_analyses=len(rows) if analysis_ids else None,
                omitted_analyses=0 if analysis_ids else None,
            ),
        ),
    )


def _successful_search_observation(call: PersistedToolCall) -> ToolObservation:
    return ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content=KnowledgeSearchModelContent(
            results=[],
            total_results=0,
            returned_results=0,
            omitted_results=0,
            truncated=False,
        ).model_dump(mode="json"),
        artifact=KnowledgeSearchToolArtifact(
            tool_name=call.name,
            arguments=call.arguments.to_python(),
            source=ToolSource(kind="industrial_api", resource="/knowledge/search"),
            outcome=KnowledgeSearchToolOutcome(
                mode=ResponseMode.COMPLETE,
                partial_data={},
                results=[],
                total_results=0,
                returned_results=0,
                omitted_results=0,
            ),
        ),
    )


def _real_technical_observation(
    tool_name: str,
    total_items: int,
) -> tuple[PersistedToolCall, ToolObservation]:
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        if tool_name == "get_rms_series":
            data = {
                "asset_id": "asset_G501",
                "point_id": "pt_G501_de",
                "unit": "mm/s",
                "baseline_reference": 2.5,
                "baseline_state": "established",
                "alarm_threshold": 2.9,
                "samples": [
                    {
                        "ts": (origin + timedelta(minutes=index)).isoformat(),
                        "value": float(index),
                    }
                    for index in range(total_items)
                ],
            }
        else:
            data = {
                "asset_id": "asset_G501",
                "point_id": "pt_G501_de",
                "peaks": [
                    {
                        "freq_hz": float(index + 1),
                        "amplitude_mm_s": float(index + 1) / 10,
                        "note": None,
                    }
                    for index in range(total_items)
                ],
                "bands_missing": [],
                "collected_at": "2026-01-03T00:00:00+00:00",
            }
        return httpx.Response(
            200,
            json={"mode": "complete", "notes": None, "data": data},
        )

    async def invoke():
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=IndustrialApiClient(
                "https://industrial.test",
                transport=httpx.MockTransport(handler),
            ),
        )
        try:
            if tool_name == "get_rms_series":
                return await execute_get_rms_series(
                    "asset_G501",
                    "pt_G501_de",
                    runtime,
                )
            return await execute_get_spectrum(
                "asset_G501",
                "pt_G501_de",
                runtime,
            )
        finally:
            await runtime.client.aclose()

    result = asyncio.run(invoke())
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id=f"call_{tool_name}_{total_items}",
        name=tool_name,
        arguments={"asset_id": "asset_G501", "point_id": "pt_G501_de"},
    )
    return call, ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content=result.content.model_dump(mode="json"),
        artifact=result.artifact,
    )


def _real_complete_observation(
    tool_name: str,
) -> tuple[PersistedToolCall, ToolObservation, str]:
    analysis = {
        "id": "an_valid_output",
        "asset_id": "asset_G501",
        "point_id": "pt_G501_de",
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
        "limitations": [],
        "model_version": "3.2.1",
        "created_at": "2026-01-02T03:04:05+00:00",
        "status": "current",
    }
    payloads = {
        "get_asset": {
            "id": "asset_G501",
            "name": "Motor principal",
            "company_id": "comp_mineracao_andes",
            "criticality": "critical",
            "plant": "Planta 1",
            "line": "Britagem",
            "parent_asset_id": None,
            "machine_type": "motor_induction",
            "rotation_rpm": 1780.0,
            "bearing_pn": "NU 310",
            "bpfo_hz": -1.0,
            "bpfi_hz": 218.1,
            "bsf_hz": 58.7,
            "ftf_hz": 11.9,
            "line_frequency_hz": 60.0,
            "sensor_status": "online",
            "points": [
                {
                    "id": "pt_G501_de",
                    "asset_id": "asset_G501",
                    "location": "DE",
                    "sensor_status": "online",
                }
            ],
        },
        "list_asset_analyses": {"analyses": [analysis]},
        "get_analysis": analysis,
        "get_baseline": {
            "id": "bs_G501_de",
            "asset_id": "asset_G501",
            "point_id": "pt_G501_de",
            "state": "established",
            "detection_mode": "baseline",
            "learnable": True,
            "established_at": "2026-01-01T00:00:00+00:00",
            "invalidated_at": None,
            "invalidation_reason": "",
            "features": [
                {
                    "feature": "rms_mm_s",
                    "reference": 2.5,
                    "tolerance": 0.4,
                }
            ],
        },
        "get_data_quality": {
            "asset_id": "asset_G501",
            "point_id": "pt_G501_de",
            "completeness": 0.82,
            "freshness_minutes": 12,
            "snr_db": -3.5,
            "staleness_flag": False,
        },
        "get_knowledge_document": {
            "id": "kb_valid_output",
            "type": "guidance",
            "title": "Orientação de rolamento",
            "body": " ",
            "tags": [],
        },
    }

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mode": "complete",
                "notes": None,
                "data": payloads[tool_name],
            },
        )

    async def invoke():
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=IndustrialApiClient(
                "https://industrial.test",
                transport=httpx.MockTransport(handler),
            ),
        )
        try:
            if tool_name == "get_asset":
                return await execute_get_asset("asset_G501", runtime)
            if tool_name == "list_asset_analyses":
                return await execute_list_asset_analyses(
                    "asset_G501", None, runtime
                )
            if tool_name == "get_analysis":
                return await execute_get_analysis("an_valid_output", runtime)
            if tool_name == "get_baseline":
                return await execute_get_baseline(
                    "asset_G501", "pt_G501_de", runtime
                )
            if tool_name == "get_data_quality":
                return await execute_get_data_quality(
                    "asset_G501", "pt_G501_de", runtime
                )
            return await execute_get_knowledge_document(
                "kb_valid_output", runtime
            )
        finally:
            await runtime.client.aclose()

    result = asyncio.run(invoke())
    arguments = {
        "get_asset": {"asset_id": "asset_G501"},
        "list_asset_analyses": {"asset_id": "asset_G501"},
        "get_analysis": {"analysis_id": "an_valid_output"},
        "get_baseline": {
            "asset_id": "asset_G501",
            "point_id": "pt_G501_de",
        },
        "get_data_quality": {
            "asset_id": "asset_G501",
            "point_id": "pt_G501_de",
        },
        "get_knowledge_document": {"document_id": "kb_valid_output"},
    }[tool_name]
    messages = {
        "get_analysis": "Detalhe a análise an_valid_output.",
        "get_knowledge_document": "Abra o documento kb_valid_output.",
        "get_baseline": "Consulte o ponto pt_G501_de deste ativo.",
        "get_data_quality": "Consulte o ponto pt_G501_de deste ativo.",
    }
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id=f"call_real_{tool_name}",
        name=tool_name,
        arguments=arguments,
    )
    observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content=result.content.model_dump(mode="json"),
        artifact=result.artifact,
    )
    return call, observation, messages.get(
        tool_name,
        "Consulte os dados completos deste ativo.",
    )


def _history_with_trusted_target_artifacts(
    calls: tuple[PersistedToolCall, ...],
    observations: tuple[ToolObservation, ...],
) -> tuple[tuple[PersistedToolCall, ...], tuple[ToolObservation, ...]]:
    analysis_ids: set[str] = set()
    document_ids: set[str] = set()
    model_ids: set[str] = set()
    point_ids: set[str] = set()
    for call in calls:
        arguments = call.arguments.to_python()
        if isinstance(arguments.get("analysis_id"), str):
            analysis_ids.add(arguments["analysis_id"])
        if isinstance(arguments.get("document_id"), str):
            document_ids.add(arguments["document_id"])
        if isinstance(arguments.get("model_id"), str):
            model_ids.add(arguments["model_id"])
        if isinstance(arguments.get("point_id"), str):
            point_ids.add(arguments["point_id"])

    authority_calls: list[PersistedToolCall] = []
    authority_observations: list[ToolObservation] = []
    if analysis_ids:
        analysis_call = PersistedToolCall(
            request_id="req_planner_01",
            call_id="call_authority_analysis_list",
            name="list_asset_analyses",
            arguments={"asset_id": "asset_G501"},
        )
        authority_calls.append(analysis_call)
        authority_observations.append(
            _analysis_list_observation(
                analysis_call,
                analysis_ids=tuple(sorted(analysis_ids)),
            )
        )
    if point_ids:
        if point_ids != {"pt_G501_de"}:
            raise AssertionError(f"fixture sem artifact de pontos para {point_ids!r}")
        point_call, point_observation, _ = _real_complete_observation("get_asset")
        authority_calls.append(point_call)
        authority_observations.append(point_observation)
    if document_ids:
        results = [
            KnowledgeSearchItem(
                id=document_id,
                type="guidance",
                title=f"Documento {document_id}",
                tags=[],
                snippet="Resultado validado.",
            )
            for document_id in sorted(document_ids)
        ]
        knowledge_call = PersistedToolCall(
            request_id="req_planner_01",
            call_id="call_authority_knowledge_search",
            name="search_knowledge",
            arguments={"query": "autoridade tipada"},
        )
        authority_calls.append(knowledge_call)
        authority_observations.append(
            ToolObservation(
                request_id=knowledge_call.request_id,
                call_id=knowledge_call.call_id,
                content=KnowledgeSearchModelContent(
                    results=results,
                    total_results=len(results),
                    returned_results=len(results),
                    omitted_results=0,
                    truncated=False,
                ).model_dump(mode="json"),
                artifact=KnowledgeSearchToolArtifact(
                    tool_name=knowledge_call.name,
                    arguments=knowledge_call.arguments.to_python(),
                    source=ToolSource(
                        kind="industrial_api",
                        resource="/knowledge/search",
                    ),
                    outcome=KnowledgeSearchToolOutcome(
                        mode=ResponseMode.COMPLETE,
                        partial_data={},
                        results=results,
                        total_results=len(results),
                        returned_results=len(results),
                        omitted_results=0,
                    ),
                ),
            )
        )
    if model_ids:
        if model_ids != {"mdl_vib_v3"}:
            raise AssertionError(f"fixture sem artifact de modelo para {model_ids!r}")
        model_call, model_observation, _, _ = _observation_with_timestamp(
            "get_model",
            "2026-01-01T00:00:00Z",
        )
        authority_calls.append(model_call)
        authority_observations.append(model_observation)
    return (
        (*authority_calls, *calls),
        (*authority_observations, *observations),
    )


def _proposal_target_authority_history(
) -> tuple[tuple[PersistedToolCall, ...], tuple[ToolObservation, ...]]:
    analysis_call = PersistedToolCall(
        request_id="req_planner_01",
        call_id="call_authority_proposal_analysis_list",
        name="list_asset_analyses",
        arguments={"asset_id": "asset_G501"},
    )
    model_call, model_observation, _, _ = _observation_with_timestamp(
        "get_model",
        "2026-01-01T00:00:00Z",
    )
    return (
        (analysis_call, model_call),
        (
            _analysis_list_observation(
                analysis_call,
                analysis_ids=("an_diag_2026",),
            ),
            model_observation,
        ),
    )


def _real_nullable_point_observation(
    tool_name: str,
) -> tuple[PersistedToolCall, ToolObservation, str]:
    partial_data_by_tool = {
        "get_asset": {"id": "asset_G501", "point_id": None},
        "list_asset_analyses": {
            "asset_id": "asset_G501",
            "point_id": None,
        },
        "get_analysis": {
            "id": "an_nullable_point",
            "asset_id": "asset_G501",
            "point_id": None,
        },
    }

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mode": "partial",
                "notes": "Ponto ausente no recorte parcial.",
                "data": partial_data_by_tool[tool_name],
            },
        )

    async def invoke():
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=IndustrialApiClient(
                "https://industrial.test",
                transport=httpx.MockTransport(handler),
            ),
        )
        try:
            if tool_name == "get_asset":
                return await execute_get_asset("asset_G501", runtime)
            if tool_name == "list_asset_analyses":
                return await execute_list_asset_analyses(
                    "asset_G501", None, runtime
                )
            return await execute_get_analysis("an_nullable_point", runtime)
        finally:
            await runtime.client.aclose()

    result = asyncio.run(invoke())
    arguments = {
        "get_asset": {"asset_id": "asset_G501"},
        "list_asset_analyses": {"asset_id": "asset_G501"},
        "get_analysis": {"analysis_id": "an_nullable_point"},
    }[tool_name]
    messages = {
        "get_asset": "Consulte o cadastro técnico deste ativo.",
        "list_asset_analyses": "Liste as análises deste ativo.",
        "get_analysis": "Detalhe a análise an_nullable_point.",
    }
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id=f"call_real_{tool_name}_nullable_point",
        name=tool_name,
        arguments=arguments,
    )
    outcome = result.artifact.outcome
    observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content={
            "mode": outcome.mode.value,
            "notes": outcome.notes,
            "partial_data": outcome.partial_data,
        },
        artifact=result.artifact,
    )
    return call, observation, messages[tool_name]


def _observation_with_timestamp(
    tool_name: str,
    timestamp: str,
) -> tuple[PersistedToolCall, ToolObservation, SupportRequest, BaseTool]:
    if tool_name in {
        "get_analysis",
        "list_asset_analyses",
        "get_baseline",
    }:
        call, observation, request_message = _real_complete_observation(tool_name)
        artifact = observation.artifact.validated_read_artifact()
        content = observation.content.to_python()
        if tool_name == "get_analysis":
            assert isinstance(artifact, AnalysisDetailToolArtifact)
            analysis = artifact.outcome.analysis.model_copy(
                update={"created_at": timestamp}
            )
            artifact = artifact.model_copy(
                update={
                    "outcome": artifact.outcome.model_copy(
                        update={"analysis": analysis}
                    )
                }
            )
            content = analysis.model_dump(mode="json")
            offered_tool = get_analysis
        elif tool_name == "list_asset_analyses":
            assert isinstance(artifact, AnalysisListToolArtifact)
            analysis = dict(artifact.outcome.analyses[0])
            analysis["created_at"] = timestamp
            artifact = artifact.model_copy(
                update={
                    "outcome": artifact.outcome.model_copy(
                        update={"analyses": [analysis]}
                    )
                }
            )
            content["analyses"][0]["created_at"] = timestamp
            offered_tool = list_asset_analyses
        else:
            assert isinstance(artifact, BaselineToolArtifact)
            baseline = artifact.outcome.baseline.model_copy(
                update={"established_at": timestamp}
            )
            artifact = artifact.model_copy(
                update={
                    "outcome": artifact.outcome.model_copy(
                        update={"baseline": baseline}
                    )
                }
            )
            content = baseline.model_dump(mode="json")
            offered_tool = get_baseline
        request = _request().model_copy(update={"message": request_message})
    elif tool_name in {"get_rms_series", "get_spectrum"}:
        call, observation = _real_technical_observation(tool_name, 10)
        artifact = observation.artifact.validated_read_artifact()
        if tool_name == "get_rms_series":
            assert isinstance(artifact, RmsToolArtifact)
            samples = list(artifact.outcome.rms.samples)
            samples[0] = samples[0].model_copy(update={"ts": timestamp})
            model_samples = list(artifact.model_content.samples)
            model_samples[0] = model_samples[0].model_copy(
                update={"ts": timestamp}
            )
            model_content = artifact.model_content.model_copy(
                update={"samples": model_samples}
            )
            outcome = artifact.outcome.model_copy(
                update={
                    "rms": artifact.outcome.rms.model_copy(
                        update={"samples": samples}
                    )
                }
            )
            offered_tool = get_rms_series
        else:
            assert isinstance(artifact, SpectrumToolArtifact)
            model_content = artifact.model_content.model_copy(
                update={"collected_at": timestamp}
            )
            outcome = artifact.outcome.model_copy(
                update={
                    "spectrum": artifact.outcome.spectrum.model_copy(
                        update={"collected_at": timestamp}
                    )
                }
            )
            offered_tool = get_spectrum
        artifact = artifact.model_copy(
            update={"outcome": outcome, "model_content": model_content}
        )
        content = model_content.model_dump(mode="json")
        request = _request().model_copy(
            update={"message": "Consulte o ponto pt_G501_de deste ativo."}
        )
    else:
        model_artifact = ModelArtifact(
            id="mdl_vib_v3",
            version="3.2.1",
            coverage=[],
            requirements={
                "min_completeness": 0.8,
                "min_snr_db": 12.0,
                "min_rotation_rpm": None,
            },
            processing_state="idle",
            last_run_at=timestamp,
        )
        call = PersistedToolCall(
            request_id="req_planner_01",
            call_id="call_model_timestamp",
            name="get_model",
            arguments={},
        )
        artifact = ModelToolArtifact(
            tool_name=call.name,
            arguments={},
            source=ToolSource(
                kind="industrial_api",
                resource="/models/mdl_vib_v3",
            ),
            outcome=ModelToolOutcome(
                mode=ResponseMode.COMPLETE,
                model=model_artifact,
            ),
        )
        content = model_artifact.model_dump(mode="json")
        request = _request()
        offered_tool = get_model
    restored = ToolObservation.model_validate_json(
        ToolObservation(
            request_id=call.request_id,
            call_id=call.call_id,
            content=content,
            artifact=artifact,
        ).model_dump_json()
    )
    return call, restored, request, offered_tool


def test_select_planner_tools_reuses_read_catalog_only_with_read_permission():
    offered = select_planner_tools(_state(), _read_runtime())

    assert offered
    assert all(any(selected is catalogued for catalogued in READ_TOOLS) for selected in offered)
    assert select_planner_tools(
        _state(permissions=frozenset()),
        _read_runtime(permissions=frozenset()),
    ) == ()


@pytest.mark.parametrize(
    "tool_name",
    [
        "get_asset",
        "list_asset_analyses",
        "get_analysis",
        "get_baseline",
        "get_data_quality",
        "get_knowledge_document",
    ],
)
def test_select_planner_tools_accepts_real_complete_outputs_after_json_round_trip(
    tool_name,
):
    call, observation, request_message = _real_complete_observation(tool_name)
    restored = ToolObservation.model_validate_json(observation.model_dump_json())
    request = _request().model_copy(update={"message": request_message})

    offered = select_planner_tools(
        _state(
            request=request,
            tool_calls=(call,),
            tool_observations=(restored,),
            with_trusted_target_artifacts=True,
        ),
        _read_runtime(),
    )

    assert offered
    assert restored.artifact.validated_read_artifact() is not None
    assert restored.content == observation.content


@pytest.mark.parametrize(
    "message",
    [
        "O log incidental cita an_diag_2026 e mdl_vib_v3.",
        "Não consulte an_diag_2026 nem kb_bearing_guidance.",
        'Texto colado: {"analysis_id":"an_diag_2026",'
        '"document_id":"kb_bearing_guidance","model_id":"mdl_vib_v3"}.',
    ],
    ids=["incidental", "negated", "pasted"],
)
def test_select_planner_tools_does_not_authorize_targets_from_free_text(message):
    permissions = frozenset({"read", "action_low", "action_high", "escalate"})
    request = _request().model_copy(update={"message": message})

    offered_names = {
        tool.name
        for tool in select_planner_tools(
            _state(permissions=permissions, request=request),
            _write_runtime(permissions=permissions),
        )
    }

    assert "get_analysis" not in offered_names
    assert "get_knowledge_document" not in offered_names
    assert "propose_reprocess_analysis" not in offered_names
    assert "propose_request_specialist_analysis" not in offered_names
    assert "propose_request_model_retraining" not in offered_names


@pytest.mark.parametrize(
    ("tool_name", "message"),
    [
        ("get_analysis", "O log incidental cita an_valid_output."),
        ("get_knowledge_document", "Não consulte kb_valid_output."),
        ("get_baseline", 'Texto colado: {"point_id":"pt_G501_de"}.'),
    ],
    ids=["incidental-analysis", "negated-document", "pasted-point"],
)
def test_planner_rejects_free_text_only_target_history_before_model(
    tool_name,
    message,
):
    call, observation, _ = _real_complete_observation(tool_name)
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request().model_copy(update={"message": message}),
                request_id="req_planner_01",
                usage=PlannerUsage(
                    request_id="req_planner_01",
                    selection_count=1,
                ),
                offered_tools=(get_asset,),
                tool_calls=(call,),
                tool_observations=(observation,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert model._events == []


def test_select_planner_tools_hides_asset_argument_tools_without_request_asset():
    request = _request().model_copy(update={"asset_id": None})

    offered_names = {
        tool.name
        for tool in select_planner_tools(
            _state(request=request),
            _read_runtime(),
        )
    }

    assert offered_names == {"get_model", "search_knowledge"}


def test_select_planner_tools_uses_only_ids_observed_for_current_request():
    old_call = PersistedToolCall(
        request_id="req_old",
        call_id="call_old",
        name="list_asset_analyses",
        arguments={"asset_id": "asset_G501"},
    )
    old_observation = _analysis_list_observation(
        old_call,
        analysis_ids=("an_old_request",),
    )
    names_with_old_only = {
        tool.name
        for tool in select_planner_tools(
            _state(
                tool_calls=(old_call,),
                tool_observations=(old_observation,),
            ),
            _read_runtime(),
        )
    }
    current_call = old_call.model_copy(
        update={"request_id": "req_planner_01", "call_id": "call_current"}
    )
    current_observation = _analysis_list_observation(
        current_call,
        analysis_ids=("an_current_request",),
    )
    names_with_current = {
        tool.name
        for tool in select_planner_tools(
            _state(
                tool_calls=(old_call, current_call),
                tool_observations=(old_observation, current_observation),
            ),
            _read_runtime(),
        )
    }

    assert "get_analysis" not in names_with_old_only
    assert "get_analysis" in names_with_current


@pytest.mark.parametrize(
    ("tool_name", "arguments", "content", "resource"),
    [
        (
            "search_knowledge",
            {"query": "x"},
            {"results": [{"id": "kb_from_short_query"}]},
            "/knowledge/search",
        ),
        (
            "search_knowledge",
            {"query": "rolamento", "unexpected": "SELECTOR_EXTRA_SENTINEL"},
            {"results": [{"id": "kb_from_extra_field"}]},
            "/knowledge/search",
        ),
        (
            "list_asset_analyses",
            {"asset_id": "asset_G501", "unexpected": "SELECTOR_LIST_SENTINEL"},
            {"analyses": [{"id": "an_from_invalid_list"}]},
            "/assets/asset_G501/analyses",
        ),
    ],
    ids=["short-query", "extra-field", "invalid-list-analyses"],
)
def test_select_planner_tools_rejects_invalid_current_producer_history(
    tool_name,
    arguments,
    content,
    resource,
):
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id=f"call_invalid_selector_{tool_name}_{len(arguments)}",
        name=tool_name,
        arguments=arguments,
    )
    observation = ToolObservation(
        request_id="req_planner_01",
        call_id=call.call_id,
        content=content,
        artifact=ToolArtifact(
            tool_name=call.name,
            arguments=arguments,
            source=ToolSource(kind="industrial_api", resource=resource),
            outcome=ToolOutcome(mode="complete"),
        ),
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        select_planner_tools(
            _state(
                tool_calls=(call,),
                tool_observations=(observation,),
            ),
            _read_runtime(),
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.usage is None


@pytest.mark.parametrize(
    ("tool_name", "arguments", "resource", "content", "use_write_runtime"),
    [
        (
            "list_asset_analyses",
            {"asset_id": "asset_G501"},
            "/assets/asset_G501/analyses",
            {"analyses": [{"id": "an_FromError"}]},
            False,
        ),
        (
            "get_model",
            {},
            "/models/mdl_vib_v3",
            {"id": "mdl_vib_v3"},
            True,
        ),
    ],
    ids=["analysis-error-cannot-unlock-detail", "model-error-cannot-unlock-proposal"],
)
def test_select_planner_tools_rejects_success_content_attached_to_error_artifact(
    tool_name,
    arguments,
    resource,
    content,
    use_write_runtime,
):
    api_error = {
        "ok": False,
        "category": "timeout",
        "code": "READ_TIMEOUT",
        "message": "Falha sanitizada.",
        "status_code": None,
    }
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id=f"call_error_{tool_name}",
        name=tool_name,
        arguments=arguments,
    )
    observation = ToolObservation(
        request_id="req_planner_01",
        call_id=call.call_id,
        content=content,
        artifact=ToolArtifact(
            tool_name=call.name,
            arguments=arguments,
            source=ToolSource(kind="industrial_api", resource=resource),
            outcome=ToolOutcome(error=api_error),
        ),
    )
    permissions = (
        frozenset({"read", "action_high"})
        if use_write_runtime
        else frozenset({"read"})
    )
    restored_state = AgentState.model_validate_json(
        _state(
            permissions=permissions,
            tool_calls=(call,),
            tool_observations=(observation,),
        ).model_dump_json()
    )
    runtime = (
        _write_runtime(permissions=permissions)
        if use_write_runtime
        else _read_runtime(permissions=permissions)
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        select_planner_tools(restored_state, runtime)

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.usage is None
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


_READ_ERROR_CASES = (
    ("get_asset", {"asset_id": "asset_G501"}, "/assets/asset_G501"),
    (
        "list_asset_analyses",
        {"asset_id": "asset_G501"},
        "/assets/asset_G501/analyses",
    ),
    ("get_analysis", {"analysis_id": "an_9906"}, "/analyses/an_9906"),
    (
        "get_baseline",
        {"asset_id": "asset_G501", "point_id": None},
        "/assets/asset_G501/baseline",
    ),
    (
        "get_rms_series",
        {"asset_id": "asset_G501", "point_id": None},
        "/assets/asset_G501/rms",
    ),
    (
        "get_spectrum",
        {"asset_id": "asset_G501", "point_id": None},
        "/assets/asset_G501/spectrum",
    ),
    (
        "get_data_quality",
        {"asset_id": "asset_G501", "point_id": None},
        "/assets/asset_G501/data-quality",
    ),
    ("get_model", {}, "/models/mdl_vib_v3"),
    ("search_knowledge", {"query": "rolamento"}, "/knowledge/search"),
    (
        "get_knowledge_document",
        {"document_id": "kb_bearing_guidance"},
        "/knowledge/kb_bearing_guidance",
    ),
)


@pytest.mark.parametrize(
    ("tool_name", "arguments", "resource"),
    _READ_ERROR_CASES,
    ids=[case[0] for case in _READ_ERROR_CASES],
)
def test_select_planner_tools_accepts_sanitized_error_from_each_read_tool(
    tool_name,
    arguments,
    resource,
):
    requested_target = {
        "get_analysis": " Consulte an_9906.",
        "get_knowledge_document": " Consulte kb_bearing_guidance.",
    }.get(tool_name, "")
    request = _request().model_copy(
        update={"message": _request().message + requested_target}
    )
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id=f"call_error_{tool_name}",
        name=tool_name,
        arguments=arguments,
    )
    error = ApiError(
        category=ApiErrorCategory.TIMEOUT,
        code="READ_TIMEOUT",
        message="A consulta excedeu o tempo limite.",
    )
    observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content={"error": error.model_dump(mode="json")},
        artifact=ToolArtifact(
            tool_name=tool_name,
            arguments=arguments,
            source=ToolSource(kind="industrial_api", resource=resource),
            outcome=ToolOutcome(error=error),
        ),
    )
    restored = ToolObservation.model_validate_json(observation.model_dump_json())

    offered = select_planner_tools(
        _state(
            request=request,
            tool_calls=(call,),
            tool_observations=(restored,),
            with_trusted_target_artifacts=True,
        ),
        _read_runtime(),
    )

    assert offered
    if tool_name == "list_asset_analyses":
        assert "get_analysis" not in {tool.name for tool in offered}


@pytest.mark.parametrize(
    ("tool_name", "arguments", "resource"),
    _READ_ERROR_CASES[:8],
    ids=[case[0] for case in _READ_ERROR_CASES[:8]],
)
def test_select_planner_tools_accepts_generic_degraded_read_artifacts(
    tool_name,
    arguments,
    resource,
):
    request = _request().model_copy(
        update={"message": "Consulte an_9906 neste ativo."}
    )
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id=f"call_degraded_{tool_name}",
        name=tool_name,
        arguments=arguments,
    )
    if tool_name == "get_model":
        content = DegradedModelContent(
            mode=ResponseMode.PARTIAL,
            notes="Resposta parcial validada.",
            model={"id": "mdl_vib_v3", "version": "3.0"},
            partial_data={},
        ).model_dump(mode="json")
        artifact = ModelToolArtifact(
            tool_name=tool_name,
            arguments=arguments,
            source=ToolSource(kind="industrial_api", resource=resource),
            outcome=ModelToolOutcome(
                mode=ResponseMode.PARTIAL,
                notes="Resposta parcial validada.",
                model={"id": "mdl_vib_v3", "version": "3.0"},
                partial_data={},
            ),
        )
    else:
        content = {
            "mode": ResponseMode.PARTIAL.value,
            "notes": "Resposta parcial validada.",
            "partial_data": {},
        }
        artifact = ToolArtifact(
            tool_name=tool_name,
            arguments=arguments,
            source=ToolSource(kind="industrial_api", resource=resource),
            outcome=ToolOutcome(
                mode=ResponseMode.PARTIAL,
                notes="Resposta parcial validada.",
                partial_data={},
            ),
        )
    observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content=content,
        artifact=artifact,
    )

    assert select_planner_tools(
        _state(
            request=request,
            tool_calls=(call,),
            tool_observations=(
                ToolObservation.model_validate_json(observation.model_dump_json()),
            ),
            with_trusted_target_artifacts=True,
        ),
        _read_runtime(),
    )


@pytest.mark.parametrize("tool_name", ["search_knowledge", "get_knowledge_document"])
def test_select_planner_tools_accepts_specialized_degraded_knowledge_artifacts(
    tool_name,
):
    if tool_name == "search_knowledge":
        arguments = {"query": "rolamento"}
        resource = "/knowledge/search"
        content = DegradedKnowledgeSearchModelContent(
            mode=ResponseMode.PARTIAL,
            notes="Busca parcial validada.",
            results=[],
            total_results=0,
            returned_results=0,
            omitted_results=0,
            truncated=False,
            partial_data={},
        ).model_dump(mode="json")
        artifact = KnowledgeSearchToolArtifact(
            tool_name=tool_name,
            arguments=arguments,
            source=ToolSource(kind="industrial_api", resource=resource),
            outcome=KnowledgeSearchToolOutcome(
                mode=ResponseMode.PARTIAL,
                notes="Busca parcial validada.",
                results=[],
                total_results=0,
                returned_results=0,
                omitted_results=0,
                partial_data={},
            ),
        )
    else:
        arguments = {"document_id": "kb_bearing_guidance"}
        resource = "/knowledge/kb_bearing_guidance"
        document = {"id": "kb_bearing_guidance", "title": "Guia parcial"}
        content = DegradedKnowledgeDocumentContent(
            mode=ResponseMode.PARTIAL,
            notes="Documento parcial validado.",
            document=document,
            partial_data={},
        ).model_dump(mode="json")
        artifact = KnowledgeDocumentToolArtifact(
            tool_name=tool_name,
            arguments=arguments,
            source=ToolSource(kind="industrial_api", resource=resource),
            outcome=KnowledgeDocumentToolOutcome(
                mode=ResponseMode.PARTIAL,
                notes="Documento parcial validado.",
                document=document,
                partial_data={},
            ),
        )
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id=f"call_degraded_{tool_name}",
        name=tool_name,
        arguments=arguments,
    )
    observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content=content,
        artifact=artifact,
    )
    request = _request().model_copy(
        update={"message": "Consulte kb_bearing_guidance neste ativo."}
    )

    assert select_planner_tools(
        _state(
            request=request,
            tool_calls=(call,),
            tool_observations=(
                ToolObservation.model_validate_json(observation.model_dump_json()),
            ),
            with_trusted_target_artifacts=True,
        ),
        _read_runtime(),
    )


def test_planner_rejects_divergent_asset_resource_before_model():
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id="call_wrong_source",
        name="get_asset",
        arguments={"asset_id": "asset_G501"},
    )
    error = ApiError(
        category=ApiErrorCategory.TIMEOUT,
        code="WRONG_SOURCE_SENTINEL",
        message="A consulta excedeu o tempo limite.",
    )
    observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content={"error": error.model_dump(mode="json")},
        artifact=ToolArtifact(
            tool_name=call.name,
            arguments=call.arguments.to_python(),
            source=ToolSource(
                kind="industrial_api",
                resource="/assets/asset_G999",
            ),
            outcome=ToolOutcome(error=error),
        ),
    )
    usage = PlannerUsage(request_id="req_planner_01", selection_count=1)
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=usage,
                offered_tools=(get_asset,),
                tool_calls=(call,),
                tool_observations=(observation,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.usage == usage
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "WRONG_SOURCE_SENTINEL" not in str(exc_info.value)
    assert model._events == []


def test_select_planner_tools_rejects_forged_rms_prompt_projection_after_cut():
    call, observation = _real_technical_observation("get_rms_series", 1_001)
    forged_content = observation.content.to_python()
    forged_content["samples"][50]["value"] = 999_999.0
    restored = ToolObservation.model_validate_json(
        ToolObservation(
            request_id=call.request_id,
            call_id=call.call_id,
            content=forged_content,
            artifact=observation.artifact,
        ).model_dump_json()
    )
    request = _request().model_copy(
        update={"message": "Consulte o ponto pt_G501_de deste ativo."}
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        select_planner_tools(
            _state(
                request=request,
                tool_calls=(call,),
                tool_observations=(restored,),
            ),
            _read_runtime(),
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_planner_rejects_forged_spectrum_prompt_projection_after_cut():
    call, observation = _real_technical_observation("get_spectrum", 201)
    forged_content = observation.content.to_python()
    forged_content["peaks"][10]["amplitude_mm_s"] = 999_999.0
    forged_content["peaks"][10]["note"] = "FORGED_SPECTRUM_SENTINEL"
    restored = ToolObservation.model_validate_json(
        ToolObservation(
            request_id=call.request_id,
            call_id=call.call_id,
            content=forged_content,
            artifact=observation.artifact,
        ).model_dump_json()
    )
    request = _request().model_copy(
        update={"message": "Consulte o ponto pt_G501_de deste ativo."}
    )
    usage = PlannerUsage(request_id="req_planner_01", selection_count=1)
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                request,
                request_id="req_planner_01",
                usage=usage,
                offered_tools=(get_spectrum,),
                tool_calls=(call,),
                tool_observations=(restored,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.usage == usage
    assert "FORGED_SPECTRUM_SENTINEL" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert model._events == []


@pytest.mark.parametrize(
    ("tool_name", "total_items"),
    [
        ("get_rms_series", 1_000),
        ("get_rms_series", 1_001),
        ("get_spectrum", 200),
        ("get_spectrum", 201),
    ],
)
def test_select_planner_tools_accepts_real_bounded_projection_at_cut_boundaries(
    tool_name,
    total_items,
):
    call, observation = _real_technical_observation(tool_name, total_items)
    restored = ToolObservation.model_validate_json(observation.model_dump_json())
    request = _request().model_copy(
        update={"message": "Consulte o ponto pt_G501_de deste ativo."}
    )

    offered = select_planner_tools(
        _state(
            request=request,
            tool_calls=(call,),
            tool_observations=(restored,),
            with_trusted_target_artifacts=True,
        ),
        _read_runtime(),
    )

    typed_artifact = restored.artifact.validated_read_artifact()
    assert offered
    assert typed_artifact is not None
    assert typed_artifact.model_content is not None
    assert typed_artifact.model_content.model_dump(mode="json") == (
        restored.content.to_python()
    )


@pytest.mark.parametrize(
    ("tool_name", "offered_tool", "total_items", "collection_name"),
    [
        ("get_rms_series", get_rms_series, 1_001, "samples"),
        ("get_spectrum", get_spectrum, 201, "peaks"),
    ],
)
def test_planner_rejects_forged_tail_of_bounded_projection_before_model(
    tool_name,
    offered_tool,
    total_items,
    collection_name,
):
    call, observation = _real_technical_observation(tool_name, total_items)
    forged_content = observation.content.to_python()
    tail = forged_content[collection_name][-1]
    if tool_name == "get_rms_series":
        tail["value"] = 999_999.0
    else:
        tail["amplitude_mm_s"] = 999_999.0
        tail["note"] = "FORGED_TAIL_MUST_NOT_LEAK"
    restored = ToolObservation.model_validate_json(
        ToolObservation(
            request_id=call.request_id,
            call_id=call.call_id,
            content=forged_content,
            artifact=observation.artifact,
        ).model_dump_json()
    )
    request = _request().model_copy(
        update={"message": "Consulte o ponto pt_G501_de deste ativo."}
    )
    usage = PlannerUsage(request_id="req_planner_01", selection_count=1)
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                request,
                request_id="req_planner_01",
                usage=usage,
                offered_tools=(offered_tool,),
                tool_calls=(call,),
                tool_observations=(restored,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.usage == usage
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "FORGED_TAIL_MUST_NOT_LEAK" not in str(exc_info.value)
    assert model._events == []


@pytest.mark.parametrize(
    ("tool_name", "offered_tool", "total_items", "collection_name"),
    [
        ("get_rms_series", get_rms_series, 1_000, "samples"),
        ("get_spectrum", get_spectrum, 200, "peaks"),
    ],
)
def test_planner_rejects_forged_interior_at_artifact_cut_before_model(
    tool_name,
    offered_tool,
    total_items,
    collection_name,
):
    call, observation = _real_technical_observation(tool_name, total_items)
    forged_content = observation.content.to_python()
    interior = forged_content[collection_name][10]
    if tool_name == "get_rms_series":
        interior["value"] = 999_999.0
    else:
        interior["amplitude_mm_s"] = 999_999.0
        interior["note"] = "FORGED_INTERIOR_MUST_NOT_LEAK"
    restored = ToolObservation.model_validate_json(
        ToolObservation(
            request_id=call.request_id,
            call_id=call.call_id,
            content=forged_content,
            artifact=observation.artifact,
        ).model_dump_json()
    )
    usage = PlannerUsage(request_id="req_planner_01", selection_count=1)
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request().model_copy(
                    update={
                        "message": "Consulte o ponto pt_G501_de deste ativo."
                    }
                ),
                request_id="req_planner_01",
                usage=usage,
                offered_tools=(offered_tool,),
                tool_calls=(call,),
                tool_observations=(restored,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.usage == usage
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "FORGED_INTERIOR_MUST_NOT_LEAK" not in str(exc_info.value)
    assert model._events == []


def test_select_planner_tools_does_not_authorize_id_from_degraded_notes():
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id="call_degraded_notes",
        name="list_asset_analyses",
        arguments={"asset_id": "asset_G501"},
    )
    observation = _analysis_list_observation(
        call,
        notes="Texto incidental cita an_injetada.",
        partial_data={},
    )

    offered_names = {
        tool.name
        for tool in select_planner_tools(
            _state(
                tool_calls=(call,),
                tool_observations=(observation,),
            ),
            _read_runtime(),
        )
    }

    assert "get_analysis" not in offered_names


def test_select_planner_tools_accepts_nullable_partial_point_without_authorizing_it():
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id="call_partial_nullable_point",
        name="get_asset",
        arguments={"asset_id": "asset_G501"},
    )
    partial_data = {"id": "asset_G501", "point_id": None}
    observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content={
            "mode": "partial",
            "notes": "Ponto ainda não informado.",
            "partial_data": partial_data,
        },
        artifact=ToolArtifact(
            tool_name=call.name,
            arguments=call.arguments.to_python(),
            source=ToolSource(
                kind="industrial_api",
                resource="/assets/asset_G501",
            ),
            outcome=ToolOutcome(
                mode=ResponseMode.PARTIAL,
                notes="Ponto ainda não informado.",
                partial_data=partial_data,
            ),
        ),
    )

    offered_names = {
        tool.name
        for tool in select_planner_tools(
            _state(
                tool_calls=(call,),
                tool_observations=(observation,),
            ),
            _read_runtime(),
        )
    }

    assert "get_baseline" in offered_names
    assert "pt_G501_de" not in str(offered_names)


def test_select_planner_tools_rejects_degraded_asset_with_divergent_primary_id():
    call, observation, _ = _real_nullable_point_observation("get_asset")
    artifact = observation.artifact.validated_read_artifact()
    assert isinstance(artifact, AssetToolArtifact)
    partial_data = {"id": "asset_G999", "point_id": None}
    forged_observation = ToolObservation.model_validate_json(
        ToolObservation(
            request_id=call.request_id,
            call_id=call.call_id,
            content={
                "mode": "partial",
                "notes": "FORGED_ASSET_SCOPE_MUST_NOT_LEAK",
                "partial_data": partial_data,
            },
            artifact=artifact.model_copy(
                update={
                    "outcome": artifact.outcome.model_copy(
                        update={
                            "notes": "FORGED_ASSET_SCOPE_MUST_NOT_LEAK",
                            "partial_data": partial_data,
                        }
                    )
                }
            ),
        ).model_dump_json()
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        select_planner_tools(
            _state(
                tool_calls=(call,),
                tool_observations=(forged_observation,),
            ),
            _read_runtime(),
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "FORGED_ASSET_SCOPE_MUST_NOT_LEAK" not in str(exc_info.value)


@pytest.mark.parametrize(
    "partial_data",
    [
        {"id": None},
        {"nested": {"assetId": "asset_G999"}},
        {"nested": {"assetId": None}},
        {"nested": {"asset-id": "asset_G999"}},
        {"nested": {"asset id": "asset_G999"}},
        {"nested": {"companyId": "comp_outro"}},
        {"nested": {"companyId": None}},
        {"nested": {"company-id": "comp_outro"}},
        {"points": "invalid"},
        {"points": ["invalid"]},
        {"points": [{"asset_id": "asset_G999"}]},
    ],
)
def test_planner_rejects_impossible_degraded_asset_scope_before_model(
    partial_data,
):
    call, observation, _ = _real_nullable_point_observation("get_asset")
    artifact = observation.artifact.validated_read_artifact()
    assert isinstance(artifact, AssetToolArtifact)
    sentinel = "FORGED_DEGRADED_ASSET_MUST_NOT_LEAK"
    forged_observation = ToolObservation.model_validate_json(
        ToolObservation(
            request_id=call.request_id,
            call_id=call.call_id,
            content={
                "mode": "partial",
                "notes": sentinel,
                "partial_data": partial_data,
            },
            artifact=artifact.model_copy(
                update={
                    "outcome": artifact.outcome.model_copy(
                        update={
                            "notes": sentinel,
                            "partial_data": partial_data,
                        }
                    )
                }
            ),
        ).model_dump_json()
    )
    usage = PlannerUsage(request_id="req_planner_01", selection_count=1)
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=usage,
                offered_tools=(get_asset,),
                tool_calls=(call,),
                tool_observations=(forged_observation,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.usage == usage
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in str(exc_info.value)
    assert model._events == []


@pytest.mark.parametrize(
    ("tool_name", "offered_tool"),
    [
        ("get_asset", get_asset),
        ("list_asset_analyses", list_asset_analyses),
        ("get_analysis", get_analysis),
    ],
)
def test_select_planner_tools_accepts_nullable_point_only_in_permitted_degraded_reads(
    tool_name,
    offered_tool,
):
    call, observation, request_message = _real_nullable_point_observation(
        tool_name
    )
    observation = ToolObservation.model_validate_json(
        observation.model_dump_json()
    )
    request = _request().model_copy(update={"message": request_message})
    tool_calls, tool_observations = _history_with_trusted_target_artifacts(
        (call,),
        (observation,),
    )

    offered = select_planner_tools(
        _state(
            request=request,
            tool_calls=tool_calls,
            tool_observations=tool_observations,
        ),
        _read_runtime(),
    )

    assert offered
    model = _RecordingPlannerModel(
        selector_response=AIMessage(content=""),
        terminal_response=PlannerTerminalDecision(
            decision=PlannerDecisionKind.GUIDE,
            stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
        ),
    )

    result = asyncio.run(
        Planner(model).ainvoke(
            request,
            request_id="req_planner_01",
            usage=PlannerUsage(
                request_id="req_planner_01",
                selection_count=len(tool_calls),
            ),
            offered_tools=(offered_tool,),
            tool_calls=tool_calls,
            tool_observations=tool_observations,
        )
    )

    assert isinstance(result, PlannerDecisionTurn)


@pytest.mark.parametrize(
    ("tool_name", "offered_tool", "resource"),
    [
        ("get_baseline", get_baseline, "/assets/asset_G501/baseline"),
        ("get_rms_series", get_rms_series, "/assets/asset_G501/rms"),
        ("get_spectrum", get_spectrum, "/assets/asset_G501/spectrum"),
        (
            "get_data_quality",
            get_data_quality,
            "/assets/asset_G501/data-quality",
        ),
    ],
)
def test_planner_rejects_nullable_point_in_degraded_technical_history_before_model(
    tool_name,
    offered_tool,
    resource,
):
    arguments = {"asset_id": "asset_G501"}
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id=f"call_{tool_name}_nullable_point",
        name=tool_name,
        arguments=arguments,
    )
    partial_data = {"nested": {"point_id": None}}
    observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content={
            "mode": "partial",
            "notes": "Ponto ausente no recorte parcial.",
            "partial_data": partial_data,
        },
        artifact=ToolArtifact(
            tool_name=tool_name,
            arguments=arguments,
            source=ToolSource(kind="industrial_api", resource=resource),
            outcome=ToolOutcome(
                mode=ResponseMode.PARTIAL,
                notes="Ponto ausente no recorte parcial.",
                partial_data=partial_data,
            ),
        ),
    )
    usage = PlannerUsage(request_id="req_planner_01", selection_count=1)
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=usage,
                offered_tools=(offered_tool,),
                tool_calls=(call,),
                tool_observations=(observation,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.usage == usage
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert model._events == []


def test_nullable_partial_point_never_authorizes_a_future_point_target():
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id="call_asset_nullable_target",
        name="get_asset",
        arguments={"asset_id": "asset_G501"},
    )
    partial_data = {"point_id": None}
    observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content={
            "mode": "partial",
            "notes": None,
            "partial_data": partial_data,
        },
        artifact=ToolArtifact(
            tool_name=call.name,
            arguments=call.arguments.to_python(),
            source=ToolSource(
                kind="industrial_api",
                resource="/assets/asset_G501",
            ),
            outcome=ToolOutcome(
                mode=ResponseMode.PARTIAL,
                partial_data=partial_data,
            ),
        ),
    )
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_baseline",
                    "args": {
                        "asset_id": "asset_G501",
                        "point_id": "pt_not_observed",
                    },
                    "id": "call_unobserved_nullable_target",
                    "type": "tool_call",
                }
            ],
        )
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=PlannerUsage(
                    request_id="req_planner_01",
                    selection_count=1,
                ),
                offered_tools=(get_baseline,),
                tool_calls=(call,),
                tool_observations=(observation,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_TOOL_ARGUMENTS


def test_planner_rejects_complete_asset_impossible_for_the_executor():
    impossible_asset = AssetArtifact(
        id="asset_G501",
        name=" ",
        company_id="comp_mineracao_andes",
        criticality="high",
        hierarchy=AssetHierarchy(
            plant=" ",
            line="Linha A",
            parent_asset_id=None,
        ),
        points=[],
        technical_configuration=TechnicalConfiguration(
            machine_type=" ",
            rotation_rpm=-1.0,
            bearing_specs=BearingSpecifications(
                part_number=None,
                bpfo_hz=None,
                bpfi_hz=None,
                bsf_hz=None,
                ftf_hz=None,
            ),
            line_frequency_hz=None,
        ),
        sensor_status=" ",
    )
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id="call_impossible_asset",
        name="get_asset",
        arguments={"asset_id": "asset_G501"},
    )
    content = AssetModelContent(
        id=impossible_asset.id,
        name=impossible_asset.name,
        criticality=impossible_asset.criticality,
        machine_type=impossible_asset.technical_configuration.machine_type,
        rotation_rpm=impossible_asset.technical_configuration.rotation_rpm,
        sensor_status=impossible_asset.sensor_status,
        points=impossible_asset.points,
    )
    observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content=content.model_dump(mode="json"),
        artifact=AssetToolArtifact(
            tool_name=call.name,
            arguments=call.arguments.to_python(),
            source=ToolSource(
                kind="industrial_api",
                resource="/assets/asset_G501",
            ),
            outcome=AssetToolOutcome(
                mode=ResponseMode.COMPLETE,
                asset=impossible_asset,
            ),
        ),
    )
    usage = PlannerUsage(request_id="req_planner_01", selection_count=1)
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=usage,
                offered_tools=(get_asset,),
                tool_calls=(call,),
                tool_observations=(observation,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.usage == usage
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert model._events == []


def test_select_rejects_complete_asset_company_that_executor_cannot_emit():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "mode": "complete",
                "notes": None,
                "data": {
                    "id": "asset_G501",
                    "name": "Motor principal",
                    "company_id": "tenant_x",
                    "criticality": "critical",
                    "plant": "Planta 1",
                    "line": "Britagem",
                    "parent_asset_id": None,
                    "machine_type": "motor_induction",
                    "rotation_rpm": 1780.0,
                    "bearing_pn": None,
                    "bpfo_hz": None,
                    "bpfi_hz": None,
                    "bsf_hz": None,
                    "ftf_hz": None,
                    "line_frequency_hz": 60.0,
                    "sensor_status": "online",
                    "points": [
                        {
                            "id": "pt_G501_de",
                            "asset_id": "asset_G501",
                            "location": "DE",
                            "sensor_status": "online",
                        }
                    ],
                },
            },
        )

    async def invoke_executor():
        runtime = ReadToolRuntime.create(
            user_id="usr_pedro",
            company_id="tenant_x",
            permissions=frozenset({"read"}),
            central_asset_id="asset_G501",
            client=IndustrialApiClient(
                "https://industrial.test",
                transport=httpx.MockTransport(handler),
            ),
        )
        try:
            return await execute_get_asset("asset_G501", runtime)
        finally:
            await runtime.client.aclose()

    executor_result = asyncio.run(invoke_executor())
    assert executor_result.error is not None
    assert executor_result.error.category is ApiErrorCategory.INVALID_RESPONSE

    call, observation, _ = _real_complete_observation("get_asset")
    artifact = observation.artifact.validated_read_artifact()
    assert isinstance(artifact, AssetToolArtifact)
    impossible_asset = artifact.outcome.asset.model_copy(
        update={"company_id": "tenant_x"}
    )
    impossible_observation = ToolObservation.model_validate_json(
        ToolObservation(
            request_id=call.request_id,
            call_id=call.call_id,
            content=observation.content,
            artifact=artifact.model_copy(
                update={
                    "outcome": artifact.outcome.model_copy(
                        update={"asset": impossible_asset}
                    )
                }
            ),
        ).model_dump_json()
    )
    request = _request().model_copy(
        update={
            "identity": Identity(
                user_id="usr_pedro",
                company_id="tenant_x",
            )
        }
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        select_planner_tools(
            _state(
                request=request,
                tool_calls=(call,),
                tool_observations=(impossible_observation,),
            ),
            _read_runtime(company_id="tenant_x"),
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_planner_rejects_impossible_complete_asset_company_before_model():
    call, observation, _ = _real_complete_observation("get_asset")
    artifact = observation.artifact.validated_read_artifact()
    assert isinstance(artifact, AssetToolArtifact)
    impossible_asset = artifact.outcome.asset.model_copy(
        update={"company_id": "tenant_x"}
    )
    impossible_observation = ToolObservation.model_validate_json(
        ToolObservation(
            request_id=call.request_id,
            call_id=call.call_id,
            content=observation.content,
            artifact=artifact.model_copy(
                update={
                    "outcome": artifact.outcome.model_copy(
                        update={"asset": impossible_asset}
                    )
                }
            ),
        ).model_dump_json()
    )
    request = _request().model_copy(
        update={
            "identity": Identity(
                user_id="usr_pedro",
                company_id="tenant_x",
            )
        }
    )
    usage = PlannerUsage(request_id="req_planner_01", selection_count=1)
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                request,
                request_id="req_planner_01",
                usage=usage,
                offered_tools=(get_asset,),
                tool_calls=(call,),
                tool_observations=(impossible_observation,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.usage == usage
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "tenant_x" not in str(exc_info.value)
    assert model._events == []


def test_select_planner_tools_rejects_complete_analysis_impossible_for_executor():
    impossible_analysis = AnalysisArtifact(
        id="an_impossible",
        asset_id="asset_G501",
        point_id="pt_G501_de",
        type="bearing_fault",
        detection_mode="symptom",
        severity="high",
        confidence=2.0,
        baseline_state_at_detection="established",
        evidence=[],
        limitations=[],
        model_version=" ",
        created_at="not-a-timestamp",
        status="current",
    )
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id="call_impossible_analysis",
        name="get_analysis",
        arguments={"analysis_id": "an_impossible"},
    )
    observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content=impossible_analysis.model_dump(mode="json"),
        artifact=AnalysisDetailToolArtifact(
            tool_name=call.name,
            arguments=call.arguments.to_python(),
            source=ToolSource(
                kind="industrial_api",
                resource="/analyses/an_impossible",
            ),
            outcome=AnalysisDetailToolOutcome(
                mode=ResponseMode.COMPLETE,
                analysis=impossible_analysis,
            ),
        ),
    )
    request = _request().model_copy(
        update={"message": "Detalhe a análise an_impossible."}
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        select_planner_tools(
            _state(
                request=request,
                tool_calls=(call,),
                tool_observations=(observation,),
            ),
            _read_runtime(),
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_select_planner_tools_rejects_analysis_timestamp_with_space_separator():
    call, observation, request_message = _real_complete_observation(
        "get_analysis"
    )
    artifact = observation.artifact.validated_read_artifact()
    assert isinstance(artifact, AnalysisDetailToolArtifact)
    impossible_timestamp = "2026-01-02 03:04:05+00:00"
    analysis = artifact.outcome.analysis.model_copy(
        update={"created_at": impossible_timestamp}
    )
    impossible_observation = ToolObservation.model_validate_json(
        ToolObservation(
            request_id=call.request_id,
            call_id=call.call_id,
            content=analysis.model_dump(mode="json"),
            artifact=artifact.model_copy(
                update={
                    "outcome": artifact.outcome.model_copy(
                        update={"analysis": analysis}
                    )
                }
            ),
        ).model_dump_json()
    )
    request = _request().model_copy(update={"message": request_message})

    with pytest.raises(PlannerProtocolError) as exc_info:
        select_planner_tools(
            _state(
                request=request,
                tool_calls=(call,),
                tool_observations=(impossible_observation,),
            ),
            _read_runtime(),
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY


@pytest.mark.parametrize(
    "tool_name",
    [
        "get_analysis",
        "list_asset_analyses",
        "get_baseline",
        "get_rms_series",
        "get_spectrum",
        "get_model",
    ],
)
def test_planner_rejects_space_separated_timestamp_before_model(tool_name):
    call, observation, request, offered_tool = _observation_with_timestamp(
        tool_name,
        "2026-01-02 03:04:05+00:00",
    )
    usage = PlannerUsage(request_id="req_planner_01", selection_count=1)
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                request,
                request_id="req_planner_01",
                usage=usage,
                offered_tools=(offered_tool,),
                tool_calls=(call,),
                tool_observations=(observation,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.usage == usage
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert model._events == []


@pytest.mark.parametrize(
    ("tool_name", "timestamp"),
    [
        (tool_name, timestamp)
        for tool_name in (
            "get_analysis",
            "list_asset_analyses",
            "get_baseline",
            "get_rms_series",
            "get_spectrum",
            "get_model",
        )
        for timestamp in (
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00Z",
        )
    ],
)
def test_select_planner_tools_accepts_canonical_timestamp_forms(
    tool_name,
    timestamp,
):
    call, observation, request, _ = _observation_with_timestamp(
        tool_name,
        timestamp,
    )

    assert select_planner_tools(
        _state(
            request=request,
            tool_calls=(call,),
            tool_observations=(observation,),
            with_trusted_target_artifacts=True,
        ),
        _read_runtime(),
    )


def test_select_planner_tools_rejects_impossible_analysis_inside_complete_list():
    call, observation, _ = _real_complete_observation("list_asset_analyses")
    artifact = observation.artifact.validated_read_artifact()
    assert isinstance(artifact, AnalysisListToolArtifact)
    impossible_analysis = AnalysisArtifact.model_validate(
        artifact.outcome.analyses[0]
    ).model_copy(
        update={"confidence": 2.0}
    )
    impossible_artifact = artifact.model_copy(
        update={
            "outcome": artifact.outcome.model_copy(
                update={"analyses": [impossible_analysis]}
            )
        }
    )
    content = observation.content.to_python()
    content["analyses"][0]["confidence"] = 2.0
    impossible_observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content=content,
        artifact=impossible_artifact,
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        select_planner_tools(
            _state(
                tool_calls=(call,),
                tool_observations=(impossible_observation,),
            ),
            _read_runtime(),
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY


def test_select_planner_tools_rejects_complete_baseline_impossible_for_executor():
    impossible_baseline = BaselineArtifact(
        id="invalid-baseline-id",
        asset_id="asset_G501",
        point_id="pt_G501_de",
        state="established",
        detection_mode="baseline",
        learnable=True,
        established_at="not-a-timestamp",
        invalidated_at=None,
        invalidation_reason=None,
        features=[
            {"feature": "rms_mm_s", "reference": -2.0, "tolerance": -1.0}
        ],
        alarm_threshold=-3.0,
    )
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id="call_impossible_baseline",
        name="get_baseline",
        arguments={"asset_id": "asset_G501", "point_id": "pt_G501_de"},
    )
    observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content=impossible_baseline.model_dump(mode="json"),
        artifact=BaselineToolArtifact(
            tool_name=call.name,
            arguments=call.arguments.to_python(),
            source=ToolSource(
                kind="industrial_api",
                resource="/assets/asset_G501/baseline",
            ),
            outcome=BaselineToolOutcome(
                mode=ResponseMode.COMPLETE,
                baseline=impossible_baseline,
            ),
        ),
    )
    request = _request().model_copy(
        update={"message": "Consulte o ponto pt_G501_de deste ativo."}
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        select_planner_tools(
            _state(
                request=request,
                tool_calls=(call,),
                tool_observations=(observation,),
            ),
            _read_runtime(),
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY


def test_select_planner_tools_rejects_complete_rms_impossible_for_executor():
    call, observation = _real_technical_observation("get_rms_series", 10)
    artifact = observation.artifact.validated_read_artifact()
    assert artifact is not None
    impossible_rms = artifact.outcome.rms.model_copy(
        update={"baseline_reference": -1.0}
    )
    impossible_artifact = artifact.model_copy(
        update={
            "outcome": artifact.outcome.model_copy(
                update={"rms": impossible_rms}
            )
        }
    )
    content = observation.content.to_python()
    content["baseline_reference"] = -1.0
    impossible_observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content=content,
        artifact=impossible_artifact,
    )
    request = _request().model_copy(
        update={"message": "Consulte o ponto pt_G501_de deste ativo."}
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        select_planner_tools(
            _state(
                request=request,
                tool_calls=(call,),
                tool_observations=(impossible_observation,),
            ),
            _read_runtime(),
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY


def test_select_planner_tools_rejects_complete_spectrum_impossible_for_executor():
    call, observation = _real_technical_observation("get_spectrum", 10)
    artifact = observation.artifact.validated_read_artifact()
    assert artifact is not None
    impossible_spectrum = artifact.outcome.spectrum.model_copy(
        update={"collected_at": "not-a-timestamp"}
    )
    impossible_artifact = artifact.model_copy(
        update={
            "outcome": artifact.outcome.model_copy(
                update={"spectrum": impossible_spectrum}
            )
        }
    )
    content = observation.content.to_python()
    content["collected_at"] = "not-a-timestamp"
    impossible_observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content=content,
        artifact=impossible_artifact,
    )
    request = _request().model_copy(
        update={"message": "Consulte o ponto pt_G501_de deste ativo."}
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        select_planner_tools(
            _state(
                request=request,
                tool_calls=(call,),
                tool_observations=(impossible_observation,),
            ),
            _read_runtime(),
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY


def test_select_planner_tools_rejects_complete_quality_impossible_for_executor():
    impossible_quality = DataQualityArtifact(
        asset_id="asset_G501",
        point_id="pt_G501_de",
        completeness=2.0,
        freshness_minutes=-1,
        snr_db=-10.0,
        staleness_flag=False,
    )
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id="call_impossible_quality",
        name="get_data_quality",
        arguments={"asset_id": "asset_G501", "point_id": "pt_G501_de"},
    )
    observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content=impossible_quality.model_dump(mode="json"),
        artifact=DataQualityToolArtifact(
            tool_name=call.name,
            arguments=call.arguments.to_python(),
            source=ToolSource(
                kind="industrial_api",
                resource="/assets/asset_G501/data-quality",
            ),
            outcome=DataQualityToolOutcome(
                mode=ResponseMode.COMPLETE,
                data_quality=impossible_quality,
            ),
        ),
    )
    request = _request().model_copy(
        update={"message": "Consulte o ponto pt_G501_de deste ativo."}
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        select_planner_tools(
            _state(
                request=request,
                tool_calls=(call,),
                tool_observations=(observation,),
            ),
            _read_runtime(),
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY


def test_select_planner_tools_rejects_complete_document_impossible_for_executor():
    impossible_document = KnowledgeDocumentContent(
        id="kb_impossible",
        type="guidance",
        title=" ",
        body="",
        tags=[""],
        returned_body_characters=0,
        omitted_body_characters=0,
        truncated=False,
    )
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id="call_impossible_document",
        name="get_knowledge_document",
        arguments={"document_id": "kb_impossible"},
    )
    observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content=impossible_document.model_dump(mode="json"),
        artifact=KnowledgeDocumentToolArtifact(
            tool_name=call.name,
            arguments=call.arguments.to_python(),
            source=ToolSource(
                kind="industrial_api",
                resource="/knowledge/kb_impossible",
            ),
            outcome=KnowledgeDocumentToolOutcome(
                mode=ResponseMode.COMPLETE,
                document=impossible_document,
            ),
        ),
    )
    request = _request().model_copy(
        update={"message": "Abra o documento kb_impossible."}
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        select_planner_tools(
            _state(
                request=request,
                tool_calls=(call,),
                tool_observations=(observation,),
            ),
            _read_runtime(),
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY


def test_planner_rejects_analysis_id_not_authorized_by_typed_request():
    request = _request().model_copy(
        update={"message": "Detalhe a análise an_permitida."}
    )
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_analysis",
                    "args": {"analysis_id": "an_nao_observada"},
                    "id": "call_wrong_analysis",
                    "type": "tool_call",
                }
            ],
        )
    )
    usage = PlannerUsage(request_id="req_planner_01")

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                request,
                offered_tools=(get_analysis,),
                request_id="req_planner_01",
                usage=usage,
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_TOOL_ARGUMENTS
    assert exc_info.value.usage == usage.model_copy(update={"selection_count": 1})
    assert str(exc_info.value) == (
        "planner protocol error: invalid_tool_arguments"
    )
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    ("tool", "request_message", "arguments"),
    [
        (
            get_knowledge_document,
            "Consulte kb_permitido.",
            {"document_id": "kb_nao_observado"},
        ),
        (
            get_asset,
            "Consulte o ativo central.",
            {"asset_id": "asset_G999"},
        ),
        (
            get_baseline,
            "Consulte o ponto pt_permitido.",
            {"asset_id": "asset_G501", "point_id": "pt_nao_observado"},
        ),
        (
            next(
                tool
                for tool in WRITE_PROPOSAL_TOOLS
                if tool.name == "propose_request_model_retraining"
            ),
            "Reavalie mdl_vib_v3.",
            {"model_id": "mdl_substituto", "justification": "Drift confirmado."},
        ),
    ],
    ids=["knowledge", "asset", "point", "hidden-model"],
)
def test_planner_rejects_untrusted_or_hidden_tool_targets(
    tool,
    request_message,
    arguments,
):
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": tool.name,
                    "args": arguments,
                    "id": "call_untrusted_target",
                    "type": "tool_call",
                }
            ],
        )
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request().model_copy(update={"message": request_message}),
                request_id="req_planner_01",
                usage=PlannerUsage(request_id="req_planner_01"),
                offered_tools=(tool,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_TOOL_ARGUMENTS
    assert exc_info.value.usage == PlannerUsage(
        request_id="req_planner_01",
        selection_count=1,
    )


def test_select_planner_tools_offers_proposals_only_with_write_runtime():
    permissions = frozenset({"read", "action_low", "action_high", "escalate"})
    request = _request().model_copy(
        update={
            "message": (
                "Pedido com alvos tipados an_diag_2026 e mdl_vib_v3."
            )
        }
    )
    tool_calls, tool_observations = _proposal_target_authority_history()
    state = _state(
        permissions=permissions,
        request=request,
        tool_calls=tool_calls,
        tool_observations=tool_observations,
    )

    read_only_names = {
        tool.name
        for tool in select_planner_tools(
            state,
            _read_runtime(permissions=permissions),
        )
    }
    write_tools = select_planner_tools(state, _write_runtime())
    write_names = {tool.name for tool in write_tools}

    assert read_only_names.isdisjoint(tool.name for tool in WRITE_PROPOSAL_TOOLS)
    assert {tool.name for tool in WRITE_PROPOSAL_TOOLS} <= write_names
    assert all(
        any(selected is catalogued for catalogued in (*READ_TOOLS, *WRITE_PROPOSAL_TOOLS))
        for selected in write_tools
    )


@pytest.mark.parametrize(
    ("permissions", "expected_names"),
    [
        (
            frozenset({"action_low"}),
            {
                "propose_reprocess_analysis",
                "propose_request_specialist_analysis",
            },
        ),
        (
            frozenset({"action_high"}),
            {
                "propose_update_asset_criticality",
                "propose_request_model_retraining",
            },
        ),
        (frozenset({"escalate"}), {"propose_escalate_case"}),
        (frozenset({"read"}), set()),
    ],
)
def test_select_planner_tools_applies_each_proposal_permission(
    permissions,
    expected_names,
):
    request = _request().model_copy(
        update={"message": "Alvos an_diag_2026 e mdl_vib_v3."}
    )
    tool_calls, tool_observations = _proposal_target_authority_history()

    offered_names = {
        tool.name
        for tool in select_planner_tools(
            _state(
                permissions=permissions,
                request=request,
                tool_calls=tool_calls,
                tool_observations=tool_observations,
            ),
            _write_runtime(permissions=permissions),
        )
        if tool.name.startswith("propose_")
    }

    assert offered_names == expected_names


@pytest.mark.parametrize(
    "runtime",
    [
        _read_runtime(user_id="usr_outro"),
        _read_runtime(company_id="comp_outra"),
        _read_runtime(central_asset_id="asset_G999"),
        _write_runtime(current_case_id="case_outro"),
    ],
    ids=["person", "company", "central-asset", "current-case"],
)
def test_select_planner_tools_fails_closed_for_trusted_scope_drift(runtime):
    with pytest.raises(PlannerProtocolError) as exc_info:
        select_planner_tools(_state(), runtime)

    assert exc_info.value.code is PlannerErrorCode.RUNTIME_SCOPE_MISMATCH


def test_planner_system_prompt_has_a_versioned_safe_role():
    normalized_prompt = PLANNER_SYSTEM_PROMPT.casefold()

    assert PLANNER_SYSTEM_PROMPT_VERSION == "planner-v1"
    assert PLANNER_SYSTEM_PROMPT_VERSION in PLANNER_SYSTEM_PROMPT
    assert "writer" in normalized_prompt
    assert "no máximo uma tool" in normalized_prompt
    assert "não invente evidência" in normalized_prompt
    assert "não executam efeito" in normalized_prompt
    assert "raciocínio interno" in normalized_prompt


def test_planner_limits_are_immutable_and_fixed_to_the_approved_budget():
    limits = PlannerLimits()

    assert limits.tool_calls == 7
    assert limits.selections == 8
    assert limits.finalizations == 1
    assert limits.context_characters == 48_000
    with pytest.raises(ValidationError):
        limits.selections = 9
    with pytest.raises(ValidationError):
        PlannerLimits(context_characters=48_001)


@pytest.mark.parametrize(
    "invalid_usage",
    [
        {"selection_count": 9},
        {"finalization_count": 2},
        {"selection_count": True},
        {"selection_count": "1"},
    ],
    ids=[
        "selections-above-limit",
        "finalizations-above-limit",
        "boolean-counter",
        "string-counter",
    ],
)
def test_planner_usage_rejects_values_above_the_fixed_budget(invalid_usage):
    with pytest.raises(ValidationError):
        PlannerUsage(request_id="req_invalid_usage", **invalid_usage)


@pytest.mark.parametrize(
    "offered_tools",
    [
        (get_asset, get_asset),
        (
            get_analysis.model_copy(update={"name": get_asset.name}),
            get_asset,
        ),
    ],
    ids=["same-object", "different-tools-and-schemas"],
)
def test_planner_rejects_duplicate_tool_names_before_calling_the_model(
    offered_tools,
):
    assert offered_tools[0].name == offered_tools[1].name
    if offered_tools[0] is not offered_tools[1]:
        assert (
            offered_tools[0].tool_call_schema.model_json_schema()
            != offered_tools[1].tool_call_schema.model_json_schema()
        )
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_asset",
                    "args": {"asset_id": "asset_G501"},
                    "id": "call_ambiguous_tool",
                    "type": "tool_call",
                }
            ],
        )
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=PlannerUsage(request_id="req_planner_01"),
                offered_tools=offered_tools,
            )
        )

    assert exc_info.value.code is PlannerErrorCode.DUPLICATE_TOOL_NAME
    assert model._events == []


def test_planner_requires_request_id_and_usage_before_calling_the_model():
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(TypeError):
        Planner(model).ainvoke(
            _request(),
            offered_tools=(get_asset,),
        )

    assert model._events == []


def test_planner_rejects_explicit_null_usage_before_calling_the_model():
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=None,  # type: ignore[arg-type]
                offered_tools=(get_asset,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_USAGE
    assert model._events == []


@pytest.mark.parametrize("prior_count", [0, 1], ids=["first", "second"])
def test_planner_uses_one_based_sequence_for_persisted_call_ids(prior_count):
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_asset",
                    "args": {"asset_id": "asset_G501"},
                    "id": "call_get_asset_01",
                    "type": "tool_call",
                }
            ],
        )
    )
    prior_calls, prior_observations = _planner_history(prior_count)

    result = asyncio.run(
        Planner(model).ainvoke(
            _request(),
            request_id="req_planner_01",
            usage=PlannerUsage(
                request_id="req_planner_01",
                selection_count=prior_count,
            ),
            offered_tools=(get_asset,),
            tool_calls=prior_calls,
            tool_observations=prior_observations,
        )
    )

    assert isinstance(result, PlannerToolTurn)
    expected_id = hashlib.sha256(
        (
            "planner-v1\0req_planner_01\0" + str(prior_count + 1)
        ).encode("utf-8")
    ).hexdigest()[:24]
    assert result.tool_call.call_id == f"call_planner_{expected_id}"
    assert result.tool_call.name == "get_asset"
    assert result.tool_call.arguments.to_python() == {"asset_id": "asset_G501"}
    assert PlannerToolTurn.model_validate_json(result.model_dump_json()) == result
    assert model._bound_tool_names == ("get_asset",)
    assert model._events == ["bind_tools", "selection_request"]
    assert isinstance(model._selection_messages[0], SystemMessage)
    assert model._selection_messages[0].content == PLANNER_SYSTEM_PROMPT


def test_planner_refuses_ninth_selection_before_calling_the_model():
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=PlannerUsage(
                    request_id="req_planner_01",
                    selection_count=8,
                ),
                offered_tools=(get_asset,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.SELECTION_LIMIT_EXCEEDED
    assert model._events == []


def test_planner_rejects_usage_from_another_request_before_model():
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_current",
                usage=PlannerUsage(request_id="req_other"),
                offered_tools=(get_asset,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_USAGE
    assert model._events == []


def test_invalid_selection_consumes_and_exposes_the_selection_budget():
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_asset",
                    "args": {"asset_id": "outside-public-schema"},
                    "id": "call_invalid_at_budget_boundary",
                    "type": "tool_call",
                }
            ],
        )
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_invalid_at_budget_boundary",
                usage=PlannerUsage(
                    request_id="req_invalid_at_budget_boundary",
                    selection_count=7,
                ),
                offered_tools=(get_asset,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_TOOL_ARGUMENTS
    assert exc_info.value.usage == PlannerUsage(
        request_id="req_invalid_at_budget_boundary",
        selection_count=8,
    )

    retry_model = _RecordingPlannerModel(selector_response=AIMessage(content=""))
    with pytest.raises(PlannerProtocolError) as retry_exc:
        asyncio.run(
            Planner(retry_model).ainvoke(
                _request(),
                request_id="req_invalid_at_budget_boundary",
                usage=exc_info.value.usage,
                offered_tools=(get_asset,),
            )
        )
    assert retry_exc.value.code is PlannerErrorCode.SELECTION_LIMIT_EXCEEDED
    assert retry_model._events == []


@pytest.mark.parametrize(
    ("prior_selections", "accepted"),
    [(7, True), (8, False)],
    ids=["below", "at-limit"],
)
def test_planner_selection_limit_boundaries(prior_selections, accepted):
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_asset",
                    "args": {"asset_id": "asset_G501"},
                    "id": "call_selection_boundary",
                    "type": "tool_call",
                }
            ],
        )
    )
    invocation = Planner(model).ainvoke(
        _request(),
        request_id="req_selection_boundary",
        usage=PlannerUsage(
            request_id="req_selection_boundary",
            selection_count=prior_selections,
        ),
        offered_tools=(get_asset,),
    )

    if accepted:
        result = asyncio.run(invocation)
        assert isinstance(result, PlannerToolTurn)
        assert result.usage is not None
        assert result.usage.selection_count == 8
        assert model._events == ["bind_tools", "selection_request"]
    else:
        with pytest.raises(PlannerProtocolError) as exc_info:
            asyncio.run(invocation)
        assert exc_info.value.code is PlannerErrorCode.SELECTION_LIMIT_EXCEEDED
        assert model._events == []


def test_planner_usage_is_json_safe_and_resets_for_a_new_request():
    state = _state()

    assert state.planner_usage == PlannerUsage(request_id="req_planner_01")
    restored = AgentState.model_validate_json(state.model_dump_json())
    assert restored.planner_usage == state.planner_usage

    continued = state.continue_with(
        request=_request().model_copy(update={"message": "Novo pedido no caso."}),
        identity=state.identity,
        permissions=state.permissions,
        request_id="req_planner_02",
        execution_id="exec_planner_02",
    )

    assert continued.planner_usage == PlannerUsage(request_id="req_planner_02")


def test_agent_state_keeps_legacy_planner_history_unattributed():
    legacy_calls, legacy_observations = _planner_history(1, request_id="req_legacy")
    legacy_wire = _state().model_dump(mode="json")
    legacy_wire.pop("planner_usage")
    legacy_wire["tool_calls"] = [
        {
            key: value
            for key, value in legacy_calls[0].model_dump(mode="json").items()
            if key != "request_id"
        }
    ]
    legacy_wire["tool_observations"] = [
        {
            key: value
            for key, value in legacy_observations[0].model_dump(mode="json").items()
            if key != "request_id"
        }
    ]

    restored = AgentState.model_validate(legacy_wire)

    assert restored.planner_usage == PlannerUsage(request_id="req_planner_01")
    assert restored.tool_calls[0].request_id is None
    assert restored.tool_observations[0].request_id is None

    continued = restored.continue_with(
        request=_request().model_copy(update={"message": "Segunda solicitação."}),
        identity=restored.identity,
        permissions=restored.permissions,
        request_id="req_planner_02",
        execution_id="exec_planner_02",
    )
    reparsed = AgentState.model_validate_json(continued.model_dump_json())

    assert reparsed.planner_usage == PlannerUsage(request_id="req_planner_02")
    assert reparsed.tool_calls[0].request_id is None
    assert reparsed.tool_observations[0].request_id is None


def test_agent_state_rejects_current_planner_history_above_tool_budget():
    calls, observations = _planner_history(8)

    with pytest.raises(ValidationError):
        _state(tool_calls=calls, tool_observations=observations)


def test_planner_refuses_eighth_tool_call_before_any_http_execution():
    calls, observations = _planner_history(7)
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_asset",
                    "args": {"asset_id": "asset_G999"},
                    "id": "call_eighth",
                    "type": "tool_call",
                }
            ],
        )
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=PlannerUsage(
                    request_id="req_planner_01",
                    selection_count=7,
                ),
                offered_tools=(get_asset,),
                tool_calls=calls,
                tool_observations=observations,
            )
        )

    assert exc_info.value.code is PlannerErrorCode.TOOL_CALL_LIMIT_EXCEEDED
    assert model._events == []


@pytest.mark.parametrize(
    ("prior_calls", "expected_code", "expected_events"),
    [
        (6, None, ["bind_tools", "selection_request"]),
        (7, PlannerErrorCode.TOOL_CALL_LIMIT_EXCEEDED, []),
        (8, PlannerErrorCode.INVALID_HISTORY, []),
    ],
    ids=["below", "at-limit", "above"],
)
def test_planner_tool_call_limit_boundaries(
    prior_calls,
    expected_code,
    expected_events,
):
    calls, observations = _planner_history(prior_calls)
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "list_asset_analyses",
                    "args": {"asset_id": "asset_G501"},
                    "id": "call_tool_boundary",
                    "type": "tool_call",
                }
            ],
        )
    )
    invocation = Planner(model).ainvoke(
        _request(),
        request_id="req_planner_01",
        usage=PlannerUsage(request_id="req_planner_01"),
        offered_tools=(list_asset_analyses,),
        tool_calls=calls,
        tool_observations=observations,
    )

    if expected_code is None:
        result = asyncio.run(invocation)
        assert isinstance(result, PlannerToolTurn)
    else:
        with pytest.raises(PlannerProtocolError) as exc_info:
            asyncio.run(invocation)
        assert exc_info.value.code is expected_code
    assert model._events == expected_events


def test_planner_rejects_canonical_repeat_but_allows_distinct_arguments():
    base_calls, base_observations = _planner_history(1)
    calls = (
        PersistedToolCall(
            request_id=base_calls[0].request_id,
            call_id=base_calls[0].call_id,
            name="search_knowledge",
            arguments={"query": "rolamento", "document_type": None},
        ),
    )
    observations = (_successful_search_observation(calls[0]),)
    repeated_model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge",
                    "args": {"query": "rolamento"},
                    "id": "provider_changed_call_id",
                    "type": "tool_call",
                }
            ],
        )
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(repeated_model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=PlannerUsage(request_id="req_planner_01"),
                offered_tools=(search_knowledge,),
                tool_calls=calls,
                tool_observations=observations,
            )
        )

    assert exc_info.value.code is PlannerErrorCode.REPEATED_TOOL_CALL

    distinct_model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge",
                    "args": {"query": "lubrificacao"},
                    "id": "call_distinct_arguments",
                    "type": "tool_call",
                }
            ],
        )
    )
    distinct = asyncio.run(
        Planner(distinct_model).ainvoke(
            _request(),
            request_id="req_planner_01",
            usage=PlannerUsage(request_id="req_planner_01"),
            offered_tools=(search_knowledge,),
            tool_calls=calls,
            tool_observations=observations,
        )
    )

    assert isinstance(distinct, PlannerToolTurn)
    assert distinct.tool_call.arguments.to_python() == {
        "query": "lubrificacao",
        "document_type": None,
    }


def test_planner_allows_provider_call_id_reused_with_distinct_arguments():
    calls, _ = _planner_history(1)
    prior_call = PersistedToolCall(
        request_id="req_planner_01",
        call_id=calls[0].call_id,
        name="search_knowledge",
        arguments={"query": "rolamento", "document_type": None},
    )
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge",
                    "args": {"query": "lubrificacao"},
                    "id": calls[0].call_id,
                    "type": "tool_call",
                }
            ],
        )
    )

    result = asyncio.run(
        Planner(model).ainvoke(
            _request(),
            request_id="req_planner_01",
            usage=PlannerUsage(request_id="req_planner_01"),
            offered_tools=(search_knowledge,),
            tool_calls=(prior_call,),
            tool_observations=(_successful_search_observation(prior_call),),
        )
    )

    assert isinstance(result, PlannerToolTurn)
    assert result.tool_call.arguments.to_python() == {
        "query": "lubrificacao",
        "document_type": None,
    }
    assert result.tool_call.call_id != prior_call.call_id
    assert model._events == ["bind_tools", "selection_request"]


def test_planner_excludes_other_request_history_from_prompt_and_fingerprint():
    old_calls, old_observations = _planner_history(7, request_id="req_old")
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_asset",
                    "args": {"asset_id": "asset_G501"},
                    "id": "call_current_same_arguments",
                    "type": "tool_call",
                }
            ],
        )
    )

    result = asyncio.run(
        Planner(model).ainvoke(
            _request(),
            request_id="req_planner_01",
            usage=PlannerUsage(request_id="req_planner_01"),
            offered_tools=(get_asset,),
            tool_calls=old_calls,
            tool_observations=old_observations,
        )
    )

    assert isinstance(result, PlannerToolTurn)
    assert result.tool_call.arguments.to_python() == {"asset_id": "asset_G501"}
    sent_context = str(model._selection_messages)
    assert "req_old" not in sent_context
    assert "asset_G500" not in sent_context
    assert '{"omitted_interactions":0}' in sent_context


def test_planner_filters_interleaved_history_as_current_request_pairs():
    current_calls, current_observations = _planner_history(2)
    old_calls, old_observations = _planner_history(2, request_id="req_old")
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "list_asset_analyses",
                    "args": {"asset_id": "asset_G501"},
                    "id": "call_after_interleaved_history",
                    "type": "tool_call",
                }
            ],
        )
    )

    result = asyncio.run(
        Planner(model).ainvoke(
            _request(),
            request_id="req_planner_01",
            usage=PlannerUsage(
                request_id="req_planner_01",
                selection_count=2,
            ),
            offered_tools=(list_asset_analyses,),
            tool_calls=(
                old_calls[0],
                current_calls[0],
                old_calls[1],
                current_calls[1],
            ),
            tool_observations=(
                old_observations[0],
                current_observations[0],
                old_observations[1],
                current_observations[1],
            ),
        )
    )

    assert isinstance(result, PlannerToolTurn)
    assert result.usage == PlannerUsage(
        request_id="req_planner_01",
        selection_count=3,
    )
    sent_context = str(model._selection_messages)
    assert "req_old" not in sent_context
    assert current_calls[0].call_id in sent_context
    assert current_calls[1].call_id in sent_context


def test_planner_rejects_current_history_from_another_asset_before_model():
    sentinel = "WRONG_ASSET_SENTINEL"
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id="call_wrong_asset_history",
        name="get_asset",
        arguments={"asset_id": "asset_G999"},
    )
    observation = ToolObservation(
        request_id="req_planner_01",
        call_id=call.call_id,
        content={"id": "asset_G999", "detail": sentinel},
        artifact=ToolArtifact(
            tool_name=call.name,
            arguments={"asset_id": "asset_G999"},
            source=ToolSource(
                kind="industrial_api",
                resource="/assets/asset_G999",
            ),
            outcome=ToolOutcome(partial_data={"detail": sentinel}),
        ),
    )
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_asset",
                    "args": {"asset_id": "asset_G501"},
                    "id": "call_after_wrong_asset",
                    "type": "tool_call",
                }
            ],
        )
    )
    usage = PlannerUsage(
        request_id="req_planner_01",
        selection_count=2,
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=usage,
                offered_tools=(get_asset,),
                tool_calls=(call,),
                tool_observations=(observation,),
            )
        )

    error = exc_info.value
    assert error.code is PlannerErrorCode.INVALID_HISTORY
    assert error.usage == usage
    assert model._events == []
    assert error.__cause__ is None
    assert error.__context__ is None
    assert sentinel not in "".join(traceback.format_exception(error))


def test_planner_rejects_unsafe_degraded_artifact_before_model():
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id="call_degraded_baseline",
        name="get_baseline",
        arguments={"asset_id": "asset_G501", "point_id": None},
    )
    observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content={
            "mode": "partial",
            "notes": "Baseline parcial.",
            "partial_data": {},
        },
        artifact=BaselineToolArtifact(
            tool_name=call.name,
            arguments=call.arguments.to_python(),
            source=ToolSource(
                kind="industrial_api",
                resource="/assets/asset_G501/baseline",
            ),
            outcome=BaselineToolOutcome(
                mode=ResponseMode.PARTIAL,
                notes="Baseline parcial.",
                partial_data={},
            ),
        ),
    )
    sentinel = "UNSAFE_DEGRADED_IDENTITY_SENTINEL"
    typed_wire = observation.artifact.typed_artifact.to_python()
    typed_wire["outcome"]["partial_data"] = {
        "identity": {"user_id": sentinel},
    }
    content_wire = {
        "mode": "partial",
        "notes": "Baseline parcial.",
        "partial_data": typed_wire["outcome"]["partial_data"],
    }
    tampered_artifact = observation.artifact.model_copy(
        update={
            "typed_artifact": JsonSnapshot(
                encoded=json.dumps(typed_wire, sort_keys=True)
            )
        }
    )
    tampered_observation = observation.model_copy(
        update={
            "artifact": tampered_artifact,
            "content": JsonSnapshot(
                encoded=json.dumps(content_wire, sort_keys=True)
            ),
        }
    )
    model = _RecordingPlannerModel(
        selector_response=AIMessage(content=""),
        terminal_response=PlannerTerminalDecision(
            decision=PlannerDecisionKind.GUIDE,
            stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
        ),
    )
    usage = PlannerUsage(request_id="req_planner_01", selection_count=1)

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=usage,
                offered_tools=(get_baseline,),
                tool_calls=(call,),
                tool_observations=(tampered_observation,),
            )
        )

    error = exc_info.value
    assert error.code is PlannerErrorCode.INVALID_HISTORY
    assert error.usage == usage
    assert model._events == []
    assert sentinel not in "".join(traceback.format_exception(error))


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        (
            "get_baseline",
            {"asset_id": "asset_G501", "point_id": "pt_nao_autorizado"},
        ),
        ("get_model", {"model_id": "mdl_target_oculto"}),
    ],
    ids=["point", "hidden-model"],
)
def test_planner_rejects_untrusted_point_or_model_in_current_history(
    tool_name,
    arguments,
):
    sentinel = f"UNTRUSTED_HISTORY_{tool_name.upper()}"
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id=f"call_untrusted_{tool_name}",
        name=tool_name,
        arguments=arguments,
    )
    observation = ToolObservation(
        request_id="req_planner_01",
        call_id=call.call_id,
        content={"detail": sentinel},
        artifact=ToolArtifact(
            tool_name=tool_name,
            arguments=arguments,
            source=ToolSource(kind="industrial_api", resource="/trusted-boundary"),
            outcome=ToolOutcome(partial_data={"detail": sentinel}),
        ),
    )
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))
    usage = PlannerUsage(request_id="req_planner_01", selection_count=1)

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=usage,
                offered_tools=(get_asset,),
                tool_calls=(call,),
                tool_observations=(observation,),
            )
        )

    error = exc_info.value
    assert error.code is PlannerErrorCode.INVALID_HISTORY
    assert error.usage == usage
    assert model._events == []
    assert error.__cause__ is None
    assert error.__context__ is None
    assert sentinel not in "".join(traceback.format_exception(error))


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("unknown_tool", {}),
        ("search_knowledge", {"query": "x"}),
        (
            "search_knowledge",
            {
                "query": "rolamento",
                "unexpected": "INVALID_HISTORY_EXTRA_FIELD",
            },
        ),
    ],
    ids=["unknown-tool", "short-query", "extra-field"],
)
def test_planner_rejects_history_outside_static_catalog_contract_before_model(
    tool_name,
    arguments,
):
    sentinel = f"INVALID_HISTORY_PAYLOAD_{tool_name.upper()}"
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id=f"call_invalid_catalog_{tool_name}_{len(arguments)}",
        name=tool_name,
        arguments=arguments,
    )
    observation = ToolObservation(
        request_id="req_planner_01",
        call_id=call.call_id,
        content={"detail": sentinel},
        artifact=ToolArtifact(
            tool_name=call.name,
            arguments=arguments,
            source=ToolSource(kind="industrial_api", resource="/invalid-history"),
            outcome=ToolOutcome(partial_data={"detail": sentinel}),
        ),
    )
    usage = PlannerUsage(
        request_id="req_planner_01",
        selection_count=1,
    )
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=usage,
                offered_tools=(search_knowledge,),
                tool_calls=(call,),
                tool_observations=(observation,),
            )
        )

    error = exc_info.value
    assert error.code is PlannerErrorCode.INVALID_HISTORY
    assert error.usage == usage
    assert model._events == []
    assert error.__cause__ is None
    assert error.__context__ is None
    accessible_text = "".join(
        (
            str(error),
            repr(error),
            repr(error.args),
            "".join(traceback.format_exception(error)),
        )
    )
    assert sentinel not in accessible_text
    extra_field_sentinel = arguments.get("unexpected")
    if extra_field_sentinel is not None:
        assert extra_field_sentinel not in accessible_text


@pytest.mark.parametrize(
    (
        "request_message",
        "history_tool",
        "history_arguments",
        "history_resource",
        "history_content",
        "next_tool",
        "next_arguments",
    ),
    [
        (
            "Detalhe a análise an_Autorizada.",
            "get_analysis",
            {"analysis_id": "an_Autorizada"},
            "/analyses/an_Autorizada",
            {"id": "an_Injetada", "point_id": "pt_da_injetada"},
            "get_analysis",
            {"analysis_id": "an_Injetada"},
        ),
        (
            "Abra o documento kb_Autorizado.",
            "get_knowledge_document",
            {"document_id": "kb_Autorizado"},
            "/knowledge/kb_Autorizado",
            {"id": "kb_Injetado"},
            "get_knowledge_document",
            {"document_id": "kb_Injetado"},
        ),
        (
            "Consulte o ponto pt_Autorizado.",
            "get_baseline",
            {"asset_id": "asset_G501", "point_id": "pt_Autorizado"},
            "/assets/asset_G501/baseline",
            {"asset_id": "asset_G501", "point_id": "pt_Injetado"},
            "get_baseline",
            {"asset_id": "asset_G501", "point_id": "pt_Injetado"},
        ),
    ],
    ids=["analysis-id-jump", "knowledge-id-jump", "point-id-jump"],
)
def test_planner_rejects_content_target_that_contradicts_typed_artifact(
    request_message,
    history_tool,
    history_arguments,
    history_resource,
    history_content,
    next_tool,
    next_arguments,
):
    request = _request().model_copy(update={"message": request_message})
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id=f"call_history_{history_tool}",
        name=history_tool,
        arguments=history_arguments,
    )
    if history_tool == "get_analysis":
        typed_artifact = AnalysisDetailToolArtifact(
            tool_name=history_tool,
            arguments=history_arguments,
            source=ToolSource(kind="industrial_api", resource=history_resource),
            outcome=AnalysisDetailToolOutcome(
                mode=ResponseMode.COMPLETE,
                analysis=AnalysisArtifact(
                    id="an_Autorizada",
                    asset_id="asset_G501",
                    point_id="pt_Autorizado",
                    type="bearing_fault",
                    detection_mode="symptom",
                    severity="high",
                    confidence=0.9,
                    baseline_state_at_detection="established",
                    evidence=[],
                    limitations=[],
                    model_version="3.0",
                    created_at="2025-05-01T10:00:00Z",
                    status="current",
                ),
            ),
        )
    elif history_tool == "get_knowledge_document":
        typed_artifact = KnowledgeDocumentToolArtifact(
            tool_name=history_tool,
            arguments=history_arguments,
            source=ToolSource(kind="industrial_api", resource=history_resource),
            outcome=KnowledgeDocumentToolOutcome(
                mode=ResponseMode.COMPLETE,
                document=KnowledgeDocumentContent(
                    id="kb_Autorizado",
                    type="guidance",
                    title="Guia autorizado",
                    body="Conteúdo autorizado.",
                    tags=[],
                    returned_body_characters=20,
                    omitted_body_characters=0,
                    truncated=False,
                ),
            ),
        )
    else:
        typed_artifact = BaselineToolArtifact(
            tool_name=history_tool,
            arguments=history_arguments,
            source=ToolSource(kind="industrial_api", resource=history_resource),
            outcome=BaselineToolOutcome(
                mode=ResponseMode.COMPLETE,
                baseline=BaselineArtifact(
                    id="bs_Autorizado",
                    asset_id="asset_G501",
                    point_id="pt_Autorizado",
                    state="established",
                    detection_mode="baseline",
                    learnable=True,
                    established_at="2025-05-01T10:00:00Z",
                    invalidated_at=None,
                    invalidation_reason=None,
                    features=[],
                    alarm_threshold=4.5,
                ),
            ),
        )
    observation = ToolObservation(
        request_id="req_planner_01",
        call_id=call.call_id,
        content=history_content,
        artifact=typed_artifact,
    )
    offered_tool = next(
        tool for tool in READ_TOOLS if tool.name == next_tool
    )
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": next_tool,
                    "args": next_arguments,
                    "id": f"call_after_{history_tool}",
                    "type": "tool_call",
                }
            ],
        )
    )
    usage = PlannerUsage(
        request_id="req_planner_01",
        selection_count=1,
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                request,
                request_id="req_planner_01",
                usage=usage,
                offered_tools=(offered_tool,),
                tool_calls=(call,),
                tool_observations=(observation,),
            )
        )

    error = exc_info.value
    assert error.code is PlannerErrorCode.INVALID_HISTORY
    assert error.usage == usage
    assert model._events == []
    assert error.__cause__ is None
    assert error.__context__ is None
    assert not any(value in str(error) for value in next_arguments.values())


def test_planner_sends_validated_canonical_history_arguments_to_model():
    query = "rolamento do transportador"
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id="call_optional_omitted",
        name="search_knowledge",
        arguments={"query": query},
    )
    observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content=KnowledgeSearchModelContent(
            results=[],
            total_results=0,
            returned_results=0,
            omitted_results=0,
            truncated=False,
        ).model_dump(mode="json"),
        artifact=KnowledgeSearchToolArtifact(
            tool_name=call.name,
            arguments={"query": query},
            source=ToolSource(kind="industrial_api", resource="/knowledge/search"),
            outcome=KnowledgeSearchToolOutcome(
                mode=ResponseMode.COMPLETE,
                partial_data={},
                results=[],
                total_results=0,
                returned_results=0,
                omitted_results=0,
            ),
        ),
    )
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "list_asset_analyses",
                    "args": {"asset_id": "asset_G501"},
                    "id": "call_after_canonical_history",
                    "type": "tool_call",
                }
            ],
        )
    )

    result = asyncio.run(
        Planner(model).ainvoke(
            _request(),
            request_id="req_planner_01",
            usage=PlannerUsage(
                request_id="req_planner_01",
                selection_count=1,
            ),
            offered_tools=(list_asset_analyses,),
            tool_calls=(call,),
            tool_observations=(observation,),
        )
    )

    assert isinstance(result, PlannerToolTurn)
    historical_call = next(
        message.tool_calls[0]
        for message in model._selection_messages
        if isinstance(message, AIMessage) and message.tool_calls
    )
    assert historical_call["args"] == {
        "query": query,
        "document_type": None,
    }


def test_planner_treats_optional_default_forms_as_duplicate_history():
    query = "rolamento do transportador"
    argument_forms = (
        {"query": query},
        {"query": query, "document_type": None},
    )
    calls = tuple(
        PersistedToolCall(
            request_id="req_planner_01",
            call_id=f"call_optional_form_{index}",
            name="search_knowledge",
            arguments=arguments,
        )
        for index, arguments in enumerate(argument_forms)
    )
    observations = tuple(
        ToolObservation(
            request_id="req_planner_01",
            call_id=call.call_id,
            content={"results": []},
            artifact=ToolArtifact(
                tool_name=call.name,
                arguments=arguments,
                source=ToolSource(
                    kind="industrial_api",
                    resource="/knowledge/search",
                ),
                outcome=ToolOutcome(partial_data={"results": []}),
            ),
        )
        for call, arguments in zip(calls, argument_forms, strict=True)
    )
    usage = PlannerUsage(
        request_id="req_planner_01",
        selection_count=2,
    )
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=usage,
                offered_tools=(search_knowledge,),
                tool_calls=calls,
                tool_observations=observations,
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.usage == usage
    assert model._events == []


def test_planner_rejects_duplicate_call_ids_in_current_history_before_model():
    calls, observations = _planner_history(2)
    duplicate_calls = (
        calls[0],
        calls[1].model_copy(update={"call_id": calls[0].call_id}),
    )
    duplicate_observations = (
        observations[0],
        observations[1].model_copy(update={"call_id": calls[0].call_id}),
    )
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=PlannerUsage(request_id="req_planner_01"),
                offered_tools=(get_asset,),
                tool_calls=duplicate_calls,
                tool_observations=duplicate_observations,
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert model._events == []


def test_planner_rejects_duplicate_fingerprints_in_current_history_before_model():
    sentinel = "DUPLICATE_HISTORY_PAYLOAD_MUST_NOT_LEAK"
    calls = tuple(
        PersistedToolCall(
            request_id="req_planner_01",
            call_id=f"call_same_intent_{index}",
            name="get_asset",
            arguments={"asset_id": "asset_G501"},
        )
        for index in range(2)
    )
    observations = tuple(
        ToolObservation(
            request_id="req_planner_01",
            call_id=call.call_id,
            content={"id": "asset_G501", "detail": sentinel},
            artifact=ToolArtifact(
                tool_name=call.name,
                arguments={"asset_id": "asset_G501"},
                source=ToolSource(
                    kind="industrial_api",
                    resource="/assets/asset_G501",
                ),
                outcome=ToolOutcome(partial_data={"detail": sentinel}),
            ),
        )
        for call in calls
    )
    usage = PlannerUsage(
        request_id="req_planner_01",
        selection_count=2,
    )
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=usage,
                offered_tools=(get_asset,),
                tool_calls=calls,
                tool_observations=observations,
            )
        )

    error = exc_info.value
    assert error.code is PlannerErrorCode.INVALID_HISTORY
    assert error.usage == usage
    assert model._events == []
    assert error.__cause__ is None
    assert error.__context__ is None
    assert sentinel not in "".join(
        (
            str(error),
            repr(error),
            repr(error.args),
            "".join(traceback.format_exception(error)),
        )
    )


def test_planner_refuses_second_finalization_before_structured_model_call():
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=PlannerUsage(
                    request_id="req_planner_01",
                    selection_count=1,
                    finalization_count=1,
                ),
                offered_tools=(get_asset,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.FINALIZATION_LIMIT_EXCEEDED
    assert model._events == []


@pytest.mark.parametrize(
    ("prior_finalizations", "accepted"),
    [(0, True), (1, False)],
    ids=["below", "at-limit"],
)
def test_planner_finalization_limit_boundaries(prior_finalizations, accepted):
    terminal = PlannerTerminalDecision(
        decision=PlannerDecisionKind.GUIDE,
        stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
    )
    model = _RecordingPlannerModel(
        selector_response=AIMessage(content=""),
        terminal_response=terminal,
    )
    invocation = Planner(model).ainvoke(
        _request(),
        request_id="req_finalization_boundary",
        usage=PlannerUsage(
            request_id="req_finalization_boundary",
            finalization_count=prior_finalizations,
        ),
        offered_tools=(get_asset,),
    )

    if accepted:
        result = asyncio.run(invocation)
        assert isinstance(result, PlannerDecisionTurn)
        assert result.usage is not None
        assert result.usage.selection_count == 1
        assert result.usage.finalization_count == 1
        assert model._events[-2:] == ["with_structured_output", "terminal_request"]
    else:
        with pytest.raises(PlannerProtocolError) as exc_info:
            asyncio.run(invocation)
        assert exc_info.value.code is PlannerErrorCode.FINALIZATION_LIMIT_EXCEEDED
        assert model._events == []


def test_planner_context_limit_is_exactly_48000_characters():
    def invoke_with_message(message: str):
        model = _RecordingPlannerModel(
            selector_response=AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_asset",
                        "args": {"asset_id": "asset_G501"},
                        "id": "call_context_boundary",
                        "type": "tool_call",
                    }
                ],
            )
        )
        request = _request().model_copy(update={"message": message})
        try:
            result = asyncio.run(
                Planner(model).ainvoke(
                    request,
                    request_id="req_context_boundary",
                    usage=PlannerUsage(request_id="req_context_boundary"),
                    offered_tools=(get_asset,),
                )
            )
        except PlannerProtocolError as error:
            return error, model
        return result, model

    baseline, _ = invoke_with_message("x")
    assert isinstance(baseline, PlannerToolTurn)
    assert baseline.context is not None
    fixed_characters = baseline.context.characters - 1

    below, _ = invoke_with_message("x" * (47_999 - fixed_characters))
    on_limit, _ = invoke_with_message("x" * (48_000 - fixed_characters))
    above, above_model = invoke_with_message("x" * (48_001 - fixed_characters))

    assert isinstance(below, PlannerToolTurn)
    assert below.context is not None
    assert below.context.characters == 47_999
    assert isinstance(on_limit, PlannerToolTurn)
    assert on_limit.context is not None
    assert on_limit.context.characters == 48_000
    assert isinstance(above, PlannerProtocolError)
    assert above.code is PlannerErrorCode.CONTEXT_LIMIT_EXCEEDED
    assert above_model._events == []


@pytest.mark.parametrize("terminal_characters", [47_999, 48_000, 48_001])
def test_planner_recalculates_exact_context_for_terminal_schema(
    terminal_characters,
):
    terminal = PlannerTerminalDecision(
        decision=PlannerDecisionKind.GUIDE,
        stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
    )

    def invoke(message: str):
        model = _RecordingPlannerModel(
            selector_response=AIMessage(content=""),
            terminal_response=terminal,
        )
        request = _request().model_copy(update={"message": message})
        try:
            result = asyncio.run(
                Planner(model).ainvoke(
                    request,
                    request_id="req_terminal_context",
                    usage=PlannerUsage(request_id="req_terminal_context"),
                    offered_tools=(get_asset,),
                )
            )
        except PlannerProtocolError as error:
            return error, model
        return result, model

    baseline, _ = invoke("x")
    assert isinstance(baseline, PlannerDecisionTurn)
    assert baseline.context is not None
    terminal_fixed = baseline.context.characters - 1
    result, model = invoke("x" * (terminal_characters - terminal_fixed))

    if terminal_characters <= 48_000:
        assert isinstance(result, PlannerDecisionTurn)
        assert result.context is not None
        assert result.context.characters == terminal_characters
    else:
        assert isinstance(result, PlannerProtocolError)
        assert result.code is PlannerErrorCode.CONTEXT_LIMIT_EXCEEDED
        assert result.usage == PlannerUsage(
            request_id="req_terminal_context",
            selection_count=1,
        )
        assert model._events == ["bind_tools", "selection_request"]


def test_planner_context_counts_the_bound_tool_wire_before_model():
    oversized_tool = get_asset.model_copy(update={"description": "x" * 48_000})
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_oversized_tool_wire",
                usage=PlannerUsage(request_id="req_oversized_tool_wire"),
                offered_tools=(oversized_tool,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.CONTEXT_LIMIT_EXCEEDED
    assert model._events == []


def test_planner_omits_old_interactions_whole_and_keeps_latest_error():
    calls = (
        PersistedToolCall(
            request_id="req_planner_01",
            call_id="call_old_complete",
            name="search_knowledge",
            arguments={"query": "histórico antigo completo"},
        ),
        PersistedToolCall(
            request_id="req_planner_01",
            call_id="call_latest_error",
            name="list_asset_analyses",
            arguments={"asset_id": "asset_G501", "status": "stale"},
        ),
    )
    latest_error = ApiError(
        category=ApiErrorCategory.TIMEOUT,
        code="LATEST_TIMEOUT_SENTINEL",
        message="tempo limite ao consultar a API industrial",
    )
    old_result = KnowledgeSearchItem(
        id="kb_old_context",
        type="guidance",
        title="OLD_SENTINEL_" + ("x" * 60_000),
        tags=[],
        snippet="resultado antigo",
    )
    observations = (
        ToolObservation(
            request_id="req_planner_01",
            call_id=calls[0].call_id,
            content=KnowledgeSearchModelContent(
                results=[old_result],
                total_results=1,
                returned_results=1,
                omitted_results=0,
                truncated=False,
            ).model_dump(mode="json"),
            artifact=KnowledgeSearchToolArtifact(
                tool_name=calls[0].name,
                arguments=calls[0].arguments.to_python(),
                source=ToolSource(
                    kind="industrial_api",
                    resource="/knowledge/search",
                ),
                outcome=KnowledgeSearchToolOutcome(
                    mode=ResponseMode.COMPLETE,
                    partial_data={},
                    results=[old_result],
                    total_results=1,
                    returned_results=1,
                    omitted_results=0,
                ),
            ),
        ),
        ToolObservation(
            request_id="req_planner_01",
            call_id=calls[1].call_id,
            content={"error": latest_error.model_dump(mode="json")},
            artifact=AnalysisListToolArtifact(
                tool_name=calls[1].name,
                arguments=calls[1].arguments.to_python(),
                source=ToolSource(
                    kind="industrial_api",
                    resource="/assets/asset_G501/analyses",
                ),
                outcome=AnalysisListToolOutcome(error=latest_error),
            ),
        ),
    )
    model = _RecordingPlannerModel(
        selector_response=AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_asset",
                    "args": {"asset_id": "asset_G501"},
                    "id": "call_after_degraded",
                    "type": "tool_call",
                }
            ],
        )
    )

    result = asyncio.run(
        Planner(model).ainvoke(
            _request(),
            request_id="req_planner_01",
            usage=PlannerUsage(
                request_id="req_planner_01",
                selection_count=2,
            ),
            offered_tools=(get_asset,),
            tool_calls=calls,
            tool_observations=observations,
        )
    )

    assert isinstance(result, PlannerToolTurn)
    assert result.context is not None
    assert result.context.omitted_interactions == 1
    sent_context = str(model._selection_messages)
    assert "OLD_SENTINEL" not in sent_context
    assert "LATEST_TIMEOUT_SENTINEL" in sent_context
    assert '{"omitted_interactions":1}' in sent_context
    assert _request().message in sent_context


def test_planner_never_compacts_away_the_latest_degraded_observation():
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id="call_latest_degraded",
        name="list_asset_analyses",
        arguments={"asset_id": "asset_G501"},
    )
    degraded_observation = _analysis_list_observation(
        call,
        notes="LATEST_DEGRADED_MUST_REMAIN" + ("x" * 48_000),
        partial_data={},
    )
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=PlannerUsage(
                    request_id="req_planner_01",
                    selection_count=1,
                ),
                offered_tools=(get_asset,),
                    tool_calls=(call,),
                tool_observations=(degraded_observation,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.CONTEXT_LIMIT_EXCEEDED
    assert model._events == []


def test_planner_discards_selector_text_and_uses_a_separate_terminal_request():
    selector_decoy = "Resposta livre que não pode virar decisão nem resposta."
    terminal = PlannerTerminalDecision(
        decision=PlannerDecisionKind.GUIDE,
        stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
    )
    model = _RecordingPlannerModel(
        selector_response=AIMessage(content=selector_decoy),
        terminal_response=terminal,
    )

    result = asyncio.run(
        Planner(model).ainvoke(
            _request(),
            request_id="req_planner_01",
            usage=PlannerUsage(request_id="req_planner_01"),
            offered_tools=(get_asset,),
        )
    )

    assert isinstance(result, PlannerDecisionTurn)
    assert result.decision == terminal
    assert model._events == [
        "bind_tools",
        "selection_request",
        "with_structured_output",
        "terminal_request",
    ]
    assert model._structured_schema is PlannerTerminalDecision
    assert all(
        selector_decoy not in str(message.content)
        for message in model._terminal_messages
    )
    assert selector_decoy not in result.model_dump_json()
    assert PlannerDecisionTurn.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    ("decision", "stop_reason", "missing_information"),
    [
        (
            PlannerDecisionKind.GUIDE,
            PlannerStopReason.SUFFICIENT_EVIDENCE,
            None,
        ),
        (
            PlannerDecisionKind.REQUEST_INFORMATION,
            PlannerStopReason.MISSING_INFORMATION,
            "Informe o ponto de medição que deve ser investigado.",
        ),
        (
            PlannerDecisionKind.REQUIRE_HUMAN_REVIEW,
            PlannerStopReason.HUMAN_REVIEW_REQUIRED,
            None,
        ),
    ],
)
def test_terminal_decision_accepts_only_coherent_stop_contracts(
    decision,
    stop_reason,
    missing_information,
):
    terminal = PlannerTerminalDecision(
        decision=decision,
        stop_reason=stop_reason,
        missing_information=missing_information,
    )

    assert terminal.decision is decision
    assert terminal.stop_reason is stop_reason
    assert terminal.missing_information == missing_information


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {"decision": "act", "stop_reason": "sufficient_evidence"},
        {"decision": "escalate", "stop_reason": "human_review_required"},
        {
            "decision": "request_confirmation",
            "stop_reason": "missing_information",
            "missing_information": "Confirme a ação.",
        },
        {
            "decision": PlannerDecisionKind.GUIDE,
            "stop_reason": PlannerStopReason.MISSING_INFORMATION,
            "missing_information": "Informe o ativo.",
        },
        {
            "decision": PlannerDecisionKind.REQUEST_INFORMATION,
            "stop_reason": PlannerStopReason.MISSING_INFORMATION,
        },
        {
            "decision": PlannerDecisionKind.REQUIRE_HUMAN_REVIEW,
            "stop_reason": PlannerStopReason.HUMAN_REVIEW_REQUIRED,
            "missing_information": "Não deve acompanhar revisão humana.",
        },
        {
            "decision": PlannerDecisionKind.REQUEST_INFORMATION,
            "stop_reason": PlannerStopReason.MISSING_INFORMATION,
            "missing_information": "x" * 301,
        },
        {
            "decision": PlannerDecisionKind.GUIDE,
            "stop_reason": PlannerStopReason.SUFFICIENT_EVIDENCE,
            "response": "Texto do writer não pertence ao planner.",
        },
    ],
)
def test_terminal_decision_rejects_unsafe_or_incoherent_contracts(invalid_payload):
    with pytest.raises(ValidationError):
        PlannerTerminalDecision.model_validate(invalid_payload)


@pytest.mark.parametrize(
    ("selector_response", "expected_code"),
    [
        (
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "unknown_tool",
                        "args": {"asset_id": "asset_G501"},
                        "id": "call_unknown_01",
                        "type": "tool_call",
                    }
                ],
            ),
            PlannerErrorCode.UNKNOWN_TOOL,
        ),
        (
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_asset",
                        "args": {"asset_id": "asset_G501"},
                        "id": "call_first",
                        "type": "tool_call",
                    },
                    {
                        "name": "get_asset",
                        "args": {"asset_id": "asset_G502"},
                        "id": "call_second",
                        "type": "tool_call",
                    },
                ],
            ),
            PlannerErrorCode.MULTIPLE_TOOL_CALLS,
        ),
        (
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_asset",
                        "args": {"asset_id": "outside-public-schema"},
                        "id": "call_invalid_arguments",
                        "type": "tool_call",
                    }
                ],
            ),
            PlannerErrorCode.INVALID_TOOL_ARGUMENTS,
        ),
        (
            AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "name": "get_asset",
                        "args": "not-json",
                        "id": "call_malformed_arguments",
                        "error": "arguments are not valid JSON",
                        "type": "invalid_tool_call",
                    }
                ],
            ),
            PlannerErrorCode.INVALID_TOOL_ARGUMENTS,
        ),
    ],
)
def test_planner_fails_closed_for_unsafe_tool_selections(
    selector_response,
    expected_code,
):
    model = _RecordingPlannerModel(selector_response=selector_response)

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=PlannerUsage(request_id="req_planner_01"),
                offered_tools=(get_asset,),
            )
        )

    assert exc_info.value.code is expected_code
    assert model._events == ["bind_tools", "selection_request"]


def test_planner_fails_closed_for_invalid_terminal_output():
    model = _RecordingPlannerModel(
        selector_response=AIMessage(content="texto livre descartado"),
        terminal_response={
            "decision": "act",
            "stop_reason": "sufficient_evidence",
            "response": "não deve chegar ao cliente",
        },
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=PlannerUsage(request_id="req_planner_01"),
                offered_tools=(get_asset,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_TERMINAL_OUTPUT
    error_text = str(exc_info.value)
    assert "texto livre descartado" not in error_text
    assert "não deve chegar ao cliente" not in error_text


def test_invalid_terminal_output_consumes_selection_and_finalization_budgets():
    model = _RecordingPlannerModel(
        selector_response=AIMessage(content="descartar"),
        terminal_response={
            "decision": "act",
            "stop_reason": "sufficient_evidence",
        },
    )

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_invalid_terminal_budget",
                usage=PlannerUsage(
                    request_id="req_invalid_terminal_budget",
                    selection_count=3,
                ),
                offered_tools=(get_asset,),
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_TERMINAL_OUTPUT
    assert exc_info.value.usage == PlannerUsage(
        request_id="req_invalid_terminal_budget",
        selection_count=4,
        finalization_count=1,
    )


@pytest.mark.parametrize(
    ("invalid_boundary", "expected_code"),
    [
        ("terminal", PlannerErrorCode.INVALID_TERMINAL_OUTPUT),
        ("arguments", PlannerErrorCode.INVALID_TOOL_ARGUMENTS),
    ],
)
def test_planner_protocol_errors_discard_sensitive_validation_failures(
    invalid_boundary,
    expected_code,
):
    sentinel = f"SENSITIVE_{invalid_boundary.upper()}_PAYLOAD"
    if invalid_boundary == "terminal":
        model = _RecordingPlannerModel(
            selector_response=AIMessage(content="texto livre descartado"),
            terminal_response={
                "decision": "act",
                "stop_reason": "sufficient_evidence",
                "response": sentinel,
            },
        )
    else:
        model = _RecordingPlannerModel(
            selector_response=AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_asset",
                        "args": {"asset_id": sentinel},
                        "id": "call_sensitive_invalid_arguments",
                        "type": "tool_call",
                    }
                ],
            )
        )

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                _request(),
                request_id="req_planner_01",
                usage=PlannerUsage(request_id="req_planner_01"),
                offered_tools=(get_asset,),
            )
        )

    error = exc_info.value
    exception_chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in exception_chain:
        exception_chain.append(current)
        current = current.__cause__ or current.__context__
    accessible_text = "\n".join(
        (
            *(str(item) for item in exception_chain),
            *(repr(item) for item in exception_chain),
            *(repr(item.args) for item in exception_chain),
            "".join(traceback.format_exception(error)),
        )
    )

    assert error.code is expected_code
    assert error.__cause__ is None
    assert error.__context__ is None
    assert exception_chain == [error]
    assert sentinel not in accessible_text


def test_planner_uses_only_persisted_next_turn_content_after_a_tool_call():
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id="call_get_asset_01",
        name="get_asset",
        arguments={"asset_id": "asset_G501"},
    )
    observation = ToolObservation(
        request_id="req_planner_01",
        call_id=call.call_id,
        content={
            "mode": "partial",
            "notes": None,
            "partial_data": {"sensor_status": "online"},
        },
        artifact=ToolArtifact(
            tool_name="get_asset",
            arguments={"asset_id": "asset_G501"},
            source=ToolSource(
                kind="industrial_api",
                resource="/assets/asset_G501",
            ),
            outcome=ToolOutcome(
                mode=ResponseMode.PARTIAL,
                partial_data={"sensor_status": "online"},
            ),
        ),
    )
    terminal = PlannerTerminalDecision(
        decision=PlannerDecisionKind.GUIDE,
        stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
    )
    model = _RecordingPlannerModel(
        selector_response=AIMessage(content="não persistir"),
        terminal_response=terminal,
    )

    result = asyncio.run(
        Planner(model).ainvoke(
            _request(),
            request_id="req_planner_01",
            usage=PlannerUsage(request_id="req_planner_01"),
            offered_tools=(get_asset,),
            tool_calls=(call,),
            tool_observations=(observation,),
        )
    )

    assert isinstance(result, PlannerDecisionTurn)
    selection_tool_message = model._selection_messages[-1]
    assert isinstance(selection_tool_message, ToolMessage)
    assert selection_tool_message.tool_call_id == call.call_id
    assert selection_tool_message.content == (
        '{"mode":"partial","notes":null,'
        '"partial_data":{"sensor_status":"online"}}'
    )
    assert "industrial_api" not in str(model._selection_messages)
    assert model._terminal_messages == model._selection_messages


def test_legacy_and_two_request_planner_state_survives_sqlite_reopen(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "planner-cycle.sqlite3"
    request = _request()
    call = PersistedToolCall(
        request_id="req_planner_01",
        call_id="call_get_asset_01",
        name="get_asset",
        arguments={"asset_id": "asset_G501"},
    )
    observation = ToolObservation(
        request_id="req_planner_01",
        call_id=call.call_id,
        content={"id": "asset_G501", "sensor_status": "online"},
        artifact=ToolArtifact(
            tool_name="get_asset",
            arguments={"asset_id": "asset_G501"},
            source=ToolSource(
                kind="industrial_api",
                resource="/assets/asset_G501",
            ),
            outcome=ToolOutcome(
                partial_data={"id": "asset_G501", "sensor_status": "online"}
            ),
        ),
    )
    legacy_call = call.model_copy(
        update={"request_id": None, "call_id": "call_legacy_unattributed"}
    )
    legacy_observation = observation.model_copy(
        update={"request_id": None, "call_id": legacy_call.call_id}
    )
    state = AgentState(
        request=request,
        identity=TrustedIdentity(
            user_id=request.identity.user_id,
            company_id=request.identity.company_id,
        ),
        permissions=frozenset({"read"}),
        request_id="req_planner_01",
        thread_id="thread_planner_01",
        execution_id="exec_planner_01",
        thread_scope=ThreadScope(
            thread_id="thread_planner_01",
            case_id=request.case_id,
            company_id=request.identity.company_id,
            user_id=request.identity.user_id,
        ),
        tool_calls=(call, legacy_call),
        tool_observations=(observation, legacy_observation),
        ledger=compile_observations(
            (observation,),
            recorded_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
        ),
        planner_usage=PlannerUsage(
            request_id="req_planner_01",
            selection_count=1,
        ),
        step_limit=3,
    )
    state = state.continue_with(
        request=request.model_copy(update={"message": "Segunda solicitação."}),
        identity=state.identity,
        permissions=state.permissions,
        request_id="req_planner_02",
        execution_id="exec_planner_02",
    )
    serialized_state = state.model_dump(mode="json")
    json.dumps(serialized_state, allow_nan=False)
    config = {
        "configurable": {
            "thread_id": state.thread_id,
            "checkpoint_ns": "",
        }
    }
    checkpoint = {
        "v": LATEST_VERSION,
        "id": "00000000-0000-6000-8000-000000000010",
        "ts": "2026-09-01T12:00:00+00:00",
        "channel_values": {"agent_state": serialized_state},
        "channel_versions": {"agent_state": "planner-v1"},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": ["agent_state"],
    }

    async def close_and_reopen():
        async with open_checkpointer(checkpoint_path) as saver:
            await saver.aput(
                config,
                checkpoint,
                {"source": "update", "step": 0, "parents": {}},
                {"agent_state": "planner-v1"},
            )
        async with open_checkpointer(checkpoint_path) as reopened_saver:
            return await reopened_saver.aget(config)

    restored_checkpoint = asyncio.run(close_and_reopen())

    assert restored_checkpoint is not None
    restored_state = AgentState.model_validate(
        restored_checkpoint["channel_values"]["agent_state"]
    )
    assert restored_state.request_id == "req_planner_02"
    assert restored_state.tool_calls == (call, legacy_call)
    assert restored_state.tool_observations == (observation, legacy_observation)
    assert tuple(item.request_id for item in restored_state.tool_calls) == (
        "req_planner_01",
        None,
    )
    assert restored_state.planner_usage == state.planner_usage
    assert restored_state.tool_observations[0].content is not None
    assert restored_state.tool_observations[0].content.to_python() == {
        "id": "asset_G501",
        "sensor_status": "online",
    }


def test_planner_rejects_duplicate_fingerprint_restored_from_sqlite(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "planner-duplicate-history.sqlite3"
    sentinel = "RESTORED_DUPLICATE_PAYLOAD_MUST_NOT_LEAK"
    calls = tuple(
        PersistedToolCall(
            request_id="req_planner_01",
            call_id=f"call_restored_duplicate_{index}",
            name="get_asset",
            arguments={"asset_id": "asset_G501"},
        )
        for index in range(2)
    )
    observations = tuple(
        ToolObservation(
            request_id="req_planner_01",
            call_id=call.call_id,
            content={"id": "asset_G501", "detail": sentinel},
            artifact=ToolArtifact(
                tool_name=call.name,
                arguments={"asset_id": "asset_G501"},
                source=ToolSource(
                    kind="industrial_api",
                    resource="/assets/asset_G501",
                ),
                outcome=ToolOutcome(partial_data={"detail": sentinel}),
            ),
        )
        for call in calls
    )
    usage = PlannerUsage(
        request_id="req_planner_01",
        selection_count=2,
    )
    state = AgentState.model_validate(
        {
            **_state(
                tool_calls=calls,
                tool_observations=observations,
            ).model_dump(mode="python"),
            "planner_usage": usage,
        }
    )
    serialized_state = state.model_dump(mode="json")
    config = {
        "configurable": {
            "thread_id": state.thread_id,
            "checkpoint_ns": "",
        }
    }
    checkpoint = {
        "v": LATEST_VERSION,
        "id": "00000000-0000-6000-8000-000000000011",
        "ts": "2026-09-01T12:01:00+00:00",
        "channel_values": {"agent_state": serialized_state},
        "channel_versions": {"agent_state": "planner-v1"},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": ["agent_state"],
    }

    async def close_and_reopen():
        async with open_checkpointer(checkpoint_path) as saver:
            await saver.aput(
                config,
                checkpoint,
                {"source": "update", "step": 0, "parents": {}},
                {"agent_state": "planner-v1"},
            )
        async with open_checkpointer(checkpoint_path) as reopened_saver:
            return await reopened_saver.aget(config)

    restored_checkpoint = asyncio.run(close_and_reopen())

    assert restored_checkpoint is not None
    restored_state = AgentState.model_validate(
        restored_checkpoint["channel_values"]["agent_state"]
    )
    assert tuple(call.call_id for call in restored_state.tool_calls) == (
        "call_restored_duplicate_0",
        "call_restored_duplicate_1",
    )
    assert restored_state.planner_usage == usage
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                restored_state.request,
                request_id=restored_state.request_id,
                usage=restored_state.planner_usage,
                offered_tools=(get_asset,),
                tool_calls=restored_state.tool_calls,
                tool_observations=restored_state.tool_observations,
            )
        )

    error = exc_info.value
    assert error.code is PlannerErrorCode.INVALID_HISTORY
    assert error.usage == usage
    assert model._events == []
    assert error.__cause__ is None
    assert error.__context__ is None
    assert sentinel not in "".join(traceback.format_exception(error))


def test_bounded_technical_projections_survive_sqlite_reopen(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "planner-bounded-projections.sqlite3"
    rms_call, rms_observation = _real_technical_observation(
        "get_rms_series", 1_001
    )
    spectrum_call, spectrum_observation = _real_technical_observation(
        "get_spectrum", 201
    )
    request = _request().model_copy(
        update={"message": "Consulte o ponto pt_G501_de deste ativo."}
    )
    state = _state(
        request=request,
        tool_calls=(rms_call, spectrum_call),
        tool_observations=(rms_observation, spectrum_observation),
        with_trusted_target_artifacts=True,
    ).model_copy(
        update={
            "planner_usage": PlannerUsage(
                request_id="req_planner_01",
                selection_count=3,
            )
        }
    )
    serialized_state = state.model_dump(mode="json")
    json.dumps(serialized_state, allow_nan=False)
    config = {
        "configurable": {
            "thread_id": state.thread_id,
            "checkpoint_ns": "",
        }
    }
    checkpoint = {
        "v": LATEST_VERSION,
        "id": "00000000-0000-6000-8000-000000000012",
        "ts": "2026-09-01T12:02:00+00:00",
        "channel_values": {"agent_state": serialized_state},
        "channel_versions": {"agent_state": "planner-v1"},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": ["agent_state"],
    }

    async def close_and_reopen():
        async with open_checkpointer(checkpoint_path) as saver:
            await saver.aput(
                config,
                checkpoint,
                {"source": "update", "step": 0, "parents": {}},
                {"agent_state": "planner-v1"},
            )
        async with open_checkpointer(checkpoint_path) as reopened_saver:
            return await reopened_saver.aget(config)

    restored_checkpoint = asyncio.run(close_and_reopen())

    assert restored_checkpoint is not None
    restored_state = AgentState.model_validate(
        restored_checkpoint["channel_values"]["agent_state"]
    )
    for observation in restored_state.tool_observations:
        artifact = observation.artifact.validated_read_artifact()
        assert artifact is not None
        if isinstance(artifact, (RmsToolArtifact, SpectrumToolArtifact)):
            assert artifact.model_content is not None
            assert artifact.model_content.model_dump(mode="json") == (
                observation.content.to_python()
            )
    assert select_planner_tools(restored_state, _read_runtime())


def test_degraded_asset_scope_is_revalidated_after_sqlite_reopen(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "planner-degraded-asset-scope.sqlite3"
    call, observation, _ = _real_nullable_point_observation("get_asset")
    artifact = observation.artifact.validated_read_artifact()
    assert isinstance(artifact, AssetToolArtifact)
    sentinel = "SQLITE_FORGED_ASSET_SCOPE_MUST_NOT_LEAK"
    partial_data = {"nested": {"assetId": "asset_G999"}}
    forged_observation = ToolObservation(
        request_id=call.request_id,
        call_id=call.call_id,
        content={
            "mode": "partial",
            "notes": sentinel,
            "partial_data": partial_data,
        },
        artifact=artifact.model_copy(
            update={
                "outcome": artifact.outcome.model_copy(
                    update={
                        "notes": sentinel,
                        "partial_data": partial_data,
                    }
                )
            }
        ),
    )
    usage = PlannerUsage(request_id="req_planner_01", selection_count=1)
    state = _state(
        tool_calls=(call,),
        tool_observations=(forged_observation,),
    ).model_copy(update={"planner_usage": usage})
    serialized_state = state.model_dump(mode="json")
    config = {
        "configurable": {
            "thread_id": state.thread_id,
            "checkpoint_ns": "",
        }
    }
    checkpoint = {
        "v": LATEST_VERSION,
        "id": "00000000-0000-6000-8000-000000000013",
        "ts": "2026-09-01T12:03:00+00:00",
        "channel_values": {"agent_state": serialized_state},
        "channel_versions": {"agent_state": "planner-v1"},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": ["agent_state"],
    }

    async def close_and_reopen():
        async with open_checkpointer(checkpoint_path) as saver:
            await saver.aput(
                config,
                checkpoint,
                {"source": "update", "step": 0, "parents": {}},
                {"agent_state": "planner-v1"},
            )
        async with open_checkpointer(checkpoint_path) as reopened_saver:
            return await reopened_saver.aget(config)

    restored_checkpoint = asyncio.run(close_and_reopen())

    assert restored_checkpoint is not None
    restored_state = AgentState.model_validate(
        restored_checkpoint["channel_values"]["agent_state"]
    )
    model = _RecordingPlannerModel(selector_response=AIMessage(content=""))

    with pytest.raises(PlannerProtocolError) as exc_info:
        asyncio.run(
            Planner(model).ainvoke(
                restored_state.request,
                request_id=restored_state.request_id,
                usage=restored_state.planner_usage,
                offered_tools=(get_asset,),
                tool_calls=restored_state.tool_calls,
                tool_observations=restored_state.tool_observations,
            )
        )

    assert exc_info.value.code is PlannerErrorCode.INVALID_HISTORY
    assert exc_info.value.usage == usage
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert sentinel not in str(exc_info.value)
    assert model._events == []
