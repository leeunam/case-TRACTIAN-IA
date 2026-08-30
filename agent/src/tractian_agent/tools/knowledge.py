"""Tools de leitura para metadados de modelo e conhecimento global."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Literal

from langchain.tools import tool
from langgraph.prebuilt import ToolRuntime
from pydantic import ConfigDict, Field, JsonValue, TypeAdapter, ValidationError, field_validator

from tractian_agent.contracts import ApiError, ResponseMode, StrictModel

from .identifiers import KnowledgeDocumentId, ModelId
from .observations import ToolArtifact, ToolOutcome, ToolSource, assert_safe_partial_json
from .runtime import ReadToolRuntime
from .timestamps import parse_aware_iso_timestamp

KnowledgeDocumentType = Literal["procedure", "glossary", "guidance"]
ProcessingState = Literal["idle", "running", "pending", "delayed", "failed"]
_DOCUMENT_ID = TypeAdapter(KnowledgeDocumentId)
_MODEL_ID = TypeAdapter(ModelId)
_DOCUMENT_TYPE = TypeAdapter(KnowledgeDocumentType)
_PROCESSING_STATE = TypeAdapter(ProcessingState)
_DEGRADED_FLAGS = frozenset({"conflict", "inconclusive"})
_SNIPPET_LIMIT = 240
_SEARCH_LIMIT = 10
_CONTENT_BODY_LIMIT = 8_000
_ARTIFACT_BODY_LIMIT = 32_000


class _ModelCoverageWire(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    machine_type: str = Field(min_length=1, pattern=r"\S")
    supported: bool
    can_learn_baseline: bool
    note: str | None = Field(default=None, min_length=1, pattern=r"\S")


class _ModelRequirementsWire(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    min_completeness: float = Field(ge=0, le=1, allow_inf_nan=False)
    min_snr_db: float = Field(ge=0, allow_inf_nan=False)
    min_rotation_rpm: float | None = Field(ge=0, allow_inf_nan=False)


class _ModelWire(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: ModelId
    version: str = Field(min_length=1, pattern=r"\S")
    coverage: list[_ModelCoverageWire]
    requirements: _ModelRequirementsWire
    processing_state: ProcessingState
    last_run_at: str | None = Field(min_length=1)

    @field_validator("coverage")
    @classmethod
    def _requires_unique_machine_types(cls, value: list[_ModelCoverageWire]) -> list[_ModelCoverageWire]:
        machine_types = [item.machine_type for item in value]
        if len(machine_types) != len(set(machine_types)):
            raise ValueError("A cobertura do modelo contém tipos de máquina duplicados.")
        return value

    @field_validator("last_run_at")
    @classmethod
    def _requires_aware_timestamp(cls, value: str | None) -> str | None:
        if value is not None:
            parse_aware_iso_timestamp(value)
        return value


class ModelCoverage(StrictModel):
    machine_type: str
    supported: bool
    can_learn_baseline: bool
    note: str | None = None


class ModelRequirements(StrictModel):
    min_completeness: float
    min_snr_db: float
    min_rotation_rpm: float | None


class ModelArtifact(StrictModel):
    id: ModelId
    version: str
    coverage: list[ModelCoverage]
    requirements: ModelRequirements
    processing_state: ProcessingState
    last_run_at: str | None


class DegradedModelContent(StrictModel):
    mode: ResponseMode
    notes: str | None
    model: dict[str, JsonValue]
    partial_data: JsonValue


class ModelToolOutcome(ToolOutcome):
    model: ModelArtifact | dict[str, JsonValue] | None = None


class ModelToolArtifact(ToolArtifact):
    outcome: ModelToolOutcome


class GetModelResult(StrictModel):
    content: ModelArtifact | DegradedModelContent | None
    artifact: ModelToolArtifact
    error: ApiError | None = None


class _KnowledgeDocumentWire(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: KnowledgeDocumentId
    type: KnowledgeDocumentType
    title: str = Field(min_length=1, pattern=r"\S")
    body: str = Field(min_length=1)
    tags: list[str]

    @field_validator("tags")
    @classmethod
    def _requires_nonblank_tags(cls, value: list[str]) -> list[str]:
        if not all(tag.strip() for tag in value):
            raise ValueError("As tags do documento devem ser não vazias.")
        return value


class KnowledgeSearchItem(StrictModel):
    id: KnowledgeDocumentId
    type: KnowledgeDocumentType
    title: str
    tags: list[str]
    snippet: str


class KnowledgeSearchModelContent(StrictModel):
    results: list[KnowledgeSearchItem]
    total_results: int = Field(ge=0)
    returned_results: int = Field(ge=0)
    omitted_results: int = Field(ge=0)
    truncated: bool


class DegradedKnowledgeSearchModelContent(StrictModel):
    mode: ResponseMode
    notes: str | None
    results: list[dict[str, JsonValue]]
    total_results: int = Field(ge=0)
    returned_results: int = Field(ge=0)
    omitted_results: int = Field(ge=0)
    truncated: bool
    partial_data: JsonValue


class KnowledgeSearchToolOutcome(ToolOutcome):
    results: list[KnowledgeSearchItem] | list[dict[str, JsonValue]] | None = None
    total_results: int | None = Field(default=None, ge=0)
    returned_results: int | None = Field(default=None, ge=0)
    omitted_results: int | None = Field(default=None, ge=0)


class KnowledgeSearchToolArtifact(ToolArtifact):
    outcome: KnowledgeSearchToolOutcome


class SearchKnowledgeResult(StrictModel):
    content: KnowledgeSearchModelContent | DegradedKnowledgeSearchModelContent | None
    artifact: KnowledgeSearchToolArtifact
    error: ApiError | None = None


class KnowledgeDocumentContent(StrictModel):
    id: KnowledgeDocumentId
    type: KnowledgeDocumentType
    title: str
    body: str
    tags: list[str]
    returned_body_characters: int = Field(ge=0)
    omitted_body_characters: int = Field(ge=0)
    truncated: bool


class DegradedKnowledgeDocumentContent(StrictModel):
    mode: ResponseMode
    notes: str | None
    document: dict[str, JsonValue]
    partial_data: JsonValue


class KnowledgeDocumentToolOutcome(ToolOutcome):
    document: KnowledgeDocumentContent | dict[str, JsonValue] | None = None


class KnowledgeDocumentToolArtifact(ToolArtifact):
    outcome: KnowledgeDocumentToolOutcome


class GetKnowledgeDocumentResult(StrictModel):
    content: KnowledgeDocumentContent | DegradedKnowledgeDocumentContent | None
    artifact: KnowledgeDocumentToolArtifact
    error: ApiError | None = None


class _GetModelToolArguments(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)

    runtime: ToolRuntime[ReadToolRuntime]


class _SearchArguments(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=2, max_length=200)
    document_type: KnowledgeDocumentType | None = None

    @field_validator("query")
    @classmethod
    def _reject_outer_whitespace(cls, value: str) -> str:
        if value != value.strip() or not value.strip():
            raise ValueError("A consulta não pode começar ou terminar com espaços.")
        return value


class _SearchKnowledgeToolArguments(_SearchArguments):
    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)

    runtime: ToolRuntime[ReadToolRuntime]


class _GetKnowledgeDocumentToolArguments(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)

    document_id: KnowledgeDocumentId
    runtime: ToolRuntime[ReadToolRuntime]


def _assert_read(runtime: ReadToolRuntime, resource: str) -> None:
    if "read" not in runtime.permissions:
        raise PermissionError(f"A permissão 'read' é necessária para consultar {resource}.")


def _validate_document_id(value: object) -> KnowledgeDocumentId:
    return _DOCUMENT_ID.validate_python(value, strict=True)


def _validated_search_arguments(
    query: object, document_type: object
) -> _SearchArguments:
    return _SearchArguments.model_validate(
        {"query": query, "document_type": document_type}, strict=True
    )


def _params(runtime: ReadToolRuntime, **parameters: str) -> dict[str, str] | None:
    result = dict(parameters)
    if runtime.seed is not None:
        result["seed"] = runtime.seed
    return result or None


def _normalize_model(model: _ModelWire) -> ModelArtifact:
    return ModelArtifact(
        id=model.id,
        version=model.version,
        coverage=[
            ModelCoverage(
                machine_type=item.machine_type,
                supported=item.supported,
                can_learn_baseline=item.can_learn_baseline,
                note=item.note,
            )
            for item in model.coverage
        ],
        requirements=ModelRequirements(**model.requirements.model_dump()),
        processing_state=model.processing_state,
        last_run_at=model.last_run_at,
    )


def _validate_degraded_model(data: JsonValue, configured_model_id: ModelId) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    assert_safe_partial_json(data)
    if not isinstance(data, Mapping):
        raise ValueError("A resposta degradada do modelo é inválida.")
    allowed = {"id", "version", "coverage", "requirements", "processing_state", "last_run_at", *_DEGRADED_FLAGS}
    if set(data) - allowed:
        raise ValueError("A resposta degradada contém um campo de topo inesperado.")
    projected: dict[str, JsonValue] = {}
    if "id" in data:
        model_id = _MODEL_ID.validate_python(data["id"], strict=True)
        if model_id != configured_model_id:
            raise ValueError("A API retornou um modelo diferente do configurado.")
        projected["id"] = model_id
    if "version" in data:
        version = data["version"]
        if not isinstance(version, str) or not version.strip():
            raise ValueError("A resposta degradada contém versão inválida.")
        projected["version"] = version
    if "processing_state" in data:
        projected["processing_state"] = _PROCESSING_STATE.validate_python(data["processing_state"], strict=True)
    if "last_run_at" in data:
        timestamp = data["last_run_at"]
        if timestamp is not None:
            if not isinstance(timestamp, str):
                raise ValueError("A resposta degradada contém timestamp inválido.")
            parse_aware_iso_timestamp(timestamp)
        projected["last_run_at"] = timestamp
    if "coverage" in data:
        coverage = data["coverage"]
        if not isinstance(coverage, list):
            raise ValueError("A resposta degradada contém cobertura inválida.")
        items: list[dict[str, JsonValue]] = []
        seen_machine_types: set[str] = set()
        for item in coverage:
            if not isinstance(item, Mapping) or set(item) - {"machine_type", "supported", "can_learn_baseline", "note"}:
                raise ValueError("A resposta degradada contém cobertura inválida.")
            projected_item: dict[str, JsonValue] = {}
            if "machine_type" in item:
                machine_type = item["machine_type"]
                if not isinstance(machine_type, str) or not machine_type.strip() or machine_type in seen_machine_types:
                    raise ValueError("A resposta degradada contém cobertura inválida.")
                seen_machine_types.add(machine_type)
                projected_item["machine_type"] = machine_type
            for key in ("supported", "can_learn_baseline"):
                if key in item:
                    if not isinstance(item[key], bool):
                        raise ValueError("A resposta degradada contém cobertura inválida.")
                    projected_item[key] = item[key]
            if "note" in item:
                note = item["note"]
                if note is not None and (not isinstance(note, str) or not note.strip()):
                    raise ValueError("A resposta degradada contém cobertura inválida.")
                projected_item["note"] = note
            items.append(projected_item)
        projected["coverage"] = items
    if "requirements" in data:
        requirements = data["requirements"]
        if not isinstance(requirements, Mapping) or set(requirements) - {"min_completeness", "min_snr_db", "min_rotation_rpm"}:
            raise ValueError("A resposta degradada contém requisitos inválidos.")
        projected_requirements: dict[str, JsonValue] = {}
        for key, minimum, maximum in (
            ("min_completeness", 0.0, 1.0),
            ("min_snr_db", 0.0, None),
            ("min_rotation_rpm", 0.0, None),
        ):
            if key not in requirements:
                continue
            value = requirements[key]
            if value is None and key == "min_rotation_rpm":
                projected_requirements[key] = None
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
                raise ValueError("A resposta degradada contém requisitos inválidos.")
            projected_requirements[key] = float(value)
        projected["requirements"] = projected_requirements
    flags = {key: data[key] for key in _DEGRADED_FLAGS if key in data}
    if not all(isinstance(value, bool) for value in flags.values()):
        raise ValueError("A resposta degradada contém uma flag inválida.")
    return projected, flags


def _snippet(body: str) -> str:
    return re.sub(r"\s+", " ", body).strip()[:_SNIPPET_LIMIT]


def _summary(document: _KnowledgeDocumentWire) -> KnowledgeSearchItem:
    return KnowledgeSearchItem(
        id=document.id,
        type=document.type,
        title=document.title,
        tags=list(document.tags),
        snippet=_snippet(document.body),
    )


def _validate_degraded_document_item(item: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    allowed = {"id", "type", "title", "body", "tags"}
    if set(item) - allowed:
        raise ValueError("A resposta degradada contém um documento inválido.")
    projected: dict[str, JsonValue] = {}
    if "id" in item:
        projected["id"] = _validate_document_id(item["id"])
    if "type" in item:
        projected["type"] = _DOCUMENT_TYPE.validate_python(item["type"], strict=True)
    if "title" in item:
        title = item["title"]
        if not isinstance(title, str) or not title.strip():
            raise ValueError("A resposta degradada contém título inválido.")
        projected["title"] = title
    if "tags" in item:
        tags = item["tags"]
        if not isinstance(tags, list) or not all(isinstance(tag, str) and tag.strip() for tag in tags):
            raise ValueError("A resposta degradada contém tags inválidas.")
        projected["tags"] = tags
    if "body" in item:
        body = item["body"]
        if not isinstance(body, str):
            raise ValueError("A resposta degradada contém corpo inválido.")
        projected["snippet"] = _snippet(body)
    return projected


def _normalize_search(data: JsonValue, *, document_type: KnowledgeDocumentType | None, degraded: bool) -> tuple[list[KnowledgeSearchItem] | list[dict[str, JsonValue]], dict[str, JsonValue]]:
    assert_safe_partial_json(data)
    if not isinstance(data, Mapping) or "results" not in data or not isinstance(data["results"], list):
        raise ValueError("A resposta de busca contém resultados inválidos.")
    allowed = {"results", *_DEGRADED_FLAGS}
    if set(data) - allowed:
        raise ValueError("A resposta de busca contém um campo de topo inesperado.")
    flags = {key: data[key] for key in _DEGRADED_FLAGS if key in data}
    if not all(isinstance(value, bool) for value in flags.values()):
        raise ValueError("A resposta de busca contém uma flag inválida.")
    seen_ids: set[str] = set()
    normalized: list[KnowledgeSearchItem] | list[dict[str, JsonValue]] = []
    for raw_item in data["results"]:
        if not isinstance(raw_item, Mapping):
            raise ValueError("A resposta de busca contém um documento inválido.")
        if not degraded or set(raw_item) == {"id", "type", "title", "body", "tags"}:
            try:
                document = _KnowledgeDocumentWire.model_validate(raw_item, strict=True)
            except ValidationError as exc:
                raise ValueError("A resposta de busca não corresponde ao contrato do documento.") from exc
            item: KnowledgeSearchItem | dict[str, JsonValue] = _summary(document)
            item_id = document.id
            item_type = document.type
        else:
            item = _validate_degraded_document_item(raw_item)
            item_id = item.get("id")
            item_type = item.get("type")
        if isinstance(item_id, str):
            if item_id in seen_ids:
                raise ValueError("A resposta de busca contém documento duplicado.")
            seen_ids.add(item_id)
        if document_type is not None and item_type is not None and item_type != document_type:
            raise ValueError("A resposta de busca não respeita o filtro de tipo.")
        normalized.append(item)  # type: ignore[arg-type]
    return normalized, flags


def _limited_search(items: list[KnowledgeSearchItem] | list[dict[str, JsonValue]]) -> tuple[list[KnowledgeSearchItem] | list[dict[str, JsonValue]], int, int, int, bool]:
    total = len(items)
    returned = items[:_SEARCH_LIMIT]
    omitted = total - len(returned)
    return returned, total, len(returned), omitted, omitted > 0


def _limited_document(document: _KnowledgeDocumentWire, limit: int) -> KnowledgeDocumentContent:
    body = document.body[:limit]
    omitted = len(document.body) - len(body)
    return KnowledgeDocumentContent(
        id=document.id,
        type=document.type,
        title=document.title,
        body=body,
        tags=list(document.tags),
        returned_body_characters=len(body),
        omitted_body_characters=omitted,
        truncated=omitted > 0,
    )


def _limited_degraded_document(item: Mapping[str, JsonValue], limit: int) -> dict[str, JsonValue]:
    projected = _validate_degraded_document_item(item)
    if "body" in item:
        body = item["body"]
        if not isinstance(body, str):  # already checked, for narrowing only
            raise ValueError("A resposta degradada contém corpo inválido.")
        trimmed = body[:limit]
        projected["body"] = trimmed
        projected["returned_body_characters"] = len(trimmed)
        projected["omitted_body_characters"] = len(body) - len(trimmed)
        projected["truncated"] = len(body) > len(trimmed)
        projected.pop("snippet", None)
    return projected


async def execute_get_model(runtime: ReadToolRuntime) -> GetModelResult:
    _assert_read(runtime, "modelos")
    model_id = _MODEL_ID.validate_python(runtime.configured_model_id, strict=True)
    path = f"/models/{model_id}"
    result = await runtime.client.query(path, response_model=_ModelWire, identity=runtime.identity, params=_params(runtime))
    common = {"tool_name": "get_model", "arguments": {}, "source": ToolSource(kind="industrial_api", resource=path)}
    if isinstance(result, ApiError):
        return GetModelResult(content=None, error=result, artifact=ModelToolArtifact(**common, outcome=ModelToolOutcome(error=result)))
    if result.mode is ResponseMode.COMPLETE:
        model = result.data
        if not isinstance(model, _ModelWire):
            raise TypeError("A resposta completa do modelo não foi validada.")
        if model.id != model_id:
            raise ValueError("A API retornou um modelo diferente do configurado.")
        normalized = _normalize_model(model)
        return GetModelResult(content=normalized, artifact=ModelToolArtifact(**common, outcome=ModelToolOutcome(mode=result.mode, notes=result.notes, model=normalized)))
    model, flags = _validate_degraded_model(result.data, model_id)
    content = DegradedModelContent(mode=result.mode, notes=result.notes, model=model, partial_data=flags)
    return GetModelResult(content=content, artifact=ModelToolArtifact(**common, outcome=ModelToolOutcome(mode=result.mode, notes=result.notes, model=model, partial_data=flags)))


async def execute_search_knowledge(query: object, document_type: object, runtime: ReadToolRuntime) -> SearchKnowledgeResult:
    arguments = _validated_search_arguments(query, document_type)
    _assert_read(runtime, "conhecimento")
    path = "/knowledge/search"
    parameters = {"q": arguments.query}
    if arguments.document_type is not None:
        parameters["type"] = arguments.document_type
    result = await runtime.client.query(path, response_model=dict[str, object], identity=runtime.identity, params=_params(runtime, **parameters))
    common = {"tool_name": "search_knowledge", "arguments": arguments.model_dump(exclude_none=True), "source": ToolSource(kind="industrial_api", resource=path)}
    if isinstance(result, ApiError):
        return SearchKnowledgeResult(content=None, error=result, artifact=KnowledgeSearchToolArtifact(**common, outcome=KnowledgeSearchToolOutcome(error=result)))
    degraded = result.mode is not ResponseMode.COMPLETE
    items, flags = _normalize_search(result.data, document_type=arguments.document_type, degraded=degraded)
    returned, total, returned_count, omitted, truncated = _limited_search(items)
    if not degraded:
        content = KnowledgeSearchModelContent(results=returned, total_results=total, returned_results=returned_count, omitted_results=omitted, truncated=truncated)  # type: ignore[arg-type]
    else:
        degraded_results = [
            item.model_dump(mode="json") if isinstance(item, KnowledgeSearchItem) else item
            for item in returned
        ]
        content = DegradedKnowledgeSearchModelContent(mode=result.mode, notes=result.notes, results=degraded_results, total_results=total, returned_results=returned_count, omitted_results=omitted, truncated=truncated, partial_data=flags)
        returned = degraded_results
    return SearchKnowledgeResult(content=content, artifact=KnowledgeSearchToolArtifact(**common, outcome=KnowledgeSearchToolOutcome(mode=result.mode, notes=result.notes, results=returned, total_results=total, returned_results=returned_count, omitted_results=omitted, partial_data=flags), truncated=truncated, omitted_items=omitted))


async def execute_get_knowledge_document(document_id: object, runtime: ReadToolRuntime) -> GetKnowledgeDocumentResult:
    document_id = _validate_document_id(document_id)
    _assert_read(runtime, "conhecimento")
    path = f"/knowledge/{document_id}"
    result = await runtime.client.query(path, response_model=_KnowledgeDocumentWire, identity=runtime.identity, params=_params(runtime))
    common = {"tool_name": "get_knowledge_document", "arguments": {"document_id": document_id}, "source": ToolSource(kind="industrial_api", resource=path)}
    if isinstance(result, ApiError):
        return GetKnowledgeDocumentResult(content=None, error=result, artifact=KnowledgeDocumentToolArtifact(**common, outcome=KnowledgeDocumentToolOutcome(error=result)))
    if result.mode is ResponseMode.COMPLETE:
        document = result.data
        if not isinstance(document, _KnowledgeDocumentWire):
            raise TypeError("A resposta completa do documento não foi validada.")
        if document.id != document_id:
            raise ValueError("A API retornou um documento diferente do solicitado.")
        content = _limited_document(document, _CONTENT_BODY_LIMIT)
        artifact_document = _limited_document(document, _ARTIFACT_BODY_LIMIT)
        return GetKnowledgeDocumentResult(content=content, artifact=KnowledgeDocumentToolArtifact(**common, outcome=KnowledgeDocumentToolOutcome(mode=result.mode, notes=result.notes, document=artifact_document), truncated=artifact_document.truncated, omitted_items=0))
    assert_safe_partial_json(result.data)
    if not isinstance(result.data, Mapping):
        raise ValueError("A resposta degradada do documento é inválida.")
    allowed = {"id", "type", "title", "body", "tags", *_DEGRADED_FLAGS}
    if set(result.data) - allowed:
        raise ValueError("A resposta degradada contém um campo de topo inesperado.")
    if "id" in result.data:
        returned_id = _validate_document_id(result.data["id"])
        if returned_id != document_id:
            raise ValueError("A API retornou um documento diferente do solicitado.")
    flags = {key: result.data[key] for key in _DEGRADED_FLAGS if key in result.data}
    if not all(isinstance(value, bool) for value in flags.values()):
        raise ValueError("A resposta degradada contém uma flag inválida.")
    document_data = {key: value for key, value in result.data.items() if key not in _DEGRADED_FLAGS}
    content_document = _limited_degraded_document(document_data, _CONTENT_BODY_LIMIT)
    artifact_document = _limited_degraded_document(document_data, _ARTIFACT_BODY_LIMIT)
    content = DegradedKnowledgeDocumentContent(mode=result.mode, notes=result.notes, document=content_document, partial_data=flags)
    return GetKnowledgeDocumentResult(content=content, artifact=KnowledgeDocumentToolArtifact(**common, outcome=KnowledgeDocumentToolOutcome(mode=result.mode, notes=result.notes, document=artifact_document, partial_data=flags), truncated=bool(artifact_document.get("truncated", False)), omitted_items=0))


def _content_and_artifact(*, content: StrictModel | None, artifact: ToolArtifact, error: ApiError | None) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    if error is not None:
        model_content: dict[str, JsonValue] = {"error": error.model_dump(mode="json")}
    elif content is not None:
        model_content = content.model_dump(mode="json")
    else:
        outcome = artifact.outcome
        model_content = {"mode": outcome.mode.value if outcome.mode is not None else None, "notes": outcome.notes, "partial_data": outcome.partial_data}
    return model_content, artifact.model_dump(mode="json")


@tool("get_model", args_schema=_GetModelToolArguments, response_format="content_and_artifact", description="Consulta o modelo de vibração configurado no contexto confiável, sua cobertura, requisitos e estado de processamento.")
async def get_model(runtime: ToolRuntime[ReadToolRuntime]) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    result = await execute_get_model(runtime.context)
    return _content_and_artifact(content=result.content, artifact=result.artifact, error=result.error)


@tool("search_knowledge", args_schema=_SearchKnowledgeToolArguments, response_format="content_and_artifact", description="Busca procedimentos, termos de glossário ou orientações no conhecimento industrial global.")
async def search_knowledge(query: str, runtime: ToolRuntime[ReadToolRuntime], document_type: KnowledgeDocumentType | None = None) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    result = await execute_search_knowledge(query, document_type, runtime.context)
    return _content_and_artifact(content=result.content, artifact=result.artifact, error=result.error)


@tool("get_knowledge_document", args_schema=_GetKnowledgeDocumentToolArguments, response_format="content_and_artifact", description="Consulta o texto de um documento de conhecimento pelo identificador retornado na busca.")
async def get_knowledge_document(document_id: KnowledgeDocumentId, runtime: ToolRuntime[ReadToolRuntime]) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    result = await execute_get_knowledge_document(document_id, runtime.context)
    return _content_and_artifact(content=result.content, artifact=result.artifact, error=result.error)
