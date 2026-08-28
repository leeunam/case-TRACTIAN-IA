"""API industrial simplificada — Challenge TRACTIAN x Inteli.

Implementa o contrato docs/api-contract.openapi.yaml. GETs de consulta retornam um
envelope probabilístico {mode, notes, data}; ações de impacto exigem justificativa.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import store
from .models import ActionResult
from .prob import Mode, envelope, resolve_mode

app = FastAPI(
    title="Tractian Industrial Support API (Challenge Inteli)",
    version="0.1.0",
    description=(
        "API industrial simplificada. GETs retornam envelope probabilístico {mode, notes, data}. "
        "Ações de impacto exigem justificativa (>=20 chars). Contexto de usuário via header x-user-id."
    ),
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ---------------------------------------------------------------------------
# Dependências
# ---------------------------------------------------------------------------
def user_id_dep(x_user_id: str | None = Header(default=None, alias="x-user-id")) -> str | None:
    """Resolve o usuário do contexto. Sem header = contexto anônimo (só leitura)."""
    return x_user_id


def require_permission(permission: str):
    """Dependência que exige que o usuário tenha uma permissão."""

    def _dep(x_user_id: str | None = Header(default=None, alias="x-user-id")) -> dict[str, Any]:
        if not x_user_id:
            raise HTTPException(401, "Header x-user-id ausente.")
        user = store.get_user(x_user_id)
        if not user:
            raise HTTPException(401, f"Usuário desconhecido: {x_user_id}")
        perms = user.get("permissions") or []
        if permission not in perms:
            raise HTTPException(403, f"Permissão necessária: {permission}")
        return user

    return _dep


def _seed(seed: str | None = Query(default=None, description="Semente para comportamento probabilístico")) -> str | None:
    return seed


def _mode_for(resource: str, category: str, seed: str | None) -> Mode:
    cfg = store.seed_config()
    return resolve_mode(
        resource=resource,
        category=category,
        seed=seed,
        overrides=cfg.get("overrides"),
        distribution=cfg.get("distribution"),
    )


# ---------------------------------------------------------------------------
# Helpers de envelope por modo
# ---------------------------------------------------------------------------
def _apply_mode(payload: dict[str, Any], mode: Mode, category: str) -> tuple[dict[str, Any], str | None]:
    """Degradou o payload conforme o modo. Retorna (data, notes).

    Conhecimento e empresa são fontes estáveis: degradam só para `partial` (trunca o body)
    e nunca somem por inteiro, para não confundir buscas que têm match real.
    """
    stable = category in ("knowledge", "company", "assets")

    if mode is Mode.COMPLETE:
        return payload, None
    if mode is Mode.PARTIAL:
        trimmed = {k: v for k, v in payload.items() if k not in _PARTIAL_DROP.get(category, ())}
        missing = [k for k in payload if k not in trimmed]
        return trimmed, f"Informação parcial: campos ausentes {missing or '(detalhes)'}"
    if mode is Mode.INCONCLUSIVE:
        if stable:
            return payload, "Resultado inconclusivo: verifique fontes complementares."
        return (
            {"inconclusive": True, **{k: v for k, v in payload.items() if k == "asset_id"}},
            "Resultado inconclusivo: dados insuficientes para concluir.",
        )
    if mode is Mode.CONFLICT:
        payload = {**payload, "conflict": True}
        return payload, "Conflito entre fontes: verifique análises especializadas."
    if mode is Mode.UNAVAILABLE:
        if stable:
            return payload, "Indisponibilidade temporária parcial; dados podem estar incompletos."
        return {}, "Indisponibilidade temporária: recurso não pôde ser recuperado."
    raise AssertionError("modo não tratado")


_PARTIAL_DROP: dict[str, tuple[str, ...]] = {
    "analyses": ("evidence", "limitations"),
    "baseline": ("features",),
    "data_quality": ("freshness_minutes",),
    "rms": ("samples",),
    "model": ("requirements", "last_run_at"),
}


# ---------------------------------------------------------------------------
# Contexto
# ---------------------------------------------------------------------------
@app.get("/companies/{company_id}", tags=["Contexto"])
def get_company(company_id: str, seed: str | None = Depends(_seed)):
    company = store.get_company(company_id)
    if not company:
        raise HTTPException(404, "Empresa não encontrada.")
    mode = _mode_for(company_id, "company", seed)
    data, notes = _apply_mode(company, mode, "company")
    return envelope(data, mode, notes)


@app.get("/companies/{company_id}/assets", tags=["Contexto"])
def list_assets_by_company(company_id: str, seed: str | None = Depends(_seed)):
    if not store.get_company(company_id):
        raise HTTPException(404, "Empresa não encontrada.")
    assets = store.list_assets_by_company(company_id)
    mode = _mode_for(company_id, "assets", seed)
    data, notes = _apply_mode({"assets": assets}, mode, "assets")
    return envelope(data, mode, notes)


@app.get("/users/me", tags=["Contexto"])
def get_current_user(user_id: str | None = Depends(user_id_dep)):
    if not user_id:
        raise HTTPException(401, "Header x-user-id ausente.")
    user = store.get_user(user_id)
    if not user:
        raise HTTPException(401, f"Usuário desconhecido: {user_id}")
    return user


# ---------------------------------------------------------------------------
# Ativos
# ---------------------------------------------------------------------------
@app.get("/assets/{asset_id}", tags=["Ativos"])
def get_asset(asset_id: str, seed: str | None = Depends(_seed)):
    asset = store.get_asset(asset_id)
    if not asset:
        raise HTTPException(404, "Ativo não encontrado.")
    asset = {**asset, "points": store.list_points(asset_id)}
    mode = _mode_for(asset_id, "asset", seed)
    data, notes = _apply_mode(asset, mode, "asset")
    return envelope(data, mode, notes)


@app.patch("/assets/{asset_id}", tags=["Ativos"])
def update_asset_config(
    asset_id: str,
    body: dict[str, Any],
    user: dict[str, Any] = Depends(require_permission("action_high")),
):
    if not store.get_asset(asset_id):
        raise HTTPException(404, "Ativo não encontrado.")
    _require_justification(body)
    return ActionResult(
        action_id=f"act_{uuid.uuid4().hex[:8]}",
        message=f"Configuração do ativo {asset_id} atualizada por {user['name']}.",
    ).model_dump()


# ---------------------------------------------------------------------------
# Análises
# ---------------------------------------------------------------------------
@app.get("/assets/{asset_id}/analyses", tags=["Análises"])
def list_analyses(asset_id: str, status: str | None = Query(default=None), seed: str | None = Depends(_seed)):
    if not store.get_asset(asset_id):
        raise HTTPException(404, "Ativo não encontrado.")
    rows = store.list_analyses(asset_id, status)
    mode = _mode_for(asset_id, "analyses", seed)
    data, notes = _apply_mode({"analyses": rows}, mode, "analyses")
    return envelope(data, mode, notes)


@app.get("/analyses/{analysis_id}", tags=["Análises"])
def get_analysis(analysis_id: str, seed: str | None = Depends(_seed)):
    analysis = store.get_analysis(analysis_id)
    if not analysis:
        raise HTTPException(404, "Análise não encontrada.")
    mode = _mode_for(analysis["asset_id"], "analyses", seed)
    data, notes = _apply_mode(analysis, mode, "analyses")
    return envelope(data, mode, notes)


@app.post("/analyses/{analysis_id}/reprocess", tags=["Análises"])
def reprocess_analysis(
    analysis_id: str,
    body: dict[str, Any],
    user: dict[str, Any] = Depends(require_permission("action_low")),
):
    analysis = store.get_analysis(analysis_id)
    if not analysis:
        raise HTTPException(404, "Análise não encontrada.")
    _require_justification(body)
    return ActionResult(
        action_id=f"act_{uuid.uuid4().hex[:8]}",
        message=f"Reprocesso da análise {analysis_id} aceito. Sucesso sem ciclo de status.",
    ).model_dump()


@app.post("/analyses/{analysis_id}/request-specialist", tags=["Análises"])
def request_specialist_analysis(
    analysis_id: str,
    body: dict[str, Any],
    user: dict[str, Any] = Depends(require_permission("action_low")),
):
    analysis = store.get_analysis(analysis_id)
    if not analysis:
        raise HTTPException(404, "Análise não encontrada.")
    _require_justification(body)
    return ActionResult(
        action_id=f"act_{uuid.uuid4().hex[:8]}",
        message=f"Análise especializada solicitada para {analysis_id}.",
    ).model_dump()


# ---------------------------------------------------------------------------
# Dados técnicos
# ---------------------------------------------------------------------------
@app.get("/assets/{asset_id}/baseline", tags=["Dados técnicos"])
def get_baseline(asset_id: str, point_id: str | None = Query(default=None), seed: str | None = Depends(_seed)):
    if not store.get_asset(asset_id):
        raise HTTPException(404, "Ativo não encontrado.")
    baseline = store.get_baseline(asset_id, point_id)
    mode = _mode_for(asset_id, "baseline", seed)
    if baseline is None:
        return envelope({"baseline": None}, Mode.INCONCLUSIVE, "Nenhum baseline para este ativo/ponto.")
    data, notes = _apply_mode(baseline, mode, "baseline")
    return envelope(data, mode, notes)


@app.get("/assets/{asset_id}/rms", tags=["Dados técnicos"])
def get_rms(asset_id: str, point_id: str | None = Query(default=None), seed: str | None = Depends(_seed)):
    if not store.get_asset(asset_id):
        raise HTTPException(404, "Ativo não encontrado.")
    baseline = store.get_baseline(asset_id, point_id)
    rows = store.get_rms(asset_id, point_id)
    payload = {
        "asset_id": asset_id,
        "point_id": point_id or (rows[0]["point_id"] if rows else None),
        "unit": "mm/s",
        "baseline_reference": (baseline.get("features") or [{}])[0].get("reference") if baseline else None,
        "baseline_state": baseline.get("state") if baseline else "not_applicable",
        "alarm_threshold": _alarm_threshold(baseline),
        "samples": [{"ts": r["ts"], "value": r["value"]} for r in rows],
    }
    mode = _mode_for(asset_id, "rms", seed)
    data, notes = _apply_mode(payload, mode, "rms")
    return envelope(data, mode, notes)


def _alarm_threshold(baseline: dict[str, Any] | None) -> float | None:
    if not baseline:
        return None
    feats = baseline.get("features") or []
    for f in feats:
        if f.get("feature") == "rms_mm_s":
            return round(f.get("reference", 0) + f.get("tolerance", 0), 3)
    return None


@app.get("/assets/{asset_id}/spectrum", tags=["Dados técnicos"])
def get_spectrum(asset_id: str, point_id: str | None = Query(default=None), seed: str | None = Depends(_seed)):
    if not store.get_asset(asset_id):
        raise HTTPException(404, "Ativo não encontrado.")
    spectrum = store.get_spectrum(asset_id, point_id)
    mode = _mode_for(asset_id, "spectrum", seed)
    if spectrum is None:
        return envelope({"spectrum": None}, Mode.INCONCLUSIVE, "Sem espectro disponível.")
    data, notes = _apply_mode(spectrum, mode, "spectrum")
    return envelope(data, mode, notes)


@app.get("/assets/{asset_id}/data-quality", tags=["Dados técnicos"])
def get_data_quality(asset_id: str, point_id: str | None = Query(default=None), seed: str | None = Depends(_seed)):
    if not store.get_asset(asset_id):
        raise HTTPException(404, "Ativo não encontrado.")
    dq = store.get_data_quality(asset_id, point_id)
    mode = _mode_for(asset_id, "data_quality", seed)
    if dq is None:
        return envelope({"data_quality": None}, Mode.INCONCLUSIVE, "Sem dados de qualidade.")
    data, notes = _apply_mode(dq, mode, "data_quality")
    return envelope(data, mode, notes)


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
@app.get("/models/{model_id}", tags=["Modelos"])
def get_model(model_id: str, seed: str | None = Depends(_seed)):
    model = store.get_model(model_id)
    if not model:
        raise HTTPException(404, "Modelo não encontrado.")
    # Agrupa campos de requisito no objeto `requirements` (alinha com o contrato OpenAPI)
    model = {
        k: v for k, v in model.items()
        if k not in ("min_completeness", "min_snr_db", "min_rotation_rpm")
    }
    model["requirements"] = {
        "min_completeness": store.get_model(model_id)["min_completeness"],
        "min_snr_db": store.get_model(model_id)["min_snr_db"],
        "min_rotation_rpm": store.get_model(model_id)["min_rotation_rpm"],
    }
    mode = _mode_for(model_id, "model", seed)
    data, notes = _apply_mode(model, mode, "model")
    return envelope(data, mode, notes)


@app.post("/models/{model_id}/request-retraining", tags=["Modelos"])
def request_retraining(
    model_id: str,
    body: dict[str, Any],
    user: dict[str, Any] = Depends(require_permission("action_high")),
):
    if not store.get_model(model_id):
        raise HTTPException(404, "Modelo não encontrado.")
    _require_justification(body)
    return ActionResult(
        action_id=f"act_{uuid.uuid4().hex[:8]}",
        message=f"Retreinamento solicitado para o modelo {model_id}.",
    ).model_dump()


# ---------------------------------------------------------------------------
# Conhecimento
# ---------------------------------------------------------------------------
@app.get("/knowledge/search", tags=["Conhecimento"])
def search_knowledge(q: str = Query(...), type: str | None = Query(default=None), seed: str | None = Depends(_seed)):
    rows = store.search_knowledge(q, type)
    mode = _mode_for(f"knowledge:{q}", "knowledge", seed)
    data, notes = _apply_mode({"results": rows}, mode, "knowledge")
    return envelope(data, mode, notes)


@app.get("/knowledge/{doc_id}", tags=["Conhecimento"])
def get_knowledge_doc(doc_id: str, seed: str | None = Depends(_seed)):
    doc = store.get_knowledge_doc(doc_id)
    if not doc:
        raise HTTPException(404, "Documento não encontrado.")
    mode = _mode_for(f"knowledge:{doc_id}", "knowledge", seed)
    data, notes = _apply_mode(doc, mode, "knowledge")
    return envelope(data, mode, notes)


# ---------------------------------------------------------------------------
# Ações / Escalonamento
# ---------------------------------------------------------------------------
@app.post("/cases/{case_id}/escalate", tags=["Ações"])
def escalate_case(
    case_id: str,
    body: dict[str, Any],
    user: dict[str, Any] = Depends(require_permission("escalate")),
):
    case = store.get_case(case_id)
    if not case:
        raise HTTPException(404, "Caso não encontrado.")
    _require_justification(body)
    return ActionResult(
        action_id=f"act_{uuid.uuid4().hex[:8]}",
        message=f"Caso {case_id} escalado para análise humana por {user['name']}.",
    ).model_dump()


# ---------------------------------------------------------------------------
# Validação de justificativa
# ---------------------------------------------------------------------------
def _require_justification(body: dict[str, Any]) -> None:
    justification = (body or {}).get("justification")
    if not justification or not isinstance(justification, str) or len(justification.strip()) < 20:
        raise HTTPException(
            400,
            "Justificativa ausente ou inválida (mínimo 20 caracteres).",
        )


@app.exception_handler(HTTPException)
async def _http_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": _err_code(exc.status_code), "message": exc.detail},
    )


def _err_code(status: int) -> str:
    return {400: "VALIDATION_ERROR", 401: "UNAUTHORIZED", 403: "FORBIDDEN", 404: "NOT_FOUND"}.get(
        status, "ERROR"
    )
