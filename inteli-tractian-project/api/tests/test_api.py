"""Testes funcionais da API — espelham os cenários de docs/test-scenarios.md.

Rodam com `pytest` (a partir de api/). Usam TestClient; dados já devem estar
gerados em ../data (rodar `python -m seed_data` antes).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

H_USER_LUCAS = {"x-user-id": "usr_lucas"}      # mechanic, action_low
H_USER_PEDRO = {"x-user-id": "usr_pedro"}      # coordinator, escalate
H_USER_BRUNO = {"x-user-id": "usr_bruno"}      # operator, read only
H_USER_ANA = {"x-user-id": "usr_ana"}          # maintenance_manager, action_high


# ---------------------------------------------------------------------------
# Contexto / Ativos
# ---------------------------------------------------------------------------
def test_get_company():
    r = client.get("/companies/comp_forja_br")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["id"] == "comp_forja_br"
    assert body["mode"] in {"complete", "partial", "conflict", "inconclusive", "unavailable"}


def test_get_company_404():
    assert client.get("/companies/inexistente").status_code == 404


def test_list_assets_by_company():
    r = client.get("/companies/comp_forja_br/assets")
    assert r.status_code == 200
    assets = r.json()["data"]["assets"]
    assert any(a["id"] == "asset_M101" for a in assets)


def test_get_asset_with_points():
    r = client.get("/assets/asset_M101")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["machine_type"] == "motor_induction"
    assert isinstance(data.get("points"), list) and data["points"]


def test_get_asset_404():
    assert client.get("/assets/inexistente").status_code == 404


def test_users_me_requires_header():
    assert client.get("/users/me").status_code == 401


def test_users_me_unknown():
    assert client.get("/users/me", headers={"x-user-id": "usr_fantasma"}).status_code == 401


def test_users_me_ok():
    r = client.get("/users/me", headers=H_USER_LUCAS)
    assert r.status_code == 200
    assert r.json()["role"] == "mechanic"


# ---------------------------------------------------------------------------
# Análises
# ---------------------------------------------------------------------------
def test_list_analyses_by_asset():
    r = client.get("/assets/asset_M205/analyses")
    assert r.status_code == 200
    rows = r.json()["data"].get("analyses", [])
    # pode estar degradado por mode, mas em complete há 2 análises conflitantes
    assert r.json()["mode"] in {"complete", "conflict", "partial", "inconclusive", "unavailable"}


def test_get_analysis_s420_falso_positivo():
    """CEN-03: análise de desbalanceamento com baseline invalidated."""
    r = client.get("/analyses/an_9903")
    assert r.status_code == 200
    data = r.json()["data"]
    # em mode complete os campos vêm; conflito adiciona flag
    if r.json()["mode"] in {"complete", "conflict"}:
        assert data["type"] == "imbalance"
        assert data["detection_mode"] == "baseline"
        assert data["baseline_state_at_detection"] == "invalidated"


def test_get_analysis_lubrification_symptom():
    """CEN-04: lubrificação é detecção sintomática (sem baseline)."""
    r = client.get("/analyses/an_9905")
    assert r.status_code == 200
    data = r.json()["data"]
    if r.json()["mode"] == "complete":
        assert data["type"] == "lubrication"
        assert data["detection_mode"] == "symptom"
        assert data["baseline_state_at_detection"] == "not_applicable"


# ---------------------------------------------------------------------------
# Dados técnicos / baseline
# ---------------------------------------------------------------------------
def test_baseline_established():
    r = client.get("/assets/asset_C710/baseline")
    assert r.status_code == 200
    assert r.json()["data"]["state"] == "established"


def test_baseline_invalidated_after_maintenance():
    r = client.get("/assets/asset_B204/baseline?seed=fixed-b204")
    assert r.status_code == 200
    body = r.json()
    # B204 sem override: mode varia, mas em complete/partial/inconclusive(conflict) o state vem
    if body["mode"] in {"complete", "conflict"}:
        data = body["data"]
        assert data["state"] == "invalidated"
        assert data["invalidation_reason"] == "maintenance_intervention"


def test_baseline_symptom_not_learnable():
    r = client.get("/assets/asset_M208/baseline")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["detection_mode"] == "symptom"
    assert data["learnable"] is False
    assert data["state"] == "learning"


def test_rms_has_alarm_threshold_from_baseline():
    r = client.get("/assets/asset_C710/rms")
    assert r.status_code == 200
    data = r.json()["data"]
    # baseline established -> deve haver threshold derivado (em mode complete/partial de rms)
    if r.json()["mode"] in {"complete", "conflict"}:
        assert data["baseline_state"] == "established"
        assert data["alarm_threshold"] is not None


def test_spectrum_has_peaks():
    r = client.get("/assets/asset_S420/spectrum")
    assert r.status_code == 200
    # em partial, peaks pode ser droppado; em complete/conflict, há picos
    if r.json()["mode"] in {"complete", "conflict"}:
        assert r.json()["data"]["peaks"]


def test_data_quality_low_for_v301():
    r = client.get("/assets/asset_V301/data-quality")
    assert r.status_code == 200
    # V301 tem qualidade baixa (mesmo em partial, completeness vem)
    data = r.json()["data"]
    assert data["completeness"] < 0.7


# ---------------------------------------------------------------------------
# Comportamento probabilístico (overrides do seed.json)
# ---------------------------------------------------------------------------
def test_override_g501_rms_unavailable():
    """CEN-01: G501 tem override rms=unavailable."""
    r = client.get("/assets/asset_G501/rms")
    assert r.json()["mode"] == "unavailable"
    assert r.json()["data"] == {}


def test_override_g501_analyses_inconclusive():
    r = client.get("/assets/asset_G501/analyses")
    assert r.json()["mode"] == "inconclusive"


def test_c710_analyses_has_pending_status():
    """CEN-02: C710 tem análise com status=pending (processamento atrasado)."""
    r = client.get("/assets/asset_C710/analyses?seed=fixed-c710")
    body = r.json()
    # o status=pending é um dado da análise, não um mode do envelope
    if body["mode"] in {"complete", "conflict"}:
        statuses = [a.get("status") for a in body["data"].get("analyses", [])]
        assert "pending" in statuses


def test_seed_determinism():
    """Mesmo seed -> mesmo mode (reprodutibilidade para a Parte 2 / avaliação)."""
    r1 = client.get("/assets/asset_M101/rms?seed=abc123")
    r2 = client.get("/assets/asset_M101/rms?seed=abc123")
    assert r1.json()["mode"] == r2.json()["mode"]


def test_seed_variation():
    """Seeds diferentes podem (não garantido) dar modes diferentes."""
    r1 = client.get("/assets/asset_M101/rms?seed=aaa")
    r2 = client.get("/assets/asset_M101/rms?seed=zzz")
    # pelo menos a API responde consistentemente ambos
    assert r1.status_code == 200 and r2.status_code == 200


def test_seed_complete_forces_complete():
    """seed=complete força modo complete em ativos sem override de cenário."""
    r = client.get("/assets/asset_M101?seed=complete")
    assert r.json()["mode"] == "complete"
    assert r.json()["data"]["machine_type"] == "motor_induction"


def test_seed_complete_does_not_override_scenario():
    """seed=complete NÃO vence overrides de cenário (G501 rms continua unavailable)."""
    r = client.get("/assets/asset_G501/rms?seed=complete")
    assert r.json()["mode"] == "unavailable"


# ---------------------------------------------------------------------------
# Ações de impacto — justificativa e permissões
# ---------------------------------------------------------------------------
def test_reprocess_requires_justification():
    r = client.post("/analyses/an_9906/reprocess", json={"justification": "curto"}, headers=H_USER_LUCAS)
    assert r.status_code == 400
    assert r.json()["code"] == "VALIDATION_ERROR"


def test_reprocess_requires_user():
    r = client.post(
        "/analyses/an_9906/reprocess",
        json={"justification": "justificativa suficientemente longa para passar"},
    )
    assert r.status_code == 401


def test_reprocess_success_no_status_cycle():
    """§5.1: chamada aceita = sucesso, sem ciclo de status."""
    r = client.post(
        "/analyses/an_9906/reprocess",
        json={"justification": "rolamento trocado na bomba B-204; baseline invalidated; RMS sadio"},
        headers=H_USER_LUCAS,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] is True
    assert body["action_id"].startswith("act_")


def test_reprocess_404():
    r = client.post(
        "/analyses/an_xxxx/reprocess",
        json={"justification": "justificativa suficientemente longa para passar"},
        headers=H_USER_LUCAS,
    )
    assert r.status_code == 404


def test_escalate_requires_permission():
    """Operador (read only) não pode escalar."""
    r = client.post(
        "/cases/case_tkt_exe_16/escalate",
        json={"justification": "caso que ultrapassa suporte remoto e exige campo"},
        headers=H_USER_BRUNO,
    )
    assert r.status_code == 403
    assert r.json()["code"] == "FORBIDDEN"


def test_escalate_success():
    r = client.post(
        "/cases/case_tkt_exe_16/escalate",
        json={"justification": "caso que ultrapassa suporte remoto e exige campo"},
        headers=H_USER_PEDRO,
    )
    assert r.status_code == 200
    assert r.json()["accepted"] is True


def test_request_retraining_requires_action_high():
    # Lucas (action_low) não pode retreinamento (action_high)
    r = client.post(
        "/models/mdl_vib_v3/request-retraining",
        json={"justification": "insights sistematicamente errados para spindle de alta rotação"},
        headers=H_USER_LUCAS,
    )
    assert r.status_code == 403


def test_request_retraining_success():
    r = client.post(
        "/models/mdl_vib_v3/request-retraining",
        json={"justification": "insights sistematicamente errados para spindle de alta rotação"},
        headers=H_USER_ANA,
    )
    assert r.status_code == 200


def test_update_asset_config_requires_action_high():
    r = client.patch(
        "/assets/asset_V301",
        json={"justification": "ventilador deixou de ser critico para producao, rebaixar criticidade", "changes": {"criticality": "medium"}},
        headers=H_USER_LUCAS,  # action_low -> 403
    )
    assert r.status_code == 403


def test_update_asset_config_success():
    r = client.patch(
        "/assets/asset_V301",
        json={"justification": "ventilador deixou de ser critico para producao, rebaixar criticidade", "changes": {"criticality": "medium"}},
        headers=H_USER_ANA,  # action_high
    )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Conhecimento
# ---------------------------------------------------------------------------
def test_knowledge_search_returns_results():
    r = client.get("/knowledge/search?q=lubrificacao")
    assert r.status_code == 200
    # conhecimento é estável: mesmo degradado, mantém matches
    results = r.json()["data"].get("results", [])
    assert len(results) >= 1


def test_knowledge_search_glossary():
    r = client.get("/knowledge/search?q=BPFO")
    assert r.status_code == 200
    assert len(r.json()["data"].get("results", [])) >= 1


def test_knowledge_doc_by_id():
    r = client.get("/knowledge/kb_glos_001")
    assert r.status_code == 200
    assert r.json()["data"]["type"] == "glossary"


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def test_model_coverage_motor_dc_no_baseline():
    """CEN-09: motor DC suportado mas can_learn_baseline=false."""
    r = client.get("/models/mdl_vib_v3")
    assert r.status_code == 200
    data = r.json()["data"]
    if r.json()["mode"] in {"complete", "partial"}:
        coverage = {c["machine_type"]: c for c in data["coverage"]}
        assert coverage["motor_dc"]["supported"] is True
        assert coverage["motor_dc"]["can_learn_baseline"] is False


def test_model_processing_state():
    r = client.get("/models/mdl_vib_v3")
    assert r.json()["data"]["processing_state"] == "delayed"
