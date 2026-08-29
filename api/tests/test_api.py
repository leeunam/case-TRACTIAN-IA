"""Testes funcionais da API — espelham os cenários de docs/test-scenarios.md.

Rodam com `pytest` (a partir de api/). Usam TestClient; dados já devem estar
gerados em ../data (rodar `python -m seed_data` antes).
"""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event, Lock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.idempotency import (
    IdempotencyStore,
    ReservationResult,
    canonical_payload_hash,
    get_idempotency_store,
)
from app.main import app

client = TestClient(app)

H_USER_LUCAS = {"x-user-id": "usr_lucas"}      # mechanic, action_low
H_USER_PEDRO = {"x-user-id": "usr_pedro"}      # coordinator, escalate
H_USER_BRUNO = {"x-user-id": "usr_bruno"}      # operator, read only
H_USER_ANA = {"x-user-id": "usr_ana"}          # maintenance_manager, action_high


@pytest.fixture(autouse=True)
def isolated_idempotency_store(tmp_path):
    test_store = IdempotencyStore(tmp_path / "idempotency.sqlite3")
    app.dependency_overrides[get_idempotency_store] = lambda: test_store
    yield test_store
    app.dependency_overrides.pop(get_idempotency_store, None)


def test_idempotency_store_uses_environment_configuration(tmp_path, monkeypatch):
    configured_path = tmp_path / "configured" / "idempotency.sqlite3"
    monkeypatch.setenv("IDEMPOTENCY_DB_PATH", str(configured_path))
    monkeypatch.setenv("IDEMPOTENCY_PROCESSING_TIMEOUT_SECONDS", "123")
    get_idempotency_store.cache_clear()

    try:
        configured_store = get_idempotency_store()
        assert configured_store.database_path == configured_path
        assert configured_store.processing_timeout_seconds == 123
        assert configured_path.exists()
    finally:
        get_idempotency_store.cache_clear()


def test_idempotency_scope_separates_user_method_and_endpoint(
    isolated_idempotency_store,
):
    payload_hash = canonical_payload_hash(
        {"justification": "rolamento substituído; solicitar novo processamento"}
    )
    common = {
        "idempotency_key": "tractian-agent:shared-across-scopes",
        "payload_hash": payload_hash,
    }

    reservations = [
        isolated_idempotency_store.reserve(
            **common,
            user_id="usr_lucas",
            method="POST",
            endpoint="/analyses/an_9906/reprocess",
        ),
        isolated_idempotency_store.reserve(
            **common,
            user_id="usr_ana",
            method="POST",
            endpoint="/analyses/an_9906/reprocess",
        ),
        isolated_idempotency_store.reserve(
            **common,
            user_id="usr_lucas",
            method="PUT",
            endpoint="/analyses/an_9906/reprocess",
        ),
        isolated_idempotency_store.reserve(
            **common,
            user_id="usr_lucas",
            method="POST",
            endpoint="/analyses/an_other/reprocess",
        ),
    ]

    assert [reservation.decision for reservation in reservations] == [
        "execute",
        "execute",
        "execute",
        "execute",
    ]


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
# Chave de idempotência — validação do cabeçalho de reprocesso
# ---------------------------------------------------------------------------


def test_reprocess_requires_idempotency_key():
    r = client.post(
        "/analyses/an_9906/reprocess",
        json={"justification": "rolamento substituído; solicitar novo processamento"},
        headers=H_USER_LUCAS,
    )
    assert r.status_code == 400
    assert r.json()["code"] == "VALIDATION_ERROR"
    assert "Idempotency-Key" in r.json()["message"]


def test_reprocess_rejects_blank_idempotency_key():
    r = client.post(
        "/analyses/an_9906/reprocess",
        json={"justification": "rolamento substituído; solicitar novo processamento"},
        headers={
            **H_USER_LUCAS,
            "Idempotency-Key": "  ",
        },
    )
    assert r.status_code == 400
    assert r.json()["code"] == "VALIDATION_ERROR"
    assert "Idempotency-Key" in r.json()["message"]


@pytest.mark.parametrize(
    "invalid_key",
    [
        "tractian-agent:chave com espaco",
        "x" * 256,
    ],
)
def test_reprocess_rejects_invalid_idempotency_key_format(invalid_key):
    r = client.post(
        "/analyses/an_9906/reprocess",
        json={"justification": "rolamento substituído; solicitar novo processamento"},
        headers={
            **H_USER_LUCAS,
            "Idempotency-Key": invalid_key,
        },
    )
    assert r.status_code == 400
    assert r.json()["code"] == "VALIDATION_ERROR"
    assert "Idempotency-Key" in r.json()["message"]


def test_reprocess_accepts_255_character_idempotency_key():
    r = client.post(
        "/analyses/an_9906/reprocess",
        json={"justification": "rolamento substituído; solicitar novo processamento"},
        headers={
            **H_USER_LUCAS,
            "Idempotency-Key": "x" * 255,
        },
    )
    assert r.status_code == 200


def test_reprocess_openapi_documents_idempotency_errors():
    operation = app.openapi()["paths"]["/analyses/{analysis_id}/reprocess"]["post"]
    validation_response = operation["responses"]["400"]
    conflict_response = operation["responses"]["409"]
    internal_error_response = operation["responses"]["500"]

    assert validation_response["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/Error"
    }
    conflict_content = conflict_response["content"]["application/json"]
    assert conflict_content["schema"] == {"$ref": "#/components/schemas/Error"}
    assert internal_error_response["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/Error"
    }
    documented_codes = {
        example["value"]["code"]
        for example in conflict_content["examples"].values()
    }
    assert documented_codes == {
        "IDEMPOTENCY_PAYLOAD_CONFLICT",
        "IDEMPOTENCY_IN_PROGRESS",
        "IDEMPOTENCY_OUTCOME_UNKNOWN",
    }


# ---------------------------------------------------------------------------
# Ações de impacto — justificativa e permissões
# ---------------------------------------------------------------------------
def test_reprocess_requires_justification():
    r = client.post(
        "/analyses/an_9906/reprocess",
        json={"justification": "curto"},
        headers={
            **H_USER_LUCAS,
            "Idempotency-Key": "test-reprocess-short-justification",
        },
    )
    assert r.status_code == 400
    assert r.json()["code"] == "VALIDATION_ERROR"


def test_reprocess_requires_user():
    r = client.post(
        "/analyses/an_9906/reprocess",
        json={"justification": "justificativa suficientemente longa para passar"},
        headers={"Idempotency-Key": "test-reprocess-missing-user"},
    )
    assert r.status_code == 401


def test_reprocess_success_no_status_cycle():
    """§5.1: chamada aceita = sucesso, sem ciclo de status."""
    r = client.post(
        "/analyses/an_9906/reprocess",
        json={"justification": "rolamento trocado na bomba B-204; baseline invalidated; RMS sadio"},
        headers={
            **H_USER_LUCAS,
            "Idempotency-Key": "test-reprocess-success",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] is True
    assert body["action_id"].startswith("act_")


def test_reprocess_replays_original_response():
    headers = {
        **H_USER_LUCAS,
        "Idempotency-Key": "tractian-agent:11111111-1111-4111-8111-111111111111",
    }
    payload = {
        "justification": "rolamento substituído; solicitar novo processamento",
    }

    first_response = client.post(
        "/analyses/an_9906/reprocess",
        json=payload,
        headers=headers,
    )
    retry_response = client.post(
        "/analyses/an_9906/reprocess",
        json=payload,
        headers=headers,
    )

    assert first_response.status_code == 200
    assert retry_response.status_code == 200
    assert retry_response.json() == first_response.json()


def test_reprocess_replays_payload_with_different_key_order(monkeypatch):
    headers = {
        **H_USER_LUCAS,
        "Idempotency-Key": "tractian-agent:canonical-payload",
    }
    first_payload = {
        "justification": "rolamento substituído; solicitar novo processamento",
        "context": {"source": "maintenance", "priority": 2},
    }
    reordered_payload = {
        "context": {"priority": 2, "source": "maintenance"},
        "justification": "rolamento substituído; solicitar novo processamento",
    }
    action_call_count = 0
    original_uuid4 = main_module.uuid.uuid4

    def counted_uuid4():
        nonlocal action_call_count
        action_call_count += 1
        return original_uuid4()

    monkeypatch.setattr(main_module.uuid, "uuid4", counted_uuid4)
    first_response = client.post(
        "/analyses/an_9906/reprocess",
        json=first_payload,
        headers=headers,
    )
    retry_response = client.post(
        "/analyses/an_9906/reprocess",
        json=reordered_payload,
        headers=headers,
    )

    assert first_response.status_code == 200
    assert retry_response.status_code == 200
    assert retry_response.json() == first_response.json()
    assert action_call_count == 1


def test_reprocess_treats_key_case_as_distinct(monkeypatch):
    payload = {
        "justification": "rolamento substituído; solicitar novo processamento",
    }
    generated_ids = iter(
        [
            UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
        ]
    )
    action_call_count = 0

    def deterministic_uuid4():
        nonlocal action_call_count
        action_call_count += 1
        return next(generated_ids)

    monkeypatch.setattr(main_module.uuid, "uuid4", deterministic_uuid4)
    lowercase_response = client.post(
        "/analyses/an_9906/reprocess",
        json=payload,
        headers={
            **H_USER_LUCAS,
            "Idempotency-Key": "tractian-agent:case-sensitive",
        },
    )
    uppercase_response = client.post(
        "/analyses/an_9906/reprocess",
        json=payload,
        headers={
            **H_USER_LUCAS,
            "Idempotency-Key": "tractian-agent:CASE-SENSITIVE",
        },
    )

    assert lowercase_response.status_code == 200
    assert uppercase_response.status_code == 200
    assert uppercase_response.json() != lowercase_response.json()
    assert action_call_count == 2


def test_reprocess_replay_survives_store_recreation(isolated_idempotency_store):
    headers = {
        **H_USER_LUCAS,
        "Idempotency-Key": "tractian-agent:33333333-3333-4333-8333-333333333333",
    }
    payload = {
        "justification": "rolamento substituído; solicitar novo processamento",
    }

    first_response = client.post(
        "/analyses/an_9906/reprocess",
        json=payload,
        headers=headers,
    )

    recreated_store = IdempotencyStore(isolated_idempotency_store.database_path)
    app.dependency_overrides[get_idempotency_store] = lambda: recreated_store
    retry_response = client.post(
        "/analyses/an_9906/reprocess",
        json=payload,
        headers=headers,
    )

    assert first_response.status_code == 200
    assert retry_response.status_code == 200
    assert retry_response.json() == first_response.json()


def test_reprocess_rejects_same_key_with_different_payload():
    headers = {
        **H_USER_LUCAS,
        "Idempotency-Key": "tractian-agent:22222222-2222-4222-8222-222222222222",
    }
    first_payload = {
        "justification": "rolamento substituído; solicitar novo processamento",
    }
    changed_payload = {
        "justification": "baseline invalidado; solicitar novo processamento",
    }

    first_response = client.post(
        "/analyses/an_9906/reprocess",
        json=first_payload,
        headers=headers,
    )
    conflict_response = client.post(
        "/analyses/an_9906/reprocess",
        json=changed_payload,
        headers=headers,
    )

    assert first_response.status_code == 200
    assert conflict_response.status_code == 409
    assert conflict_response.json()["code"] == "IDEMPOTENCY_PAYLOAD_CONFLICT"


def test_reprocess_concurrent_retry_does_not_duplicate_action(monkeypatch):
    headers = {
        **H_USER_LUCAS,
        "Idempotency-Key": "tractian-agent:44444444-4444-4444-8444-444444444444",
    }
    payload = {
        "justification": "rolamento substituído; solicitar novo processamento",
    }
    action_started = Event()
    release_action = Event()
    call_lock = Lock()
    call_count = 0
    original_uuid4 = main_module.uuid.uuid4

    def controlled_uuid4():
        nonlocal call_count
        with call_lock:
            call_count += 1
            is_first_call = call_count == 1
        if is_first_call:
            action_started.set()
            assert release_action.wait(timeout=5)
        return original_uuid4()

    monkeypatch.setattr(main_module.uuid, "uuid4", controlled_uuid4)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(
            client.post,
            "/analyses/an_9906/reprocess",
            json=payload,
            headers=headers,
        )
        assert action_started.wait(timeout=5)
        concurrent_response = client.post(
            "/analyses/an_9906/reprocess",
            json=payload,
            headers=headers,
        )
        release_action.set()
        first_response = first_future.result(timeout=5)

    replay_response = client.post(
        "/analyses/an_9906/reprocess",
        json=payload,
        headers=headers,
    )

    assert first_response.status_code == 200
    assert concurrent_response.status_code == 409
    assert concurrent_response.json()["code"] == "IDEMPOTENCY_IN_PROGRESS"
    assert replay_response.json() == first_response.json()
    assert call_count == 1


def test_reprocess_unknown_reservation_decision_fails_closed(
    isolated_idempotency_store,
    monkeypatch,
):
    headers = {
        **H_USER_LUCAS,
        "Idempotency-Key": "tractian-agent:unknown-decision",
    }
    payload = {
        "justification": "rolamento substituído; solicitar novo processamento",
    }
    original_reserve = isolated_idempotency_store.reserve
    action_call_count = 0

    def reserve_with_unknown_decision(**kwargs):
        reservation = original_reserve(**kwargs)
        return ReservationResult("unexpected", reservation.record)

    def unexpected_uuid4():
        nonlocal action_call_count
        action_call_count += 1
        raise AssertionError("uma decisão desconhecida não deve liberar a ação")

    monkeypatch.setattr(
        isolated_idempotency_store,
        "reserve",
        reserve_with_unknown_decision,
    )
    monkeypatch.setattr(main_module.uuid, "uuid4", unexpected_uuid4)
    client_without_server_exceptions = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = client_without_server_exceptions.post(
        "/analyses/an_9906/reprocess",
        json=payload,
        headers=headers,
    )

    assert response.status_code == 500
    assert action_call_count == 0


def test_reprocess_failure_marks_outcome_uncertain_and_blocks_retry(monkeypatch):
    headers = {
        **H_USER_LUCAS,
        "Idempotency-Key": "tractian-agent:55555555-5555-4555-8555-555555555555",
    }
    payload = {
        "justification": "rolamento substituído; solicitar novo processamento",
    }
    call_count = 0

    def failing_uuid4():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("falha simulada durante a ação")

    monkeypatch.setattr(main_module.uuid, "uuid4", failing_uuid4)
    client_without_server_exceptions = TestClient(
        app,
        raise_server_exceptions=False,
    )

    first_response = client_without_server_exceptions.post(
        "/analyses/an_9906/reprocess",
        json=payload,
        headers=headers,
    )
    retry_response = client_without_server_exceptions.post(
        "/analyses/an_9906/reprocess",
        json=payload,
        headers=headers,
    )

    assert first_response.status_code == 500
    assert first_response.json() == {
        "code": "INTERNAL_ERROR",
        "message": "Erro interno durante o processamento.",
    }
    assert retry_response.status_code == 409
    assert retry_response.json()["code"] == "IDEMPOTENCY_OUTCOME_UNKNOWN"
    assert call_count == 1


def test_reprocess_stale_processing_becomes_uncertain(
    isolated_idempotency_store,
    monkeypatch,
):
    headers = {
        **H_USER_LUCAS,
        "Idempotency-Key": "tractian-agent:66666666-6666-4666-8666-666666666666",
    }
    payload = {
        "justification": "rolamento substituído; solicitar novo processamento",
    }
    payload_hash = canonical_payload_hash(payload)
    isolated_idempotency_store.reserve(
        idempotency_key=headers["Idempotency-Key"],
        user_id=H_USER_LUCAS["x-user-id"],
        method="POST",
        endpoint="/analyses/an_9906/reprocess",
        payload_hash=payload_hash,
    )
    stale_time = datetime.now(timezone.utc) - timedelta(
        seconds=isolated_idempotency_store.processing_timeout_seconds + 1
    )
    with sqlite3.connect(isolated_idempotency_store.database_path) as connection:
        connection.execute(
            "UPDATE idempotency_records SET updated_at = ?",
            (stale_time.isoformat(),),
        )

    action_call_count = 0

    def unexpected_uuid4():
        nonlocal action_call_count
        action_call_count += 1
        raise AssertionError("uma reserva vencida não deve repetir a ação")

    monkeypatch.setattr(main_module.uuid, "uuid4", unexpected_uuid4)
    retry_response = client.post(
        "/analyses/an_9906/reprocess",
        json=payload,
        headers=headers,
    )
    stored_record = isolated_idempotency_store.find(
        idempotency_key=headers["Idempotency-Key"],
        user_id=H_USER_LUCAS["x-user-id"],
        method="POST",
        endpoint="/analyses/an_9906/reprocess",
    )

    assert retry_response.status_code == 409
    assert retry_response.json()["code"] == "IDEMPOTENCY_OUTCOME_UNKNOWN"
    assert stored_record is not None
    assert stored_record.status == "uncertain"
    assert action_call_count == 0


def test_reprocess_expired_record_allows_new_execution(
    isolated_idempotency_store,
    monkeypatch,
):
    headers = {
        **H_USER_LUCAS,
        "Idempotency-Key": "tractian-agent:77777777-7777-4777-8777-777777777777",
    }
    payload = {
        "justification": "rolamento substituído; solicitar novo processamento",
    }
    action_call_count = 0
    generated_ids = iter(
        [
            UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc"),
            UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd"),
        ]
    )

    def counted_uuid4():
        nonlocal action_call_count
        action_call_count += 1
        return next(generated_ids)

    monkeypatch.setattr(main_module.uuid, "uuid4", counted_uuid4)
    first_response = client.post(
        "/analyses/an_9906/reprocess",
        json=payload,
        headers=headers,
    )
    expired_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    with sqlite3.connect(isolated_idempotency_store.database_path) as connection:
        connection.execute(
            "UPDATE idempotency_records SET expires_at = ?",
            (expired_at.isoformat(),),
        )

    response_after_expiry = client.post(
        "/analyses/an_9906/reprocess",
        json=payload,
        headers=headers,
    )

    assert first_response.status_code == 200
    assert response_after_expiry.status_code == 200
    assert response_after_expiry.json() != first_response.json()
    assert action_call_count == 2


def test_expired_generation_cannot_modify_new_reservation(
    isolated_idempotency_store,
):
    scope = {
        "idempotency_key": "tractian-agent:generation-guard",
        "user_id": H_USER_LUCAS["x-user-id"],
        "method": "POST",
        "endpoint": "/analyses/an_9906/reprocess",
    }
    payload_hash = canonical_payload_hash(
        {"justification": "rolamento substituído; solicitar novo processamento"}
    )
    isolated_idempotency_store.reserve(
        **scope,
        payload_hash=payload_hash,
    )
    old_created_at = datetime.now(timezone.utc) - timedelta(days=8)
    old_expires_at = datetime.now(timezone.utc) - timedelta(days=1)
    with sqlite3.connect(isolated_idempotency_store.database_path) as connection:
        connection.execute(
            """
            UPDATE idempotency_records
            SET created_at = ?, updated_at = ?, expires_at = ?
            """,
            (
                old_created_at.isoformat(),
                old_created_at.isoformat(),
                old_expires_at.isoformat(),
            ),
        )

    new_reservation = isolated_idempotency_store.reserve(
        **scope,
        payload_hash=payload_hash,
    )
    assert new_reservation.decision == "execute"

    with pytest.raises(RuntimeError):
        isolated_idempotency_store.complete(
            **scope,
            payload_hash=payload_hash,
            reservation_created_at=old_created_at.isoformat(),
            response_status=200,
            response_body={"action_id": "act_old"},
        )
    isolated_idempotency_store.mark_uncertain(
        **scope,
        payload_hash=payload_hash,
        reservation_created_at=old_created_at.isoformat(),
    )
    record_after_old_worker = isolated_idempotency_store.find(**scope)

    assert record_after_old_worker is not None
    assert record_after_old_worker.status == "processing"
    assert record_after_old_worker.created_at == new_reservation.record.created_at

    isolated_idempotency_store.complete(
        **scope,
        payload_hash=payload_hash,
        reservation_created_at=new_reservation.record.created_at,
        response_status=200,
        response_body={"action_id": "act_new"},
    )
    final_record = isolated_idempotency_store.find(**scope)
    assert final_record is not None
    assert final_record.status == "completed"
    assert final_record.response_body == '{"action_id":"act_new"}'


def test_reprocess_retry_after_response_loss_replays_committed_response(
    isolated_idempotency_store,
    monkeypatch,
):
    headers = {
        **H_USER_LUCAS,
        "Idempotency-Key": "tractian-agent:88888888-8888-4888-8888-888888888888",
    }
    payload = {
        "justification": "rolamento substituído; solicitar novo processamento",
    }
    action_call_count = 0
    committed_response_body = None
    original_uuid4 = main_module.uuid.uuid4
    original_complete = isolated_idempotency_store.complete

    def counted_uuid4():
        nonlocal action_call_count
        action_call_count += 1
        return original_uuid4()

    def complete_then_lose_response(**kwargs):
        nonlocal committed_response_body
        committed_response_body = kwargs["response_body"]
        original_complete(**kwargs)
        raise TimeoutError("resposta perdida depois do commit")

    monkeypatch.setattr(main_module.uuid, "uuid4", counted_uuid4)
    monkeypatch.setattr(
        isolated_idempotency_store,
        "complete",
        complete_then_lose_response,
    )
    client_without_server_exceptions = TestClient(
        app,
        raise_server_exceptions=False,
    )

    lost_response = client_without_server_exceptions.post(
        "/analyses/an_9906/reprocess",
        json=payload,
        headers=headers,
    )
    retry_response = client_without_server_exceptions.post(
        "/analyses/an_9906/reprocess",
        json=payload,
        headers=headers,
    )
    stored_record = isolated_idempotency_store.find(
        idempotency_key=headers["Idempotency-Key"],
        user_id=H_USER_LUCAS["x-user-id"],
        method="POST",
        endpoint="/analyses/an_9906/reprocess",
    )

    assert lost_response.status_code == 500
    assert retry_response.status_code == 200
    assert retry_response.json() == committed_response_body
    assert stored_record is not None
    assert stored_record.status == "completed"
    assert action_call_count == 1


def test_reprocess_404():
    r = client.post(
        "/analyses/an_xxxx/reprocess",
        json={"justification": "justificativa suficientemente longa para passar"},
        headers={
            **H_USER_LUCAS,
            "Idempotency-Key": "test-reprocess-analysis-not-found",
        },
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
