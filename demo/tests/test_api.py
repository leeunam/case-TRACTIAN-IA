from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tractian_demo.app import create_app
from tractian_demo.contracts import Persona
from tractian_demo.settings import DemoSettings


def settings(tmp_path: Path) -> DemoSettings:
    cases = tmp_path / "cases.json"
    cases.write_text(
        '[{"id":"case_public_1","ticket_id":"TKT-1",'
        '"company_id":"comp_1","user_id":"usr_1",'
        '"asset_id":"asset_1","message":"Mensagem original"}]',
        encoding="utf-8",
    )
    return DemoSettings(
        database_path=tmp_path / "demo.sqlite3",
        public_cases_path=cases,
        industrial_api_url="http://industrial.test",
        allowed_origins=("http://localhost:5173",),
        public_app_url="http://localhost:5173",
    )


@pytest.fixture
def client(tmp_path: Path):
    async def personas() -> tuple[Persona, ...]:
        return (
            Persona(
                id="usr_1", name="Pessoa", profile="requester",
                company_id="comp_1", permissions=frozenset({"read"}),
            ),
            Persona(
                id="tractian_reviewer", name="Equipe TRACTIAN",
                profile="tractian", company_id=None,
                permissions=frozenset({"technical_review"}),
            ),
        )

    with TestClient(create_app(settings(tmp_path), persona_loader=personas)) as client:
        yield client


def test_rest_contract_creates_case_and_enqueues_message(client: TestClient) -> None:
    created = client.post("/v1/cases", json={"source_case_id": "case_public_1"})
    assert created.status_code == 201
    case_id = created.json()["id"]

    response = client.post(
        f"/v1/cases/{case_id}/messages",
        json={
            "persona_id": "usr_1",
            "content": "Olá agente",
            "idempotency_key": "browser-click-1",
        },
    )
    assert response.status_code == 202
    assert response.json()["execution"]["status"] == "queued"

    detail = client.get(f"/v1/cases/{case_id}").json()
    assert [message["content"] for message in detail["messages"]] == ["Olá agente"]
    assert detail["executions"][0]["status"] == "queued"


def test_public_case_is_immutable_and_cors_is_allowlisted(client: TestClient) -> None:
    response = client.post(
        "/v1/cases/case_public_1/messages",
        json={"persona_id": "usr_1", "content": "x", "idempotency_key": "1"},
        headers={"origin": "http://evil.example"},
    )
    assert response.status_code == 409
    assert response.json() == {
        "detail": {"code": "IMMUTABLE_PUBLIC_CASE", "message": "Duplique o caso público antes de conversar."}
    }
    assert "access-control-allow-origin" not in response.headers


def test_personas_and_config_never_expose_secrets(client: TestClient) -> None:
    config = client.get("/v1/demo/config").json()
    assert config["mode"] == "live"
    assert config["warning"].startswith("Demonstração")
    assert "api_key" not in str(config).lower()
    assert [item["profile"] for item in client.get("/v1/personas").json()] == [
        "requester", "tractian"
    ]
