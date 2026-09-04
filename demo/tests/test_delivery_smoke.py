from __future__ import annotations

from pathlib import Path
import json

import pytest

from tractian_demo.delivery_smoke import (
    run_slack_decision_smoke,
    run_slack_e2e_from_environment,
)


class RecordingSlackClient:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    async def send_message(self, *, channel_id: str, text: str) -> str:
        self.messages.append((channel_id, text))
        return f"message-{len(self.messages)}"


@pytest.mark.anyio
async def test_slack_decision_smoke_covers_both_audiences_once(tmp_path: Path) -> None:
    client = RecordingSlackClient()
    report = await run_slack_decision_smoke(
        workspace=tmp_path,
        slack_client=client,
        public_app_url="http://localhost:5173",
        tractian_channel="channel-tractian",
        authority_channel="channel-authority",
    )

    assert report.status == "passed"
    assert report.database_isolated is True
    assert [item.audience for item in report.scenarios] == ["tractian", "authority"]
    assert [item.channel_id for item in report.scenarios] == [
        "channel-tractian",
        "channel-authority",
    ]
    assert all(item.delivery_status == "delivered" for item in report.scenarios)
    assert all(item.external_id for item in report.scenarios)
    assert all(item.wrong_persona_status == 403 for item in report.scenarios)
    assert all(item.replay_status == 200 for item in report.scenarios)
    assert all(item.conflict_status == 409 for item in report.scenarios)
    assert all(item.resume_execution_count == 1 for item in report.scenarios)
    assert all(item.resume_status == "completed" for item in report.scenarios)
    assert all("?decision=decision_" in text for _, text in client.messages)
    assert all("segredo" not in text.lower() for _, text in client.messages)


def test_slack_e2e_missing_configuration_is_not_success(tmp_path: Path) -> None:
    output = tmp_path / "slack-e2e.json"

    exit_code = run_slack_e2e_from_environment({}, output_path=output)

    assert exit_code == 2
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact == {
        "version": "slack-decision-smoke-v1",
        "status": "skipped",
        "reason": "missing_configuration",
        "missing": [
            "SLACK_AUTHORITY_CHANNEL_ID",
            "SLACK_MCP_ACCESS_TOKEN",
            "SLACK_TRACTIAN_CHANNEL_ID",
        ],
        "database_isolated": True,
        "scenarios": [],
    }


def test_slack_e2e_cli_writes_safe_passed_artifact(tmp_path: Path) -> None:
    output = tmp_path / "slack-e2e.json"
    client = RecordingSlackClient()

    exit_code = run_slack_e2e_from_environment(
        {
            "SLACK_MCP_ACCESS_TOKEN": "super-secret-token",
            "SLACK_TRACTIAN_CHANNEL_ID": "channel-tractian",
            "SLACK_AUTHORITY_CHANNEL_ID": "channel-authority",
        },
        output_path=output,
        client_factory=lambda _http, _token: client,
    )

    wire = output.read_text(encoding="utf-8")
    artifact = json.loads(wire)
    assert exit_code == 0
    assert artifact["status"] == "passed"
    assert len(artifact["scenarios"]) == 2
    assert "super-secret-token" not in wire
