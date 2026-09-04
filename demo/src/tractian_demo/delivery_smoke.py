from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from tractian_demo.app import create_app
from tractian_demo.contracts import (
    AgentRunProjection,
    CaseMessage,
    DecisionCandidate,
    DemoCase,
    Execution,
    Persona,
)
from tractian_demo.settings import DemoSettings
from tractian_demo.slack_mcp import SlackMcpClient
from tractian_demo.slack_worker import SlackDeliveryWorker
from tractian_demo.smoke_artifacts import write_smoke_artifact
from tractian_demo.worker import DemoWorker


class SlackClient(Protocol):
    async def send_message(self, *, channel_id: str, text: str) -> str: ...


class SmokeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SlackScenarioReport(SmokeModel):
    audience: Literal["tractian", "authority"]
    channel_id: str = Field(min_length=1, pattern=r"^\S+$")
    decision_id: str = Field(pattern=r"^decision_\S+$")
    notification_id: str = Field(pattern=r"^notification_\S+$")
    delivery_status: Literal["delivered"]
    external_id: str = Field(min_length=1, pattern=r"^\S+$")
    wrong_persona_status: Literal[403]
    replay_status: Literal[200]
    conflict_status: Literal[409]
    resume_execution_count: Literal[1]
    resume_status: Literal["completed"]


class SlackDecisionSmokeReport(SmokeModel):
    version: Literal["slack-decision-smoke-v1"] = "slack-decision-smoke-v1"
    status: Literal["passed"] = "passed"
    database_isolated: Literal[True]
    scenarios: tuple[SlackScenarioReport, SlackScenarioReport]


class _ScenarioExecutor:
    async def execute(
        self, *, case: DemoCase, message: CaseMessage, execution: Execution
    ) -> AgentRunProjection:
        if execution.resume_decision_id is not None:
            return AgentRunProjection(
                assistant_message="A decisão foi aplicada uma única vez no fluxo simulado.",
                decision="guide",
                trace_id=f"smoke:{execution.id}",
                provider="groq",
                fallback_reason=None,
                evidence_count=1,
                limitation_count=0,
                tool_names=(),
            )
        authority = message.content == "smoke:authority"
        candidate = DecisionCandidate(
            audience="authority" if authority else "tractian",
            kind="action_authorization" if authority else "technical_review",
            summary=(
                "Autorizar alteração simulada de criticidade."
                if authority
                else "Revisar bloqueio técnico simulado."
            ),
            scope={"smoke_scenario": "authority" if authority else "tractian"},
            required_permission="action_high" if authority else None,
            resume_kind="delegated_action" if authority else "technical_review",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        )
        return AgentRunProjection(
            assistant_message="A execução aguarda uma decisão humana simulada.",
            decision="require_human_review",
            trace_id=f"smoke:{execution.id}",
            provider="groq",
            fallback_reason=None,
            evidence_count=1,
            limitation_count=1,
            tool_names=(),
            decision_candidate=candidate,
        )


async def _personas() -> tuple[Persona, ...]:
    return (
        Persona(
            id="smoke_requester",
            name="Solicitante simulado",
            profile="requester",
            company_id="smoke_company",
            permissions=frozenset({"read"}),
        ),
        Persona(
            id="smoke_tractian",
            name="Especialista TRACTIAN simulado",
            profile="tractian",
            company_id=None,
            permissions=frozenset({"technical_review"}),
        ),
        Persona(
            id="smoke_authority",
            name="Autoridade simulada",
            profile="authority",
            company_id="smoke_company",
            permissions=frozenset({"read", "action_high"}),
        ),
    )


async def run_slack_decision_smoke(
    *,
    workspace: Path,
    slack_client: SlackClient,
    public_app_url: str,
    tractian_channel: str,
    authority_channel: str,
) -> SlackDecisionSmokeReport:
    workspace.mkdir(parents=True, exist_ok=True)
    public_cases_path = workspace / "public-cases.json"
    database_path = workspace / "smoke.sqlite3"
    public_cases_path.write_text(
        json.dumps(
            [
                {
                    "id": "case_smoke_public",
                    "ticket_id": "SMOKE-1",
                    "company_id": "smoke_company",
                    "user_id": "smoke_requester",
                    "asset_id": "smoke_asset",
                    "message": "Caso sintético do smoke de entrega.",
                }
            ]
        ),
        encoding="utf-8",
    )
    settings = DemoSettings(
        database_path=database_path,
        public_cases_path=public_cases_path,
        industrial_api_url="http://industrial.invalid",
        public_app_url=public_app_url,
        slack_tractian_channel=tractian_channel,
        slack_authority_channel=authority_channel,
        slack_access_token_configured=True,
    )
    app = create_app(settings, persona_loader=_personas)
    scenarios: list[SlackScenarioReport] = []
    async with app.router.lifespan_context(app):
        repository = app.state.repository
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://demo.test"
        ) as api:
            worker = DemoWorker(
                repository, _ScenarioExecutor(), worker_id="smoke-agent"
            )
            slack_worker = SlackDeliveryWorker(
                repository,
                slack_client,
                worker_id="smoke-slack",
                public_app_url=public_app_url,
                tractian_channel=tractian_channel,
                authority_channel=authority_channel,
            )
            for audience, correct_persona, wrong_persona, channel in (
                ("tractian", "smoke_tractian", "smoke_requester", tractian_channel),
                ("authority", "smoke_authority", "smoke_tractian", authority_channel),
            ):
                created = await api.post(
                    "/v1/cases", json={"source_case_id": "case_smoke_public"}
                )
                created.raise_for_status()
                case_id = created.json()["id"]
                enqueued = await api.post(
                    f"/v1/cases/{case_id}/messages",
                    json={
                        "persona_id": "smoke_requester",
                        "content": f"smoke:{audience}",
                        "idempotency_key": f"smoke-{audience}",
                    },
                )
                enqueued.raise_for_status()
                if not await worker.run_once():
                    raise RuntimeError("AGENT_WORK_NOT_CLAIMED")
                visible = await api.get(
                    "/v1/decisions", params={"persona_id": correct_persona}
                )
                visible.raise_for_status()
                decision = next(
                    item for item in visible.json() if item["case_id"] == case_id
                )
                wrong = await api.post(
                    f"/v1/decisions/{decision['id']}/resolve",
                    json={"persona_id": wrong_persona, "resolution": "approve"},
                )
                if not await slack_worker.run_once():
                    raise RuntimeError("SLACK_WORK_NOT_CLAIMED")
                notification = repository.get_outbox_for_decision(decision["id"])
                resolved = await api.post(
                    f"/v1/decisions/{decision['id']}/resolve",
                    json={"persona_id": correct_persona, "resolution": "approve"},
                )
                resolved.raise_for_status()
                replay = await api.post(
                    f"/v1/decisions/{decision['id']}/resolve",
                    json={"persona_id": correct_persona, "resolution": "approve"},
                )
                conflict = await api.post(
                    f"/v1/decisions/{decision['id']}/resolve",
                    json={"persona_id": correct_persona, "resolution": "reject"},
                )
                before_resume = await api.get(f"/v1/cases/{case_id}")
                before_resume.raise_for_status()
                resume_count = sum(
                    item["resume_decision_id"] == decision["id"]
                    for item in before_resume.json()["executions"]
                )
                if not await worker.run_once():
                    raise RuntimeError("RESUME_WORK_NOT_CLAIMED")
                after_resume = await api.get(f"/v1/cases/{case_id}")
                after_resume.raise_for_status()
                resume = next(
                    item
                    for item in after_resume.json()["executions"]
                    if item["resume_decision_id"] == decision["id"]
                )
                scenarios.append(
                    SlackScenarioReport(
                        audience=audience,
                        channel_id=channel,
                        decision_id=decision["id"],
                        notification_id=notification.id,
                        delivery_status=notification.status.value,
                        external_id=notification.external_id or "",
                        wrong_persona_status=wrong.status_code,
                        replay_status=replay.status_code,
                        conflict_status=conflict.status_code,
                        resume_execution_count=resume_count,
                        resume_status=resume["status"],
                    )
                )
            if await worker.run_once():
                raise RuntimeError("UNEXPECTED_REMAINING_WORK")
    return SlackDecisionSmokeReport(
        database_isolated=database_path.parent == workspace,
        scenarios=(scenarios[0], scenarios[1]),
    )


def run_slack_e2e_from_environment(
    environment: Mapping[str, str],
    *,
    output_path: Path,
    client_factory: Callable[[httpx.AsyncClient, str], SlackClient] | None = None,
) -> int:
    required = (
        "SLACK_AUTHORITY_CHANNEL_ID",
        "SLACK_MCP_ACCESS_TOKEN",
        "SLACK_TRACTIAN_CHANNEL_ID",
    )
    missing = sorted(
        name for name in required if not environment.get(name, "").strip()
    )
    if missing:
        write_smoke_artifact(
            output_path,
            {
                "version": "slack-decision-smoke-v1",
                "status": "skipped",
                "reason": "missing_configuration",
                "missing": missing,
                "database_isolated": True,
                "scenarios": [],
            },
        )
        return 2

    async def execute(workspace: Path) -> SlackDecisionSmokeReport:
        async with httpx.AsyncClient(timeout=20) as http:
            factory = client_factory or (
                lambda client, token: SlackMcpClient(
                    http=client, access_token=token
                )
            )
            return await run_slack_decision_smoke(
                workspace=workspace,
                slack_client=factory(http, environment["SLACK_MCP_ACCESS_TOKEN"]),
                public_app_url=environment.get(
                    "PUBLIC_APP_URL", "http://127.0.0.1:5173"
                ),
                tractian_channel=environment["SLACK_TRACTIAN_CHANNEL_ID"],
                authority_channel=environment["SLACK_AUTHORITY_CHANNEL_ID"],
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix="tractian-slack-e2e-", dir=output_path.parent
        ) as temporary:
            report = asyncio.run(execute(Path(temporary)))
    except Exception:
        write_smoke_artifact(
            output_path,
            {
                "version": "slack-decision-smoke-v1",
                "status": "failed",
                "reason": "slack_or_contract_failure",
                "missing": [],
                "database_isolated": True,
                "scenarios": [],
            },
        )
        return 1
    write_smoke_artifact(output_path, report.model_dump(mode="json"))
    return 0


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    output = root / ".run" / "smoke" / "slack-e2e.json"
    exit_code = run_slack_e2e_from_environment(dict(os.environ), output_path=output)
    print(f"status={'passed' if exit_code == 0 else 'not_passed'} report={output}")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
