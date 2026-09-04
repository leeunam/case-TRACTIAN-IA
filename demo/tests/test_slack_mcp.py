import json
from types import SimpleNamespace

import httpx
import pytest

from tractian_demo.slack_mcp import SlackMcpClient, SlackMcpProtocolError
from tractian_demo.slack_worker import SlackDeliveryWorker


@pytest.mark.anyio
async def test_slack_mcp_discovers_send_tool_and_returns_external_id() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if body.get("method") == "initialize":
            return httpx.Response(
                200,
                headers={"mcp-session-id": "session-1"},
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "serverInfo": {"name": "slack", "version": "1"},
                    },
                },
            )
        if body.get("method") == "notifications/initialized":
            return httpx.Response(202)
        if body.get("method") == "tools/list":
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {
                        "tools": [
                            {
                                "name": "slack_send_message",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "channel_id": {"type": "string"},
                                        "message": {"type": "string"},
                                    },
                                    "required": ["channel_id", "message"],
                                },
                            }
                        ]
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "result": {
                    "content": [{"type": "text", "text": "sent"}],
                    "structuredContent": {"message_ts": "123.456"},
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        external_id = await SlackMcpClient(
            http=http, access_token="xoxp-not-persisted"
        ).send_message(
            channel_id="C123",
            text="Decisão decision_1: http://localhost/decisions/decision_1",
        )
    assert external_id == "123.456"
    assert requests[-1]["params"]["name"] == "slack_send_message"
    assert "xoxp-not-persisted" not in str(requests)


@pytest.mark.anyio
async def test_slack_success_without_external_id_is_uncertain() -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                headers={"mcp-session-id": "s"},
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "serverInfo": {"name": "slack", "version": "1"},
                    },
                },
            ),
            httpx.Response(202),
            httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {
                        "tools": [
                            {
                                "name": "send_message",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"channel": {}, "text": {}},
                                    "required": ["channel", "text"],
                                },
                            }
                        ]
                    },
                },
            ),
            httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "result": {"content": [{"type": "text", "text": "ok"}]},
                },
            ),
        ]
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: next(responses))
    ) as http:
        with pytest.raises(SlackMcpProtocolError, match="SLACK_EXTERNAL_ID_MISSING"):
            await SlackMcpClient(http=http, access_token="token").send_message(
                channel_id="C1", text="safe"
            )


@pytest.mark.anyio
async def test_slack_accepts_streamable_http_sse_response() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        if body.get("method") == "notifications/initialized":
            return httpx.Response(202)
        if body.get("method") == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "serverInfo": {"name": "slack", "version": "1"},
            }
        elif body.get("method") == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "send_message",
                        "inputSchema": {"properties": {"channel": {}, "text": {}}},
                    }
                ]
            }
        else:
            result = {"structuredContent": {"message_ts": "789.012"}}
        rpc = json.dumps({"jsonrpc": "2.0", "id": body.get("id"), "result": result})
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                **({"mcp-session-id": "sse-session"} if calls == 1 else {}),
            },
            text=f"event: message\ndata: {rpc}\n\n",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        external_id = await SlackMcpClient(
            http=http, access_token="token"
        ).send_message(channel_id="C1", text="safe")
    assert external_id == "789.012"


class _OutboxRepository:
    def __init__(self) -> None:
        self.finished: tuple | None = None

    def claim_outbox(self, **_: object):
        return SimpleNamespace(
            id="notification_1",
            audience="tractian",
            decision_id="decision_1",
            payload={"category": "technical_review", "summary": "Revisar caso"},
        )

    def finish_outbox(self, *args: object, **kwargs: object) -> None:
        self.finished = (args, kwargs)


class _HttpFailureClient:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    async def send_message(self, **_: object) -> str:
        request = httpx.Request("POST", "https://mcp.slack.com/mcp")
        response = httpx.Response(self.status_code, request=request)
        raise httpx.HTTPStatusError("failed", request=request, response=response)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(400, "failed"), (429, "uncertain"), (503, "uncertain")],
)
async def test_slack_worker_closes_http_failures(
    status_code: int, expected: str
) -> None:
    repository = _OutboxRepository()
    worker = SlackDeliveryWorker(
        repository,  # type: ignore[arg-type]
        _HttpFailureClient(status_code),  # type: ignore[arg-type]
        worker_id="worker",
        public_app_url="http://localhost:5173",
        tractian_channel="C1",
        authority_channel="C2",
    )

    assert await worker.run_once() is True
    assert repository.finished is not None
    assert repository.finished[1]["status"].value == expected
    assert repository.finished[1]["error_code"] == f"SLACK_HTTP_{status_code}"
