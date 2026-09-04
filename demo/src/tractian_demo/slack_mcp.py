from __future__ import annotations

from typing import Any
import json

import httpx


SLACK_MCP_ENDPOINT = "https://mcp.slack.com/mcp"


class SlackMcpProtocolError(RuntimeError):
    pass


class SlackMcpClient:
    """Cliente mínimo do transporte Streamable HTTP do MCP oficial do Slack."""

    def __init__(self, *, http: httpx.AsyncClient, access_token: str) -> None:
        if not access_token.strip():
            raise ValueError("SLACK_MCP_ACCESS_TOKEN is required")
        self._http = http
        self._token = access_token
        self._session_id: str | None = None
        self._next_id = 1

    def _headers(self) -> dict[str, str]:
        values = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            values["Mcp-Session-Id"] = self._session_id
        return values

    async def _rpc(
        self, method: str, params: dict[str, object] | None = None
    ) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        response = await self._http.post(
            SLACK_MCP_ENDPOINT,
            headers=self._headers(),
            json={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            },
        )
        response.raise_for_status()
        session = response.headers.get("mcp-session-id")
        if session:
            self._session_id = session
        if "text/event-stream" in response.headers.get("content-type", ""):
            data_lines = [
                line.removeprefix("data:").strip()
                for line in response.text.splitlines()
                if line.startswith("data:")
            ]
            if not data_lines:
                raise SlackMcpProtocolError("SLACK_MCP_INVALID_RESPONSE")
            try:
                payload = json.loads(data_lines[-1])
            except json.JSONDecodeError as error:
                raise SlackMcpProtocolError("SLACK_MCP_INVALID_RESPONSE") from error
        else:
            try:
                payload = response.json()
            except ValueError as error:
                raise SlackMcpProtocolError("SLACK_MCP_INVALID_RESPONSE") from error
        if not isinstance(payload, dict):
            raise SlackMcpProtocolError("SLACK_MCP_INVALID_RESPONSE")
        if payload.get("error") is not None:
            raise SlackMcpProtocolError("SLACK_MCP_RPC_ERROR")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise SlackMcpProtocolError("SLACK_MCP_INVALID_RESPONSE")
        return result

    async def _initialize(self) -> None:
        if self._session_id is not None:
            return
        await self._rpc(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "tractian-case-center", "version": "0.1.0"},
            },
        )
        await self._http.post(
            SLACK_MCP_ENDPOINT,
            headers=self._headers(),
            json={
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            },
        )

    async def send_message(self, *, channel_id: str, text: str) -> str:
        await self._initialize()
        listed = await self._rpc("tools/list")
        tools = listed.get("tools")
        if not isinstance(tools, list):
            raise SlackMcpProtocolError("SLACK_TOOLS_MISSING")
        tool = next(
            (
                item
                for item in tools
                if isinstance(item, dict)
                and "send" in str(item.get("name", "")).lower()
                and "message" in str(item.get("name", "")).lower()
            ),
            None,
        )
        if tool is None:
            raise SlackMcpProtocolError("SLACK_SEND_TOOL_MISSING")
        properties = tool.get("inputSchema", {}).get("properties", {})
        channel_key = "channel_id" if "channel_id" in properties else "channel"
        text_key = "message" if "message" in properties else "text"
        result = await self._rpc(
            "tools/call",
            {
                "name": tool["name"],
                "arguments": {channel_key: channel_id, text_key: text},
            },
        )
        if result.get("isError") is True:
            raise SlackMcpProtocolError("SLACK_TOOL_REJECTED")
        structured = result.get("structuredContent", {})
        external_id = (
            structured.get("message_ts") if isinstance(structured, dict) else None
        )
        if not isinstance(external_id, str) or not external_id:
            raise SlackMcpProtocolError("SLACK_EXTERNAL_ID_MISSING")
        return external_id
