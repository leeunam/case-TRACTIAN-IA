from __future__ import annotations

import asyncio
import os

import httpx

from tractian_demo.settings import DemoSettings
from tractian_demo.slack_mcp import SlackMcpClient


async def main() -> None:
    environment = dict(os.environ)
    settings = DemoSettings.from_env(environment)
    token = environment.get("SLACK_MCP_ACCESS_TOKEN", "")
    if (
        not token
        or not settings.slack_tractian_channel
        or not settings.slack_authority_channel
    ):
        raise SystemExit(
            "Configure SLACK_MCP_ACCESS_TOKEN e os dois SLACK_*_CHANNEL_ID no .env."
        )
    async with httpx.AsyncClient(timeout=20) as http:
        client = SlackMcpClient(http=http, access_token=token)
        for audience, channel in (
            ("equipe TRACTIAN", settings.slack_tractian_channel),
            ("autoridade da empresa", settings.slack_authority_channel),
        ):
            external_id = await client.send_message(
                channel_id=channel,
                text=(
                    f"[smoke] Canal de {audience} validado. "
                    f"Decisões permanecem na central: {settings.public_app_url}/"
                ),
            )
            print(f"ok audience={audience!r} external_id={external_id!r}")


if __name__ == "__main__":
    asyncio.run(main())
