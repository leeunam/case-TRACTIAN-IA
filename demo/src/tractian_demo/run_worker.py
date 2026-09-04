from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import httpx

from tractian_agent.checkpoint import open_checkpointer
from tractian_agent.client import IndustrialApiClient
from tractian_agent.graph import build_agent_graph
from tractian_agent.groq_provider import GroqModelProvider
from tractian_agent.model_provider import ModelConfig
from tractian_agent.nvidia_nim_provider import NvidiaNimModelProvider
from tractian_agent.planner import Planner
from tractian_agent.writer import Writer

from tractian_demo.agent_executor import LiveAgentExecutor
from tractian_demo.repository import DemoRepository
from tractian_demo.settings import DemoSettings
from tractian_demo.worker import DemoWorker


def _provider(name: str, environment: dict[str, str]):
    if name == "groq":
        return GroqModelProvider.from_env(environment, max_retries=0)
    if name == "nvidia-nim":
        return NvidiaNimModelProvider.from_env(environment)
    raise ValueError("provider desconhecido")


async def main() -> None:
    environment = dict(os.environ)
    settings = DemoSettings.from_env(environment)
    repository = DemoRepository(settings.database_path)
    repository.open()
    model_config = ModelConfig(
        model_id=settings.planner_model,
        temperature=0.0,
        timeout_seconds=30.0,
        max_output_tokens=512,
    )
    provider = _provider(settings.primary_provider, environment)
    planner = Planner(provider.create_chat_model(model_config))
    writer = Writer(provider.create_chat_model(model_config))
    worker_id = f"worker_{uuid4().hex}"
    try:
        async with (
            open_checkpointer(settings.checkpoint_path) as saver,
            IndustrialApiClient(settings.industrial_api_url) as industrial_client,
            httpx.AsyncClient(base_url=settings.industrial_api_url, timeout=5) as identity_client,
        ):
            graph = build_agent_graph(saver, planner=planner, writer=writer)
            worker = DemoWorker(
                repository,
                LiveAgentExecutor(
                    graph=graph, industrial_client=industrial_client,
                    identity_client=identity_client, provider=settings.primary_provider,
                ),
                worker_id=worker_id,
            )
            while True:
                worked = await worker.run_once()
                if not worked:
                    await asyncio.sleep(0.25)
    finally:
        repository.close()


if __name__ == "__main__":
    asyncio.run(main())
