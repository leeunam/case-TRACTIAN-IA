from __future__ import annotations

import asyncio
from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any, Literal

import httpx
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, ConfigDict

from tractian_agent.groq_provider import GroqModelProvider
from tractian_agent.model_provider import ModelConfig
from tractian_agent.nvidia_nim_provider import NvidiaNimModelProvider
from tractian_demo.provider_router import FallbackTracker, with_availability_fallback
from tractian_demo.smoke_artifacts import write_smoke_artifact


class FallbackSmokeStageError(RuntimeError):
    def __init__(self, stage: str) -> None:
        self.stage = stage
        super().__init__(f"{stage}_failed")


class FallbackSmokeReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal["provider-fallback-smoke-v1"] = "provider-fallback-smoke-v1"
    status: Literal["passed"] = "passed"
    groq_probe: Literal["passed"] = "passed"
    nim_probe: Literal["passed"] = "passed"
    routed_provider: Literal["nvidia-nim"] = "nvidia-nim"
    fallback_reason: Literal["timeout", "rate_limit", "network", "server"]
    controlled_fault: bool = True


def _require_response(value: object) -> None:
    content = getattr(value, "content", value)
    if not str(content).strip():
        raise ValueError("EMPTY_PROVIDER_RESPONSE")


async def run_controlled_fallback_smoke(
    *,
    groq: Runnable[Any, Any],
    nim: Runnable[Any, Any],
    unavailable_primary: Runnable[Any, Any],
    input_value: object,
) -> FallbackSmokeReport:
    try:
        _require_response(await groq.ainvoke(input_value))
    except Exception as error:
        raise FallbackSmokeStageError("groq_probe") from error
    try:
        _require_response(await nim.ainvoke(input_value))
    except Exception as error:
        raise FallbackSmokeStageError("nim_probe") from error
    tracker = FallbackTracker()
    routed = with_availability_fallback(
        unavailable_primary,
        nim,
        tracker=tracker,
    )
    try:
        _require_response(await routed.ainvoke(input_value))
    except Exception as error:
        raise FallbackSmokeStageError("routed_fallback") from error
    if tracker.reason is None:
        raise RuntimeError("FALLBACK_NOT_OBSERVED")
    return FallbackSmokeReport(fallback_reason=tracker.reason)


def run_fallback_from_environment(
    environment: Mapping[str, str], *, output_path: Path
) -> int:
    required = ("GROQ_API_KEY", "NVIDIA_API_KEY")
    missing = sorted(
        name for name in required if not environment.get(name, "").strip()
    )
    if missing:
        write_smoke_artifact(
            output_path,
            {
                "version": "provider-fallback-smoke-v1",
                "status": "skipped",
                "reason": "missing_configuration",
                "missing": missing,
                "controlled_fault": True,
            },
        )
        return 2

    async def unavailable(_: object) -> object:
        raise httpx.ReadTimeout("controlled availability fault")

    async def execute() -> FallbackSmokeReport:
        model_id = environment.get(
            "DEMO_FALLBACK_SMOKE_MODEL", "openai/gpt-oss-20b"
        )
        config = ModelConfig(
            model_id=model_id,
            temperature=0.0,
            timeout_seconds=30.0,
            max_output_tokens=512,
        )
        groq = GroqModelProvider.from_env(
            environment, max_retries=0
        ).create_chat_model(config)
        nim = NvidiaNimModelProvider.from_env(environment).create_chat_model(config)
        prompt = [HumanMessage(content="Responda somente com: OK")]
        return await run_controlled_fallback_smoke(
            groq=groq,
            nim=nim,
            unavailable_primary=RunnableLambda(unavailable),
            input_value=prompt,
        )

    try:
        report = asyncio.run(execute())
    except Exception as error:
        write_smoke_artifact(
            output_path,
            {
                "version": "provider-fallback-smoke-v1",
                "status": "failed",
                "reason": (
                    str(error)
                    if isinstance(error, FallbackSmokeStageError)
                    else "provider_or_routing_failure"
                ),
                "missing": [],
                "controlled_fault": True,
            },
        )
        return 1
    write_smoke_artifact(output_path, report.model_dump(mode="json"))
    return 0


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    output = root / ".run" / "smoke" / "provider-fallback.json"
    exit_code = run_fallback_from_environment(dict(os.environ), output_path=output)
    print(f"status={'passed' if exit_code == 0 else 'not_passed'} report={output}")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
