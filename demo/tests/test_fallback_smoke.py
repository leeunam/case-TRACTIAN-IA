from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from langchain_core.runnables import RunnableLambda

from tractian_demo.fallback_smoke import (
    FallbackSmokeStageError,
    run_controlled_fallback_smoke,
    run_fallback_from_environment,
)


@pytest.mark.anyio
async def test_controlled_fallback_smoke_proves_live_probes_and_routing() -> None:
    calls: list[str] = []

    async def groq_probe(_: object) -> str:
        calls.append("groq")
        return "groq-ok"

    async def nim_probe(_: object) -> str:
        calls.append("nim")
        return "nim-ok"

    async def unavailable(_: object) -> str:
        calls.append("controlled-timeout")
        raise httpx.ReadTimeout("controlled availability fault")

    report = await run_controlled_fallback_smoke(
        groq=RunnableLambda(groq_probe),
        nim=RunnableLambda(nim_probe),
        unavailable_primary=RunnableLambda(unavailable),
        input_value="smoke",
    )

    assert report.status == "passed"
    assert report.groq_probe == "passed"
    assert report.nim_probe == "passed"
    assert report.routed_provider == "nvidia-nim"
    assert report.fallback_reason == "timeout"
    assert calls == ["groq", "nim", "controlled-timeout", "nim"]


def test_fallback_smoke_missing_keys_writes_skipped_artifact(tmp_path: Path) -> None:
    output = tmp_path / "fallback.json"

    exit_code = run_fallback_from_environment({}, output_path=output)

    assert exit_code == 2
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "version": "provider-fallback-smoke-v1",
        "status": "skipped",
        "reason": "missing_configuration",
        "missing": ["GROQ_API_KEY", "NVIDIA_API_KEY"],
        "controlled_fault": True,
    }


@pytest.mark.anyio
async def test_fallback_smoke_identifies_stage_without_exposing_provider_error() -> None:
    async def passed(_: object) -> str:
        return "ok"

    async def failed(_: object) -> str:
        raise RuntimeError("secret raw provider response")

    with pytest.raises(FallbackSmokeStageError) as captured:
        await run_controlled_fallback_smoke(
            groq=RunnableLambda(passed),
            nim=RunnableLambda(failed),
            unavailable_primary=RunnableLambda(passed),
            input_value="smoke",
        )

    assert captured.value.stage == "nim_probe"
    assert str(captured.value) == "nim_probe_failed"
    assert "secret" not in str(captured.value)
