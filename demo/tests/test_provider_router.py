import httpx
import pytest
from langchain_core.runnables import RunnableLambda

from tractian_demo.provider_router import (
    FallbackTracker,
    availability_reason,
    with_availability_fallback,
)


class StatusError(RuntimeError):
    def __init__(self, status_code: int):
        self.status_code = status_code


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.ConnectError("offline"), "network"),
        (StatusError(429), "rate_limit"),
        (StatusError(503), "server"),
    ],
)
async def test_fallback_only_for_availability_failures(
    error: Exception, reason: str
) -> None:
    tracker = FallbackTracker()

    async def fail(_: object):
        raise error

    routed = with_availability_fallback(
        RunnableLambda(fail), RunnableLambda(lambda _: "nim-ok"), tracker=tracker
    )
    assert await routed.ainvoke({}) == "nim-ok"
    assert tracker.reason == reason


@pytest.mark.anyio
async def test_protocol_or_quality_failure_never_uses_fallback() -> None:
    tracker = FallbackTracker()
    called = False

    async def invalid(_: object):
        raise ValueError("invalid structured output")

    async def fallback(_: object):
        nonlocal called
        called = True
        return "must not run"

    routed = with_availability_fallback(
        RunnableLambda(invalid), RunnableLambda(fallback), tracker=tracker
    )
    with pytest.raises(ValueError, match="structured"):
        await routed.ainvoke({})
    assert called is False
    assert tracker.reason is None
    assert availability_reason(ValueError("schema")) is None
