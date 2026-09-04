from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from pydantic import ConfigDict


FallbackReason = Literal["timeout", "rate_limit", "network", "server"]


def availability_reason(error: BaseException) -> FallbackReason | None:
    if isinstance(error, (httpx.TimeoutException, TimeoutError)):
        return "timeout"
    if isinstance(error, (httpx.NetworkError, ConnectionError)):
        return "network"
    status = getattr(error, "status_code", None)
    if status == 429:
        return "rate_limit"
    if isinstance(status, int) and 500 <= status <= 599:
        return "server"
    name = type(error).__name__.lower()
    if "ratelimit" in name:
        return "rate_limit"
    if "timeout" in name:
        return "timeout"
    if "connection" in name:
        return "network"
    return None


@dataclass
class FallbackTracker:
    reason: FallbackReason | None = None

    def reset(self) -> None:
        self.reason = None


def with_availability_fallback(
    primary: Runnable[Any, Any],
    fallback: Runnable[Any, Any],
    *,
    tracker: FallbackTracker,
) -> Runnable[Any, Any]:
    async def route(value: Any, config: RunnableConfig) -> Any:
        try:
            return await primary.ainvoke(value, config=config)
        except BaseException as error:
            reason = availability_reason(error)
            if reason is None:
                raise
            tracker.reason = reason
            return await fallback.ainvoke(value, config=config)

    return RunnableLambda(route)


class AvailabilityFallbackChatModel(BaseChatModel):
    """Modelo LangChain que troca provider apenas em indisponibilidade fechada."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary: BaseChatModel
    fallback: BaseChatModel
    tracker: FallbackTracker

    @property
    def _llm_type(self) -> str:
        return "availability-fallback"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return self.primary._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        except BaseException as error:
            reason = availability_reason(error)
            if reason is None:
                raise
            self.tracker.reason = reason
            return self.fallback._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return await self.primary._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        except BaseException as error:
            reason = availability_reason(error)
            if reason is None:
                raise
            self.tracker.reason = reason
            return await self.fallback._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable[Any, Any]:
        return with_availability_fallback(
            self.primary.bind_tools(tools, **kwargs),
            self.fallback.bind_tools(tools, **kwargs),
            tracker=self.tracker,
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:
        return with_availability_fallback(
            self.primary.with_structured_output(schema, **kwargs),
            self.fallback.with_structured_output(schema, **kwargs),
            tracker=self.tracker,
        )
