import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool

from tractian_agent.evaluation.contracts import BenchmarkInput
from tractian_agent.evaluation.live_experiment import (
    LiveExperimentOptions,
    UserRuntimeProfile,
    _apply_model_pacing,
    _fetch_user_profile,
    _live_rate_limiter,
    run_live_experiment,
)
from tractian_agent.planner import (
    PlannerDecisionKind,
    PlannerStopReason,
    PlannerTerminalDecision,
)
from tractian_agent.state import AgentDecision, WriterDraft, WriterNextStep


class _LocalPlannerModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "local-live-experiment-planner"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise AssertionError("o planner deve usar os wrappers públicos")

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> RunnableLambda:
        async def select(_: list[BaseMessage]) -> AIMessage:
            return AIMessage(content="done")

        return RunnableLambda(select)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> RunnableLambda:
        assert schema is PlannerTerminalDecision
        assert include_raw is False

        async def finalize(_: list[BaseMessage]) -> PlannerTerminalDecision:
            return PlannerTerminalDecision(
                decision=PlannerDecisionKind.GUIDE,
                stop_reason=PlannerStopReason.SUFFICIENT_EVIDENCE,
            )

        return RunnableLambda(finalize)


class _LocalWriterModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "local-live-experiment-writer"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise AssertionError("o writer deve usar saída estruturada")

    def bind_tools(self, *args: Any, **kwargs: Any) -> RunnableLambda:
        raise AssertionError("o writer não pode receber tools")

    def with_structured_output(
        self,
        schema: object,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> RunnableLambda:
        assert schema is WriterDraft
        assert include_raw is False

        async def write(_: list[BaseMessage]) -> WriterDraft:
            return WriterDraft(
                decision=AgentDecision.GUIDE,
                next_step=WriterNextStep.MONITOR,
            )

        return RunnableLambda(write)


class _LocalProvider:
    def __init__(self) -> None:
        self._calls = 0

    def create_chat_model(self, config):
        self._calls += 1
        return _LocalPlannerModel() if self._calls == 1 else _LocalWriterModel()


def test_live_model_pacing_is_explicit_and_can_be_disabled() -> None:
    shared_limiter = _live_rate_limiter(20.0)
    paced_planner = _apply_model_pacing(_LocalPlannerModel(), shared_limiter)
    paced_writer = _apply_model_pacing(_LocalWriterModel(), shared_limiter)
    unpaced = _apply_model_pacing(_LocalWriterModel(), None)

    assert paced_planner.rate_limiter is shared_limiter
    assert paced_writer.rate_limiter is shared_limiter
    assert shared_limiter is not None
    assert shared_limiter.requests_per_second == pytest.approx(1 / 20)
    assert shared_limiter.max_bucket_size == 1
    assert unpaced.rate_limiter is None


def _case() -> BenchmarkInput:
    return BenchmarkInput(
        id="case_tkt_ctx_02",
        ticket_id="TKT-CTX-02",
        company_id="comp_aurora",
        user_id="usr_lucas",
        asset_id="asset_B204",
        message="O que significa BPFO?",
    )


def test_live_runtime_fetches_permissions_from_trusted_api_boundary() -> None:
    async def scenario():
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["x-user-id"] == "usr_lucas"
            return httpx.Response(
                200,
                json={
                    "id": "usr_lucas",
                    "company_id": "comp_aurora",
                    "name": "Lucas",
                    "role": "mechanic",
                    "permissions": ["read", "action_low"],
                },
            )

        async with httpx.AsyncClient(
            base_url="https://simulator.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            return await _fetch_user_profile(client, _case())

    profile = asyncio.run(scenario())

    assert profile.permissions == frozenset({"read", "action_low"})


def test_live_runtime_rejects_cross_company_identity() -> None:
    async def scenario():
        async with httpx.AsyncClient(
            base_url="https://simulator.test",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={
                        "id": "usr_lucas",
                        "company_id": "comp_other",
                        "name": "Lucas",
                        "role": "mechanic",
                        "permissions": ["read"],
                    },
                )
            ),
        ) as client:
            return await _fetch_user_profile(client, _case())

    with pytest.raises(ValueError, match="diverge"):
        asyncio.run(scenario())


def test_live_experiment_can_repeat_the_same_destination_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    output_dir = tmp_path / "live-experiment"
    provider_options: list[tuple[int, int]] = []

    def local_provider(name, environment, max_retries, output_parse_retries):
        provider_options.append((max_retries, output_parse_retries))
        return _LocalProvider()

    monkeypatch.setattr(
        "tractian_agent.evaluation.live_experiment._provider",
        local_provider,
    )
    monkeypatch.setattr(
        "tractian_agent.evaluation.live_experiment._apply_model_pacing",
        lambda model, _: model,
    )

    async def local_profile(_, benchmark: BenchmarkInput) -> UserRuntimeProfile:
        return UserRuntimeProfile(
            id=benchmark.user_id,
            company_id=benchmark.company_id,
            role="test-role",
            permissions=frozenset({"read"}),
        )

    monkeypatch.setattr(
        "tractian_agent.evaluation.live_experiment._fetch_user_profile",
        local_profile,
    )

    async def run_twice():
        options = LiveExperimentOptions(
            provider="groq",
            api_base_url="https://simulator.test",
        )
        first = await run_live_experiment(
            root=root,
            config_path=root / "eval/experiment-config.json",
            output_dir=output_dir,
            code_revision="test-revision",
            dirty=False,
            environment={},
            options=options,
        )
        second = await run_live_experiment(
            root=root,
            config_path=root / "eval/experiment-config.json",
            output_dir=output_dir,
            code_revision="test-revision",
            dirty=False,
            environment={},
            options=options,
        )
        return first, second

    first, second = asyncio.run(run_twice())

    assert first.profile == second.profile == "live-groq"
    assert first.total_runs == second.total_runs == 34
    assert second.programmatic_report_path.exists()
    assert provider_options == [(3, 2), (3, 2)]
