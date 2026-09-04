"""Experimento real do agente contra simulador e provider configurados."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter
from pydantic import Field

from tractian_agent.checkpoint import open_checkpointer
from tractian_agent.client import IndustrialApiClient
from tractian_agent.evaluation.artifacts import (
    build_programmatic_artifact,
    write_json_artifact,
)
from tractian_agent.evaluation.calibration import (
    BlindCandidate,
    build_blind_review_packet,
)
from tractian_agent.evaluation.checks import run_programmatic_checks
from tractian_agent.evaluation.contracts import BenchmarkInput, EvaluationModel
from tractian_agent.evaluation.dataset import load_public_dataset
from tractian_agent.evaluation.experiment_config import (
    build_experiment_manifest,
    load_experiment_config,
)
from tractian_agent.evaluation.offline_experiment import OfflineExperimentResult
from tractian_agent.evaluation.runner import (
    AgentCaseExecutor,
    execute_before_loading_references,
    isolated_evaluation_checkpoint,
)
from tractian_agent.graph import build_agent_graph
from tractian_agent.groq_provider import GroqModelProvider
from tractian_agent.model_provider import ModelProvider
from tractian_agent.nvidia_nim_provider import NvidiaNimModelProvider
from tractian_agent.planner import Planner
from tractian_agent.tools.runtime import Permission, WriteToolRuntime
from tractian_agent.writer import Writer


class UserRuntimeProfile(EvaluationModel):
    id: str = Field(min_length=1, pattern=r"^usr_\S+$")
    company_id: str = Field(min_length=1, pattern=r"^comp_\S+$")
    role: str = Field(min_length=1, pattern=r"\S")
    permissions: frozenset[Permission]


class _UserApiProfile(EvaluationModel):
    id: str = Field(min_length=1, pattern=r"^usr_\S+$")
    company_id: str = Field(min_length=1, pattern=r"^comp_\S+$")
    name: str = Field(min_length=1, pattern=r"\S")
    role: str = Field(min_length=1, pattern=r"\S")
    permissions: tuple[Permission, ...]


@dataclass(frozen=True)
class LiveExperimentOptions:
    provider: Literal["groq", "nvidia-nim"]
    api_base_url: str


def _provider(
    name: str,
    environment: dict[str, str],
    max_retries: int,
    output_parse_retries: int,
) -> ModelProvider:
    if name == "groq":
        return GroqModelProvider.from_env(
            environment,
            max_retries=max_retries,
            output_parse_retries=output_parse_retries,
        )
    if name == "nvidia-nim":
        if max_retries or output_parse_retries:
            raise ValueError("retries do NVIDIA NIM não foram configurados")
        return NvidiaNimModelProvider.from_env(environment)
    raise ValueError("provider do agente deve ser groq ou nvidia-nim")


def _live_rate_limiter(pacing_seconds: float) -> BaseRateLimiter | None:
    if pacing_seconds == 0:
        return None
    return InMemoryRateLimiter(
        requests_per_second=1 / pacing_seconds,
        check_every_n_seconds=min(0.1, pacing_seconds),
        max_bucket_size=1,
    )


def _apply_model_pacing(
    model: BaseChatModel,
    rate_limiter: BaseRateLimiter | None,
) -> BaseChatModel:
    """Compartilha a mesma cadência entre os papéis do agente."""

    model.rate_limiter = rate_limiter
    return model


async def _fetch_user_profile(
    client: httpx.AsyncClient,
    benchmark: BenchmarkInput,
) -> UserRuntimeProfile:
    response = await client.get(
        "/users/me",
        headers={"x-user-id": benchmark.user_id},
    )
    response.raise_for_status()
    wire = _UserApiProfile.model_validate_json(
        response.content,
        strict=True,
    )
    profile = UserRuntimeProfile(
        id=wire.id,
        company_id=wire.company_id,
        role=wire.role,
        permissions=frozenset(wire.permissions),
    )
    if profile.id != benchmark.user_id or profile.company_id != benchmark.company_id:
        raise ValueError("identidade retornada pela API diverge do caso público")
    return profile


async def run_live_experiment(
    *,
    root: Path,
    config_path: Path,
    output_dir: Path,
    code_revision: str,
    dirty: bool,
    environment: dict[str, str],
    options: LiveExperimentOptions,
) -> OfflineExperimentResult:
    """Executa os 17 casos no grafo LLM real e só depois carrega referências."""

    config = load_experiment_config(config_path)
    provider_spec = next(
        item for item in config.live_agents if item.provider == options.provider
    )
    model_provider = _provider(
        options.provider,
        environment,
        provider_spec.max_retries,
        provider_spec.output_parse_retries,
    )
    shared_rate_limiter = _live_rate_limiter(provider_spec.pacing_seconds)
    planner = Planner(
        _apply_model_pacing(
            model_provider.create_chat_model(provider_spec.planner),
            shared_rate_limiter,
        )
    )
    writer = Writer(
        _apply_model_pacing(
            model_provider.create_chat_model(provider_spec.writer),
            shared_rate_limiter,
        )
    )
    public_dataset = load_public_dataset(root / config.public_cases_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with isolated_evaluation_checkpoint(output_dir) as checkpoint_path:
        async with open_checkpointer(checkpoint_path) as saver:
            graph = build_agent_graph(saver, planner=planner, writer=writer)
            async with (
                IndustrialApiClient(options.api_base_url) as industrial_client,
                httpx.AsyncClient(
                    base_url=options.api_base_url,
                    follow_redirects=False,
                ) as identity_client,
            ):

                async def runtime_factory(case: BenchmarkInput) -> WriteToolRuntime:
                    profile = await _fetch_user_profile(identity_client, case)
                    return WriteToolRuntime.create(
                        user_id=profile.id,
                        company_id=profile.company_id,
                        permissions=profile.permissions,
                        central_asset_id=case.asset_id,
                        current_case_id=case.id,
                        client=industrial_client,
                    )

                executor = AgentCaseExecutor(
                    graph=graph,
                    runtime_factory=runtime_factory,
                    experiment_id=config.experiment_id,
                    step_limit=config.step_limit,
                )
                completed = await execute_before_loading_references(
                    public_dataset,
                    executor.execute,
                    reference_path=root / config.reference_cases_path,
                    repeat=config.repetitions,
                    max_concurrency=config.max_concurrency,
                )

    check_report = await run_programmatic_checks(completed)
    artifact = build_programmatic_artifact(
        check_report,
        experiment_id=config.experiment_id,
        config_digest=config.digest(),
    )
    benchmark_by_id = {case.inputs.id: case.inputs for case in public_dataset.cases}
    candidates = tuple(
        BlindCandidate(
            run_id=item.run_id,
            benchmark=benchmark_by_id[item.case_id],
            output=item.output,
        )
        for item in artifact.cases
    )
    blind_packet = build_blind_review_packet(
        candidates,
        sample_size=config.human_sample_size,
    )
    manifest = build_experiment_manifest(
        config,
        root=root,
        code_revision=code_revision,
        dirty=dirty,
    )
    manifest_path = output_dir / "manifest.json"
    report_path = output_dir / "programmatic-report.json"
    blind_path = output_dir / "blind-review-packet.json"
    write_json_artifact(manifest_path, manifest)
    write_json_artifact(report_path, artifact)
    write_json_artifact(blind_path, blind_packet)
    return OfflineExperimentResult(
        profile=f"live-{options.provider}",
        total_cases=len(public_dataset.cases),
        total_runs=artifact.total_runs,
        judges="not_run",
        human_calibration="awaiting_labels",
        manifest_path=manifest_path,
        programmatic_report_path=report_path,
        blind_packet_path=blind_path,
    )
