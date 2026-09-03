"""Experimento local reproduzível que não depende de juiz nem credencial."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

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
from tractian_agent.evaluation.dataset import load_public_dataset
from tractian_agent.evaluation.experiment_config import (
    build_experiment_manifest,
    load_experiment_config,
)
from tractian_agent.evaluation.runner import (
    AgentCaseExecutor,
    execute_before_loading_references,
)
from tractian_agent.graph import build_agent_graph
from tractian_agent.tools.runtime import ReadToolRuntime


@dataclass(frozen=True)
class OfflineExperimentResult:
    profile: str
    total_cases: int
    total_runs: int
    judges: str
    human_calibration: str
    manifest_path: Path
    programmatic_report_path: Path
    blind_packet_path: Path


async def run_offline_experiment(
    *,
    root: Path,
    config_path: Path,
    output_dir: Path,
    code_revision: str,
    dirty: bool,
) -> OfflineExperimentResult:
    """Executa o fallback real para provar runner, isolamento e relatórios."""

    config = load_experiment_config(config_path)
    public_dataset = load_public_dataset(root / config.public_cases_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    def forbidden_http(request: httpx.Request) -> httpx.Response:
        raise AssertionError(
            f"o perfil deterministic-fallback não pode chamar HTTP: {request.url}"
        )

    async with open_checkpointer(output_dir / "checkpoints.sqlite3") as saver:
        graph = build_agent_graph(saver)
        async with IndustrialApiClient(
            "https://offline-evaluation.invalid",
            transport=httpx.MockTransport(forbidden_http),
        ) as client:
            executor = AgentCaseExecutor(
                graph=graph,
                runtime_factory=lambda case: ReadToolRuntime.create(
                    user_id=case.user_id,
                    company_id=case.company_id,
                    permissions=frozenset({"read"}),
                    central_asset_id=case.asset_id,
                    client=client,
                ),
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
    candidates = tuple(
        BlindCandidate(
            run_id=f"run_{index:03d}",
            benchmark=result.inputs,
            output=result.output,
        )
        for index, result in enumerate(completed.execution_report.cases, start=1)
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
        profile="deterministic-fallback",
        total_cases=len(public_dataset.cases),
        total_runs=artifact.total_runs,
        judges="not_run",
        human_calibration="awaiting_labels",
        manifest_path=manifest_path,
        programmatic_report_path=report_path,
        blind_packet_path=blind_path,
    )
