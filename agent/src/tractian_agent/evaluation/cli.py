"""CLI dos experimentos offline, comparação de providers e calibração."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from io import TextIOBase
import os
from pathlib import Path
import subprocess
import sys

from pydantic import TypeAdapter

from tractian_agent.evaluation.artifacts import write_json_artifact
from tractian_agent.evaluation.artifacts import ProgrammaticArtifact
from tractian_agent.evaluation.calibration import (
    BlindCandidate,
    BlindReviewPacket,
    CalibrationReport,
    HumanLabel,
    HumanLabelTemplate,
    JudgeScore,
    build_blind_review_packet,
    build_human_label_template,
    calibrate_thresholds,
)
from tractian_agent.evaluation.dataset import load_public_dataset, load_reference_cases
from tractian_agent.evaluation.experiment_config import load_experiment_config
from tractian_agent.evaluation.experiment_analysis import compare_all_thresholds
from tractian_agent.evaluation.judge_runner import (
    JudgeInputRecord,
    OfflineJudgeReport,
    conservative_calibration_score,
    run_offline_judge_records,
)
from tractian_agent.evaluation.judge_provider import GroqJudgeModelProvider
from tractian_agent.evaluation.judges import BlindResultJudge, TrajectoryJudge
from tractian_agent.evaluation.live_experiment import (
    LiveExperimentOptions,
    run_live_experiment,
)
from tractian_agent.evaluation.offline_experiment import run_offline_experiment
from tractian_agent.evaluation.provider_benchmark import (
    compare_provider_reports,
    run_provider_benchmark,
)
from tractian_agent.groq_provider import GroqModelProvider
from tractian_agent.nvidia_nim_provider import NvidiaNimModelProvider


_LABELS = TypeAdapter(list[HumanLabel])
_SCORES = TypeAdapter(list[JudgeScore])


def _load_human_labels(path: Path) -> tuple[HumanLabel, ...]:
    payload = path.read_text(encoding="utf-8")
    if payload.lstrip().startswith("{"):
        template = HumanLabelTemplate.model_validate_json(payload, strict=True)
        if any(item.approved is None for item in template.labels):
            raise ValueError("todos os rótulos humanos devem ser preenchidos")
        return tuple(
            HumanLabel(
                review_id=item.review_id,
                approved=item.approved,
                reason=item.reason,
            )
            for item in template.labels
        )
    return tuple(_LABELS.validate_json(payload, strict=True))


def _default_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _git_state(root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, dirty


async def _run_providers(
    *,
    root: Path,
    config_path: Path,
    output_path: Path,
    environment: Mapping[str, str],
) -> int:
    config = load_experiment_config(config_path)
    specs = {item.provider: item for item in config.providers}
    providers = (
        GroqModelProvider.from_env(environment),
        NvidiaNimModelProvider.from_env(environment),
    )
    reports = (
        await run_provider_benchmark(providers[0], specs["groq"]),
        await run_provider_benchmark(providers[1], specs["nvidia-nim"]),
    )
    comparison = compare_provider_reports(reports)
    write_json_artifact(output_path, comparison)
    return 0 if all(role.passed for report in reports for role in report.roles) else 1


def _configured_provider(
    name: str,
    environment: Mapping[str, str],
) -> GroqModelProvider | NvidiaNimModelProvider:
    if name == "groq":
        return GroqModelProvider.from_env(environment)
    if name == "nvidia-nim":
        return NvidiaNimModelProvider.from_env(environment)
    raise ValueError("provider deve ser groq ou nvidia-nim")


async def _run_judges(
    *,
    root: Path,
    config_path: Path,
    report_path: Path,
    output_path: Path,
    scores_path: Path,
    comparison_path: Path,
    provider_name: str,
    environment: Mapping[str, str],
) -> None:
    config = load_experiment_config(config_path)
    artifact = ProgrammaticArtifact.model_validate_json(
        report_path.read_text(encoding="utf-8"),
        strict=True,
    )
    if artifact.config_digest != config.digest():
        raise ValueError("relatório programático pertence a outra configuração")
    public = load_public_dataset(root / config.public_cases_path)
    benchmark_by_id = {case.inputs.id: case.inputs for case in public.cases}
    references = load_reference_cases(root / config.reference_cases_path)
    records = tuple(
        JudgeInputRecord(
            run_id=item.run_id,
            benchmark=benchmark_by_id[item.case_id],
            observed=item.output,
            reference=references[item.case_id],
        )
        for item in artifact.cases
    )
    if provider_name != config.judges.provider:
        raise ValueError("provider dos juízes diverge da configuração congelada")
    provider = (
        GroqJudgeModelProvider.from_env(environment)
        if provider_name == "groq"
        else _configured_provider(provider_name, environment)
    )
    judge_report = await run_offline_judge_records(
        records,
        blind_result_judge=BlindResultJudge(
            provider.create_chat_model(config.judges.blind_result)
        ),
        trajectory_judge=TrajectoryJudge(
            provider.create_chat_model(config.judges.trajectory)
        ),
        thresholds=config.thresholds,
        provider=provider_name,
        blind_model_id=config.judges.blind_result.model_id,
        trajectory_model_id=config.judges.trajectory.model_id,
        pacing_seconds=config.judges.pacing_seconds,
    )
    candidates = tuple(
        BlindCandidate(
            run_id=item.run_id,
            benchmark=benchmark_by_id[item.case_id],
            output=item.output,
        )
        for item in artifact.cases
    )
    packet = build_blind_review_packet(
        candidates,
        sample_size=config.human_sample_size,
    )
    judged_by_run = {item.run_id: item for item in judge_report.cases}
    scores = tuple(
        JudgeScore(
            review_id=review.review_id,
            score=conservative_calibration_score(
                judged_by_run[candidate.run_id].blind_result,
                judged_by_run[candidate.run_id].trajectory,
            ),
        )
        for review, candidate in zip(
            packet.items,
            candidates[: config.human_sample_size],
            strict=True,
        )
    )
    write_json_artifact(output_path, judge_report)
    write_json_artifact(
        comparison_path,
        compare_all_thresholds(artifact, judge_report),
    )
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(
        TypeAdapter(tuple[JudgeScore, ...]).dump_json(scores, indent=2).decode()
        + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tractian-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    offline = subparsers.add_parser("offline")
    offline.add_argument("--root", type=Path, default=_default_root())
    offline.add_argument("--config", type=Path)
    offline.add_argument("--output-dir", type=Path)
    offline.add_argument("--code-revision")
    offline.add_argument("--clean", action="store_true")

    live = subparsers.add_parser("live")
    live.add_argument("--root", type=Path, default=_default_root())
    live.add_argument("--config", type=Path)
    live.add_argument("--output-dir", type=Path)
    live.add_argument("--provider", choices=("groq", "nvidia-nim"), default="groq")
    live.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    live.add_argument("--code-revision")
    live.add_argument("--clean", action="store_true")

    providers = subparsers.add_parser("providers")
    providers.add_argument("--root", type=Path, default=_default_root())
    providers.add_argument("--config", type=Path)
    providers.add_argument("--output", type=Path)

    judges = subparsers.add_parser("judges")
    judges.add_argument("--root", type=Path, default=_default_root())
    judges.add_argument("--config", type=Path)
    judges.add_argument("--programmatic-report", type=Path)
    judges.add_argument("--output", type=Path)
    judges.add_argument("--scores-output", type=Path)
    judges.add_argument("--comparison-output", type=Path)
    judges.add_argument(
        "--provider",
        choices=("groq", "nvidia-nim"),
        default="groq",
    )

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--labels", type=Path, required=True)
    calibrate.add_argument("--scores", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)

    labels_template = subparsers.add_parser("labels-template")
    labels_template.add_argument("--packet", type=Path, required=True)
    labels_template.add_argument("--output", type=Path, required=True)

    layers = subparsers.add_parser("layers")
    layers.add_argument("--programmatic-report", type=Path, required=True)
    layers.add_argument("--judge-report", type=Path, required=True)
    layers.add_argument("--calibration-report", type=Path)
    layers.add_argument("--output", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    output: TextIOBase | None = None,
) -> int:
    """Executa subcomandos sem imprimir exceções, prompts ou credenciais."""

    args = _parser().parse_args(argv)
    environment = dict(os.environ if environment is None else environment)
    output = sys.stdout if output is None else output
    if args.command in {"offline", "live"}:
        root = args.root.resolve()
        if args.command == "live":
            required_key = (
                "GROQ_API_KEY"
                if args.provider == "groq"
                else "NVIDIA_API_KEY"
            )
            local_nim = args.provider == "nvidia-nim" and environment.get(
                "NVIDIA_NIM_BASE_URL", ""
            ).startswith(("http://127.0.0.1", "http://localhost", "http://[::1]"))
            if not environment.get(required_key) and not local_nim:
                print(
                    "status=skipped reason=missing_agent_configuration "
                    f"provider={args.provider}",
                    file=output,
                )
                return 0
        config_path = args.config or root / "eval/experiment-config.json"
        config = load_experiment_config(config_path)
        output_dir = args.output_dir or root / ".run/evaluation" / config.experiment_id
        if args.code_revision is None:
            code_revision, dirty = _git_state(root)
        else:
            code_revision = args.code_revision
            dirty = not args.clean
        experiment = run_offline_experiment
        kwargs = {}
        if args.command == "live":
            experiment = run_live_experiment
            kwargs = {
                "environment": environment,
                "options": LiveExperimentOptions(
                    provider=args.provider,
                    api_base_url=args.api_base_url,
                ),
            }
        try:
            result = asyncio.run(
                experiment(
                    root=root,
                    config_path=config_path,
                    output_dir=output_dir,
                    code_revision=code_revision,
                    dirty=dirty,
                    **kwargs,
                )
            )
        except Exception as error:
            if args.command == "offline":
                raise
            print(
                "status=failed stage=live_experiment "
                f"error_type={type(error).__name__}",
                file=output,
            )
            return 1
        print(
            f"status=completed profile={result.profile} "
            f"cases={result.total_cases} runs={result.total_runs}",
            file=output,
        )
        print(f"manifest={result.manifest_path}", file=output)
        print(f"programmatic_report={result.programmatic_report_path}", file=output)
        print(f"blind_packet={result.blind_packet_path}", file=output)
        return 0

    if args.command == "providers":
        root = args.root.resolve()
        config_path = args.config or root / "eval/experiment-config.json"
        output_path = args.output or root / ".run/evaluation/provider-comparison.json"
        missing = []
        if not environment.get("GROQ_API_KEY"):
            missing.append("groq")
        if not environment.get("NVIDIA_API_KEY") and not environment.get(
            "NVIDIA_NIM_BASE_URL"
        ):
            missing.append("nvidia-nim")
        if missing:
            print(
                "status=skipped reason=missing_provider_configuration "
                f"providers={','.join(missing)}",
                file=output,
            )
            return 0
        try:
            exit_code = asyncio.run(
                _run_providers(
                    root=root,
                    config_path=config_path,
                    output_path=output_path,
                    environment=environment,
                )
            )
        except Exception as error:
            print(
                "status=failed stage=provider_benchmark "
                f"error_type={type(error).__name__}",
                file=output,
            )
            return 1
        print(f"status={'completed' if exit_code == 0 else 'failed'} report={output_path}", file=output)
        return exit_code

    if args.command == "judges":
        root = args.root.resolve()
        config_path = args.config or root / "eval/experiment-config.json"
        config = load_experiment_config(config_path)
        directory = root / ".run/evaluation" / config.experiment_id
        report_path = args.programmatic_report or directory / "programmatic-report.json"
        output_path = args.output or directory / "judge-report.json"
        scores_path = args.scores_output or directory / "judge-scores.json"
        comparison_path = (
            args.comparison_output or directory / "evaluation-layers.json"
        )
        required_key = (
            "GROQ_API_KEY" if args.provider == "groq" else "NVIDIA_API_KEY"
        )
        if not environment.get(required_key):
            print(
                "status=skipped reason=missing_judge_configuration "
                f"provider={args.provider}",
                file=output,
            )
            return 0
        try:
            asyncio.run(
                _run_judges(
                    root=root,
                    config_path=config_path,
                    report_path=report_path,
                    output_path=output_path,
                    scores_path=scores_path,
                    comparison_path=comparison_path,
                    provider_name=args.provider,
                    environment=environment,
                )
            )
        except Exception as error:
            print(
                "status=failed stage=offline_judges "
                f"error_type={type(error).__name__}",
                file=output,
            )
            return 1
        print(
            f"status=completed report={output_path} scores={scores_path} "
            f"comparison={comparison_path}",
            file=output,
        )
        return 0

    if args.command == "labels-template":
        packet = BlindReviewPacket.model_validate_json(
            args.packet.read_text(encoding="utf-8"),
            strict=True,
        )
        write_json_artifact(args.output, build_human_label_template(packet))
        print(f"status=completed template={args.output}", file=output)
        return 0

    if args.command == "layers":
        programmatic = ProgrammaticArtifact.model_validate_json(
            args.programmatic_report.read_text(encoding="utf-8"),
            strict=True,
        )
        judges = OfflineJudgeReport.model_validate_json(
            args.judge_report.read_text(encoding="utf-8"),
            strict=True,
        )
        calibration = (
            CalibrationReport.model_validate_json(
                args.calibration_report.read_text(encoding="utf-8"),
                strict=True,
            )
            if args.calibration_report is not None
            else None
        )
        write_json_artifact(
            args.output,
            compare_all_thresholds(
                programmatic,
                judges,
                calibration=calibration,
            ),
        )
        print(f"status=completed comparison={args.output}", file=output)
        return 0

    labels = _load_human_labels(args.labels)
    scores = tuple(
        _SCORES.validate_json(args.scores.read_text(encoding="utf-8"), strict=True)
    )
    report = calibrate_thresholds(
        labels,
        scores,
        thresholds=(0.7, 0.8, 0.9),
    )
    write_json_artifact(args.output, report)
    print(
        f"status=completed chosen_threshold={report.chosen_threshold:.1f} "
        f"report={args.output}",
        file=output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
