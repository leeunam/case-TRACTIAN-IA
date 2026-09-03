from io import StringIO
import json
from pathlib import Path

from tractian_agent.evaluation.cli import main


def test_offline_cli_runs_reproducible_experiment_without_credentials(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    output = StringIO()

    exit_code = main(
        [
            "offline",
            "--root",
            str(root),
            "--config",
            str(root / "eval/experiment-config.json"),
            "--output-dir",
            str(tmp_path / "experiment"),
            "--code-revision",
            "test-revision",
            "--clean",
        ],
        environment={},
        output=output,
    )

    assert exit_code == 0
    assert "status=completed profile=deterministic-fallback cases=17 runs=34" in output.getvalue()

    template_path = tmp_path / "human-labels.json"
    template_output = StringIO()
    assert main(
        [
            "labels-template",
            "--packet",
            str(tmp_path / "experiment/blind-review-packet.json"),
            "--output",
            str(template_path),
        ],
        environment={},
        output=template_output,
    ) == 0
    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert len(template["labels"]) == 24
    assert all(item["approved"] is None for item in template["labels"])
    assert "score" not in template_path.read_text(encoding="utf-8")


def test_provider_cli_skips_safely_when_credentials_are_absent(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = StringIO()

    exit_code = main(
        [
            "providers",
            "--root",
            str(root),
            "--output",
            str(tmp_path / "providers.json"),
        ],
        environment={},
        output=output,
    )

    assert exit_code == 0
    assert output.getvalue().strip() == (
        "status=skipped reason=missing_provider_configuration "
        "providers=groq,nvidia-nim"
    )
    assert not (tmp_path / "providers.json").exists()


def test_live_cli_skips_safely_without_selected_provider_credentials(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    output = StringIO()

    exit_code = main(
        [
            "live",
            "--root",
            str(root),
            "--output-dir",
            str(tmp_path / "live"),
            "--provider",
            "groq",
        ],
        environment={},
        output=output,
    )

    assert exit_code == 0
    assert output.getvalue().strip() == (
        "status=skipped reason=missing_agent_configuration provider=groq"
    )
    assert not (tmp_path / "live").exists()


def test_judge_cli_skips_safely_without_selected_provider_credentials(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    output = StringIO()

    exit_code = main(
        ["judges", "--root", str(root), "--provider", "nvidia-nim"],
        environment={},
        output=output,
    )

    assert exit_code == 0
    assert output.getvalue().strip() == (
        "status=skipped reason=missing_judge_configuration provider=nvidia-nim"
    )


def test_provider_cli_accepts_nvidia_hosted_configuration_without_base_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    output = StringIO()

    async def fake_run_providers(**kwargs) -> int:
        assert kwargs["environment"]["NVIDIA_API_KEY"] == "nvidia-secret"
        return 0

    monkeypatch.setattr(
        "tractian_agent.evaluation.cli._run_providers",
        fake_run_providers,
    )
    report_path = tmp_path / "providers.json"
    exit_code = main(
        ["providers", "--root", str(root), "--output", str(report_path)],
        environment={
            "GROQ_API_KEY": "groq-secret",
            "NVIDIA_API_KEY": "nvidia-secret",
        },
        output=output,
    )

    assert exit_code == 0
    assert output.getvalue().strip() == (
        f"status=completed report={report_path}"
    )


def test_calibration_cli_writes_metrics_from_human_labels_and_judge_scores(
    tmp_path: Path,
) -> None:
    labels_path = tmp_path / "labels.json"
    scores_path = tmp_path / "scores.json"
    report_path = tmp_path / "calibration.json"
    labels_path.write_text(
        json.dumps(
            {
                "schema_version": "human-calibration-label-template-v1",
                "labels": [
                    {
                        "review_id": f"review_{index:02d}",
                        "approved": index < 10,
                        "reason": "rótulo humano cego",
                    }
                    for index in range(20)
                ],
            }
        ),
        encoding="utf-8",
    )
    scores_path.write_text(
        json.dumps(
            [
                {
                    "review_id": f"review_{index:02d}",
                    "score": 0.9 if index < 10 else 0.2,
                }
                for index in range(20)
            ]
        ),
        encoding="utf-8",
    )
    output = StringIO()

    exit_code = main(
        [
            "calibrate",
            "--labels",
            str(labels_path),
            "--scores",
            str(scores_path),
            "--output",
            str(report_path),
        ],
        environment={},
        output=output,
    )

    assert exit_code == 0
    assert "status=completed chosen_threshold=0.7" in output.getvalue()
    assert json.loads(report_path.read_text(encoding="utf-8"))[
        "chosen_threshold"
    ] == 0.7
