import asyncio
import json
from pathlib import Path

from tractian_agent.evaluation.offline_experiment import run_offline_experiment


def test_offline_experiment_runs_all_17_cases_twice_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]

    result = asyncio.run(
        run_offline_experiment(
            root=root,
            config_path=root / "eval/experiment-config.json",
            output_dir=tmp_path / "experiment",
            code_revision="test-revision",
            dirty=False,
        )
    )

    assert result.total_cases == 17
    assert result.total_runs == 34
    assert result.profile == "deterministic-fallback"
    assert result.judges == "not_run"
    assert result.human_calibration == "awaiting_labels"
    assert result.manifest_path.exists()
    assert result.programmatic_report_path.exists()
    assert result.blind_packet_path.exists()

    report = json.loads(result.programmatic_report_path.read_text(encoding="utf-8"))
    assert report["total_runs"] == 34
    assert len({item["case_id"] for item in report["cases"]}) == 17
    assert all(len(item["checks"]) == 10 for item in report["cases"])

    blind = json.loads(result.blind_packet_path.read_text(encoding="utf-8"))
    assert len(blind["items"]) == 24
    assert "judge" not in json.dumps(blind).casefold()
    assert "score" not in json.dumps(blind).casefold()
