from pathlib import Path

from tractian_agent.evaluation.experiment_config import (
    build_experiment_manifest,
    load_experiment_config,
)


def test_versioned_experiment_config_freezes_all_reproducibility_inputs() -> None:
    root = Path(__file__).resolve().parents[2]

    config = load_experiment_config(root / "eval/experiment-config.json")
    manifest = build_experiment_manifest(
        config,
        root=root,
        code_revision="abc123",
        dirty=False,
    )

    assert config.version == "evaluation-experiment-v1"
    assert config.repetitions == 2
    assert config.thresholds == (0.7, 0.8, 0.9)
    assert config.human_sample_size == 24
    assert {provider.provider for provider in config.providers} == {
        "groq",
        "nvidia-nim",
    }
    assert all(
        provider.planner.model_id == "openai/gpt-oss-20b"
        and provider.writer.model_id == "openai/gpt-oss-20b"
        for provider in config.providers
    )
    assert config.judges.provider == "groq"
    assert config.judges.blind_result.model_id == "openai/gpt-oss-120b"
    assert config.judges.trajectory.model_id == "openai/gpt-oss-120b"
    assert {pin.component for pin in config.versions} >= {
        "dataset",
        "planner_prompt",
        "writer_prompt",
        "blind_result_rubric",
        "trajectory_rubric",
        "programmatic_checks",
    }
    assert manifest.code_revision == "abc123"
    assert manifest.dirty is False
    assert {item.path for item in manifest.files} == {
        "agent-input/cases.json",
        "eval/expected-paths.json",
    }
    assert all(item.digest.startswith("sha256:v1:") for item in manifest.files)
    assert {item.name for item in manifest.packages} >= {
        "pydantic-evals",
        "langchain",
        "langgraph",
    }
    wire = manifest.model_dump_json()
    assert "api_key" not in wire.casefold()
    assert "secret" not in wire.casefold()
