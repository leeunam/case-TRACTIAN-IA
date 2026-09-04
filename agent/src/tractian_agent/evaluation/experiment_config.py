"""Configuração congelada e manifesto verificável de cada experimento."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path, PurePosixPath
import platform
from typing import Literal

from pydantic import Field, field_validator, model_validator

from tractian_agent.evaluation.contracts import EvaluationModel
from tractian_agent.evaluation.provider_benchmark import ProviderBenchmarkSpec
from tractian_agent.model_provider import ModelConfig


class VersionPin(EvaluationModel):
    component: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    version: str = Field(min_length=1, pattern=r"^\S+$")


class JudgeModelSpec(EvaluationModel):
    provider: Literal["groq", "nvidia-nim"]
    blind_result: ModelConfig
    trajectory: ModelConfig
    pacing_seconds: float = Field(ge=0, le=60, allow_inf_nan=False)


class LiveAgentSpec(EvaluationModel):
    """Modelos do agente real, separados do benchmark entre providers."""

    provider: Literal["groq", "nvidia-nim"]
    pacing_seconds: float = Field(ge=0, le=60, allow_inf_nan=False)
    max_retries: int = Field(ge=0, le=5, strict=True)
    output_parse_retries: int = Field(ge=0, le=3, strict=True)
    planner: ModelConfig
    writer: ModelConfig


class ExperimentConfig(EvaluationModel):
    version: Literal["evaluation-experiment-v5"]
    experiment_id: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    public_cases_path: str
    reference_cases_path: str
    repetitions: int = Field(ge=1, le=10, strict=True)
    max_concurrency: int = Field(ge=1, le=16, strict=True)
    step_limit: int = Field(ge=3, le=128, strict=True)
    thresholds: tuple[float, ...] = Field(min_length=1)
    human_sample_size: int = Field(ge=20, le=30, strict=True)
    versions: tuple[VersionPin, ...] = Field(min_length=6)
    live_agents: tuple[LiveAgentSpec, ...] = Field(min_length=2)
    providers: tuple[ProviderBenchmarkSpec, ...] = Field(min_length=2)
    judges: JudgeModelSpec

    @field_validator("public_cases_path", "reference_cases_path")
    @classmethod
    def _require_safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not value:
            raise ValueError("caminho de experimento deve ser relativo à raiz")
        return value

    @model_validator(mode="after")
    def _require_unique_and_ordered_configuration(self) -> "ExperimentConfig":
        if tuple(sorted(set(self.thresholds))) != self.thresholds:
            raise ValueError("limiares devem ser únicos e crescentes")
        if any(not 0 <= item <= 1 for item in self.thresholds):
            raise ValueError("limiares devem estar entre 0 e 1")
        if len({item.component for item in self.versions}) != len(self.versions):
            raise ValueError("componentes versionados devem ser únicos")
        if len({item.provider for item in self.providers}) != len(self.providers):
            raise ValueError("providers devem ser únicos")
        if len({item.provider for item in self.live_agents}) != len(
            self.live_agents
        ):
            raise ValueError("providers live devem ser únicos")
        if {item.provider for item in self.live_agents} != {
            item.provider for item in self.providers
        }:
            raise ValueError("providers live e de benchmark devem coincidir")
        return self

    def digest(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"sha256:v1:{hashlib.sha256(payload).hexdigest()}"


class FilePin(EvaluationModel):
    path: str = Field(min_length=1, pattern=r"^\S+$")
    digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")


class PackagePin(EvaluationModel):
    name: str = Field(min_length=1, pattern=r"^\S+$")
    version: str = Field(min_length=1, pattern=r"^\S+$")


class ExperimentManifest(EvaluationModel):
    version: Literal["evaluation-manifest-v1"] = "evaluation-manifest-v1"
    experiment_id: str = Field(min_length=1, pattern=r"^[a-z0-9_-]+$")
    config_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    code_revision: str = Field(min_length=1, pattern=r"^\S+$")
    dirty: bool
    files: tuple[FilePin, ...]
    versions: tuple[VersionPin, ...]
    packages: tuple[PackagePin, ...]


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Valida a configuração congelada sem interpolar ambiente ou segredos."""

    return ExperimentConfig.model_validate_json(
        path.read_text(encoding="utf-8"),
        strict=True,
    )


def _file_pin(root: Path, relative_path: str) -> FilePin:
    path = root / relative_path
    payload = path.read_bytes()
    return FilePin(
        path=relative_path,
        digest=f"sha256:v1:{hashlib.sha256(payload).hexdigest()}",
    )


def build_experiment_manifest(
    config: ExperimentConfig,
    *,
    root: Path,
    code_revision: str,
    dirty: bool,
) -> ExperimentManifest:
    """Fixa arquivos, código e dependências observados na execução."""

    package_names = (
        "pydantic",
        "pydantic-evals",
        "langchain",
        "langgraph",
        "langchain-groq",
        "langchain-openai",
    )
    packages = [
        PackagePin(name="python", version=platform.python_version()),
        *(PackagePin(name=name, version=version(name)) for name in package_names),
    ]
    return ExperimentManifest(
        experiment_id=config.experiment_id,
        config_digest=config.digest(),
        code_revision=code_revision,
        dirty=dirty,
        files=(
            _file_pin(root, config.public_cases_path),
            _file_pin(root, config.reference_cases_path),
        ),
        versions=config.versions,
        packages=tuple(packages),
    )
