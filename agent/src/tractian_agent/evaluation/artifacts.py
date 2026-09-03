"""Artefatos JSON reproduzíveis e independentes da UI de observabilidade."""

from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
import tempfile

from pydantic import Field
from pydantic_evals.reporting import EvaluationReport

from tractian_agent.evaluation.contracts import (
    EvaluationModel,
    EvaluationOutput,
    ExpectedCase,
    ProgrammaticSubject,
)


class CheckRecord(EvaluationModel):
    dimension: str = Field(min_length=1, pattern=r"^\S+$")
    passed: bool
    reason: str | None = None
    evaluator_version: str | None = None


class CaseProgrammaticReport(EvaluationModel):
    run_id: str = Field(min_length=1, pattern=r"^\S+$")
    case_id: str = Field(min_length=1, pattern=r"^case_[a-z0-9_]+$")
    output: EvaluationOutput
    checks: tuple[CheckRecord, ...]
    passed: bool


class DimensionSummary(EvaluationModel):
    dimension: str = Field(min_length=1, pattern=r"^\S+$")
    total: int = Field(ge=0, strict=True)
    passed: int = Field(ge=0, strict=True)
    failed: int = Field(ge=0, strict=True)


class ProgrammaticArtifact(EvaluationModel):
    version: str = "programmatic-report-v1"
    experiment_id: str = Field(min_length=1, pattern=r"^\S+$")
    config_digest: str = Field(pattern=r"^sha256:v1:[0-9a-f]{64}$")
    total_runs: int = Field(ge=0, strict=True)
    passed_runs: int = Field(ge=0, strict=True)
    cases: tuple[CaseProgrammaticReport, ...]
    dimensions: tuple[DimensionSummary, ...]


def build_programmatic_artifact(
    report: EvaluationReport[ProgrammaticSubject, EvaluationOutput, ExpectedCase],
    *,
    experiment_id: str,
    config_digest: str,
) -> ProgrammaticArtifact:
    """Converte o relatório da biblioteca em contrato estável do projeto."""

    if report.failures:
        raise ValueError("o relatório contém execuções de checks com falha")
    cases = []
    dimension_values: dict[str, list[bool]] = defaultdict(list)
    for run_index, result in enumerate(report.cases, start=1):
        checks = tuple(
            CheckRecord(
                dimension=name,
                passed=assertion.value,
                reason=assertion.reason,
                evaluator_version=assertion.evaluator_version,
            )
            for name, assertion in sorted(result.assertions.items())
        )
        for item in checks:
            dimension_values[item.dimension].append(item.passed)
        cases.append(
            CaseProgrammaticReport(
                run_id=f"run_{run_index:03d}",
                case_id=result.inputs.benchmark_input.id,
                output=result.output,
                checks=checks,
                passed=all(item.passed for item in checks),
            )
        )
    dimensions = tuple(
        DimensionSummary(
            dimension=name,
            total=len(values),
            passed=sum(values),
            failed=len(values) - sum(values),
        )
        for name, values in sorted(dimension_values.items())
    )
    return ProgrammaticArtifact(
        experiment_id=experiment_id,
        config_digest=config_digest,
        total_runs=len(cases),
        passed_runs=sum(item.passed for item in cases),
        cases=tuple(cases),
        dimensions=dimensions,
    )


def write_json_artifact(path: Path, artifact: EvaluationModel) -> None:
    """Grava JSON completo por substituição atômica no mesmo diretório."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = artifact.model_dump_json(indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
