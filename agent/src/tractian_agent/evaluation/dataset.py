"""Carregamento do dataset público sem acesso ao gabarito."""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter
from pydantic_evals import Case, Dataset

from tractian_agent.evaluation.contracts import (
    BenchmarkInput,
    EvaluationOutput,
    ExpectedCase,
)


_PUBLIC_CASES_ADAPTER = TypeAdapter(list[BenchmarkInput])
_REFERENCE_CASES_ADAPTER = TypeAdapter(list[ExpectedCase])


def load_public_dataset(
    input_path: Path,
) -> Dataset[BenchmarkInput, EvaluationOutput, None]:
    """Converte somente o pacote sanitizado em casos sem saída esperada."""

    inputs = _PUBLIC_CASES_ADAPTER.validate_json(
        input_path.read_text(encoding="utf-8"), strict=True
    )
    return Dataset(
        name="tractian-industrial-support-v1",
        cases=[Case(name=item.id, inputs=item) for item in inputs],
    )


def load_reference_cases(reference_path: Path) -> dict[str, ExpectedCase]:
    """Carrega o gabarito por uma interface que o executor não recebe."""

    references = _REFERENCE_CASES_ADAPTER.validate_json(
        reference_path.read_text(encoding="utf-8"), strict=True
    )
    by_id = {item.id: item for item in references}
    if len(by_id) != len(references):
        raise ValueError("o gabarito contém IDs de caso duplicados")
    return by_id
