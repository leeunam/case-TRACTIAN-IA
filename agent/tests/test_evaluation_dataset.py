import json
from pathlib import Path

from pydantic_evals import Dataset

from tractian_agent.evaluation.contracts import (
    BenchmarkInput,
    EvaluationOutput,
    ExpectedCase,
)
from tractian_agent.evaluation.dataset import load_public_dataset, load_reference_cases


def test_public_dataset_loads_sanitized_cases_without_expected_outputs(
    tmp_path: Path,
) -> None:
    public_cases = [
        {
            "id": "case_tkt_ctx_02",
            "ticket_id": "TKT-CTX-02",
            "company_id": "comp_aurora",
            "user_id": "usr_lucas",
            "asset_id": "asset_B204",
            "message": "O que significa BPFO?",
        }
    ]
    input_path = tmp_path / "cases.json"
    input_path.write_text(json.dumps(public_cases), encoding="utf-8")

    dataset = load_public_dataset(input_path)

    assert isinstance(dataset, Dataset)
    assert dataset.name == "tractian-industrial-support-v1"
    assert len(dataset.cases) == 1
    case = dataset.cases[0]
    assert case.name == "case_tkt_ctx_02"
    assert case.inputs == BenchmarkInput.model_validate(public_cases[0])
    assert case.expected_output is None
    assert dataset.evaluators == []
    assert Dataset[BenchmarkInput, EvaluationOutput, None]


def test_reference_cases_are_loaded_by_a_separate_post_execution_interface(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / "expected-paths.json"
    reference_path.write_text(
        json.dumps(
            [
                {
                    "id": "case_tkt_ctx_02",
                    "ticket_id": "TKT-CTX-02",
                    "root_question": "Definir BPFO.",
                    "mode": "partial",
                    "expected_path": [
                        {
                            "step": "GET /knowledge/search?q=BPFO",
                            "note": "localizar o glossário",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    references = load_reference_cases(reference_path)

    assert references == {
        "case_tkt_ctx_02": ExpectedCase(
            id="case_tkt_ctx_02",
            ticket_id="TKT-CTX-02",
            root_question="Definir BPFO.",
            mode="partial",
            expected_path=(
                {
                    "step": "GET /knowledge/search?q=BPFO",
                    "note": "localizar o glossário",
                },
            ),
        )
    }


def test_repository_dataset_contains_the_same_17_public_and_reference_cases() -> None:
    root = Path(__file__).resolve().parents[2]

    public = load_public_dataset(root / "agent-input/cases.json")
    references = load_reference_cases(root / "eval/expected-paths.json")

    assert len(public.cases) == 17
    assert {case.inputs.id for case in public.cases} == set(references)
    assert all(case.expected_output is None for case in public.cases)
    public_wire = (root / "agent-input/cases.json").read_text(encoding="utf-8")
    assert "root_question" not in public_wire
    assert "expected_path" not in public_wire
