from tractian_agent.evaluation.calibration import (
    BlindCandidate,
    HumanLabel,
    JudgeScore,
    build_blind_review_packet,
    build_human_label_template,
    calibrate_thresholds,
)
from tractian_agent.evaluation.contracts import BenchmarkInput, EvaluationOutput


def _candidate(index: int) -> BlindCandidate:
    case_id = f"case_calibration_{index:02d}"
    return BlindCandidate(
        run_id=f"run_{index:02d}",
        benchmark=BenchmarkInput(
            id=case_id,
            ticket_id=f"TKT-CAL-{index:02d}",
            company_id="comp_aurora",
            user_id="usr_lucas",
            asset_id="asset_B204",
            message=f"Solicitação sintética {index}",
        ),
        output=EvaluationOutput(
            case_id=case_id,
            ticket_id=f"TKT-CAL-{index:02d}",
            decision="guide",
            message=f"Resposta candidata {index}",
        ),
    )


def test_blind_packet_has_20_to_30_outputs_and_no_judge_or_golden_fields() -> None:
    packet = build_blind_review_packet(
        tuple(_candidate(index) for index in range(24)),
        sample_size=24,
    )

    assert len(packet.items) == 24
    wire = packet.model_dump_json()
    assert "score" not in wire
    assert "judge" not in wire
    assert "expected" not in wire
    assert "trace" not in wire
    assert len({item.review_id for item in packet.items}) == 24

    template = build_human_label_template(packet)
    assert all(item.approved is None and item.reason == "" for item in template.labels)
    template_wire = template.model_dump_json()
    assert "score" not in template_wire
    assert "expected" not in template_wire


def test_calibration_compares_same_scores_at_three_thresholds() -> None:
    labels = tuple(
        HumanLabel(
            review_id=f"review_{index:02d}",
            approved=index < 10,
            reason="rótulo humano cego",
        )
        for index in range(20)
    )
    scores = (
        *(0.95 for _ in range(8)),
        0.85,
        0.75,
        0.85,
        0.75,
        *(0.2 for _ in range(8)),
    )
    judge_scores = tuple(
        JudgeScore(review_id=f"review_{index:02d}", score=score)
        for index, score in enumerate(scores)
    )

    report = calibrate_thresholds(
        labels,
        judge_scores,
        thresholds=(0.7, 0.8, 0.9),
    )

    assert [metric.threshold for metric in report.metrics] == [0.7, 0.8, 0.9]
    assert [metric.false_approved for metric in report.metrics] == [2, 1, 0]
    assert [metric.false_rejected for metric in report.metrics] == [0, 1, 2]
    assert [metric.raw_agreement for metric in report.metrics] == [0.9, 0.9, 0.9]
    assert [metric.cohens_kappa for metric in report.metrics] == [0.8, 0.8, 0.8]
    assert [metric.review_rate for metric in report.metrics] == [0.4, 0.5, 0.6]
    assert report.chosen_threshold == 0.9
    assert "falsos aprovados" in report.rationale
    assert report.limitations == (
        "A referência humana individual foi produzida por uma única pessoa avaliadora; não mede concordância entre especialistas.",
    )
