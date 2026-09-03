"""Lote cego e métricas determinísticas da calibração humana."""

from __future__ import annotations

import hashlib

from pydantic import Field

from tractian_agent.evaluation.contracts import (
    BenchmarkInput,
    EvaluationModel,
    EvaluationOutput,
)


class BlindCandidate(EvaluationModel):
    run_id: str = Field(min_length=1, pattern=r"^\S+$")
    benchmark: BenchmarkInput
    output: EvaluationOutput


class BlindReviewItem(EvaluationModel):
    review_id: str = Field(pattern=r"^review_[0-9a-f]{24}$")
    case_id: str = Field(min_length=1, pattern=r"^case_[a-z0-9_]+$")
    ticket_id: str = Field(min_length=1, pattern=r"^TKT-[A-Za-z0-9-]+$")
    request_message: str = Field(min_length=1, pattern=r"\S")
    response_decision: str = Field(min_length=1, pattern=r"^\S+$")
    response_message: str = Field(min_length=1, pattern=r"\S")


class BlindReviewPacket(EvaluationModel):
    schema_version: str = "human-calibration-blind-v1"
    items: tuple[BlindReviewItem, ...] = Field(min_length=20, max_length=30)


class HumanLabel(EvaluationModel):
    review_id: str = Field(min_length=1, pattern=r"^\S+$")
    approved: bool
    reason: str = Field(min_length=1, max_length=500, pattern=r"\S")


class HumanLabelDraft(EvaluationModel):
    review_id: str = Field(min_length=1, pattern=r"^\S+$")
    approved: bool | None = None
    reason: str = ""


class HumanLabelTemplate(EvaluationModel):
    schema_version: str = "human-calibration-label-template-v1"
    labels: tuple[HumanLabelDraft, ...] = Field(min_length=20, max_length=30)


class JudgeScore(EvaluationModel):
    review_id: str = Field(min_length=1, pattern=r"^\S+$")
    score: float = Field(ge=0, le=1, allow_inf_nan=False)


class CalibrationMetric(EvaluationModel):
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    sample_size: int = Field(ge=20, le=30, strict=True)
    raw_agreement: float = Field(ge=0, le=1, allow_inf_nan=False)
    cohens_kappa: float = Field(ge=-1, le=1, allow_inf_nan=False)
    false_approved: int = Field(ge=0, strict=True)
    false_rejected: int = Field(ge=0, strict=True)
    review_rate: float = Field(ge=0, le=1, allow_inf_nan=False)


class CalibrationReport(EvaluationModel):
    metrics: tuple[CalibrationMetric, ...]
    chosen_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    rationale: str = Field(min_length=1, pattern=r"\S")
    limitations: tuple[str, ...]


def _review_id(candidate: BlindCandidate) -> str:
    material = (
        f"human-calibration-blind-v1\0{candidate.run_id}\0"
        f"{candidate.benchmark.id}\0{candidate.output.message}"
    ).encode("utf-8")
    return f"review_{hashlib.sha256(material).hexdigest()[:24]}"


def build_blind_review_packet(
    candidates: tuple[BlindCandidate, ...],
    *,
    sample_size: int,
) -> BlindReviewPacket:
    """Cria lote sem notas, gabarito ou trace para rotulagem humana."""

    if not 20 <= sample_size <= 30:
        raise ValueError("a calibração exige de 20 a 30 saídas")
    if len(candidates) < sample_size:
        raise ValueError("não há saídas suficientes para o lote cego")
    selected = candidates[:sample_size]
    items = tuple(
        BlindReviewItem(
            review_id=_review_id(candidate),
            case_id=candidate.benchmark.id,
            ticket_id=candidate.benchmark.ticket_id,
            request_message=candidate.benchmark.message,
            response_decision=candidate.output.decision,
            response_message=candidate.output.message,
        )
        for candidate in selected
    )
    if len({item.review_id for item in items}) != len(items):
        raise ValueError("o lote cego contém saídas duplicadas")
    return BlindReviewPacket(items=items)


def build_human_label_template(
    packet: BlindReviewPacket,
) -> HumanLabelTemplate:
    """Cria campos vazios preservando a cegueira em relação aos juízes."""

    return HumanLabelTemplate(
        labels=tuple(HumanLabelDraft(review_id=item.review_id) for item in packet.items)
    )


def _cohens_kappa(
    *,
    human_approved: int,
    judge_approved: int,
    agreement: int,
    sample_size: int,
) -> float:
    observed = agreement / sample_size
    human_rate = human_approved / sample_size
    judge_rate = judge_approved / sample_size
    expected = human_rate * judge_rate + (1 - human_rate) * (1 - judge_rate)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def calibrate_thresholds(
    labels: tuple[HumanLabel, ...],
    judge_scores: tuple[JudgeScore, ...],
    *,
    thresholds: tuple[float, ...],
) -> CalibrationReport:
    """Compara rótulos cegos com os mesmos scores em todos os cortes."""

    if not 20 <= len(labels) <= 30:
        raise ValueError("a calibração exige de 20 a 30 rótulos humanos")
    if not thresholds:
        raise ValueError("ao menos um limiar é obrigatório")
    labels_by_id = {item.review_id: item for item in labels}
    scores_by_id = {item.review_id: item for item in judge_scores}
    if len(labels_by_id) != len(labels) or len(scores_by_id) != len(judge_scores):
        raise ValueError("IDs duplicados na calibração")
    if set(labels_by_id) != set(scores_by_id):
        raise ValueError("rótulos humanos e scores cobrem conjuntos diferentes")
    metrics = []
    human_approved = sum(item.approved for item in labels)
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("limiares devem estar entre 0 e 1")
        agreement = 0
        false_approved = 0
        false_rejected = 0
        judge_approved = 0
        for review_id, human in labels_by_id.items():
            predicted = scores_by_id[review_id].score >= threshold
            judge_approved += predicted
            agreement += predicted == human.approved
            false_approved += predicted and not human.approved
            false_rejected += not predicted and human.approved
        size = len(labels)
        metrics.append(
            CalibrationMetric(
                threshold=threshold,
                sample_size=size,
                raw_agreement=round(agreement / size, 12),
                cohens_kappa=round(
                    _cohens_kappa(
                        human_approved=human_approved,
                        judge_approved=judge_approved,
                        agreement=agreement,
                        sample_size=size,
                    ),
                    12,
                ),
                false_approved=false_approved,
                false_rejected=false_rejected,
                review_rate=round((size - judge_approved) / size, 12),
            )
        )
    chosen = min(
        metrics,
        key=lambda item: (
            item.false_approved,
            item.false_rejected,
            item.review_rate,
            item.threshold,
        ),
    )
    return CalibrationReport(
        metrics=tuple(metrics),
        chosen_threshold=chosen.threshold,
        rationale=(
            f"O limiar {chosen.threshold:.1f} minimiza primeiro falsos aprovados "
            f"({chosen.false_approved}), depois falsos reprovados "
            f"({chosen.false_rejected}) e taxa de revisão ({chosen.review_rate:.1%})."
        ),
        limitations=(
            "A referência humana individual foi produzida por uma única pessoa avaliadora; não mede concordância entre especialistas.",
        ),
    )
