"""Contratos persistíveis do benchmark offline."""

from typing import Literal

from pydantic import ConfigDict, Field, JsonValue, model_validator

from tractian_agent.contracts import StrictModel


class EvaluationModel(StrictModel):
    """Base estrita e imutável para artefatos de avaliação."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BenchmarkInput(EvaluationModel):
    """Únicos campos que o runner pode entregar ao agente avaliado."""

    id: str = Field(min_length=1, pattern=r"^case_[a-z0-9_]+$")
    ticket_id: str = Field(min_length=1, pattern=r"^TKT-[A-Za-z0-9-]+$")
    company_id: str = Field(min_length=1, pattern=r"^comp_[a-z0-9_]+$")
    user_id: str = Field(min_length=1, pattern=r"^usr_[a-z0-9_]+$")
    asset_id: str = Field(min_length=1, pattern=r"^asset_[A-Za-z0-9]+$")
    message: str = Field(min_length=1, pattern=r"\S")


class EvaluationOutput(EvaluationModel):
    """Resultado e trajetória observáveis de uma execução do agente."""

    case_id: str = Field(min_length=1, pattern=r"^case_[a-z0-9_]+$")
    ticket_id: str = Field(
        default="TKT-UNKNOWN", min_length=1, pattern=r"^TKT-[A-Za-z0-9-]+$"
    )
    decision: Literal[
        "guide",
        "act",
        "request_information",
        "request_confirmation",
        "require_human_review",
        "escalate",
    ] = "require_human_review"
    message: str = Field(
        default="Execução sem resultado público.", min_length=1, pattern=r"\S"
    )
    permissions: tuple[
        Literal["read", "action_low", "action_high", "escalate"], ...
    ] = ()
    steps: tuple["ObservedStep", ...] = ()
    step_count: int = Field(default=0, ge=0, strict=True)
    step_limit: int = Field(default=1, gt=0, strict=True)
    planner_selection_count: int = Field(default=0, ge=0, le=8, strict=True)
    planner_finalization_count: int = Field(default=0, ge=0, le=1, strict=True)
    writer_attempts: int = Field(default=0, ge=0, le=2, strict=True)
    gate_outcome: (
        Literal[
            "release",
            "request_information",
            "request_confirmation",
            "require_human_review",
        ]
        | None
    ) = None
    evidence_ids: tuple[str, ...] = ()
    limitation_refs: tuple[str, ...] = ()
    failure_codes: tuple[str, ...] = ()
    duration_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class ObservedStep(EvaluationModel):
    """Chamada de tool ligada, quando aplicável, ao efeito HTTP observado."""

    ordinal: int = Field(gt=0, strict=True)
    call_id: str = Field(min_length=1, pattern=r"^\S+$")
    tool_name: str = Field(min_length=1, pattern=r"^\S+$")
    arguments: dict[str, JsonValue]
    method: Literal["GET", "POST", "PATCH"] | None = None
    resource: str | None = Field(default=None, pattern=r"^/\S*$")
    outcome: Literal["success", "error"]
    error_code: str | None = Field(default=None, pattern=r"^\S+$")

    @model_validator(mode="after")
    def _require_coherent_transport_and_outcome(self) -> "ObservedStep":
        if (self.method is None) != (self.resource is None):
            raise ValueError("method e resource devem existir juntos")
        if (self.outcome == "error") != (self.error_code is not None):
            raise ValueError("error_code deve existir somente para erro")
        return self


class ProgrammaticSubject(EvaluationModel):
    """Saída já concluída que a segunda etapa entrega aos avaliadores."""

    benchmark_input: BenchmarkInput
    observed: EvaluationOutput


class ExpectedPathStep(EvaluationModel):
    """Passo de referência reservado ao processo avaliador."""

    step: str = Field(pattern=r"^(GET|POST|PATCH) /[^\r\n]+$")
    note: str = Field(min_length=1, pattern=r"\S")


class ExpectedCase(EvaluationModel):
    """Gabarito carregado somente pela fase posterior à execução."""

    id: str = Field(min_length=1, pattern=r"^case_[a-z0-9_]+$")
    ticket_id: str = Field(min_length=1, pattern=r"^TKT-[A-Za-z0-9-]+$")
    root_question: str = Field(min_length=1, pattern=r"\S")
    mode: Literal[
        "complete",
        "partial",
        "inconclusive",
        "conflict",
        "unavailable",
        "pending",
        "stale",
    ]
    expected_path: tuple[ExpectedPathStep, ...] = Field(min_length=1)
