"""Tools que criam propostas de escrita sem executar qualquer efeito."""

from __future__ import annotations

from typing import Final, Literal

from langchain_core.tools import StructuredTool
from pydantic import ConfigDict, JsonValue

from tractian_agent.contracts import StrictModel
from tractian_agent.tools.identifiers import AnalysisId
from tractian_agent.write_policy import (
    AssetCriticality,
    EscalateCaseProposal,
    ReprocessProposal,
    RequestModelRetrainingProposal,
    RequestSpecialistAnalysisProposal,
    UpdateAssetCriticalityProposal,
    WriteProposal,
)


WriteProposalToolName = Literal[
    "propose_reprocess_analysis",
    "propose_request_specialist_analysis",
    "propose_update_asset_criticality",
    "propose_request_model_retraining",
    "propose_escalate_case",
]


class WriteProposalContent(StrictModel):
    """Conteúdo explícito entregue ao modelo após uma proposta."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["proposed"] = "proposed"
    proposal: WriteProposal


class WriteProposalArtifact(StrictModel):
    """Registro que distingue proposta de efeito industrial executado."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["write_proposal"] = "write_proposal"
    tool_name: WriteProposalToolName
    proposal: WriteProposal
    effect_executed: Literal[False] = False


class ProposeReprocessAnalysisArguments(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    analysis_id: AnalysisId
    justification: str


class ProposeRequestSpecialistAnalysisArguments(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    analysis_id: AnalysisId
    justification: str


class ProposeUpdateAssetCriticalityArguments(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    criticality: AssetCriticality
    justification: str


class ProposeJustificationOnlyArguments(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    justification: str


class _WriteProposalStructuredTool(StructuredTool):
    """Preserva no schema público a rejeição explícita de campos extras."""

    @property
    def tool_call_schema(self):
        return self.args_schema


def _content_and_artifact(
    *,
    tool_name: WriteProposalToolName,
    proposal: WriteProposal,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    content = WriteProposalContent(proposal=proposal)
    artifact = WriteProposalArtifact(tool_name=tool_name, proposal=proposal)
    return content.model_dump(mode="json"), artifact.model_dump(mode="json")


def _propose_reprocess_analysis(
    analysis_id: AnalysisId,
    justification: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    proposal = ReprocessProposal(
        analysis_id=analysis_id,
        justification=justification,
    )
    return _content_and_artifact(
        tool_name="propose_reprocess_analysis",
        proposal=proposal,
    )


def _propose_request_specialist_analysis(
    analysis_id: AnalysisId,
    justification: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    proposal = RequestSpecialistAnalysisProposal(
        analysis_id=analysis_id,
        justification=justification,
    )
    return _content_and_artifact(
        tool_name="propose_request_specialist_analysis",
        proposal=proposal,
    )


def _propose_update_asset_criticality(
    criticality: AssetCriticality,
    justification: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    proposal = UpdateAssetCriticalityProposal(
        criticality=criticality,
        justification=justification,
    )
    return _content_and_artifact(
        tool_name="propose_update_asset_criticality",
        proposal=proposal,
    )


def _propose_request_model_retraining(
    justification: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    proposal = RequestModelRetrainingProposal(justification=justification)
    return _content_and_artifact(
        tool_name="propose_request_model_retraining",
        proposal=proposal,
    )


def _propose_escalate_case(
    justification: str,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    proposal = EscalateCaseProposal(justification=justification)
    return _content_and_artifact(
        tool_name="propose_escalate_case",
        proposal=proposal,
    )


propose_reprocess_analysis = _WriteProposalStructuredTool.from_function(
    func=_propose_reprocess_analysis,
    name="propose_reprocess_analysis",
    args_schema=ProposeReprocessAnalysisArguments,
    response_format="content_and_artifact",
    description=(
        "Propõe reprocessar uma análise após mudança de dados ou contexto. "
        "A tool apenas registra a proposta e não executa a ação."
    ),
)
propose_request_specialist_analysis = _WriteProposalStructuredTool.from_function(
    func=_propose_request_specialist_analysis,
    name="propose_request_specialist_analysis",
    args_schema=ProposeRequestSpecialistAnalysisArguments,
    response_format="content_and_artifact",
    description=(
        "Propõe solicitar análise especializada para uma análise existente. "
        "A tool apenas registra a proposta e não executa a ação."
    ),
)
propose_update_asset_criticality = _WriteProposalStructuredTool.from_function(
    func=_propose_update_asset_criticality,
    name="propose_update_asset_criticality",
    args_schema=ProposeUpdateAssetCriticalityArguments,
    response_format="content_and_artifact",
    description=(
        "Propõe atualizar somente a criticidade do ativo central. "
        "A tool apenas registra a proposta e não executa a ação."
    ),
)
propose_request_model_retraining = _WriteProposalStructuredTool.from_function(
    func=_propose_request_model_retraining,
    name="propose_request_model_retraining",
    args_schema=ProposeJustificationOnlyArguments,
    response_format="content_and_artifact",
    description=(
        "Propõe solicitar retreinamento do modelo configurado. "
        "A tool apenas registra a proposta e não executa a ação."
    ),
)
propose_escalate_case = _WriteProposalStructuredTool.from_function(
    func=_propose_escalate_case,
    name="propose_escalate_case",
    args_schema=ProposeJustificationOnlyArguments,
    response_format="content_and_artifact",
    description=(
        "Propõe escalar o caso atual para atendimento humano. "
        "A tool apenas registra a proposta e não executa a ação."
    ),
)

WRITE_PROPOSAL_TOOLS: Final[tuple[StructuredTool, ...]] = (
    propose_reprocess_analysis,
    propose_request_specialist_analysis,
    propose_update_asset_criticality,
    propose_request_model_retraining,
    propose_escalate_case,
)
