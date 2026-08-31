from copy import deepcopy
import json

import pytest
from pydantic import TypeAdapter, ValidationError

from tractian_agent.tools import WRITE_PROPOSAL_TOOLS
from tractian_agent.write_policy import (
    EscalateCaseProposal,
    ReprocessProposal,
    RequestModelRetrainingProposal,
    RequestSpecialistAnalysisProposal,
    TrustedWriteContext,
    UpdateAssetCriticalityProposal,
    WriteProposal,
)


PROPOSALS = (
    ReprocessProposal(
        analysis_id="an_9906",
        justification="Há dados novos para reprocessar esta análise.",
    ),
    RequestSpecialistAnalysisProposal(
        analysis_id="an_9906",
        justification="A limitação registrada exige análise especializada.",
    ),
    UpdateAssetCriticalityProposal(
        criticality="critical",
        justification="O impacto operacional exige criticidade mais alta.",
    ),
    RequestModelRetrainingProposal(
        justification="Erros sistemáticos sustentam solicitar novo treinamento.",
    ),
    EscalateCaseProposal(
        justification="O caso ultrapassa o atendimento remoto disponível.",
    ),
)


@pytest.mark.parametrize("proposal", PROPOSALS)
def test_each_proposal_is_frozen_extra_forbid_and_round_trips_as_its_variant(
    proposal,
):
    adapter = TypeAdapter(WriteProposal)

    restored = adapter.validate_json(adapter.dump_json(proposal))

    assert restored == proposal
    assert type(restored) is type(proposal)
    with pytest.raises(ValidationError):
        proposal.justification = "conteúdo mutado"
    with pytest.raises(ValidationError):
        type(proposal).model_validate(
            {**proposal.model_dump(mode="python"), "client": "não permitido"}
        )


@pytest.mark.parametrize(
    ("proposal_type", "payload"),
    [
        (
            ReprocessProposal,
            {
                "action": "request_specialist_analysis",
                "analysis_id": "an_9906",
                "justification": "Justificativa válida para o contrato.",
            },
        ),
        (
            RequestSpecialistAnalysisProposal,
            {
                "action": "reprocess_analysis",
                "analysis_id": "an_9906",
                "justification": "Justificativa válida para o contrato.",
            },
        ),
        (
            UpdateAssetCriticalityProposal,
            {
                "action": "escalate_case",
                "criticality": "high",
                "justification": "Justificativa válida para o contrato.",
            },
        ),
    ],
)
def test_proposal_action_discriminator_is_fixed_by_the_contract(
    proposal_type,
    payload,
):
    with pytest.raises(ValidationError):
        proposal_type.model_validate(payload)


@pytest.mark.parametrize(
    ("proposal_type", "payload"),
    [
        (
            ReprocessProposal,
            {"analysis_id": "analysis_9906", "justification": "válida"},
        ),
        (
            RequestSpecialistAnalysisProposal,
            {"analysis_id": "asset_G501", "justification": "válida"},
        ),
        (
            UpdateAssetCriticalityProposal,
            {"criticality": "urgent", "justification": "válida"},
        ),
    ],
)
def test_proposals_reject_invalid_selectable_identifiers_and_criticality(
    proposal_type,
    payload,
):
    with pytest.raises(ValidationError):
        proposal_type.model_validate(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"central_asset_id": "an_9906"},
        {"current_case_id": "ticket_04"},
        {"configured_model_id": "asset_G501"},
    ],
)
def test_trusted_write_context_validates_each_hidden_target_kind(changes):
    payload = {
        "central_asset_id": "asset_G501",
        "current_case_id": "case_tkt_inv_04",
        "configured_model_id": "mdl_vib_v3",
    }
    payload.update(changes)

    with pytest.raises(ValidationError):
        TrustedWriteContext.model_validate(payload)


def test_tools_do_not_mutate_the_argument_objects_received_from_the_caller():
    arguments_by_tool = {
        "propose_reprocess_analysis": {
            "analysis_id": "an_9906",
            "justification": "Há dados novos para reprocessar esta análise.",
        },
        "propose_request_specialist_analysis": {
            "analysis_id": "an_9906",
            "justification": "A limitação registrada exige análise especializada.",
        },
        "propose_update_asset_criticality": {
            "criticality": "critical",
            "justification": "O impacto operacional exige criticidade mais alta.",
        },
        "propose_request_model_retraining": {
            "justification": "Erros sistemáticos sustentam solicitar novo treinamento.",
        },
        "propose_escalate_case": {
            "justification": "O caso ultrapassa o atendimento remoto disponível.",
        },
    }
    original = deepcopy(arguments_by_tool)

    for tool in WRITE_PROPOSAL_TOOLS:
        result = tool.invoke(arguments_by_tool[tool.name])
        json.dumps(result, allow_nan=False)

    assert arguments_by_tool == original
