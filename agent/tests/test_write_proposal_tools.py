import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from tractian_agent.tools import WRITE_PROPOSAL_TOOLS
from tractian_agent.tools.writes import (
    propose_escalate_case,
    propose_reprocess_analysis,
    propose_request_model_retraining,
    propose_request_specialist_analysis,
    propose_update_asset_criticality,
)


def _invoke_with_tool_node(tool, arguments):
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode((tool,)))
    graph.add_edge(START, "tools")
    compiled = graph.compile()
    return compiled.invoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool.name,
                            "args": arguments,
                            "id": f"call_{tool.name}",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }
    )["messages"][-1]


def test_reprocess_proposal_runs_through_tool_node_without_runtime_or_effect():
    message = _invoke_with_tool_node(
        propose_reprocess_analysis,
        {
            "analysis_id": "an_9906",
            "justification": "Há dados novos para reprocessar esta análise.",
        },
    )

    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call_propose_reprocess_analysis"
    assert json.loads(message.content) == {
        "status": "proposed",
        "proposal": {
            "action": "reprocess_analysis",
            "analysis_id": "an_9906",
            "justification": "Há dados novos para reprocessar esta análise.",
        },
    }
    assert message.artifact == {
        "kind": "write_proposal",
        "tool_name": "propose_reprocess_analysis",
        "proposal": {
            "action": "reprocess_analysis",
            "analysis_id": "an_9906",
            "justification": "Há dados novos para reprocessar esta análise.",
        },
        "effect_executed": False,
    }


@pytest.mark.parametrize(
    ("tool", "arguments", "expected_proposal"),
    [
        (
            propose_request_specialist_analysis,
            {
                "analysis_id": "an_9906",
                "justification": "A limitação registrada exige análise especializada.",
            },
            {
                "action": "request_specialist_analysis",
                "analysis_id": "an_9906",
                "justification": "A limitação registrada exige análise especializada.",
            },
        ),
        (
            propose_update_asset_criticality,
            {
                "criticality": "critical",
                "justification": "O impacto operacional exige criticidade mais alta.",
            },
            {
                "action": "update_asset_criticality",
                "criticality": "critical",
                "justification": "O impacto operacional exige criticidade mais alta.",
            },
        ),
        (
            propose_request_model_retraining,
            {
                "justification": "Erros sistemáticos sustentam solicitar novo treinamento.",
            },
            {
                "action": "request_model_retraining",
                "justification": "Erros sistemáticos sustentam solicitar novo treinamento.",
            },
        ),
        (
            propose_escalate_case,
            {
                "justification": "O caso ultrapassa o atendimento remoto disponível.",
            },
            {
                "action": "escalate_case",
                "justification": "O caso ultrapassa o atendimento remoto disponível.",
            },
        ),
    ],
)
def test_each_hidden_target_proposal_runs_without_runtime_or_effect(
    tool,
    arguments,
    expected_proposal,
):
    message = _invoke_with_tool_node(tool, arguments)

    assert isinstance(message, ToolMessage)
    assert json.loads(message.content) == {
        "status": "proposed",
        "proposal": expected_proposal,
    }
    assert message.artifact == {
        "kind": "write_proposal",
        "tool_name": tool.name,
        "proposal": expected_proposal,
        "effect_executed": False,
    }


def test_write_proposal_catalog_is_ordered_static_unique_and_has_exact_schemas():
    expected_fields = {
        "propose_reprocess_analysis": {"analysis_id", "justification"},
        "propose_request_specialist_analysis": {"analysis_id", "justification"},
        "propose_update_asset_criticality": {"criticality", "justification"},
        "propose_request_model_retraining": {"justification"},
        "propose_escalate_case": {"justification"},
    }
    expected_names = tuple(expected_fields)
    forbidden_fields = {
        "action",
        "runtime",
        "context",
        "identity",
        "user_id",
        "company_id",
        "permissions",
        "approval",
        "idempotency_key",
        "central_asset_id",
        "current_case_id",
        "configured_model_id",
        "case_id",
        "asset_id",
        "model_id",
        "url",
        "method",
        "headers",
    }

    assert isinstance(WRITE_PROPOSAL_TOOLS, tuple)
    assert tuple(tool.name for tool in WRITE_PROPOSAL_TOOLS) == expected_names
    assert len({tool.name for tool in WRITE_PROPOSAL_TOOLS}) == 5
    for tool in WRITE_PROPOSAL_TOOLS:
        schema = tool.tool_call_schema.model_json_schema()
        fields = set(schema.get("properties", {}))

        assert schema["additionalProperties"] is False
        assert fields == expected_fields[tool.name]
        assert fields.isdisjoint(forbidden_fields)
