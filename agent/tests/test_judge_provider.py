from unittest.mock import patch

from langchain_groq import ChatGroq

from tractian_agent.evaluation.judge_provider import GroqJudgeModelProvider
from tractian_agent.evaluation.judges import BlindResultJudgment
from tractian_agent.model_provider import ModelConfig


def test_groq_judge_provider_uses_json_mode_without_hidden_retry() -> None:
    provider = GroqJudgeModelProvider.from_env({"GROQ_API_KEY": "secret"})
    model = provider.create_chat_model(
        ModelConfig(
            model_id="openai/gpt-oss-120b",
            temperature=0.0,
            timeout_seconds=30.0,
            max_output_tokens=1024,
        )
    )

    with patch.object(ChatGroq, "with_structured_output") as structured:
        model.with_structured_output(BlindResultJudgment)

    structured.assert_called_once_with(
        BlindResultJudgment,
        method="json_mode",
        include_raw=False,
    )
    assert model.max_retries == 0
