"""Contrato independente de provider para modelos de linguagem."""

from typing import Final, Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import ConfigDict, Field

from tractian_agent.contracts import StrictModel


class ModelConfig(StrictModel):
    """Configuração comum compartilhada pelos adapters de modelo."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    model_id: str = Field(strict=True, min_length=1, pattern=r"^\S+$")
    temperature: float = Field(strict=True, ge=0, le=2, allow_inf_nan=False)
    timeout_seconds: float = Field(strict=True, gt=0, allow_inf_nan=False)
    max_output_tokens: int = Field(gt=0, strict=True)


INITIAL_PLANNER_MODEL_CONFIG: Final = ModelConfig(
    model_id="openai/gpt-oss-120b",
    temperature=0.0,
    timeout_seconds=30.0,
    max_output_tokens=512,
)


class ModelProvider(Protocol):
    """Contrato comum implementado por qualquer adapter de modelo."""

    def create_chat_model(self, config: ModelConfig) -> BaseChatModel:
        """Cria um modelo LangChain usando a configuração validada."""
        ...
