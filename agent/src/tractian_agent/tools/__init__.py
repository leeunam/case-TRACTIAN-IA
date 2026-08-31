"""Catálogos públicos das tools determinísticas."""

from typing import Final

from langchain_core.tools import BaseTool

from .assets import get_asset
from .analyses import get_analysis, list_asset_analyses
from .technical import (
    get_baseline,
    get_data_quality,
    get_rms_series,
    get_spectrum,
)
from .knowledge import get_knowledge_document, get_model, search_knowledge


READ_TOOLS: Final[tuple[BaseTool, ...]] = (
    get_asset,
    list_asset_analyses,
    get_analysis,
    get_baseline,
    get_rms_series,
    get_spectrum,
    get_data_quality,
    get_model,
    search_knowledge,
    get_knowledge_document,
)

__all__ = [
    "get_asset",
    "list_asset_analyses",
    "get_analysis",
    "get_baseline",
    "get_rms_series",
    "get_spectrum",
    "get_data_quality",
    "get_model",
    "search_knowledge",
    "get_knowledge_document",
    "READ_TOOLS",
    "propose_reprocess_analysis",
    "propose_request_specialist_analysis",
    "propose_update_asset_criticality",
    "propose_request_model_retraining",
    "propose_escalate_case",
    "WRITE_PROPOSAL_TOOLS",
]


_WRITE_EXPORTS = frozenset(
    {
        "propose_reprocess_analysis",
        "propose_request_specialist_analysis",
        "propose_update_asset_criticality",
        "propose_request_model_retraining",
        "propose_escalate_case",
        "WRITE_PROPOSAL_TOOLS",
    }
)


def __getattr__(name: str):
    """Carrega tools de proposta sem acoplar a política ao catálogo de tools."""
    if name not in _WRITE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import writes

    return getattr(writes, name)
