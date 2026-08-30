"""Catálogo público das tools determinísticas de leitura."""

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
]
