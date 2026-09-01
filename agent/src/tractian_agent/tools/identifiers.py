"""Identificadores aceitos nas tools, com formatos deliberadamente restritos."""
from typing import Annotated

from pydantic import StringConstraints

AssetId = Annotated[
    str,
    StringConstraints(pattern=r"^asset_[A-Za-z0-9_-]{1,64}$"),
]

CompanyId = Annotated[
    str,
    StringConstraints(pattern=r"^comp_[A-Za-z0-9_-]{1,64}$"),
]

PointId = Annotated[
    str,
    StringConstraints(pattern=r"^pt_[A-Za-z0-9_-]{1,64}$"),
]

AnalysisId = Annotated[
    str,
    StringConstraints(pattern=r"^an_[A-Za-z0-9_-]{1,64}$"),
]

ModelId = Annotated[
    str,
    StringConstraints(pattern=r"^mdl_[A-Za-z0-9_-]{1,64}$"),
]

CaseId = Annotated[
    str,
    StringConstraints(pattern=r"^case_[A-Za-z0-9_-]{1,64}$"),
]

KnowledgeDocumentId = Annotated[
    str,
    StringConstraints(pattern=r"^kb_[A-Za-z0-9_-]{1,64}$"),
]
