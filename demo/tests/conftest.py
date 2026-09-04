from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def anyio_backend() -> tuple[str, dict[str, bool]]:
    """Executa testes async no backend que desperta callbacks entre threads."""

    return "asyncio", {"use_uvloop": True}
