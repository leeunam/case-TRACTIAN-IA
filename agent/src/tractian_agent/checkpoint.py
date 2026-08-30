"""Ciclo de vida seguro do checkpointer SQLite de desenvolvimento."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


DEFAULT_CHECKPOINT_PATH = Path(".run/agent-checkpoints.sqlite3")


def create_checkpoint_serializer() -> JsonPlusSerializer:
    """Cria um serializer sem pickle nem importação de módulos arbitrários."""
    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_json_modules=None,
        allowed_msgpack_modules=None,
    )


@asynccontextmanager
async def open_checkpointer(
    path: str | Path = DEFAULT_CHECKPOINT_PATH,
) -> AsyncIterator[AsyncSqliteSaver]:
    """Mantém saver e conexão abertos somente dentro do contexto assíncrono."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
        saver.serde = create_checkpoint_serializer()
        yield saver
