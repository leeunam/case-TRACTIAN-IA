"""Ciclo de vida seguro do checkpointer SQLite de desenvolvimento."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / ".run/agent-checkpoints.sqlite3"
_OWNER_ATTRIBUTE = "_tractian_local_checkpoint_owner"


@dataclass
class _LockEntry:
    lock: asyncio.Lock
    users: int = 0


class LocalCheckpointOwner:
    """Owner efêmero de um processo/event loop, sem lease multiprocesso."""

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._entries: dict[str, _LockEntry] = {}
        self._registry_lock = asyncio.Lock()

    @asynccontextmanager
    async def thread_lock(self, thread_id: str) -> AsyncIterator[None]:
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("owner do checkpoint pertence a outro event loop")
        async with self._registry_lock:
            entry = self._entries.get(thread_id)
            if entry is None:
                entry = _LockEntry(lock=asyncio.Lock())
                self._entries[thread_id] = entry
            entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            async with self._registry_lock:
                entry.users -= 1
                if entry.users == 0:
                    del self._entries[thread_id]


_ACTIVE_OWNERS: dict[Path, LocalCheckpointOwner] = {}
_ACTIVE_OWNERS_LOCK = threading.Lock()


def _claim_namespace(
    checkpoint_path: Path,
    owner: LocalCheckpointOwner,
) -> Path:
    namespace = checkpoint_path.resolve(strict=False)
    with _ACTIVE_OWNERS_LOCK:
        if namespace in _ACTIVE_OWNERS:
            raise RuntimeError(
                "namespace do checkpoint já possui owner local ativo; "
                "reutilize o saver aberto"
            )
        _ACTIVE_OWNERS[namespace] = owner
    return namespace


def _release_namespace(namespace: Path, owner: LocalCheckpointOwner) -> None:
    with _ACTIVE_OWNERS_LOCK:
        if _ACTIVE_OWNERS.get(namespace) is owner:
            del _ACTIVE_OWNERS[namespace]


def get_checkpoint_owner(
    checkpointer: BaseCheckpointSaver[str],
) -> LocalCheckpointOwner:
    """Obtém o owner instalado pela fronteira segura de lifecycle."""
    owner = getattr(checkpointer, _OWNER_ATTRIBUTE, None)
    if not isinstance(owner, LocalCheckpointOwner):
        raise TypeError("checkpointer não gerenciado; construa-o com open_checkpointer")
    return owner


def create_checkpoint_serializer() -> JsonPlusSerializer:
    """Cria um serializer sem pickle nem importação de módulos arbitrários."""
    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_json_modules=None,
        allowed_msgpack_modules=None,
    )


@asynccontextmanager
async def open_checkpointer(
    path: str | Path | None = None,
) -> AsyncIterator[AsyncSqliteSaver]:
    """Abre um owner único por namespace no processo e event loop atuais."""
    checkpoint_path = DEFAULT_CHECKPOINT_PATH if path is None else Path(path)
    owner = LocalCheckpointOwner()
    namespace = _claim_namespace(checkpoint_path, owner)
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as saver:
            saver.serde = create_checkpoint_serializer()
            setattr(saver, _OWNER_ATTRIBUTE, owner)
            yield saver
    finally:
        _release_namespace(namespace, owner)
