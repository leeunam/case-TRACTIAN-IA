import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from tractian_agent.checkpoint import create_checkpoint_serializer, open_checkpointer


class _ArbitraryModel(BaseModel):
    value: str


class _PickleOnly:
    def __reduce__(self):
        return (str, ("não deve executar",))


def test_checkpoint_serializer_never_uses_pickle_or_recreates_arbitrary_modules():
    serializer = create_checkpoint_serializer()

    with pytest.raises(TypeError, match="not msgpack serializable"):
        serializer.dumps_typed(_PickleOnly())

    encoded = serializer.dumps_typed(_ArbitraryModel(value="observável"))
    decoded = serializer.loads_typed(encoded)

    assert decoded == {"value": "observável"}
    assert not isinstance(decoded, _ArbitraryModel)


def test_default_checkpointer_path_creates_parent_and_closes_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)

    async def scenario():
        async with open_checkpointer() as saver:
            await saver.aget({"configurable": {"thread_id": "path_probe"}})
            assert Path(".run/agent-checkpoints.sqlite3").is_file()
            with pytest.raises(TypeError, match="not msgpack serializable"):
                saver.serde.dumps_typed(_PickleOnly())

        with pytest.raises(ValueError, match="no active connection"):
            await saver.conn.execute("SELECT 1")

    asyncio.run(scenario())
