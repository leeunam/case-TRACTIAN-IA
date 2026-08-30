import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

import tractian_agent.checkpoint as checkpoint_module
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
    project_root = Path(__file__).resolve().parents[2]
    assert checkpoint_module.DEFAULT_CHECKPOINT_PATH == (
        project_root / ".run/agent-checkpoints.sqlite3"
    )

    foreign_cwd = tmp_path / "foreign-cwd"
    foreign_cwd.mkdir()
    temporary_default = tmp_path / "project-root/.run/agent-checkpoints.sqlite3"
    monkeypatch.setattr(
        checkpoint_module,
        "DEFAULT_CHECKPOINT_PATH",
        temporary_default,
    )
    monkeypatch.chdir(foreign_cwd)

    async def scenario():
        async with open_checkpointer() as saver:
            await saver.aget({"configurable": {"thread_id": "path_probe"}})
            assert temporary_default.is_file()
            assert not Path(".run/agent-checkpoints.sqlite3").exists()
            with pytest.raises(TypeError, match="not msgpack serializable"):
                saver.serde.dumps_typed(_PickleOnly())

        with pytest.raises(ValueError, match="no active connection"):
            await saver.conn.execute("SELECT 1")

        explicit_path = Path("explicit/checkpoints.sqlite3")
        async with open_checkpointer(explicit_path) as explicit_saver:
            await explicit_saver.aget(
                {"configurable": {"thread_id": "explicit_path_probe"}}
            )
        assert (foreign_cwd / explicit_path).is_file()

    asyncio.run(scenario())


def test_checkpoint_namespace_has_only_one_active_local_owner(tmp_path: Path):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"

    async def scenario():
        async with open_checkpointer(checkpoint_path):
            with pytest.raises(
                RuntimeError,
                match="namespace do checkpoint já possui owner local ativo",
            ):
                async with open_checkpointer(checkpoint_path):
                    pass

        async with open_checkpointer(checkpoint_path) as reopened_saver:
            await reopened_saver.aget(
                {"configurable": {"thread_id": "owner_reopened"}}
            )

    asyncio.run(scenario())
