import asyncio
from pathlib import Path

import pytest
from langgraph.checkpoint.base import LATEST_VERSION
from pydantic import BaseModel

import tractian_agent.checkpoint as checkpoint_module
from tractian_agent.checkpoint import create_checkpoint_serializer, open_checkpointer
from tractian_agent.contracts import Identity, SupportRequest
from tractian_agent.state import AgentState, ThreadScope
from tractian_agent.tools.runtime import TrustedIdentity
from tractian_agent.write_policy import ReprocessProposal, TrustedWriteContext


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
            await reopened_saver.aget({"configurable": {"thread_id": "owner_reopened"}})

    asyncio.run(scenario())


def test_sqlite_checkpoint_reopens_the_legacy_pending_reprocess_shape(
    tmp_path: Path,
):
    checkpoint_path = tmp_path / "legacy-pending-proposal.sqlite3"
    state = AgentState(
        request=SupportRequest(
            case_id="case_tkt_inv_04",
            ticket_id="TKT-INV-04",
            asset_id="asset_G501",
            message="Reprocesse a análise após a troca do rolamento.",
            identity=Identity(
                user_id="usr_pedro",
                company_id="comp_mineracao_andes",
            ),
        ),
        identity=TrustedIdentity(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
        ),
        permissions=frozenset({"read", "action_low"}),
        request_id="req_legacy_01",
        thread_id="thread_case_tkt_inv_04",
        execution_id="exec_legacy_01",
        thread_scope=ThreadScope(
            thread_id="thread_case_tkt_inv_04",
            case_id="case_tkt_inv_04",
            company_id="comp_mineracao_andes",
            user_id="usr_pedro",
        ),
        trusted_write_context=TrustedWriteContext(
            central_asset_id="asset_G501",
            current_case_id="case_tkt_inv_04",
            configured_model_id="mdl_vib_v3",
        ),
        step_limit=3,
        pending_proposal=ReprocessProposal(
            analysis_id="an_9906",
            justification="Rolamento substituído; solicitar novo processamento.",
        ),
    )
    legacy_payload = state.model_dump(mode="json")
    del legacy_payload["pending_proposal"]["action"]
    assert legacy_payload["pending_proposal"] == {
        "analysis_id": "an_9906",
        "justification": "Rolamento substituído; solicitar novo processamento.",
    }
    config = {
        "configurable": {
            "thread_id": state.thread_id,
            "checkpoint_ns": "",
        }
    }
    checkpoint = {
        "v": LATEST_VERSION,
        "id": "00000000-0000-6000-8000-000000000001",
        "ts": "2026-08-30T12:00:00+00:00",
        "channel_values": {"agent_state": legacy_payload},
        "channel_versions": {"agent_state": "legacy-v1"},
        "versions_seen": {},
        "pending_sends": [],
        "updated_channels": ["agent_state"],
    }

    async def scenario():
        async with open_checkpointer(checkpoint_path) as saver:
            await saver.aput(
                config,
                checkpoint,
                {"source": "update", "step": 0, "parents": {}},
                {"agent_state": "legacy-v1"},
            )

        async with open_checkpointer(checkpoint_path) as reopened_saver:
            restored_checkpoint = await reopened_saver.aget(config)
        return restored_checkpoint

    restored_checkpoint = asyncio.run(scenario())
    assert restored_checkpoint is not None

    restored = AgentState.model_validate(
        restored_checkpoint["channel_values"]["agent_state"]
    )

    assert restored.pending_proposal == state.pending_proposal
    assert restored.pending_proposal.action == "reprocess_analysis"
