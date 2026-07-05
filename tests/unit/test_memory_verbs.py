import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from collivind.cli.commands.memory import (
    add,
    context,
    export_cmd,
    forget,
    get,
    import_cmd,
    invalidate,
    update,
)
from collivind.config import CollivindConfig
from collivind.engine.memory_manager import MemoryManager
from collivind.models import MemoryCategory, MemoryNode


def _make_manager(existing=None):
    vs, gs, ep = MagicMock(), MagicMock(), MagicMock()
    gs.get_memory.return_value = existing
    ep.embed.return_value = [0.1]
    vs.search.return_value = []
    manager = MemoryManager(vs, gs, ep, CollivindConfig())
    return manager, vs, gs, ep


def _node(**kw):
    defaults = dict(id="m-1", content="old", summary="old", category=MemoryCategory.FACT)
    defaults.update(kw)
    return MemoryNode(**defaults)


# --- engine ---


def test_update_memory_reembeds_on_content_change():
    node = _node()
    manager, vs, gs, ep = _make_manager(existing=node)
    gs.update_memory.return_value = _node(content="new")

    result = manager.update_memory("m-1", content="new")

    assert result.content == "new"
    ep.embed.assert_called_once()
    vs.upsert.assert_called_once()


def test_update_memory_tags_only_skips_reembed():
    manager, vs, gs, ep = _make_manager(existing=_node())
    gs.update_memory.return_value = _node(tags=["a"])

    manager.update_memory("m-1", tags=["a"])

    ep.embed.assert_not_called()
    vs.upsert.assert_not_called()


def test_update_memory_missing_returns_none():
    manager, _, gs, _ = _make_manager(existing=None)
    assert manager.update_memory("nope", content="x") is None
    gs.update_memory.assert_not_called()


def test_forget_deletes_both_stores():
    manager, vs, gs, _ = _make_manager(existing=_node())
    assert manager.forget("m-1") is True
    gs.delete_memory.assert_called_once_with("m-1")
    vs.delete.assert_called_once_with("m-1")


def test_forget_missing_returns_false():
    manager, vs, gs, _ = _make_manager(existing=None)
    assert manager.forget("nope") is False
    gs.delete_memory.assert_not_called()


def test_export_and_import_roundtrip():
    node = _node()
    manager, _, gs, _ = _make_manager(existing=None)
    gs.get_timeline.return_value = [node]

    records = manager.export_memories("default")
    assert records == [node.to_dict()]

    gs.create_memory.return_value = node
    count = manager.import_memories(records)
    assert count == 1
    gs.create_memory.assert_called_once()


# --- cli ---


def test_cli_add_json():
    manager = MagicMock()
    manager.add_memory.return_value = _node(id="new-1")
    with patch("collivind.cli.commands.memory._manager", return_value=manager):
        result = CliRunner().invoke(add, ["Postgres chosen", "-c", "decision", "-t", "db, infra", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output)["id"] == "new-1"
    created = manager.add_memory.call_args.args[0]
    assert created.category == MemoryCategory.DECISION
    assert created.tags == ["db", "infra"]


def test_cli_get_missing_exits_nonzero():
    manager = MagicMock()
    manager.graph_store.get_memory.return_value = None
    with patch("collivind.cli.commands.memory._manager", return_value=manager):
        result = CliRunner().invoke(get, ["nope"])
    assert result.exit_code == 1


def test_cli_context_prints_block():
    manager = MagicMock()
    manager.get_context.return_value = "CONTEXT BLOCK"
    with patch("collivind.cli.commands.memory._manager", return_value=manager):
        result = CliRunner().invoke(context, ["auth flow"])
    assert result.exit_code == 0
    assert "CONTEXT BLOCK" in result.output


def test_cli_update_passes_fields():
    manager = MagicMock()
    manager.update_memory.return_value = _node(content="new")
    with patch("collivind.cli.commands.memory._manager", return_value=manager):
        result = CliRunner().invoke(update, ["m-1", "--content", "new", "-t", "x,y"])
    assert result.exit_code == 0
    kwargs = manager.update_memory.call_args.kwargs
    assert kwargs["content"] == "new"
    assert kwargs["tags"] == ["x", "y"]


def test_cli_invalidate():
    manager = MagicMock()
    with patch("collivind.cli.commands.memory._manager", return_value=manager):
        result = CliRunner().invoke(invalidate, ["m-1", "--reason", "superseded"])
    assert result.exit_code == 0
    manager.invalidate.assert_called_once_with("m-1", "", "superseded")


def test_cli_forget_confirmed():
    manager = MagicMock()
    manager.forget.return_value = True
    with patch("collivind.cli.commands.memory._manager", return_value=manager):
        result = CliRunner().invoke(forget, ["m-1", "--yes"])
    assert result.exit_code == 0
    manager.forget.assert_called_once_with("m-1")


def test_cli_export_import_files(tmp_path):
    node = _node()
    manager = MagicMock()
    manager.export_memories.return_value = [node.to_dict()]
    out = tmp_path / "mem.jsonl"

    with patch("collivind.cli.commands.memory._manager", return_value=manager):
        result = CliRunner().invoke(export_cmd, ["-o", str(out)])
    assert result.exit_code == 0
    assert json.loads(out.read_text().strip())["id"] == "m-1"

    manager.import_memories.return_value = 1
    with patch("collivind.cli.commands.memory._manager", return_value=manager):
        result = CliRunner().invoke(import_cmd, [str(out)])
    assert result.exit_code == 0
    assert manager.import_memories.call_args.args[0][0]["id"] == "m-1"


def test_cli_import_rejects_bad_json(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    with patch("collivind.cli.commands.memory._manager", return_value=MagicMock()):
        result = CliRunner().invoke(import_cmd, [str(bad)])
    assert result.exit_code == 1


def test_get_context_respects_max_tokens():
    from collivind.models import SearchResult

    manager, _, _, _ = _make_manager()
    big = _node(content="x" * 4000)
    small = _node(id="m-2", content="y" * 40)
    results = [
        SearchResult(memory=small, score=0.9, vector_score=0.9, graph_score=0.0),
        SearchResult(memory=big, score=0.8, vector_score=0.8, graph_score=0.0),
    ]
    with patch.object(manager, "search", return_value=results):
        unbudgeted = manager.get_context("q")
        budgeted = manager.get_context("q", max_tokens=100)

    assert "x" * 100 in unbudgeted
    assert "y" * 40 in budgeted
    assert "x" * 100 not in budgeted  # big result dropped to fit the budget


def test_cli_sync_merges_and_writes_stable_order(tmp_path):
    from collivind.cli.commands.memory import sync

    teammate = _node(id="t-1", content="teammate fact").to_dict()
    (tmp_path / "default.jsonl").write_text(json.dumps(teammate) + "\n")

    mine = _node(id="a-1").to_dict()
    mine["created_at"] = "2026-01-01T00:00:00+00:00"
    theirs = dict(teammate, created_at="2025-01-01T00:00:00+00:00")

    manager = MagicMock()
    manager.import_memories.return_value = 1
    manager.export_memories.return_value = [mine, theirs]  # newest first

    with patch("collivind.cli.commands.memory._manager", return_value=manager):
        result = CliRunner().invoke(sync, [str(tmp_path)])

    assert result.exit_code == 0
    assert manager.import_memories.call_args.args[0] == [teammate]
    lines = [json.loads(line) for line in (tmp_path / "default.jsonl").read_text().splitlines()]
    assert [r["id"] for r in lines] == ["t-1", "a-1"]  # oldest first, stable


def test_cli_sync_first_run_creates_file(tmp_path):
    from collivind.cli.commands.memory import sync

    manager = MagicMock()
    manager.export_memories.return_value = [_node().to_dict()]

    with patch("collivind.cli.commands.memory._manager", return_value=manager):
        result = CliRunner().invoke(sync, [str(tmp_path / "shared"), "-p", "proj"])

    assert result.exit_code == 0
    manager.import_memories.assert_not_called()
    assert (tmp_path / "shared" / "proj.jsonl").exists()
