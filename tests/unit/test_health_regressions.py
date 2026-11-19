"""Regression guards for the failures reported in #37, #38 and #39.

Each one was a failure that nothing surfaced; these tests pin the surfaces.
"""

import json
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from collivind.cli.commands import status as status_mod
from collivind.config import CollivindConfig
from collivind.engine.memory_manager import MemoryManager
from collivind.exceptions import BatchPartiallyStoredError, CollivindError
from collivind.mcp.tools import CollivindTools

GOOD = {"content": "batch probe good", "summary": "good", "category": "fact"}


def _manager():
    return MemoryManager(MagicMock(), MagicMock(), MagicMock(), CollivindConfig())


# --- #37: batch_add must not partially apply silently ----------------------


def test_invalid_item_stores_nothing(monkeypatch):
    manager = _manager()
    add = MagicMock()
    monkeypatch.setattr(manager, "add_memory", add)
    with pytest.raises(ValueError, match=r"memory\[1\].*nothing in the batch was stored"):
        manager.batch_add_memories([GOOD, {**GOOD, "confidence": 42}])
    add.assert_not_called()


def test_missing_field_names_the_item(monkeypatch):
    manager = _manager()
    monkeypatch.setattr(manager, "add_memory", MagicMock())
    with pytest.raises(ValueError, match=r"memory\[0\]: missing field 'summary'"):
        manager.batch_add_memories([{"content": "x", "category": "fact"}])


def test_storage_failure_mid_batch_reports_what_landed(monkeypatch):
    manager = _manager()
    calls = iter([MagicMock(id="m1"), CollivindError("neo4j down")])

    def add(*a, **kw):
        r = next(calls)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(manager, "add_memory", add)
    with pytest.raises(BatchPartiallyStoredError) as exc:
        manager.batch_add_memories([GOOD, GOOD])
    assert exc.value.stored_ids == ["m1"] and exc.value.failed_index == 1


def test_batch_tool_returns_stored_ids_on_partial_failure():
    manager = MagicMock()
    manager.batch_add_memories.side_effect = BatchPartiallyStoredError(["m1"], 1, RuntimeError("down"))
    res = json.loads(CollivindTools(manager).handle_call("collivind_batch_add", {"memories": [GOOD, GOOD]}))
    assert res["partial"] is True and res["stored_memory_ids"] == ["m1"] and res["failed_index"] == 1


# --- #39: backend errors must point at the cause ----------------------------


def test_backend_error_carries_status_hint():
    manager = MagicMock()
    manager.search_with_status.side_effect = CollivindError("Failed to get memory from Neo4j after retries: refused")
    res = json.loads(CollivindTools(manager).handle_call("collivind_search", {"query": "anything"}))
    assert "Neo4j" in res["error"] and "collivind status" in res["hint"]


def _container(name, status="running", restarting=False, restarts=0, health=None, exit_code=0):
    state = {"Status": status, "Restarting": restarting, "ExitCode": exit_code}
    if health:
        state["Health"] = {"Status": health}
    return {"Name": f"/{name}", "State": state, "RestartCount": restarts}


@pytest.mark.parametrize(
    "container, expect",
    [
        (_container("c"), ""),
        (_container("c", health="healthy"), ""),
        (_container("c", status="restarting", restarting=True, restarts=67, exit_code=1), "crash-looping (67"),
        (_container("c", status="exited", restarts=67, exit_code=1), "crash-looping"),
        (_container("c", status="exited"), "exited"),
        (_container("c", health="unhealthy"), "unhealthy"),
    ],
)
def test_container_problem(container, expect):
    problem = status_mod.container_problem(container)
    assert (expect in problem) if expect else problem == ""


def _docker_config():
    cfg = CollivindConfig()
    cfg.mode = "docker"
    return cfg


def _all_backends_ok(monkeypatch):
    monkeypatch.setattr(status_mod, "load_config", _docker_config)
    monkeypatch.setattr(
        status_mod,
        "check_all_services",
        lambda c: {s: {"status": "ok", "message": "ok"} for s in status_mod.DOCKER_SERVICES},
    )


def test_status_fails_on_crash_loop_even_when_probes_pass(monkeypatch):
    _all_backends_ok(monkeypatch)
    monkeypatch.setattr(
        status_mod,
        "_container_states",
        lambda names: [
            _container(names[0]),
            _container(names[1], status="restarting", restarting=True, restarts=67, exit_code=1),
            _container(names[2]),
        ],
    )
    result = CliRunner().invoke(status_mod.status)
    assert result.exit_code == 1
    assert "collivind-neo4j: crash-looping (67 restarts" in result.output


def test_status_fails_when_docker_unreachable(monkeypatch):
    _all_backends_ok(monkeypatch)
    monkeypatch.setattr(status_mod, "_container_states", lambda names: None)
    result = CliRunner().invoke(status_mod.status)
    assert result.exit_code == 1 and "docker: not reachable" in result.output


def test_status_passes_when_everything_is_up(monkeypatch):
    _all_backends_ok(monkeypatch)
    monkeypatch.setattr(status_mod, "_container_states", lambda names: [_container(n) for n in names])
    result = CliRunner().invoke(status_mod.status)
    assert result.exit_code == 0, result.output


# --- #38: status must report Claude Code MCP registration -------------------


@pytest.fixture
def claude_home(tmp_path, monkeypatch):
    (tmp_path / ".claude").mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)


@pytest.mark.parametrize(
    "servers, ok, expect",
    [
        ({}, True, "no collivind server registered"),
        ({"collivind": "✓ Connected"}, True, "✓ claude MCP: collivind"),
        ({"plugin:collivind:collivind": "✓ Connected"}, True, "plugin:collivind:collivind"),
        ({"collivind": "✗ Failed to connect"}, False, "not answering"),
        ({"collivind": "✓ Connected", "plugin:collivind:collivind": "✓ Connected"}, False, "registered twice"),
    ],
)
def test_claude_mcp_states(claude_home, monkeypatch, capsys, servers, ok, expect):
    monkeypatch.setattr(status_mod, "_claude_mcp_listing", lambda: ({"github": "✓ Connected", **servers}, ""))
    assert status_mod._status_claude_mcp() is ok
    assert expect in capsys.readouterr().out


def test_claude_mcp_listing_unreadable_is_reported_not_fatal(claude_home, monkeypatch, capsys):
    monkeypatch.setattr(status_mod, "_claude_mcp_listing", lambda: (None, "claude CLI not found"))
    assert status_mod._status_claude_mcp() is True
    assert "could not check (claude CLI not found)" in capsys.readouterr().out
