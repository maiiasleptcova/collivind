"""#35: a reminder is not a save. Checkpoints stay pending until a write confirms them."""

import json
import threading
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from collivind import capture
from collivind.cli.commands.hook import hook
from collivind.mcp.tools import CollivindTools


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return tmp_path


def _stop(session="s1", agent="claude", **payload):
    result = CliRunner().invoke(
        hook,
        ["stop", "--threshold", "3", "--agent", agent],
        input=json.dumps({"session_id": session, **payload}),
    )
    assert result.exit_code == 0, result.output
    return json.loads(result.output) if result.output.strip() else None


def _checkpoint_id(block):
    return block["reason"].split('checkpoint_id: "')[1].split('"')[0]


def _save(checkpoint_id, memories=None):
    manager = MagicMock()
    manager.batch_add_memories.return_value = ["m1"]
    tools = CollivindTools(manager, session_id="mcp-1")
    good = [{"content": "c", "summary": "s", "category": "fact"}]
    args = {"memories": good if memories is None else memories, "checkpoint_id": checkpoint_id}
    return json.loads(tools.handle_call("collivind_batch_add", args))


def test_pending_checkpoint_is_rerequested_until_a_write_confirms_it():
    assert _stop() is None and _stop() is None
    first = _stop()
    cid = _checkpoint_id(first)

    # The agent ignored the request: no new checkpoint, the same one is re-asked.
    assert [_stop(), _stop()] == [None, None]
    again = _stop()
    assert _checkpoint_id(again) == cid

    assert _save(cid)["checkpoint"] == "confirmed"
    _stop()  # sees the confirmation and resumes normal counting
    state = capture.read_state()["sessions"]["s1"]
    assert "pending" not in state and state["last_confirmed"] == cid


def test_empty_batch_records_checked_nothing_new():
    for _ in range(2):
        _stop()
    cid = _checkpoint_id(_stop())
    res = _save(cid, memories=[])
    assert res["checkpoint"] == "checked, nothing new"
    assert cid in capture.confirmed_ids()


def test_failed_write_does_not_confirm():
    for _ in range(2):
        _stop()
    cid = _checkpoint_id(_stop())
    manager = MagicMock()
    manager.batch_add_memories.side_effect = RuntimeError("neo4j down")
    CollivindTools(manager).handle_call(
        "collivind_batch_add",
        {"memories": [{"content": "c", "summary": "s", "category": "fact"}], "checkpoint_id": cid},
    )
    assert cid not in capture.confirmed_ids()
    assert capture.summary()["last_fail"]["error"] == "neo4j down"


def test_own_continuation_never_chains_another_block():
    for _ in range(5):
        assert _stop(stop_hook_active=True) is None
    assert capture.read_state()["sessions"] == {}


def test_sessions_keep_independent_counters():
    _stop("a"), _stop("a"), _stop("b")
    sessions = capture.read_state()["sessions"]
    assert sessions["a"]["count"] == 2 and sessions["b"]["count"] == 1


def test_concurrent_hooks_do_not_lose_counts():
    def bump():
        with capture.locked_state() as state:
            s = capture.session(state, "x", "claude")
            s["count"] = s.get("count", 0) + 1

    threads = [threading.Thread(target=bump) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert capture.read_state()["sessions"]["x"]["count"] == 40


def test_compaction_requests_a_save_on_the_next_session_start(monkeypatch):
    monkeypatch.setattr("collivind.cli.commands.hook._load_manager", lambda: MagicMock(get_timeline=lambda *a, **k: []))
    runner = CliRunner()
    runner.invoke(hook, ["precompact", "--agent", "codex"], input=json.dumps({"session_id": "s1", "trigger": "manual"}))
    pending = capture.read_state()["sessions"]["s1"]["pending"]

    result = runner.invoke(
        hook, ["session-start", "--agent", "codex"], input=json.dumps({"session_id": "s1", "source": "compact"})
    )
    context = json.loads(result.output)["hookSpecificOutput"]["additionalContext"]
    assert "just compacted" in context and pending["id"] in context


def test_ordinary_session_start_asks_for_nothing(monkeypatch):
    monkeypatch.setattr("collivind.cli.commands.hook._load_manager", lambda: MagicMock(get_timeline=lambda *a, **k: []))
    result = CliRunner().invoke(hook, ["session-start"], input=json.dumps({"session_id": "s1", "source": "startup"}))
    assert result.output.strip() == ""


def test_quiet_and_verbose_prompts():
    assert "silently" in capture.extraction_prompt("abc", quiet=True)
    assert "acknowledge it in one short line" in capture.extraction_prompt("abc", quiet=False)


def test_hooks_leave_heartbeats(monkeypatch):
    _stop(agent="codex")
    assert "codex:Stop" in capture.read_state()["heartbeats"]


def _status_output():
    import contextlib
    from io import StringIO

    from collivind.cli.commands import status as status_mod

    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        status_mod._status_capture()
    return buf.getvalue()


def test_status_says_codex_hooks_never_ran_and_how_to_trust_them(home):
    from collivind.cli.commands.hook import install_hooks

    (home / ".codex").mkdir()
    install_hooks(tool="codex")
    out = _status_output()
    assert "configured but never ran — trust unknown" in out and "Settings → Hooks" in out and "/hooks" in out


def test_status_separates_requested_from_confirmed(home):
    for _ in range(2):
        _stop()
    cid = _checkpoint_id(_stop())
    out = _status_output()
    assert "no confirmed save recorded yet" in out and f"checkpoint {cid}" in out and "not yet confirmed" in out

    _save(cid)
    _stop()
    out = _status_output()
    assert "last confirmed save" in out and "not yet confirmed" not in out


def test_status_flags_codex_inline_duplicates(home):
    from collivind.cli.commands.hook import install_hooks

    (home / ".codex").mkdir()
    install_hooks(tool="codex")
    (home / ".codex" / "config.toml").write_text(
        '[[hooks.Stop]]\n[[hooks.Stop.hooks]]\ntype = "command"\ncommand = "collivind hook stop --threshold 15"\n'
    )
    assert "also declared inline" in _status_output()


def _enable_codex_plugin(home):
    (home / ".codex").mkdir(exist_ok=True)
    (home / ".codex" / "config.toml").write_text('[plugins."collivind@collivind"]\nenabled = true\n')


def test_codex_plugin_replaces_user_level_hooks(home):
    from collivind.cli.commands.hook import hook_health, install_all_hooks
    from collivind.config import HooksConfig

    _enable_codex_plugin(home)
    assert "codex" not in install_all_hooks(HooksConfig())
    codex = next(h for h in hook_health() if h["tool"] == "codex")
    assert codex["source"] == "plugin" and "Stop" in codex["events"]


def test_codex_plugin_plus_hooks_json_is_flagged_as_double(home):
    from collivind.cli.commands.hook import hook_health, install_hooks

    (home / ".codex").mkdir()
    install_hooks(tool="codex")
    _enable_codex_plugin(home)
    codex = next(h for h in hook_health() if h["tool"] == "codex")
    assert codex["source"] == "both"


def test_unanswered_checkpoint_backs_off_then_is_recorded_as_failed():
    """An agent without the MCP tool must not be nagged for the whole session TTL."""
    for _ in range(2):
        assert _stop() is None
    cid = _checkpoint_id(_stop())

    asks = []
    for _ in range(200):
        block = _stop()
        if block:
            asks.append(_checkpoint_id(block))
        if "pending" not in capture.read_state()["sessions"]["s1"]:
            break
    assert asks == [cid] * capture.MAX_RETRIES  # capped, not forever

    failed = [e for e in capture.iter_history() if e["checkpoint_id"] == cid]
    assert len(failed) == 1 and failed[0]["ok"] is False and "re-ask" in failed[0]["error"]
    assert capture.summary()["pending"] == []

    # a later interval starts a fresh checkpoint
    later = [_stop() for _ in range(3)]
    assert later[:2] == [None, None] and _checkpoint_id(later[2]) != cid


def test_reasks_back_off():
    for _ in range(3):
        _stop()
    gaps, since = [], 0
    for _ in range(40):  # before the give-up, so only re-asks are seen
        since += 1
        if _stop():
            gaps.append(since)
            since = 0
    assert gaps == [capture.RETRY_AFTER * 2**i for i in range(capture.MAX_RETRIES)], gaps
