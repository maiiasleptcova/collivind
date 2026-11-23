"""GitHub Copilot hooks: payload shapes, output contracts, installer, status."""

import contextlib
import json
import subprocess
from io import StringIO
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from collivind import capture
from collivind.cli.commands import status as status_mod
from collivind.cli.commands.hook import (
    COPILOT_EVENTS,
    get_copilot_hooks_path,
    hook,
    hook_health,
    install_all_hooks,
    install_hooks,
)
from collivind.config import HooksConfig
from collivind.models import MemoryCategory, MemoryNode


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.chdir(repo)
    return tmp_path


def _run(args, payload=None):
    result = CliRunner().invoke(hook, args, input=json.dumps(payload) if payload is not None else None)
    assert result.exit_code == 0, result.output
    return json.loads(result.output) if result.output.strip() else None


def _manager(memories=(), context="No relevant context found"):
    manager = MagicMock()
    manager.get_timeline.return_value = list(memories)
    manager.get_context.return_value = context
    return manager


# --- payloads and output contracts -------------------------------------------------


def test_session_start_emits_top_level_additional_context(monkeypatch):
    memory = MemoryNode(content="c", summary="Postgres chosen", category=MemoryCategory.DECISION)
    monkeypatch.setattr("collivind.cli.commands.hook._load_manager", lambda: _manager([memory]))
    out = _run(["session-start", "--agent", "copilot"], {"sessionId": "s1", "source": "startup", "timestamp": 1})
    assert set(out) == {"additionalContext"}  # Copilot consumes only this key
    assert "[decision] Postgres chosen" in out["additionalContext"]
    assert "copilot:SessionStart" in capture.read_state()["heartbeats"]


def test_session_start_warns_when_store_is_down(monkeypatch):
    def boom():
        raise RuntimeError("qdrant refused")

    monkeypatch.setattr("collivind.cli.commands.hook._load_manager", boom)
    out = _run(["session-start", "--agent", "copilot"], {"sessionId": "s1", "source": "startup"})
    assert "UNAVAILABLE" in out["additionalContext"] and "qdrant refused" in out["additionalContext"]


@pytest.mark.parametrize("key", ["sessionId", "session_id"])
def test_stop_reads_both_session_id_shapes(key):
    for _ in range(2):
        assert _run(["stop", "--agent", "copilot", "--threshold", "3"], {key: "abc"}) is None
    out = _run(["stop", "--agent", "copilot", "--threshold", "3"], {key: "abc"})
    assert out["decision"] == "block" and "collivind_extraction" in out["reason"]
    assert "abc" in capture.read_state()["sessions"]  # not collapsed into "default"


def test_stop_hook_active_never_chains_a_block():
    for _ in range(5):
        assert (
            _run(["stop", "--agent", "copilot", "--threshold", "2"], {"sessionId": "s", "stop_hook_active": True})
            is None
        )
    assert "count" not in capture.read_state()["sessions"].get("s", {})


def test_compaction_is_requested_on_the_very_next_stop():
    _run(["precompact", "--agent", "copilot"], {"sessionId": "s1", "trigger": "auto"})
    pending = capture.read_state()["sessions"]["s1"]["pending"]
    assert pending["trigger"] == "compaction:auto" and pending["due"]

    # Our own continuation in between must not consume the request.
    assert _run(["stop", "--agent", "copilot"], {"sessionId": "s1", "stop_hook_active": True}) is None
    out = _run(["stop", "--agent", "copilot"], {"sessionId": "s1", "stop_hook_active": False})
    assert out["decision"] == "block" and pending["id"] in out["reason"]

    # Then normal backoff: not re-asked on the following stop.
    assert _run(["stop", "--agent", "copilot"], {"sessionId": "s1"}) is None
    assert "due" not in capture.read_state()["sessions"]["s1"]["pending"]


@pytest.mark.parametrize("agent", ["claude", "codex", "opencode"])
def test_other_agents_keep_waiting_for_session_start_after_compaction(agent):
    _run(["precompact", "--agent", agent], {"session_id": "s1", "trigger": "auto"})
    assert "due" not in capture.read_state()["sessions"]["s1"]["pending"]
    assert _run(["stop", "--agent", agent], {"session_id": "s1"}) is None


def test_prompt_recall_prepends_to_transformed_prompt(monkeypatch):
    monkeypatch.setattr(
        "collivind.cli.commands.hook._load_manager", lambda: _manager(["x"], "Postgres was chosen for transactions")
    )
    payload = {"sessionId": "s", "prompt": "which database did we choose?", "transformedPrompt": "<ctx/>\nwhich db?"}
    out = _run(["user-prompt", "--agent", "copilot"], payload)
    assert set(out) == {"modifiedTransformedPrompt"}
    text = out["modifiedTransformedPrompt"]
    assert text.startswith("<collivind_recall>") and text.endswith(payload["transformedPrompt"])


def test_prompt_recall_never_replaces_without_the_original(monkeypatch):
    monkeypatch.setattr("collivind.cli.commands.hook._load_manager", lambda: _manager(["x"], "relevant memory text"))
    for transformed in (None, "", "   "):
        payload = {"prompt": "which database did we choose?", "transformedPrompt": transformed}
        assert _run(["user-prompt", "--agent", "copilot"], payload) is None


def test_prompt_recall_silent_when_nothing_relevant(monkeypatch):
    monkeypatch.setattr("collivind.cli.commands.hook._load_manager", lambda: _manager(["x"]))
    payload = {"prompt": "which database did we choose?", "transformedPrompt": "which database did we choose?"}
    assert _run(["user-prompt", "--agent", "copilot"], payload) is None


# --- installer -----------------------------------------------------------------------


def _validate_copilot_hooks_doc(doc):
    """The documented config schema: version 1, camelCase events, command entries."""
    assert doc["version"] == 1
    assert set(doc["hooks"]) <= {
        "sessionStart", "sessionEnd", "userPromptSubmitted", "userPromptTransformed", "preToolUse",
        "postToolUse", "agentStop", "subagentStop", "subagentStart", "preCompact", "errorOccurred",
    }  # fmt: skip
    for entries in doc["hooks"].values():
        assert isinstance(entries, list) and entries
        for e in entries:
            assert set(e) <= {"type", "bash", "powershell", "command", "cwd", "env", "timeoutSec", "matcher"}
            assert e["type"] == "command" and isinstance(e["bash"], str) and isinstance(e["powershell"], str)
            assert isinstance(e["timeoutSec"], int) and 0 < e["timeoutSec"] <= 60


def test_install_writes_user_level_file_idempotently(home):
    (home / ".copilot" / "hooks").mkdir(parents=True)
    other = home / ".copilot" / "hooks" / "other.json"
    other.write_text('{"version": 1, "hooks": {}}')

    events = install_hooks(tool="copilot")
    first = get_copilot_hooks_path().read_text()
    install_hooks(tool="copilot")
    assert get_copilot_hooks_path().read_text() == first
    assert other.read_text() == '{"version": 1, "hooks": {}}'

    doc = json.loads(first)
    _validate_copilot_hooks_doc(doc)
    assert set(events) == set(doc["hooks"]) == set(COPILOT_EVENTS.values())
    assert doc["hooks"]["agentStop"][0]["bash"].endswith("hook stop --agent copilot")
    assert doc["hooks"]["userPromptTransformed"][0]["bash"].endswith("hook user-prompt --agent copilot")
    assert doc["hooks"]["sessionStart"][0]["powershell"].startswith("& '")


def test_install_honours_copilot_home_and_disabled_events(home, monkeypatch):
    monkeypatch.setenv("COPILOT_HOME", str(home / "alt"))
    events = install_hooks(tool="copilot", enable_user_prompt=False, enable_precompact=False)
    doc = json.loads((home / "alt" / "hooks" / "collivind.json").read_text())
    assert set(events) == set(doc["hooks"]) == {"sessionStart", "agentStop"}


def test_workspace_install_uses_a_portable_command(home):
    install_hooks(tool="copilot", workspace=True)
    doc = json.loads((home / "repo" / ".github" / "hooks" / "collivind.json").read_text())
    _validate_copilot_hooks_doc(doc)
    assert doc["hooks"]["agentStop"][0]["bash"] == "collivind hook stop --agent copilot"
    assert not get_copilot_hooks_path().exists()


def test_workspace_flag_requires_copilot():
    result = CliRunner().invoke(hook, ["install", "--tool", "claude", "--workspace"])
    assert result.exit_code != 0 and "--tool copilot" in result.output


def test_auto_install_detects_copilot_or_vscode(home):
    assert "copilot" not in install_all_hooks(HooksConfig())
    (home / ".vscode").mkdir()
    assert "copilot" in install_all_hooks(HooksConfig())
    assert get_copilot_hooks_path().exists()


# --- status --------------------------------------------------------------------------


def _status():
    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        status_mod._status_hooks()
        status_mod._status_capture()
    return buf.getvalue()


def test_status_rows_configured_then_ran(home):
    (home / ".copilot").mkdir()
    out = _status()
    assert "✗ copilot: no collivind hooks registered" in out

    install_hooks(tool="copilot")
    out = _status()
    assert "✓ copilot: PreCompact, SessionStart, Stop, UserPromptSubmit" in out
    assert "preview" in out
    assert "copilot: PreCompact, SessionStart, Stop, UserPromptSubmit configured but has not run yet" in out

    _run(["stop", "--agent", "copilot"], {"sessionId": "s"})
    assert "✓ copilot: Stop last ran" in _status()


def test_status_flags_user_plus_workspace_double_registration(home):
    (home / ".copilot").mkdir()
    install_hooks(tool="copilot")
    install_hooks(tool="copilot", workspace=True)
    h = {h["tool"]: h for h in hook_health()}["copilot"]
    assert h["source"] == "both"
    assert "fires twice" in _status()


def test_status_flags_repo_claude_settings_run_by_copilot(home):
    (home / ".copilot").mkdir()
    install_hooks(tool="copilot")
    (home / "repo" / ".claude").mkdir()
    (home / "repo" / ".claude" / "settings.local.json").write_text(
        json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "collivind hook stop"}]}]}})
    )
    out = _status()
    assert "VS Code's Copilot harness runs those too" in out and "Stop" in out


def test_status_flags_unresolvable_command(home):
    (home / ".copilot" / "hooks").mkdir(parents=True)
    get_copilot_hooks_path().write_text(
        json.dumps(
            {
                "version": 1,
                "hooks": {"agentStop": [{"type": "command", "bash": "/gone/collivind hook stop --agent copilot"}]},
            }
        )
    )
    assert "✗ copilot: Stop registered but the command no longer resolves" in _status()


def test_outside_a_repo_user_claude_settings_are_not_a_repo_double(home, monkeypatch):
    """From ~ (no repo), ~/.claude/settings.json is user config, not a repo file Copilot also runs."""
    from collivind.cli.commands.hook import copilot_warnings

    (home / ".copilot").mkdir()
    install_all_hooks(HooksConfig())  # writes ~/.claude/settings.json and the Copilot file
    monkeypatch.chdir(home)
    assert copilot_warnings() == []
    result = CliRunner().invoke(hook, ["install", "--tool", "copilot", "--workspace"])
    assert result.exit_code != 0 and "git repository" in result.output
    assert not (home / ".github").exists()


def test_workspace_command_that_does_not_resolve_is_flagged(home, monkeypatch):
    monkeypatch.setattr("collivind.cli.commands.hook.shutil.which", lambda _: None)
    install_hooks(tool="copilot", workspace=True)
    assert "✗ copilot: PreCompact, SessionStart, Stop, UserPromptSubmit registered but the command" in _status()


def test_mcp_registered_where_vscode_and_the_cli_both_read_it(home):
    from collivind.cli.commands import init as init_mod

    assert not init_mod.register_copilot_cli_mcp()  # nothing installed: create nothing
    assert not (home / ".copilot").exists()
    (home / ".vscode").mkdir()
    assert init_mod.register_copilot_cli_mcp()
    assert not init_mod.register_copilot_cli_mcp()  # idempotent
    entry = json.loads((home / ".copilot" / "mcp-config.json").read_text())["mcpServers"]["collivind"]
    assert entry["type"] == "local" and entry["tools"] == ["*"]
    assert entry["args"] == ["-m", "collivind.mcp.server"]
