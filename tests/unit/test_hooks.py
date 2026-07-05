import json

from click.testing import CliRunner

from collivind.cli.commands.hook import get_state_file, hook, install_hooks


def test_hook_stop(tmp_path, monkeypatch):
    # Mock home directory to tmp_path
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    runner = CliRunner()

    # Run 14 times
    for _ in range(14):
        result = runner.invoke(hook, ["stop", "--threshold", "15"])
        assert result.exit_code == 0
        assert "collivind_extraction" not in result.output

    # 15th time should trigger with a Stop-hook block decision (plain stdout
    # is discarded by Claude Code)
    result = runner.invoke(hook, ["stop", "--threshold", "15"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["decision"] == "block"
    assert "collivind_extraction" in payload["reason"]

    # State should reset
    state_file = get_state_file()
    assert state_file.exists()
    with open(state_file) as f:
        state = json.load(f)
    assert state["message_count"] == 0


def test_hook_precompact(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    runner = CliRunner()

    result = runner.invoke(hook, ["precompact"])
    assert result.exit_code == 0
    assert "collivind_urgent_extraction" in result.output


def _read_settings(tmp_path):
    with open(tmp_path / ".claude" / "settings.json") as f:
        return json.load(f)


def test_install_hooks_creates_settings(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    events = install_hooks(enable_stop=True, enable_precompact=True, save_interval=10)
    assert set(events) == {"Stop", "PreCompact", "SessionStart", "UserPromptSubmit"}

    settings = _read_settings(tmp_path)
    stop_cmds = [h["command"] for e in settings["hooks"]["Stop"] for h in e["hooks"]]
    assert stop_cmds == ["collivind hook stop --threshold 10"]
    pc_cmds = [h["command"] for e in settings["hooks"]["PreCompact"] for h in e["hooks"]]
    assert pc_cmds == ["collivind hook precompact"]


def test_install_hooks_idempotent_and_updates_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    install_hooks(save_interval=15)
    install_hooks(save_interval=20)

    settings = _read_settings(tmp_path)
    stop_cmds = [h["command"] for e in settings["hooks"]["Stop"] for h in e["hooks"]]
    assert stop_cmds == ["collivind hook stop --threshold 20"]


def test_install_hooks_preserves_existing_settings(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "model": "opus",
                "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "other-tool run"}]}]},
            }
        )
    )

    install_hooks()

    settings = _read_settings(tmp_path)
    assert settings["model"] == "opus"
    stop_cmds = [h["command"] for e in settings["hooks"]["Stop"] for h in e["hooks"]]
    assert "other-tool run" in stop_cmds
    assert any("collivind hook stop" in c for c in stop_cmds)


def test_install_hooks_respects_disabled_flags(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    events = install_hooks(enable_stop=False, enable_precompact=True, enable_session_start=False,
                           enable_user_prompt=False)
    assert events == ["PreCompact"]
    settings = _read_settings(tmp_path)
    assert "Stop" not in settings["hooks"]


def test_install_hooks_rejects_corrupt_settings(tmp_path, monkeypatch):
    import click
    import pytest

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("{not json")

    with pytest.raises(click.ClickException):
        install_hooks()


def test_session_start_emits_context_index(monkeypatch):
    from collivind.models import MemoryCategory, MemoryNode

    manager = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    manager.get_timeline.return_value = [
        MemoryNode(
            content="We picked Postgres over Mongo for transactions",
            summary="Postgres chosen for transactions",
            category=MemoryCategory.DECISION,
        ),
    ]
    monkeypatch.setattr("collivind.cli.commands.hook._load_manager", lambda: manager)

    result = CliRunner().invoke(hook, ["session-start"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    ctx = payload["hookSpecificOutput"]["additionalContext"]
    assert payload["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "[decision] Postgres chosen" in ctx


def test_session_start_silent_when_empty_or_broken(monkeypatch):
    manager = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    manager.get_timeline.return_value = []
    monkeypatch.setattr("collivind.cli.commands.hook._load_manager", lambda: manager)
    result = CliRunner().invoke(hook, ["session-start"])
    assert result.exit_code == 0
    assert result.output.strip() == ""

    def boom():
        raise RuntimeError("backend down")

    monkeypatch.setattr("collivind.cli.commands.hook._load_manager", boom)
    result = CliRunner().invoke(hook, ["session-start"])
    assert result.exit_code == 0
    assert result.output.strip() == ""


def test_install_hooks_registers_session_start(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    events = install_hooks()
    assert "SessionStart" in events
    settings = _read_settings(tmp_path)
    cmds = [h["command"] for e in settings["hooks"]["SessionStart"] for h in e["hooks"]]
    assert cmds == ["collivind hook session-start"]


def test_install_hooks_codex_session_start_only(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    events = install_hooks(tool="codex")
    assert set(events) == {"SessionStart", "UserPromptSubmit"}

    with open(tmp_path / ".codex" / "hooks.json") as f:
        settings = json.load(f)
    assert set(settings["hooks"].keys()) == {"SessionStart", "UserPromptSubmit"}
    cmds = [h["command"] for e in settings["hooks"]["SessionStart"] for h in e["hooks"]]
    assert cmds == ["collivind hook session-start"]


def test_install_all_hooks_detects_codex(tmp_path, monkeypatch):
    from collivind.cli.commands.hook import install_all_hooks
    from collivind.config import HooksConfig

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    results = install_all_hooks(HooksConfig())
    assert "codex" not in results  # no ~/.codex dir

    (tmp_path / ".codex").mkdir()
    results = install_all_hooks(HooksConfig())
    assert set(results["codex"]) == {"SessionStart", "UserPromptSubmit"}
    assert set(results["claude"]) == {"Stop", "PreCompact", "SessionStart", "UserPromptSubmit"}


def _prompt_payload(prompt):
    return json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": prompt})


def test_user_prompt_injects_relevant_context(monkeypatch):
    manager = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    manager.get_context.return_value = "--- Collivind Context ---\n[DECISION] Postgres chosen"
    monkeypatch.setattr("collivind.cli.commands.hook._load_manager", lambda: manager)

    result = CliRunner().invoke(hook, ["user-prompt"], input=_prompt_payload("build the database layer for orders"))
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "Postgres chosen" in payload["hookSpecificOutput"]["additionalContext"]
    assert manager.get_context.call_args.kwargs["max_tokens"] == 800


def test_user_prompt_silent_on_short_prompt_or_no_match(monkeypatch):
    manager = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    monkeypatch.setattr("collivind.cli.commands.hook._load_manager", lambda: manager)

    result = CliRunner().invoke(hook, ["user-prompt"], input=_prompt_payload("ok"))
    assert result.exit_code == 0 and result.output.strip() == ""
    manager.get_context.assert_not_called()

    manager.get_context.return_value = "No relevant context found in Collivind."
    result = CliRunner().invoke(hook, ["user-prompt"], input=_prompt_payload("something totally unrelated here"))
    assert result.exit_code == 0 and result.output.strip() == ""


def test_user_prompt_silent_on_garbage_stdin_or_broken_backend(monkeypatch):
    result = CliRunner().invoke(hook, ["user-prompt"], input="{not json")
    assert result.exit_code == 0 and result.output.strip() == ""

    def boom():
        raise RuntimeError("backend down")

    monkeypatch.setattr("collivind.cli.commands.hook._load_manager", boom)
    result = CliRunner().invoke(hook, ["user-prompt"], input=_prompt_payload("a long enough prompt to search"))
    assert result.exit_code == 0 and result.output.strip() == ""
