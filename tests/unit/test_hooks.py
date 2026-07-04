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
    assert set(events) == {"Stop", "PreCompact"}

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
    settings_path.write_text(json.dumps({
        "model": "opus",
        "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "other-tool run"}]}]},
    }))

    install_hooks()

    settings = _read_settings(tmp_path)
    assert settings["model"] == "opus"
    stop_cmds = [h["command"] for e in settings["hooks"]["Stop"] for h in e["hooks"]]
    assert "other-tool run" in stop_cmds
    assert any("collivind hook stop" in c for c in stop_cmds)


def test_install_hooks_respects_disabled_flags(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    events = install_hooks(enable_stop=False, enable_precompact=True)
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
