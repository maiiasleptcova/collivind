import json

from click.testing import CliRunner

from collivind.cli.commands.hook import get_state_file, hook


def test_hook_stop(tmp_path, monkeypatch):
    # Mock home directory to tmp_path
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    runner = CliRunner()
    
    # Run 14 times
    for _ in range(14):
        result = runner.invoke(hook, ["stop", "--threshold", "15"])
        assert result.exit_code == 0
        assert "collivind_extraction" not in result.output
        
    # 15th time should trigger
    result = runner.invoke(hook, ["stop", "--threshold", "15"])
    assert result.exit_code == 0
    assert "collivind_extraction" in result.output
    
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
