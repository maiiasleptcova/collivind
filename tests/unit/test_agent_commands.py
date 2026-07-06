
from click.testing import CliRunner

from collivind.cli.commands.agent_commands import commands, install_all_commands, install_commands


def test_install_claude_commands(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    written = install_commands("claude")

    save = tmp_path / ".claude" / "commands" / "mem-save.md"
    recall = tmp_path / ".claude" / "commands" / "mem-recall.md"
    assert set(written) == {save, recall}
    assert "$ARGUMENTS" in save.read_text()
    assert save.read_text().startswith("---\ndescription:")
    assert "collivind_get_context" in recall.read_text()


def test_install_codex_commands(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    written = install_commands("codex")
    assert all(p.parent == tmp_path / ".codex" / "prompts" for p in written)
    assert "$ARGUMENTS" in written[0].read_text()


def test_install_copilot_commands_into_project(tmp_path):
    written = install_commands("copilot", project_dir=tmp_path)
    names = sorted(p.name for p in written)
    assert names == ["mem-recall.prompt.md", "mem-save.prompt.md"]
    assert all(p.parent == tmp_path / ".github" / "prompts" for p in written)


def test_install_all_detects_codex(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert list(install_all_commands()) == ["claude"]
    (tmp_path / ".codex").mkdir()
    assert list(install_all_commands()) == ["claude", "codex"]


def test_cli_install_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    runner = CliRunner()
    assert runner.invoke(commands, ["install", "--tool", "claude"]).exit_code == 0
    result = runner.invoke(commands, ["install", "--tool", "claude"])
    assert result.exit_code == 0
    assert "mem-save.md" in result.output
