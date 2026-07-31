"""Issue #21: opencode and Copilot each want a different MCP schema.

opencode : ~/.config/opencode/opencode.json   key "mcp",     command is a LIST
copilot  : ~/.vscode/mcp.json                 key "servers", command + args
codex    : ~/.codex/config.toml               [mcp_servers.*] (already, #13)

All three hold the user's own settings, so every registrar must append and
never rewrite.
"""

import json

import pytest

from collivind.cli.commands.init import register_copilot_mcp, register_opencode_mcp


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return tmp_path


class TestOpencode:
    def _config(self, home):
        d = home / ".config" / "opencode"
        d.mkdir(parents=True)
        return d / "opencode.json"

    def test_registers_with_opencode_schema(self, home):
        self._config(home)
        assert register_opencode_mcp() is True
        entry = json.loads((home / ".config/opencode/opencode.json").read_text())["mcp"]["collivind"]
        assert entry["type"] == "local"
        assert entry["enabled"] is True
        assert isinstance(entry["command"], list), "opencode takes a command list, not a string"
        assert entry["command"][1:] == ["-m", "collivind.mcp.server"]

    def test_preserves_existing_settings(self, home):
        p = self._config(home)
        p.write_text(json.dumps({"model": "anthropic/claude", "mcp": {"other": {"type": "local", "command": ["x"]}}}))
        register_opencode_mcp()
        conf = json.loads(p.read_text())
        assert conf["model"] == "anthropic/claude"
        assert conf["mcp"]["other"]["command"] == ["x"]
        assert "collivind" in conf["mcp"]

    def test_idempotent(self, home):
        self._config(home)
        assert register_opencode_mcp() is True
        before = (home / ".config/opencode/opencode.json").read_text()
        assert register_opencode_mcp() is False
        assert (home / ".config/opencode/opencode.json").read_text() == before

    def test_skips_when_opencode_absent(self, home):
        assert register_opencode_mcp() is False
        assert not (home / ".config" / "opencode").exists(), "must not create the dir for non-users"

    def test_refuses_unparseable_config(self, home):
        p = self._config(home)
        p.write_text("{not json")
        assert register_opencode_mcp() is False
        assert p.read_text() == "{not json"


class TestCopilot:
    def test_registers_with_vscode_schema(self, home):
        (home / ".vscode").mkdir()
        assert register_copilot_mcp() is True
        entry = json.loads((home / ".vscode/mcp.json").read_text())["servers"]["collivind"]
        assert entry["type"] == "stdio"
        assert entry["args"] == ["-m", "collivind.mcp.server"]
        assert entry["command"] != "python3", "a bare interpreter may not resolve"

    def test_preserves_existing_servers(self, home):
        (home / ".vscode").mkdir()
        p = home / ".vscode" / "mcp.json"
        p.write_text(json.dumps({"servers": {"playwright": {"type": "stdio", "command": "npx", "args": ["-y", "x"]}}}))
        register_copilot_mcp()
        conf = json.loads(p.read_text())
        assert conf["servers"]["playwright"]["command"] == "npx"
        assert "collivind" in conf["servers"]

    def test_idempotent(self, home):
        (home / ".vscode").mkdir()
        assert register_copilot_mcp() is True
        assert register_copilot_mcp() is False

    def test_skips_when_vscode_absent(self, home):
        assert register_copilot_mcp() is False
        assert not (home / ".vscode").exists()
