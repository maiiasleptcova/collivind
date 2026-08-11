"""Issue #21: opencode and Copilot each want a different MCP schema.

opencode : ~/.config/opencode/opencode.json   key "mcp",     command is a LIST
copilot  : ~/.vscode/mcp.json                 key "servers", command + args
codex    : ~/.codex/config.toml               [mcp_servers.*] (already, #13)

All three hold the user's own settings, so every registrar must append and
never rewrite.
"""

import json
import sys

import pytest

from collivind.cli.commands.init import _register_mcp_server, register_copilot_mcp, register_opencode_mcp


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


class TestClaudeCode:
    """Issue #33: the other three registrars resolve an absolute interpreter,
    but Claude Code's `claude mcp add` still passed a bare `python3`.

    Claude Code spawns the MCP server with its own environment, so `python3`
    resolves to whichever interpreter that environment finds first — typically
    the system one, which has no collivind installed. The server then dies at
    import and every tool reports the store as unavailable.
    """

    @pytest.fixture
    def argv(self, home, monkeypatch):
        """Capture the argv handed to `claude mcp add`."""
        seen = []

        def fake_run(cmd, **kw):
            seen.append(cmd)
            return type("R", (), {"returncode": 0, "stdout": b"", "stderr": b""})()

        monkeypatch.setattr("collivind.cli.commands.init.subprocess.run", fake_run)
        _register_mcp_server()
        assert seen, "claude mcp add was never invoked"
        return seen[0]

    def test_registers_with_an_absolute_interpreter(self, argv):
        assert "python3" not in argv, f"a bare interpreter may not resolve: {argv}"
        assert sys.executable in argv

    def test_still_runs_the_server_as_a_module(self, argv):
        """The interpreter changes; what it is told to run must not."""
        assert argv[-2:] == ["-m", "collivind.mcp.server"]
        assert argv[:3] == ["claude", "mcp", "add"]


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
