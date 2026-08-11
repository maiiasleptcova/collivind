"""Issue #21: opencode and Copilot each want a different MCP schema.

opencode : ~/.config/opencode/opencode.json   key "mcp",     command is a LIST
copilot  : ~/.vscode/mcp.json                 key "servers", command + args
codex    : ~/.codex/config.toml               [mcp_servers.*] (already, #13)

All three hold the user's own settings, so every registrar must append and
never rewrite.
"""

import json
import subprocess
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
    """Issue #33: registration with Claude Code was broken in two ways at once.

    1. It passed a bare `python3`. Claude Code spawns the MCP server with its
       own environment, so that resolves to whichever interpreter comes first
       on PATH — normally the system one, which has no collivind installed.
       The server then dies at import and every tool reports the store as
       unavailable.
    2. It passed `--global`, which the CLI does not accept (`claude mcp add`
       takes `-s/--scope local|user|project`). Verified against Claude Code
       2.1.227: the command exits non-zero with "unknown option '--global'"
       and writes nothing, so registration had never actually happened — and
       `capture_output` hid the reason behind a bare "registration failed".
    """

    @pytest.fixture
    def calls(self, home, monkeypatch):
        """Every argv `_register_mcp_server` hands to subprocess."""
        seen = []

        def fake_run(cmd, **kw):
            seen.append(cmd)
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        monkeypatch.setattr("collivind.cli.commands.init.subprocess.run", fake_run)
        _register_mcp_server()
        assert seen, "the claude CLI was never invoked"
        return seen

    @pytest.fixture
    def add_argv(self, calls):
        add = [c for c in calls if c[:3] == ["claude", "mcp", "add"]]
        assert add, f"no `claude mcp add` among {calls}"
        return add[0]

    def test_registers_with_an_absolute_interpreter(self, add_argv):
        assert "python3" not in add_argv, f"a bare interpreter may not resolve: {add_argv}"
        assert sys.executable in add_argv

    def test_uses_a_scope_the_cli_accepts(self, add_argv):
        """`--global` is rejected outright, so the whole command never ran."""
        assert "--global" not in add_argv, "claude mcp add has no --global flag"
        assert add_argv[3:5] == ["-s", "user"], f"expected user scope, got {add_argv}"

    def test_still_runs_the_server_as_a_module(self, add_argv):
        """The interpreter changes; what it is told to run must not."""
        assert add_argv[-2:] == ["-m", "collivind.mcp.server"]

    def test_drops_a_stale_entry_before_adding(self, calls):
        """`claude mcp add` refuses a name that already exists rather than
        updating it, so without this the users who actually hit #33 — the ones
        carrying a broken python3 entry — could never repair it by upgrading
        and re-running init."""
        kinds = [c[:3] for c in calls if c[0] == "claude"]
        assert ["claude", "mcp", "remove"] in kinds, f"stale entry never dropped: {calls}"
        assert kinds.index(["claude", "mcp", "remove"]) < kinds.index(["claude", "mcp", "add"])

    def test_surfaces_why_registration_failed(self, home, monkeypatch, capsys):
        """A swallowed stderr is how `--global` survived: init reported a bare
        'registration failed' for every run and nobody could see the cause."""

        def fake_run(cmd, **kw):
            if cmd[:3] == ["claude", "mcp", "add"]:
                raise subprocess.CalledProcessError(1, cmd, stderr="error: unknown option '--nonsense'")
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        monkeypatch.setattr("collivind.cli.commands.init.subprocess.run", fake_run)
        _register_mcp_server()
        assert "unknown option '--nonsense'" in capsys.readouterr().out


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
