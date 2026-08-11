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

from collivind.cli.commands.init import (
    _register_mcp_server,
    claude_mcp_servers,
    register_copilot_mcp,
    register_opencode_mcp,
)

# Verbatim `claude mcp list` output (Claude Code 2.1.227). Names may contain
# colons; the health-check header and the multi-scope warning lines are not
# server entries.
CLAUDE_LIST = """Checking MCP server health…

claude.ai Google Drive: https://drivemcp.googleapis.com/mcp/v1 - ✔ Connected
github: npx -y @modelcontextprotocol/server-github - ✔ Connected
plugin:collivind:collivind: /Users/x/.claude/plugins/cache/collivind/0.5.3/bin/collivind-run --mcp - ✔ Connected
"""


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


class TestClaudeServerNames:
    """Parsing `claude mcp list`, which is the only usable signal: `claude mcp
    get <name>` exits 0 even for a server that does not exist."""

    def _listing(self, monkeypatch, stdout, returncode=0):
        def fake_run(cmd, **kw):
            return type("R", (), {"returncode": returncode, "stdout": stdout, "stderr": ""})()

        monkeypatch.setattr("collivind.cli.commands.init.subprocess.run", fake_run)

    def test_reads_names_that_contain_colons(self, monkeypatch):
        """A plugin server is `plugin:collivind:collivind`. Splitting on the
        first colon rather than the first colon-space truncates that to
        `plugin`, and the duplicate check then never fires."""
        self._listing(monkeypatch, CLAUDE_LIST)
        servers, problem = claude_mcp_servers()
        assert problem == ""
        assert set(servers) == {
            "claude.ai Google Drive",
            "github",
            "plugin:collivind:collivind",
        }
        assert servers["plugin:collivind:collivind"] == "✔ Connected"

    def test_ignores_the_multi_scope_warning_lines(self, monkeypatch):
        self._listing(
            monkeypatch,
            'x: /a/python -m y - Failed\n├ Server "x" is defined in multiple scopes\n└ Keep the correct one\n',
        )
        servers, _ = claude_mcp_servers()
        assert set(servers) == {"x"}

    def test_returns_none_when_the_cli_is_missing(self, monkeypatch):
        def boom(cmd, **kw):
            raise FileNotFoundError("claude")

        monkeypatch.setattr("collivind.cli.commands.init.subprocess.run", boom)
        servers, problem = claude_mcp_servers()
        assert servers is None and "not found" in problem

    def test_returns_none_on_a_failing_cli(self, monkeypatch):
        """An empty set would read as 'nothing registered' and trigger an add."""
        self._listing(monkeypatch, "", returncode=1)
        servers, problem = claude_mcp_servers()
        assert servers is None and problem, "the reason must survive"


class TestClaudeCode:
    """Issue #33 turned out to be three faults on one line.

    1. A bare `python3`. Claude Code spawns the server with its own
       environment, so that resolves to the system interpreter, which has no
       collivind: the server dies at import and every tool reports the store
       as unavailable.
    2. `--global`, which the CLI does not accept (it takes `-s/--scope`).
       Verified against Claude Code 2.1.227: "unknown option '--global'",
       non-zero exit, nothing written — so this had never registered anything,
       and `capture_output` hid the reason.
    3. Registering unconditionally. The plugin already serves collivind as
       `plugin:collivind:collivind`, and `add` cannot update an entry in
       place, so a blind add duplicates the server for everyone following the
       recommended install.
    """

    @pytest.fixture
    def run_with(self, home, monkeypatch):
        """Run registration against a given `claude mcp list` output, and
        return every argv that reached subprocess."""

        def go(listing):
            seen = []

            def fake_run(cmd, **kw):
                seen.append(cmd)
                stdout = listing if cmd[:3] == ["claude", "mcp", "list"] else ""
                return type("R", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()

            monkeypatch.setattr("collivind.cli.commands.init.subprocess.run", fake_run)
            _register_mcp_server()
            return seen

        return go

    @pytest.fixture
    def add_argv(self, run_with):
        calls = run_with("github: npx -y whatever - Connected\n")
        add = [c for c in calls if c[:3] == ["claude", "mcp", "add"]]
        assert add, f"no `claude mcp add` among {calls}"
        return add[0]

    def test_registers_with_an_absolute_interpreter(self, add_argv):
        assert "python3" not in add_argv, f"a bare interpreter may not resolve: {add_argv}"
        assert sys.executable in add_argv

    def test_uses_a_scope_the_cli_accepts(self, add_argv):
        """`--global` is rejected outright, so the command never ran at all.

        Asserted whole rather than by slice: truncating the argv is how the
        wrong `-s` value slipped past a green suite once already.
        """
        assert "--global" not in add_argv, "claude mcp add has no --global flag"
        assert add_argv == [
            "claude",
            "mcp",
            "add",
            "-s",
            "user",
            "collivind",
            "--",
            sys.executable,
            "-m",
            "collivind.mcp.server",
        ]

    def test_leaves_the_plugins_server_alone(self, run_with):
        """The recommended install already provides one. Adding on top gives
        two identical servers and two copies of every tool in context."""
        calls = run_with(CLAUDE_LIST)
        assert not [c for c in calls if c[:3] == ["claude", "mcp", "add"]], f"duplicated the plugin server: {calls}"

    def test_leaves_a_hand_made_entry_alone(self, run_with):
        """`add` cannot update in place, and its collision check is per-scope,
        so a same-named entry in another scope would silently keep winning."""
        calls = run_with("collivind: /usr/bin/python3 -m collivind.mcp.server - Failed to connect\n")
        assert not [c for c in calls if c[:3] == ["claude", "mcp", "add"]], f"clobbered an existing entry: {calls}"

    def test_never_removes_anything(self, run_with):
        """A blind remove can strip a working registration and, if the add
        then fails, leave the user with none at all."""
        for listing in (CLAUDE_LIST, "github: npx -y whatever - Connected\n"):
            calls = run_with(listing)
            assert not [c for c in calls if c[:3] == ["claude", "mcp", "remove"]], f"destructive: {calls}"

    def test_names_what_is_already_registered(self, run_with, capsys):
        """Silence here reads as 'did nothing'; the user needs to know which
        server is answering."""
        run_with(CLAUDE_LIST)
        assert "plugin:collivind:collivind" in capsys.readouterr().out

    def test_skips_cleanly_when_the_cli_is_absent(self, home, monkeypatch, capsys):
        def boom(cmd, **kw):
            raise FileNotFoundError("claude")

        monkeypatch.setattr("collivind.cli.commands.init.subprocess.run", boom)
        _register_mcp_server()
        assert "claude CLI not found" in capsys.readouterr().out

    def test_surfaces_why_registration_failed(self, home, monkeypatch, capsys):
        """A swallowed stderr is how `--global` survived: init reported a bare
        'registration failed' every run and nobody could see the cause."""

        def fake_run(cmd, **kw):
            if cmd[:3] == ["claude", "mcp", "add"]:
                raise subprocess.CalledProcessError(1, cmd, stderr="error: not allowed by enterprise policy")
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        monkeypatch.setattr("collivind.cli.commands.init.subprocess.run", fake_run)
        _register_mcp_server()
        assert "not allowed by enterprise policy" in capsys.readouterr().out

    def test_a_dead_entry_is_not_reported_as_healthy(self, run_with, capsys):
        """The person who filed #33 has exactly this entry. Telling them it is
        `already registered`, in green, from the very command they run to fix
        it, is worse than the pre-PR behaviour, which at least tried to add.
        """
        run_with("collivind: /usr/bin/python3 -m collivind.mcp.server - ✘ Failed to connect\n")
        out = capsys.readouterr().out
        assert "not answering" in out, out
        assert "claude mcp remove collivind" in out, "no repair path offered"

    def test_a_live_entry_is_reported_as_healthy(self, run_with, capsys):
        run_with(CLAUDE_LIST)
        out = capsys.readouterr().out
        assert "already registered as plugin:collivind:collivind" in out
        assert "not answering" not in out

    @pytest.mark.parametrize(
        ("exc", "returncode", "expected"),
        [
            (FileNotFoundError("claude"), 0, "claude CLI not found"),
            (subprocess.TimeoutExpired("claude", 60), 0, "timed out"),
            (None, 1, "unable to read config"),
        ],
    )
    def test_an_unreadable_listing_reports_its_own_cause(self, home, monkeypatch, capsys, exc, returncode, expected):
        """All three used to print `claude CLI not found`, so a hung listing or
        a broken config was diagnosed as a missing install — and registration
        was skipped with no fallback command to recover with."""

        def fake_run(cmd, **kw):
            if exc:
                raise exc
            return type("R", (), {"returncode": returncode, "stdout": "", "stderr": "error: unable to read config"})()

        monkeypatch.setattr("collivind.cli.commands.init.subprocess.run", fake_run)
        _register_mcp_server()
        out = capsys.readouterr().out
        assert expected in out, out
        assert "Run manually: claude mcp add -s user collivind" in out, "no recovery command offered"

    def test_the_listing_is_decoded_as_text(self, home, monkeypatch):
        """Nothing pinned this: without text=True the parser gets bytes and
        `collivind init` dies with AttributeError on the first line.

        Keyed to the `list` call specifically — asserting over every call
        passes on the `add` call's own text=True while the listing stays
        undecoded, which is exactly how this slipped through once.
        """
        seen = {}

        def fake_run(cmd, **kw):
            seen[tuple(cmd[:3])] = kw
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        monkeypatch.setattr("collivind.cli.commands.init.subprocess.run", fake_run)
        _register_mcp_server()
        listing = seen.get(("claude", "mcp", "list"))
        assert listing is not None, f"the listing was never requested: {list(seen)}"
        assert listing.get("text") is True, "the listing must be decoded, not bytes"
        assert listing.get("timeout"), "an unbounded listing hangs init on a slow server"


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
