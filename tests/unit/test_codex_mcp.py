"""Issue #13: `collivind init` wired Codex halfway — hooks and prompts, but
never the MCP server."""

import tomllib

import pytest

from collivind.cli.commands.init import register_codex_mcp

EXISTING = """\
model = "gpt-5.5"
approval_policy = "on-request"

[mcp_servers.shopify-dev-mcp]
command = "npx"
args = ["-y", "@shopify/dev-mcp@latest"]

[mcp_servers.shopify-dev-mcp.tools.search_docs_chunks]
approval_mode = "approve"
"""


@pytest.fixture
def codex(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    d = tmp_path / ".codex"
    d.mkdir()
    return d / "config.toml"


def test_registers_when_absent(codex):
    codex.write_text(EXISTING)
    assert register_codex_mcp() is True

    conf = tomllib.loads(codex.read_text())
    entry = conf["mcp_servers"]["collivind"]
    assert entry["args"] == ["-m", "collivind.mcp.server"]
    assert entry["command"]


def test_preserves_every_existing_setting(codex):
    """The file holds the user's own config — losing it would be far worse
    than the bug being fixed."""
    codex.write_text(EXISTING)
    register_codex_mcp()

    conf = tomllib.loads(codex.read_text())
    assert conf["model"] == "gpt-5.5"
    assert conf["approval_policy"] == "on-request"
    assert conf["mcp_servers"]["shopify-dev-mcp"]["command"] == "npx"
    assert conf["mcp_servers"]["shopify-dev-mcp"]["tools"]["search_docs_chunks"]["approval_mode"] == "approve"


def test_idempotent(codex):
    codex.write_text(EXISTING)
    register_codex_mcp()
    first = codex.read_text()
    assert register_codex_mcp() is False, "already registered — must not append twice"
    assert codex.read_text() == first
    assert first.count("[mcp_servers.collivind]") == 1


def test_creates_config_when_missing(codex):
    assert not codex.exists()
    assert register_codex_mcp() is True
    assert tomllib.loads(codex.read_text())["mcp_servers"]["collivind"]["args"]


def test_skips_when_codex_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)  # no ~/.codex
    assert register_codex_mcp() is False
    assert not (tmp_path / ".codex").exists(), "must not create ~/.codex for non-users"


def test_refuses_to_touch_unparseable_config(codex):
    """Better to leave a hand-mangled file alone than append into it blind."""
    codex.write_text("[mcp_servers.broken\nthis is not toml")
    before = codex.read_text()
    assert register_codex_mcp() is False
    assert codex.read_text() == before


def test_uses_absolute_interpreter_not_bare_python3(codex):
    """The PATH lesson from #4: Codex spawns servers with its own env."""
    codex.write_text(EXISTING)
    register_codex_mcp()
    cmd = tomllib.loads(codex.read_text())["mcp_servers"]["collivind"]["command"]
    assert cmd != "python3", "a bare interpreter may not resolve in Codex's environment"


class TestStatusReporting:
    def _run(self):
        import click
        from click.testing import CliRunner

        from collivind.cli.commands.status import _status_codex_mcp

        @click.command()
        def probe():
            _status_codex_mcp()

        return CliRunner().invoke(probe)

    def test_reports_missing_registration(self, codex):
        codex.write_text(EXISTING)
        out = self._run().output
        assert "not registered" in out and "collivind init" in out

    def test_reports_registered(self, codex):
        codex.write_text(EXISTING)
        register_codex_mcp()
        assert "registered" in self._run().output
        assert "not registered" not in self._run().output

    def test_silent_for_non_codex_users(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
        assert self._run().output.strip() == ""

    def test_flags_broken_toml(self, codex):
        codex.write_text("[mcp_servers.broken\nnot toml")
        assert "not valid TOML" in self._run().output
