"""The Claude Code plugin is JSON + a shell shim, so pytest is the only thing
standing between a typo and a plugin that silently fails to load."""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "plugin"
RUNNER = PLUGIN / "bin" / "collivind-run"


@pytest.mark.parametrize(
    "rel",
    [
        "plugin/.claude-plugin/plugin.json",
        "plugin/.mcp.json",
        "plugin/hooks/hooks.json",
        ".claude-plugin/marketplace.json",
    ],
)
def test_manifests_are_valid_json(rel):
    json.loads((ROOT / rel).read_text())


def test_plugin_name_matches_marketplace_entry():
    """A mismatch here installs under one name and resolves under another."""
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    entry = next(p for p in market["plugins"] if p["source"] == "./plugin")
    assert entry["name"] == manifest["name"]


def test_plugin_version_matches_package_version():
    """The plugin pins its own version; drift means users stop getting updates."""
    from collivind.version import __version__

    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["version"] == __version__


def test_every_hook_command_points_at_the_runner():
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]
    commands = [h["command"] for entries in hooks.values() for e in entries for h in e["hooks"]]
    assert commands, "no hook commands registered"
    for c in commands:
        assert "${CLAUDE_PLUGIN_ROOT}" in c, f"not plugin-root relative: {c}"
        assert "/bin/collivind-run" in c, f"bypasses the resolver: {c}"


def test_runner_is_executable():
    assert os.access(RUNNER, os.X_OK)


def test_runner_fails_loudly_when_collivind_is_absent(tmp_path):
    """It must never exit 0 having done nothing — that is the silent-failure
    mode this whole plugin is meant to avoid."""
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    r = subprocess.run([str(RUNNER), "hook", "user-prompt"], capture_output=True, text=True, env=env)
    assert r.returncode == 127
    assert "not found" in r.stderr


def test_runner_execs_the_resolved_binary(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "collivind").write_text('#!/bin/sh\necho "ran $*"\n')
    (fake / "collivind").chmod(0o755)
    env = {"PATH": f"{fake}:/usr/bin:/bin", "HOME": str(tmp_path)}
    r = subprocess.run([str(RUNNER), "hook", "user-prompt"], capture_output=True, text=True, env=env)
    assert r.returncode == 0
    assert r.stdout.strip() == "ran hook user-prompt"


def test_mcp_flag_runs_the_module_not_a_subcommand(tmp_path):
    """`collivind mcp-server` does not exist; the server is a python module."""
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "collivind").write_text("#!/bin/sh\nexit 1\n")
    (fake / "collivind").chmod(0o755)
    (fake / "python").write_text('#!/bin/sh\necho "py $*"\n')
    (fake / "python").chmod(0o755)
    env = {"PATH": f"{fake}:/usr/bin:/bin", "HOME": str(tmp_path)}
    r = subprocess.run([str(RUNNER), "--mcp"], capture_output=True, text=True, env=env)
    assert r.stdout.strip() == "py -m collivind.mcp.server"


class TestMcpInterpreterResolution:
    """`uv tool install` puts a symlink in ~/.local/bin pointing into
    ~/.local/share/uv/tools/<pkg>/bin/. Deriving the interpreter from the link
    instead of its target picked a bare python3 that cannot import collivind,
    so the plugin's MCP server failed to start."""

    def _layout(self, tmp_path):
        """Reproduce uv's shape: real venv bin + a symlink dir on PATH."""
        venv = tmp_path / "uvtools" / "collivind-memory" / "bin"
        venv.mkdir(parents=True)
        (venv / "collivind").write_text("#!/bin/sh\nexit 1\n")
        (venv / "collivind").chmod(0o755)
        (venv / "python").write_text('#!/bin/sh\necho "venv-python $*"\n')
        (venv / "python").chmod(0o755)

        linkdir = tmp_path / "localbin"
        linkdir.mkdir()
        (linkdir / "collivind").symlink_to(venv / "collivind")
        return linkdir

    def test_mcp_uses_the_interpreter_beside_the_symlink_target(self, tmp_path):
        linkdir = self._layout(tmp_path)
        env = {"PATH": f"{linkdir}:/usr/bin:/bin", "HOME": str(tmp_path)}
        r = subprocess.run([str(RUNNER), "--mcp"], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        assert r.stdout.strip() == "venv-python -m collivind.mcp.server", (
            "must resolve the symlink, not use a bare python3"
        )

    def test_mcp_fails_loudly_when_no_interpreter_is_beside_the_binary(self, tmp_path):
        """Silently falling back to a python that cannot import collivind is
        how this bug hid — the server 'started' and then died."""
        bindir = tmp_path / "bin"
        bindir.mkdir()
        (bindir / "collivind").write_text("#!/bin/sh\nexit 1\n")
        (bindir / "collivind").chmod(0o755)
        env = {"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(tmp_path)}
        r = subprocess.run([str(RUNNER), "--mcp"], capture_output=True, text=True, env=env)
        assert r.returncode == 127
        assert "no interpreter beside" in r.stderr
