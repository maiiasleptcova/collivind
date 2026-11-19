"""Regression guards for the independent review of release 0.11."""

import builtins
import importlib
import json
import subprocess
import sys

from click.testing import CliRunner


def test_daemon_spawn_cannot_import_from_the_project_dir(tmp_path, monkeypatch):
    """#1 (HIGH): `python -m` from the hook's cwd imported a repo's own argparse.py."""
    from collivind.storage import embed_daemon

    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: calls.append((cmd, kw)))
    embed_daemon.spawn(tmp_path / "embed.sock", "m", 60)
    cmd, kw = calls[0]
    assert cmd[1] == "-I" and kw["cwd"] == str(tmp_path)


def test_isolated_mode_really_ignores_a_planted_module(tmp_path):
    evil = tmp_path / "repo"
    evil.mkdir()
    (evil / "argparse.py").write_text("open('PWNED', 'w').write('x')\n")
    subprocess.run([sys.executable, "-I", "-c", "import argparse"], cwd=evil, check=True)
    assert not (evil / "PWNED").exists()


def test_capture_imports_without_fcntl(monkeypatch):
    """#2: Windows has no fcntl; the MCP server and CLI must still import."""
    real_import = builtins.__import__

    def no_fcntl(name, *a, **kw):
        if name == "fcntl":
            raise ImportError("no fcntl")
        return real_import(name, *a, **kw)

    import collivind.capture as capture

    monkeypatch.setattr(builtins, "__import__", no_fcntl)
    reloaded = importlib.reload(capture)
    assert reloaded.fcntl is None
    monkeypatch.setattr(builtins, "__import__", real_import)
    importlib.reload(capture)


def test_interval_of_one_cannot_loop_opencode(tmp_path, monkeypatch):
    """#3: at save_interval=1 every reply to a save request triggered the next one."""
    from collivind import capture
    from collivind.cli.commands.hook import hook

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    blocks = 0
    for _ in range(4):
        out = CliRunner().invoke(hook, ["stop", "--threshold", "1", "--agent", "opencode", "--session", "s"]).output
        if out.strip():
            blocks += 1
            cid = json.loads(out)["reason"].split('checkpoint_id: "')[1].split('"')[0]
            capture.record(cid, True, [])
    assert blocks <= 2


def test_compaction_request_survives_a_recall_failure(tmp_path, monkeypatch):
    """#7: a dead store dropped the post-compaction save request."""
    from collivind.cli.commands.hook import hook

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    def boom():
        raise RuntimeError("store down")

    monkeypatch.setattr("collivind.cli.commands.hook._load_manager", boom)
    out = CliRunner().invoke(hook, ["session-start"], input=json.dumps({"session_id": "s", "source": "compact"}))
    context = json.loads(out.output)["hookSpecificOutput"]["additionalContext"]
    assert "UNAVAILABLE" in context and "just compacted" in context
