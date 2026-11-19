"""#35/#34: the opencode capture plugin maps opencode events onto collivind hooks.

opencode itself is not needed: the plugin is loaded in Node with stand-ins for
the two things opencode hands it (the Bun `$` shell and the SDK client), and
the shell stand-in runs the real `collivind hook ...` commands.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

from collivind.cli.commands import hook as hook_mod

NODE = shutil.which("node")

HARNESS = r"""
import { execFileSync } from "node:child_process";
const mod = await import(process.argv[2]);
const prompts = [];
const $ = (strings, ...values) => {
  const argv = values.flat();
  const run = () => {
    try { return execFileSync(argv[0], argv.slice(1), { encoding: "utf8", env: process.env }); }
    catch (e) { return e.stdout || ""; }
  };
  const chain = { quiet: () => chain, nothrow: () => chain, text: async () => run() };
  return chain;
};
const client = { session: { prompt: async (req) => prompts.push(req) } };
const hooks = await mod.CollivindCapture({ client, $ });
for (let i = 0; i < Number(process.argv[3]); i++) {
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "oc1" } } });
}
const output = { context: [] };
await hooks["experimental.session.compacting"]({ sessionID: "oc2" }, output);
await hooks.event({ event: { type: "session.compacted", properties: { sessionID: "oc2" } } });
console.log(JSON.stringify({ prompts, context: output.context }));
"""


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_plugin_drives_real_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / ".config" / "opencode").mkdir(parents=True)
    # The plugin must call a collivind that sees the same temp HOME.
    shim = tmp_path / "collivind"
    shim.write_text(f'#!/bin/sh\nexec {sys.executable} -m collivind.cli.main "$@"\n')
    shim.chmod(0o755)
    monkeypatch.setattr(hook_mod, "collivind_bin", lambda: str(shim))

    assert hook_mod.install_hooks(tool="opencode") == ["Stop", "PreCompact"]
    plugin = hook_mod.get_opencode_plugin_path()
    harness = tmp_path / "harness.mjs"
    harness.write_text(HARNESS)
    env = {**os.environ, "HOME": str(tmp_path)}
    out = subprocess.run(
        [NODE, str(harness), plugin.as_uri(), "15"], capture_output=True, text=True, env=env, timeout=120, check=True
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])

    texts = [r["body"]["parts"][0]["text"] for r in result["prompts"]]
    ids = [r["path"]["id"] for r in result["prompts"]]
    assert ids == ["oc1", "oc2"]  # one interval checkpoint, one post-compaction request
    assert "checkpoint_id" in texts[0] and "just compacted" in texts[1]
    assert all(r["body"]["parts"][0]["synthetic"] for r in result["prompts"])
    assert result["context"] and "summary" in result["context"][0]


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_reply_to_a_save_prompt_does_not_count(tmp_path, monkeypatch):
    """The synthetic save prompt makes opencode reply and go idle again; that
    idle is our own continuation (Claude Code's stop_hook_active) and must not
    advance the interval or the re-ask counter."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / ".config" / "opencode").mkdir(parents=True)
    shim = tmp_path / "collivind"
    shim.write_text(f'#!/bin/sh\nexec {sys.executable} -m collivind.cli.main "$@"\n')
    shim.chmod(0o755)
    monkeypatch.setattr(hook_mod, "collivind_bin", lambda: str(shim))
    hook_mod.install_hooks(tool="opencode")
    harness = tmp_path / "harness.mjs"
    harness.write_text(HARNESS)
    env = {**os.environ, "HOME": str(tmp_path)}
    # 15 idles reach the default interval and send the save prompt; the 16th
    # is opencode going idle after answering it.
    subprocess.run(
        [NODE, str(harness), hook_mod.get_opencode_plugin_path().as_uri(), "16"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=True,
    )

    session = json.loads((tmp_path / ".collivind" / "hook_state.json").read_text())["sessions"]["oc1"]
    assert session["pending"]["since"] == 0
