"""#35/#34: the opencode plugin maps opencode events onto collivind hooks.

opencode itself is not needed: the plugin is loaded in Node with stand-ins for
the two things opencode hands it (the Bun `$` shell and the SDK client), and
the plugin runs the real `collivind hook ...` commands. For recall, the store
behind those commands is a fake (FAKE_CLI) so no model or Docker is needed.
"""

import json
import os
import re
import shutil
import subprocess
import sys

import pytest

from collivind.cli.commands import hook as hook_mod

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")

STANDINS = r"""
import { execFile } from "node:child_process";
const mod = await import(process.argv[2]);
const prompts = [];
const history = {};  // sessionID -> messages client.session.messages returns
const $ = (strings, ...values) => {
  const argv = values.flat();
  const run = () => new Promise((resolve) =>
    execFile(argv[0], argv.slice(1), { encoding: "utf8", env: process.env }, (e, stdout) => resolve(stdout || ""))
      .stdin.end());  // Bun's $ gives no stdin; the hooks read it to EOF
  const chain = { quiet: () => chain, nothrow: () => chain, text: run };
  return chain;
};
const client = { session: {
  prompt: async (req) => prompts.push(req),
  messages: async ({ path }) => ({ data: history[path.id] ?? [] }),
} };
const hooks = await mod.CollivindCapture({ client, $ });
"""

HARNESS = (
    STANDINS
    + r"""
for (let i = 0; i < Number(process.argv[3]); i++) {
  await hooks.event({ event: { type: "session.idle", properties: { sessionID: "oc1" } } });
}
const output = { context: [] };
await hooks["experimental.session.compacting"]({ sessionID: "oc2" }, output);
await hooks.event({ event: { type: "session.compacted", properties: { sessionID: "oc2" } } });
console.log(JSON.stringify({ prompts, context: output.context }));
"""
)

# Each step sends one user message through chat.message and records the
# parts added to it, or runs an event.
RECALL_HARNESS = (
    STANDINS
    + r"""
// The id opencode gives the prompt's own text part, generated just now.
const ownID = () => "prt_" + (BigInt(Date.now()) * 0x1000n + 1n).toString(16).padStart(12, "0") + "zzzzzzzzzzzzzz";
const send = async (sessionID, text, synthetic = false) => {
  const own = { id: ownID(), type: "text", text, synthetic };
  const output = { message: { id: "msg_1", sessionID, role: "user" }, parts: [own] };
  const started = Date.now();
  await hooks["chat.message"]({ sessionID }, output);
  return { ms: Date.now() - started, own: own.id, added: output.parts.slice(1) };
};
const event = (type, properties) => hooks.event({ event: { type, properties } });
const steps = [];
const long = "how is the opencode plugin deployed";
steps.push(await send("s1", long));
steps.push(await send("s1", "and how does the recall timeout work"));
steps.push(await send("s1", '<collivind_extraction checkpoint="ck1">save now</collivind_extraction>', true));
steps.push(await send("s1", "ok"));
await event("session.compacted", { sessionID: "s1" });
steps.push(await send("s1", '<collivind_extraction checkpoint="ck2">save now</collivind_extraction>', true));
steps.push(await send("s1", long));
history.s2 = [{ info: { role: "user" }, parts: [] }];  // opencode restarted mid-session
steps.push(await send("s2", long));
await event("session.deleted", { info: { id: "s1" } });
steps.push(await send("s1", long));
console.log(JSON.stringify({ steps, prompts }));
"""
)

FAKE_CLI = r"""
import os, sys, time, types
from collivind.cli.commands import hook

class Manager:
    def get_timeline(self, project, limit=8):
        if os.environ.get("FAKE_EMPTY"):
            return []
        return [types.SimpleNamespace(category="fact", summary="opencode recall runs in chat.message", content="")]

    def get_context(self, query, **kwargs):
        time.sleep(float(os.environ.get("FAKE_DELAY", "0")))
        return "[FACT] recalled for: " + query[:60]

hook._load_manager = lambda: Manager()
from collivind.cli.main import cli
cli()
"""

PART_ID = re.compile(r"^prt_[0-9a-f]{12}[0-9A-Za-z]{14}$")


def _install(tmp_path, monkeypatch, fake=False):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / ".config" / "opencode").mkdir(parents=True)
    # The plugin must call a collivind that sees the same temp HOME.
    shim = tmp_path / "collivind"
    if fake:
        (tmp_path / "fake_cli.py").write_text(FAKE_CLI)
        shim.write_text(f'#!/bin/sh\nexec {sys.executable} {tmp_path / "fake_cli.py"} "$@"\n')
    else:
        shim.write_text(f'#!/bin/sh\nexec {sys.executable} -m collivind.cli.main "$@"\n')
    shim.chmod(0o755)
    monkeypatch.setattr(hook_mod, "collivind_bin", lambda: str(shim))
    return hook_mod.install_hooks(tool="opencode")


def _run(tmp_path, script, *args, **env):
    harness = tmp_path / "harness.mjs"
    harness.write_text(script)
    plugin = hook_mod.get_opencode_plugin_path().as_uri()
    out = subprocess.run(
        [NODE, str(harness), plugin, *args],
        capture_output=True,
        text=True,
        env={**os.environ, "HOME": str(tmp_path), **env},
        timeout=120,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_plugin_drives_real_hooks(tmp_path, monkeypatch):
    assert _install(tmp_path, monkeypatch) == ["Stop", "PreCompact", "SessionStart", "UserPromptSubmit"]
    result = _run(tmp_path, HARNESS, "15")

    texts = [r["body"]["parts"][0]["text"] for r in result["prompts"]]
    ids = [r["path"]["id"] for r in result["prompts"]]
    assert ids == ["oc1", "oc2"]  # one interval checkpoint, one post-compaction request
    assert "checkpoint_id" in texts[0] and "just compacted" in texts[1]
    assert all(r["body"]["parts"][0]["synthetic"] for r in result["prompts"])
    assert result["context"] and "summary" in result["context"][0]


def test_reply_to_a_save_prompt_does_not_count(tmp_path, monkeypatch):
    """The synthetic save prompt makes opencode reply and go idle again; that
    idle is our own continuation (Claude Code's stop_hook_active) and must not
    advance the interval or the re-ask counter."""
    _install(tmp_path, monkeypatch)
    # 15 idles reach the default interval and send the save prompt; the 16th
    # is opencode going idle after answering it.
    _run(tmp_path, HARNESS, "16")

    session = json.loads((tmp_path / ".collivind" / "hook_state.json").read_text())["sessions"]["oc1"]
    assert session["pending"]["since"] == 0


def test_chat_message_injects_index_and_recall(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, fake=True)
    steps = _run(tmp_path, RECALL_HARNESS)["steps"]
    first, second, own_save, short, own_save2, after_compact, restarted, after_delete = steps

    # First message of a session: index and recall, in one synthetic part.
    assert len(first["added"]) == 1
    part = first["added"][0]
    assert part["synthetic"] and part["type"] == "text"
    assert part["sessionID"] == "s1" and part["messageID"] == "msg_1"
    assert "<collivind_memory_index>" in part["text"] and "<collivind_recall>" in part["text"]
    assert "recalled for: how is the opencode plugin deployed" in part["text"]

    # Later messages: recall only, once.
    assert len(second["added"]) == 1
    assert "<collivind_recall>" in second["added"][0]["text"]
    assert "<collivind_memory_index>" not in second["added"][0]["text"]

    # Never into our own save prompts; short prompts skip recall.
    assert own_save["added"] == [] and own_save2["added"] == [] and short["added"] == []

    # After a compaction the index comes back on the next real message.
    assert "<collivind_memory_index>" in after_compact["added"][0]["text"]
    # A session with earlier user messages (opencode restarted) gets no index.
    assert "<collivind_memory_index>" not in restarted["added"][0]["text"]
    # session.deleted forgets the session.
    assert "<collivind_memory_index>" in after_delete["added"][0]["text"]

    # Ids look like opencode's and sort after the prompt's own part.
    for step in steps:
        for p in step["added"]:
            assert PART_ID.match(p["id"]) and p["id"] > step["own"]

    beats = json.loads((tmp_path / ".collivind" / "hook_state.json").read_text())["heartbeats"]
    assert {"opencode:SessionStart", "opencode:UserPromptSubmit"} <= set(beats)


def test_slow_recall_fails_open(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, fake=True)
    steps = _run(tmp_path, RECALL_HARNESS, FAKE_DELAY="5", COLLIVIND_RECALL_TIMEOUT_MS="1500")["steps"]

    # The index (fast) still lands; the stalled recall is dropped at the deadline.
    assert "<collivind_memory_index>" in steps[0]["added"][0]["text"]
    assert "<collivind_recall>" not in steps[0]["added"][0]["text"]
    assert steps[1]["added"] == []
    assert all(s["ms"] < 3000 for s in steps)


def test_empty_store_adds_nothing(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, fake=True)
    steps = _run(tmp_path, RECALL_HARNESS, FAKE_EMPTY="1")["steps"]
    assert all(s["added"] == [] for s in steps)


def test_disabled_recall_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / ".config" / "opencode").mkdir(parents=True)
    events = hook_mod.install_hooks(tool="opencode", enable_session_start=False, enable_user_prompt=False)
    assert events == ["Stop", "PreCompact"]
    health = next(h for h in hook_mod.hook_health() if h["tool"] == "opencode")
    assert health["events"] == ["Stop", "PreCompact"]
