// Collivind automatic capture and recall for opencode — installed by `collivind hook install`.
// Regenerated on every install; edits here are overwritten.
//
// opencode has no command hooks, so this plugin maps its events onto the same
// `collivind hook ...` commands Claude Code and Codex run (#35):
//   session.idle                      -> hook stop         (periodic checkpoint)
//   experimental.session.compacting   -> hook precompact   (mark a checkpoint due)
//   session.compacted                 -> hook compacted    (ask for the save)
//   chat.message, first in a session  -> hook session-start (memory index)
//   chat.message                      -> hook user-prompt  (recall for the prompt)
// A save request is sent as a synthetic prompt; it counts as saved only when
// collivind_batch_add confirms its checkpoint id. The reply to that prompt
// ends in another session.idle; it is our own continuation (Claude Code's
// stop_hook_active), so it is passed as --continuation and never counted.
// Recall runs before the user's message is sent, so it has a hard deadline and
// fails open: a slow or broken store never holds up the message.
// Every failure is swallowed: memory must never break the editor.
import { spawn } from "node:child_process";

const BIN = __COLLIVIND_BIN__;
const ENABLED = __COLLIVIND_ENABLED__;
const RECALL_TIMEOUT_MS = Number(process.env.COLLIVIND_RECALL_TIMEOUT_MS) || 5000;

// Same shape as opencode's ascending ids (src/id/id.ts): "prt_" + 12 hex of
// (ms * 0x1000 + counter) + 14 base62. Parts are reloaded ordered by id, so
// ours must sort after the prompt's own parts. Not exported: opencode calls
// every function this module exports as a plugin.
const BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
let lastMs = 0;
let counter = 0;
const partID =(after = []) => {
  const ms = Date.now();
  if (ms !== lastMs) [lastMs, counter] = [ms, 0];
  counter++;
  let n = BigInt(ms) * 0x1000n + BigInt(counter);
  for (const id of after) {
    if (typeof id !== "string" || !id.startsWith("prt_")) continue;
    const prev = BigInt("0x" + (id.slice(4, 16).match(/^[0-9a-f]{12}$/)?.[0] ?? "0"));
    if (prev >= n) n = prev + 1n;
  }
  let rand = "";
  for (let i = 0; i < 14; i++) rand += BASE62[Math.floor(Math.random() * 62)];
  return "prt_" + n.toString(16).padStart(12, "0").slice(-12) + rand;
};

// Run a collivind hook with stdin; resolve its JSON, or null on any failure
// or once the deadline passes (the process is killed then).
const call = (args, input, deadline) =>
  new Promise((resolve) => {
    let out = "";
    let child;
    const timer = setTimeout(() => {
      try {
        child?.kill();
      } catch {}
      resolve(null);
    }, Math.max(0, deadline - Date.now()));
    const done = (value) => {
      clearTimeout(timer);
      resolve(value);
    };
    try {
      child = spawn(BIN, args, { stdio: ["pipe", "pipe", "ignore"] });
      child.on("error", () => done(null));
      child.stdout.on("data", (d) => (out += d));
      child.on("close", () => {
        try {
          done(out.trim() ? JSON.parse(out) : null);
        } catch {
          done(null);
        }
      });
      child.stdin.on("error", () => {});
      child.stdin.end(input);
    } catch {
      done(null);
    }
  });

const before = (promise, deadline) =>
  Promise.race([promise, new Promise((r) => setTimeout(() => r(null), Math.max(0, deadline - Date.now())))]);

export const CollivindCapture = async ({ client, $ }) => {
  const run = async (args) => {
    try {
      const out = await $`${BIN} ${args}`.quiet().nothrow().text();
      return out.trim() ? JSON.parse(out) : null;
    } catch {
      return null;
    }
  };
  const replying = new Set();
  const ask = async (sessionID, text) => {
    // Marked before sending: the reply's idle can fire before prompt() resolves.
    replying.add(sessionID);
    try {
      await client.session.prompt({
        path: { id: sessionID },
        body: { parts: [{ type: "text", text, synthetic: true }] },
      });
    } catch {
      replying.delete(sessionID);
    }
  };

  // Sessions that already got the memory index, and those due one again
  // because a compaction dropped it from the context.
  const indexed = new Set();
  const reindex = new Set();
  const wantsIndex = async (sessionID, deadline) => {
    if (reindex.delete(sessionID)) return indexed.add(sessionID);
    if (indexed.has(sessionID)) return false;
    indexed.add(sessionID);
    // Not seen since this process started: first only if the session has no
    // earlier user message (opencode restarted mid-session otherwise). The
    // current message is not saved yet when chat.message runs.
    try {
      const res = await before(client.session.messages({ path: { id: sessionID } }), deadline);
      const messages = res?.data ?? res;
      return Array.isArray(messages) && !messages.some((m) => (m.info ?? m)?.role === "user");
    } catch {
      return false;
    }
  };

  const hooks = {
    event: async ({ event }) => {
      const props = event?.properties ?? {};
      const id = props.sessionID ?? (event?.type === "session.deleted" ? props.info?.id : undefined);
      if (!id) return;
      if (event.type === "session.idle" && ENABLED.stop) {
        const args = ["hook", "stop", "--agent", "opencode", "--session", id];
        if (replying.delete(id)) args.push("--continuation");
        const out = await run(args);
        if (out?.decision === "block") await ask(id, out.reason);
      } else if (event.type === "session.compacted") {
        if (indexed.has(id)) reindex.add(id);
        if (!ENABLED.precompact) return;
        const out = await run(["hook", "compacted", "--agent", "opencode", "--session", id]);
        if (out?.prompt) await ask(id, out.prompt);
      } else if (event.type === "session.deleted") {
        indexed.delete(id);
        reindex.delete(id);
        replying.delete(id);
      }
    },
    "experimental.session.compacting": async (input, output) => {
      if (!ENABLED.precompact) return;
      await run(["hook", "precompact", "--agent", "opencode", "--session", input.sessionID]);
      output.context.push(
        "Keep every durable fact, decision, correction and open follow-up from this session in the summary: " +
          "Collivind asks for them to be saved right after compaction.",
      );
    },
    "chat.message": async (input, output) => {
      try {
        const sessionID = input?.sessionID;
        const parts = output?.parts;
        if (!sessionID || !Array.isArray(parts)) return;
        const prompt = parts
          .filter((p) => p?.type === "text" && typeof p.text === "string")
          .map((p) => p.text)
          .join("\n");
        // Our own save requests, and anything already carrying collivind context.
        if (prompt.includes("<collivind_")) return;
        const deadline = Date.now() + RECALL_TIMEOUT_MS;
        const jobs = [];
        if (ENABLED.sessionStart && (await wantsIndex(sessionID, deadline))) {
          jobs.push(call(["hook", "session-start", "--agent", "opencode"], "{}", deadline));
        }
        if (ENABLED.userPrompt) {
          const payload = JSON.stringify({ prompt, session_id: sessionID });
          jobs.push(call(["hook", "user-prompt", "--agent", "opencode"], payload, deadline));
        }
        const texts = (await Promise.all(jobs))
          .map((out) => out?.hookSpecificOutput?.additionalContext)
          .filter((t) => typeof t === "string" && t.trim());
        if (!texts.length) return;
        parts.push({
          id: partID(parts.map((p) => p?.id)),
          sessionID,
          messageID: output.message?.id,
          type: "text",
          text: [...new Set(texts)].join("\n"),
          synthetic: true,
        });
      } catch {}
    },
  };
  return hooks;
};
