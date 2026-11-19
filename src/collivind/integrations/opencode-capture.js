// Collivind automatic capture for opencode — installed by `collivind hook install`.
// Regenerated on every install; edits here are overwritten.
//
// opencode has no command hooks, so this plugin maps its events onto the same
// `collivind hook ...` commands Claude Code and Codex run (#35):
//   session.idle                      -> hook stop       (periodic checkpoint)
//   experimental.session.compacting   -> hook precompact (mark a checkpoint due)
//   session.compacted                 -> hook compacted  (ask for the save)
// A save request is sent as a synthetic prompt; it counts as saved only when
// collivind_batch_add confirms its checkpoint id. The reply to that prompt
// ends in another session.idle; it is our own continuation (Claude Code's
// stop_hook_active), so it is passed as --continuation and never counted.
// Every failure is swallowed: memory capture must never break the editor.
const BIN = __COLLIVIND_BIN__;

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

  return {
    event: async ({ event }) => {
      const id = event?.properties?.sessionID;
      if (!id) return;
      if (event.type === "session.idle") {
        const args = ["hook", "stop", "--agent", "opencode", "--session", id];
        if (replying.delete(id)) args.push("--continuation");
        const out = await run(args);
        if (out?.decision === "block") await ask(id, out.reason);
      } else if (event.type === "session.compacted") {
        const out = await run(["hook", "compacted", "--agent", "opencode", "--session", id]);
        if (out?.prompt) await ask(id, out.prompt);
      }
    },
    "experimental.session.compacting": async (input, output) => {
      await run(["hook", "precompact", "--agent", "opencode", "--session", input.sessionID]);
      output.context.push(
        "Keep every durable fact, decision, correction and open follow-up from this session in the summary: " +
          "Collivind asks for them to be saved right after compaction.",
      );
    },
  };
};
