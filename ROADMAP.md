# Collivind Roadmap

What's left after the 2026-07 iteration (direct CLI verbs, export/import,
SessionStart recall index, Codex support, token-budgeted context, `user_id`
scoping, git-based `collivind sync`). Ordered by ambition — each phase is a
planning pass of its own before code.

## Positioning (why these bets)

Market research (July 2026) found three gaps nobody fills:

1. **Team memory with permissions and provenance** — Cognee is the only player
   with real ACLs (dataset-level, buggy); everyone else is single-user or
   admin-pushed prose. `anthropics/claude-code#38536` ("Shared Team Memory")
   is the loudest open demand signal.
2. **Knowledge exchange between orgs/users** — portability formats exist
   (OKF, MIF, Letta .af), registries don't. First credible registry wins.
3. **Git-like CLI ergonomics** — funded platforms are SDK/dashboard-first;
   loved CLI tools are tiny indies. Nobody does both.

Collivind's standing advantages: graph memory in OSS (mem0 removed theirs),
zero-LLM-cost extraction (delegated to the host agent), local-first embedded
mode. Keep those free; never paywall export, audit, or sync — the incumbents'
paywalls there are our adoption wedge.

Platform-native memory (Claude Code auto-memory, Codex cloud memory, Cursor
Memories) will keep commoditizing single-user capture. Do not compete there.

## Phase 2 — Team memory (the wedge)

The git-based `collivind sync` is v1. The real thing:

- **Sync server** (`collivind remote add <url>` / `push` / `pull` / `merge`):
  local-first with a shared remote. Start with per-user API keys; the
  E2E-encrypted variant (Atuin model: client-held key, ciphertext-only
  server) is the trust differentiator nobody offers.
- **Preserve ids across machines** — current sync dedups by content; proper
  sync needs stable ids + conflict resolution (vector clock or
  last-writer-wins with SUPERSEDES edges).
- **Per-memory visibility**: `private | team | org` field, enforced at the
  server, filterable everywhere.
- **Promote/review flow**: `collivind share <id> --to team` and
  `collivind review` (curation queue) — one bad write must not poison the
  team store. Curation beats raw capture (Byterover lesson).
- **Provenance surfacing**: who/which agent/which session wrote it,
  confidence, correction history. The graph already stores the seams
  (`user_id`, `session_id`, SUPERSEDES chains); expose them in search
  results and a `collivind blame <id>` view.
- **SessionNode persistence** (deliberately skipped so far — YAGNI until the
  server needs session-level provenance).
- **Multi-user MCP**: HTTP transport with OAuth per the 2025-11-25 MCP spec;
  stdio stays single-user.

## Pro licensing (blocker for monetization)

`collivind-pro` currently ships MIT with zero entitlement checks — "paid" only
by not being published. Decide and implement:

- License-key check (offline-verifiable signature; no phone-home in line with
  the local-first story), or keep pro closed-source on a private index.
- Open-core split rule of thumb: single-user everything + sync + export stays
  OSS; hosted team server, SSO, org admin, and analytics (importance,
  consolidation, lifecycle, health) are pro.

## Phase 3 — Knowledge packs ("npm for memory")

- Pack format: versioned JSONL bundle + manifest (name, semver, license,
  provenance attestation). Track OKF/MIF — adopt, don't invent, unless they
  stall.
- `collivind pack publish acme/postgres-lessons` / `collivind install
  acme/postgres-lessons` with subscribe-for-updates.
- Graph+vector-native import is the edge over markdown-only registries.
- Needs Phase 2's permission layer first for private/team packs.

## Deferred smaller items

- **mypy burn-down**: the CI type-check step is `continue-on-error` — 57
  pre-existing errors in 14 files (mostly Optional handling). Fix per-module,
  then make the step blocking.

- **Codex extraction hooks**: only SessionStart is registered for Codex —
  verify Codex's Stop-hook `decision: block` semantics, then enable
  extraction there too.
- **PreCompact limitation**: Claude Code does not inject PreCompact stdout
  into the conversation; the current prompt likely never reaches the model.
  Options: switch to transcript-export capture (claude-mem approach) or
  verify current behavior against latest hooks docs.
- **Hook PATH assumption**: registered hook commands call bare `collivind`;
  pipx/venv installs may not expose it to the hook's shell. Consider
  absolute-path registration.
- **Graph expansion is N+1** (`find_related_memories` per entity per seed):
  batch it before the 10k-memory performance target is tested.
- **Perf/scale test suite**: the design spec's 500ms @ 10k memories target
  has no test. Add a seeded benchmark.
- **Import fidelity**: `collivind import` drops entity links (memory nodes
  only) and rewrites ids. Fine for v1 backup; fix alongside sync-server id
  stability.
- **`get_entity` / relationships export**: entities and edges deserve
  export/import too once ids are stable.
- **Benchmark honesty**: when publishing numbers (LoCoMo/LongMemEval), report
  the full picture — the mempalace launch collapse is the cautionary tale.
