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

- **Local daemon / loopback HTTP bridge**: the latency half shipped (#16).
  Embedded mode now keeps the model resident in `collivind.storage.embed_daemon`
  (unix socket `~/.collivind/embed.sock`, 0600, idle exit after 30 min; a
  data dir too long for AF_UNIX falls back to a private per-user dir under
  /tmp),
  started on demand by the first embed; a warm hook embeds in ~0.01s instead
  of a 6-9s model load. Still open: it serves embeddings only — the store is
  still opened per process (cheap, ~0.1s), and the first prompt after the
  daemon exits pays one in-process load. A full loopback API is not needed
  unless that changes.
- **Pre-compaction capture**: since 0.11 Claude Code and Codex both get
  Stop (periodic checkpoints) and PreCompact + SessionStart(`compact`)
  capture, but no host lets PreCompact speak to the model, so the save runs
  *after* compaction from the summary. Anything the summary drops is lost;
  transcript-export capture (claude-mem approach) would close that.
- **Graph expansion batching shipped**: `GraphStore.get_neighbors_batch` /
  `find_related_memories_batch` (per-item defaults on the ABC, one query each
  in SQLite and Neo4j) replace the per-seed, per-entity calls; Neo4j id
  lookups now go through the `Memory|Entity|Session` id indexes instead of an
  AllNodesScan. Still one `get_memory` per vector hit (`limit * 3`); batch
  that too if a remote graph makes those round trips show.
- **Perf/scale test suite shipped**: `benchmarks/search_bench.py` seeds 10k
  memories (Zipf-skewed entities, token-hash embedder, no model). Search
  p50/p95 at 10k: embedded 117/148 ms → 40/48 ms; Neo4j+Qdrant 1709/2071 ms
  → 59/69 ms (M-series laptop, 2026-11-19). `tests/unit/test_search_perf.py`
  pins equivalence and the graph statement count per search. Not covered:
  real embedding-model latency, 100k+ memories, concurrent searches.
- **Import fidelity** — done: `collivind export` writes a v2 bundle (header +
  memories, entities, edges, vector ids/payloads, vectors optional) through
  `GraphStore.iter_*` / `VectorStore.iter_all`, and `import` restores it by id
  via the same `restore` path as `collivind migrate`, adding only what the
  target lacks. v1 files still import through the pipeline. `collivind sync`
  still uses v1 (content-level merge, new ids); id-stable sync belongs to the
  sync server.
- **`get_entity` / relationships export** — done: entities and edges are in the
  v2 bundle with every field.
- **Benchmark honesty**: when publishing numbers (LoCoMo/LongMemEval), report
  the full picture — the mempalace launch collapse is the cautionary tale.
