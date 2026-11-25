# Collivind

<!-- TODO: link the website once GitHub Pages is enabled (see .github/workflows/pages.yml):
     Website: https://maiiasleptcova.github.io/collivind/ -->

Graph-based memory layer for AI coding assistants. Your AI assistant forgets everything between sessions — Collivind gives it persistent, structured memory.

Collivind stores facts, decisions, patterns, errors, and architecture choices from your coding sessions in a graph database with vector search. It runs entirely local with no LLM API keys required.

## Quick Start

### Docker mode (recommended)

```bash
pip install collivind-memory
collivind init
```

This starts Qdrant (vector search), Neo4j (graph), and a sentence-transformers embedding service in Docker containers.

If Qdrant and Neo4j are **already running** — inside the container collivind
itself runs in, or managed by you — `init` detects them and skips container
setup entirely. Point `~/.collivind/config.toml` at their endpoints (they
default to `localhost:6333` and `bolt://localhost:7687`); no Docker daemon is
needed in that case.

### Embedded mode (no Docker)

For devcontainers, Codespaces, CI/CD, or anywhere Docker isn't available:

```bash
pip install "collivind-memory[embedded]"
```

Set mode in `~/.collivind/config.toml`:

```toml
[collivind]
mode = "embedded"
```

Then initialize:

```bash
collivind init
```

This uses SQLite for both vectors and the graph, and loads the embedding model directly — zero Docker dependency.

> **Concurrent sessions:** embedded mode is multi-process safe — any number of agent sessions, CLI commands, and hooks can share the store, and a write from one session is immediately visible to the others.
>
> **Resident model:** hooks are short-lived processes, so the first embed starts a small background process that keeps the model loaded (`~/.collivind/embed.sock`, owner-only; if the data dir path is too long for a unix socket it uses a private per-user dir under `/tmp`, which `collivind embed-daemon status` names); later hooks embed in milliseconds instead of reloading the model every prompt. It exits after 30 idle minutes (`daemon_idle_timeout` under `[embeddings]`). Manage it with `collivind embed-daemon start|stop|status`; turn it off with `daemon = false` or `COLLIVIND_EMBED_DAEMON=0`. The same process also holds the search reranker (a second, ~90 MB model downloaded on its first start); while it is loading, searches use the first-stage ranking. With the daemon off, the reranker loads in-process on the first search. `rerank = false` under `[search]` stops it being loaded at all.
>
> **Upgrading from ≤ 0.1.x:** older versions kept vectors in a single-process Qdrant store (`~/.collivind/qdrant_data`). It is migrated automatically on first use and left in place as a backup; restart any agent sessions still running the old version once after upgrading. If you later downgrade, memories written after the upgrade live only in the new store — carry them over with `collivind export` / `import`.

> **Docker mode reranker:** the embeddings container serves the reranker too (`POST /rerank`). An embeddings image built before 0.19 has no such endpoint, so search keeps the first-stage ranking until `collivind docker up` rebuilds it (the container's memory limit rises from 512 MB to 768 MB for the second model).

### Switching modes

Each mode has its own store, so changing `mode` shows an empty graph until you copy your data across:

```bash
collivind docker up                                       # target must be up and healthy
collivind migrate --from embedded --to docker --dry-run   # counts only, writes no records
collivind migrate --from embedded --to docker
# then set mode = "docker" in ~/.collivind/config.toml
```

`migrate` copies every memory in every project — superseded versions included — plus entities, memory↔entity and memory↔memory edges (supersedes, contradicts, …) and vectors, **with ids preserved**, so entity links and version chains survive. It reads both sides from your one `config.toml` (per-backend `provider` overrides are ignored so `--from`/`--to` pick each mode's own stores). The source is never modified, re-running upserts by id instead of duplicating, and it ends by printing source/target counts per kind and exiting 1 if any source record is missing from the target. Stored vectors are copied as-is when both sides use the same embedding model and dimension (embedded's in-process model and the docker embeddings service are both `BAAI/bge-small-en-v1.5`); otherwise every vector is re-embedded with the target's provider and it says so. Edges pointing at a node that no longer exists are listed and skipped. Modes are `embedded` and `docker`; `remote` is not offered because it shares the `[qdrant]`/`[neo4j]` sections with docker. `--dry-run` writes no records, though opening the target may create its empty schema or Qdrant collection.

### npm (alternative)

```bash
npx collivind-memory init
```

## Claude Code Integration

### As a Claude Code plugin (recommended)

```
/plugin marketplace add https://github.com/maiiasleptcova/collivind
/plugin install collivind@collivind
/reload-plugins
```

Use the full HTTPS URL. The `owner/repo` shorthand clones over SSH by
default, so `maiiasleptcova/collivind` fails unless you have a GitHub SSH key
loaded (or set `CLAUDE_CODE_PLUGIN_PREFER_HTTPS=1`).

`install` takes `plugin@marketplace` — not a URL. Both names here are
`collivind`, hence `collivind@collivind`. Check it landed with `/plugin`.

The plugin registers the MCP server, all four hooks (SessionStart,
UserPromptSubmit, Stop, PreCompact) and the `/mem-save` and `/mem-recall`
commands in one step, and keeps them in sync with the plugin version. Toggle
it from `/plugin` without editing `~/.claude/settings.json`.

The plugin drives the `collivind` executable, so the package still needs to be
installed (`uv tool install /path/to/collivind` from a checkout). It resolves
that binary itself rather than relying on PATH, since Claude Code spawns hooks
with its own environment.

### Web UI: the memory explorer

```
collivind web            # on the host, any mode
```

Opens a local explorer on `127.0.0.1:8765` over the same store your hooks and
MCP tools use (it reads `~/.collivind/config.toml` and names the mode in the
header). Three views, keyboard reachable (`/` search, `g` `l` `p` to switch,
`v` `e` `s` for the graph's Overview, Everything and Summaries modes, `Esc` to deselect):

- **Graph**: memories and the entities they mention, with entity links,
  `RELATES_TO`/`CAUSED_BY`/`DEPENDS_ON`, `CONTRADICTS` (dashed) and version
  chains (`SUPERSEDES`, dotted; superseded versions drawn hollow). Pan, zoom,
  drag; hover lights up a node's neighbours; click opens it in the side panel.
  Filter by project, category, entity type and date, hide superseded versions,
  or switch to a local graph of 1 to 3 hops around the selection. Canvas
  rendering with a built-in force layout stays interactive at a few thousand
  nodes; past 3,000 memories the newest are loaded and the sidebar says so
  (pick a project to see the rest).
- **Summaries** (graph mode, `s`): only summary memories, drawn as hexagons
  with their member count. Clicking one (or picking it in the sidebar list,
  which is keyboard reachable) opens its text in the side panel, lists its
  detail memories, and draws them around it; click it again to fold them. A
  detail memory's panel shows the summary it belongs to and links back.
- **Search**: typing filters node labels live; Enter runs the hybrid semantic
  search and lights the results up on the graph (and lists them in the List view).
- **Edit and link**: the side panel edits content, summary, category, tags and
  confidence (same id, re-embedded like `collivind update`), forgets with a
  confirmation, walks the version chain, and adds or removes links between two
  memories, a memory and an entity, or two entities. Shift-drag from one node
  to another to start a link. Version edges are history and cannot be unlinked.
- **Projects**: every project with its memory counts, plus a display name,
  description, graph colour, repository URL or path, and tags. These live in the
  store itself (one `project` entity per project, id `project:<name>`), so
  `export`, `import` and `migrate` carry them.

Light and dark themes follow the system, or pick one from the header.

**Security.** There is no login: anything that can reach the port can read and
edit every memory, so it binds loopback only. Requests whose `Host` is not
`localhost`/`127.0.0.1`/`[::1]` are refused (DNS rebinding), and every write
must be JSON with an `Origin` equal to the server's own (cross-site pages and
other local ports cannot write). A strict Content-Security-Policy forbids
inline and third-party scripts.

**In Docker.** `collivind docker up` also starts a `web` container next to
Qdrant, Neo4j and the embedder, built from the collivind package installed on
the host (no PyPI needed) and published on `127.0.0.1:8765` only. It mounts
your `config.toml` read-only and reaches the other services by name. Stop a
host-side `collivind web` first, or give it `--port`, since both default to 8765.

**Embedded mode in a container** is refused on macOS. The embedded store is
SQLite in WAL mode, whose locks only work when every process shares one kernel.
Docker Desktop and Colima share your data dir with the Linux VM over virtiofs
or FUSE, and a test with the host CLI and a container writing the same database
over Colima's virtiofs lost half the writes and corrupted the file. The image
checks the mount type of the data dir and exits with an explanation. On a
Linux host, where a bind mount is the same kernel, it is supported:

```
cd ~/.collivind && docker build -f Dockerfile.web --build-arg EMBEDDED=1 -t collivind-web .
docker run --rm -p 127.0.0.1:8765:8765 --user "$(id -u):$(id -g)" \
  -e COLLIVIND_MODE=embedded -v ~/.collivind:/home/collivind/.collivind collivind-web
```

(`collivind docker up` writes `Dockerfile.web` and the package copy into the
data dir first.)

### Agent support

| Agent | Memory tools (MCP) | Automatic recall | Periodic capture | Compaction capture |
|---|---|---|---|---|
| Claude Code | yes | yes — SessionStart + UserPromptSubmit | yes — Stop hook | yes — PreCompact marks it, SessionStart (`compact`) asks |
| Codex | yes | yes — after you trust the hooks | yes — Stop hook, after you trust it | yes — PreCompact marks it, SessionStart (`compact`) asks |
| opencode | yes | yes — plugin on `chat.message` (index on a session's first message, recall on every prompt) | yes — plugin on `session.idle` | yes — plugin on `session.compacted` |
| GitHub Copilot (VS Code agent mode, CLI, app; hooks in preview in VS Code) | yes | yes — sessionStart index + userPromptTransformed | yes — agentStop hook | yes — preCompact marks it, the next agentStop asks |

`collivind init` registers the MCP server for every agent it finds installed,
plus the hooks (Claude Code, Codex, GitHub Copilot) or the capture plugin (opencode,
`~/.config/opencode/plugins/collivind-capture.js`). See *GitHub Copilot* below
for what its preview hooks can and cannot do.

In opencode, recall runs before the message is sent, so it has a 5 s deadline
(`COLLIVIND_RECALL_TIMEOUT_MS`) and the message goes without memory context
when it is missed. The plugin uses opencode's V1 plugin API (opencode 1.x);
opencode's V2 session hooks are not in a published release yet.

**What "capture" means, precisely.** A hook cannot write memories itself —
extraction needs the agent's model. So a checkpoint *asks* the agent to call
`collivind_batch_add` with a checkpoint id, and only a successful call carrying
that id confirms it. Until then the checkpoint stays pending and is re-asked
with backoff (after 3, 6, then 12 more responses); it is never counted as saved
because a reminder was shown. If it is still unconfirmed after the third re-ask
and one more wait, it is recorded as failed and the next interval starts a new
checkpoint, so an agent without the collivind tool is not nagged indefinitely.
`collivind status` shows the last confirmed save, any pending checkpoint, and the
last failed one.

**Compaction.** Neither Claude Code nor Codex lets a PreCompact hook talk to
the model (Codex ignores its output), so collivind cannot make the save happen
*before* compaction. PreCompact marks a checkpoint due, and the SessionStart hook
that runs right after compaction (`source: "compact"`) asks the agent to save
from the compaction summary plus anything since. Knowledge the summary dropped is
not recoverable; periodic checkpoints keep that window small.

**Quiet capture** (default, `quiet_capture = true` in `[hooks]`) tells the agent
to save without announcing or confirming routine checkpoints, and to speak up only
on an error. It changes the agent's instructions, not the host UI: a Stop-hook
checkpoint is still a continuation turn, and the host may show the tool call.
Set `quiet_capture = false` for a one-line confirmation after each save.

#### Codex: trust the hooks after installing or upgrading

Codex skips hooks until you review and trust them — installing them is not
enough, and restarting Codex does not trust them. After `collivind init` or
`collivind hook install`:

- **Desktop app:** Settings → Hooks (or **Review hooks** in the composer), then trust the collivind entries.
- **CLI:** run `/hooks` and trust them.

**Or install the Codex plugin**, which lists the hooks under Settings → Hooks →
**From Plugins → Collivind** instead of anonymous User config entries:

```bash
codex plugin marketplace add maiiasleptcova/collivind
codex plugin add collivind@collivind
```

With the plugin enabled, `collivind init` / `hook install` stop writing
`~/.codex/hooks.json`; if both exist `collivind status` flags the double
registration. Plugin hooks still need the same trust review.

Codex records trust against a hash of each hook definition, so an upgrade that
changes the hook commands asks for review again (the commands carry no tunables
for exactly this reason: change the interval in `config.toml`, not the hook).
Collivind cannot read Codex's trust store, so `collivind status` reports each hook
as *configured but never ran — trust unknown* until it has actually executed, and
the last time it ran once it has.

#### GitHub Copilot (preview)

VS Code agent mode, the Copilot CLI and the Copilot app share one hook system,
which VS Code still marks as **preview**; an organization policy can
switch it off. When `~/.copilot` or `~/.vscode` exists, `collivind init` (or
`collivind hook install --tool copilot`) writes `~/.copilot/hooks/collivind.json`
(`$COPILOT_HOME/hooks/` if set) and registers the MCP server in
`~/.copilot/mcp-config.json`, which the CLI, the app and VS Code's Copilot harness
all read — checkpoints ask for `collivind_batch_add`, so the agent needs the tool,
not just the hooks. Not yet verified in a live Copilot run. The file is collivind's own, so
re-running rewrites it and leaves your other hook files alone.

- **sessionStart** injects the memory index (or the "memory unavailable" warning).
- **userPromptTransformed** prepends recalled context to the model-facing prompt.
  Copilot ignores the output of `userPromptSubmitted` config hooks, so this is the
  only per-prompt route; it keeps the prompt intact and, being stored in session
  history, the recalled block stays in the transcript.
- **agentStop** asks for periodic checkpoints (`decision: "block"`), and never
  chains a block onto its own continuation (`stop_hook_active`).
- **preCompact** cannot talk to the model and Copilot fires no sessionStart after
  compaction, so it marks the checkpoint due and the *next* agentStop asks for it
  right away instead of waiting for the interval.

`collivind hook install --tool copilot --workspace` writes
`.github/hooks/collivind.json` at the current git repo's root instead — the only location
the Copilot cloud agent reads. It calls a plain `collivind` (an absolute path
means nothing on another machine), so the cloud agent needs collivind installed
and a reachable store; locally, having both files fires every hook twice and
`collivind status` says so. VS Code's Copilot harness also runs hooks from the
repo's `.claude/settings.json` / `.claude/settings.local.json`; collivind entries
there fire under Copilot as well, and `status` flags them.

`collivind status` reports what is registered where — including Claude Code's MCP server, flagging a dead or duplicated entry — and, in docker mode, each container's state, so a crash-looping container is named rather than showing up only as "connection refused". It exits 1 when anything is broken, so it can run from cron or a script.

### Manual setup

Register Collivind as an MCP server:

```bash
claude mcp add -s user collivind -- /path/to/python -m collivind.mcp.server
```

Use an interpreter that can import collivind — agents spawn the server with
their own environment, where a bare `python3` usually resolves to the system
one and the server dies at import. `collivind init` does this for you and
fills in the right interpreter automatically.

`collivind init` registers this only when Claude Code has no collivind server
already — including the one the plugin provides — since `claude mcp add`
cannot update an entry in place and would leave you with two.

Already have an entry pointing at the wrong interpreter? Replace it yourself;
`claude mcp remove` finds it in whichever scope it lives in, which matters
because a `local`-scope entry outranks a `user`-scope one:

```bash
claude mcp remove collivind
collivind init
```

### Hooks

- **SessionStart** — injects a compact index (~100 tokens) of your project's stored knowledge at the start of every session; after a compaction it also asks for the compaction checkpoint
- **UserPromptSubmit** — searches memory with every prompt you type and injects the relevant context: up to `recall_limit` memories (`[hooks]`, default 20), packed into a `recall_max_tokens` budget (default 2,400; `--max-tokens` overrides it). On coding-length memories (~100–120 tokens) a relevant prompt typically injects about 12 memories, about 1,300 tokens. With the reranker on, memories it scores below `recall_min_score` (`[search]`, default 0.00002) are dropped, so a prompt unrelated to anything stored injects nothing at all.
- **Stop** — every `save_interval` responses (default 15, per session), asks the agent to checkpoint; once a session holds `summary_after` (default 8) memories no summary covers, the save's response asks the agent to write the session summary too
- **PreCompact** — marks a compaction checkpoint due (see *Compaction* above)

Registered for Claude Code (via the plugin, or `~/.claude/settings.json`), for Codex in `~/.codex/hooks.json` when `~/.codex` exists, and for GitHub Copilot in `~/.copilot/hooks/collivind.json` (camelCase events: sessionStart, userPromptTransformed, agentStop, preCompact). If any hook finds the store down it says so in the session instead of recalling nothing.

### In-session commands

`collivind init` also installs manual slash commands, so you can save and recall memory mid-session without leaving your agent:

- `/mem-save <text>` — store knowledge (or, with no text, extract it from the current conversation)
- `/mem-recall <topic>` — pull relevant stored knowledge into the session

Installed for Claude Code (`~/.claude/commands`), Codex (`~/.codex/prompts`) and opencode (`~/.config/opencode/commands`) automatically; for VS Code Copilot run `collivind commands install --tool copilot` inside a repo (writes `.github/prompts/`).

`collivind init` registers both hooks in `~/.claude/settings.json` automatically. To (re)register manually:

```bash
collivind hook install
```

Tune the extraction interval and toggles in the `[hooks]` section of `~/.collivind/config.toml`.

## MCP Tools

Once connected, Claude Code gets these tools:

| Tool | Description |
|------|-------------|
| `collivind_status` | Health check and memory stats |
| `collivind_add_memory` | Store a fact, decision, pattern, or error |
| `collivind_batch_add` | Store multiple memories at once |
| `collivind_search` | Semantic + graph hybrid search |
| `collivind_get_context` | Get relevant context as formatted text |
| `collivind_get_entity` | Find all memories about a named entity |
| `collivind_get_timeline` | Chronological view of project knowledge |
| `collivind_invalidate` | Mark a memory as outdated/superseded |
| `collivind_forget` | Delete a memory permanently |
| `collivind_find_contradictions` | List memories close enough to need checking against one (update, conflict or neighbour) |
| `collivind_mark_conflict` | Flag two memories as a genuine, unresolved conflict for later review |
| `collivind_get_version_chain` | Full supersession history of a memory |
| `collivind_summarize` | Read a session's (or topic's) member memories, then store the summary the agent writes for them |
| `collivind_extract` | Get an extraction prompt for raw text |
| `collivind_extract_save` | Parse and store an LLM extraction response |

## CLI Commands

```bash
collivind init              # Setup and start services
collivind status            # Health check: backends, containers, hooks, MCP registration (exit 1 if unhealthy)
collivind search "query"    # Search memories from terminal
collivind add "content" -c decision -t db,infra  # Store a memory
collivind get <id>          # Show one memory
collivind context "query"   # Formatted context block (pipe into prompts)
collivind update <id> --content "..."  # Update a memory (re-embeds)
collivind update <id> -c decision      # Reclassify (also re-embeds)
collivind update <id> -t ""            # Clear every tag
collivind invalidate <id> -r outdated  # Mark outdated, keep history
collivind forget <id>       # Delete permanently (with confirmation)
collivind export -o backup.jsonl  # Lossless backup: every project, ids kept
collivind import backup.jsonl  # Restore by id; only adds what is missing
collivind migrate --from embedded --to docker  # Copy everything between modes, ids kept
collivind sync ./shared-memory  # Two-way team sync via a git-tracked dir
collivind projects list     # Projects, counts, and which one this directory maps to
collivind projects rename default my-app --match "my-app|billing"  # Relabel (dry run; --apply writes)
collivind reset             # Wipe all data (with confirmation)
collivind docker up|down|logs  # Container management (docker mode), including the web UI
collivind web               # Local memory explorer on 127.0.0.1:8765
collivind hook install      # Register Claude Code hooks
collivind summarize --session <id>     # A session's memories and its current summary
collivind summarize --session <id> --text - < summary.txt  # Store the summary you wrote (one "- " fact per line)
```

Memory verbs take `--json` for scripting and agent use.

### Projects

Memories are filed under the project of the directory the agent works in, so an agent never has to pass `project_id`. The first of these wins: the `COLLIVIND_PROJECT` environment variable, `project = "name"` in a `.collivind.toml` at or above the directory (up to the repo root), the repo name of the `origin` remote, the main checkout's folder name (worktrees count as their repo), the directory's own name outside git, then `default`. Recall searches the current project first and falls back to `default`, where memories saved before 0.15 sit until you relabel them with `collivind projects rename`. It is a dry run unless you add `--apply`; every version of a memory moves together, ids, links and vectors are kept, and an interrupted run finishes when you run it again.

### Backup and restore

`collivind export` writes a versioned JSONL bundle: a header line (format version, collivind version, embedding model and dimension, per-kind counts), then every memory — superseded versions and their chain fields included — every entity, every memory↔entity and memory↔memory edge, and which memories are searchable, **all with their ids**. `-p/--project` limits it to one project's memories, the entities they link to and the edges between them (edges into other projects are left out and counted). Vectors are large and can be rebuilt, so they are only written with `--vectors`; without them, or when the bundle came from a different embedding model, `import` re-embeds with this install's model. The bundle holds no configuration or credentials, and export is never limited.

`collivind import` validates the whole file first — a truncated, damaged or newer-format bundle is rejected before anything is written — then restores by id, adding only records the target lacks, so re-importing is safe and never reverts edits made since. It prints bundle/target/missing counts per kind and exits 1 if anything is still missing; `--dry-run` writes nothing. Files from older versions (one memory per line) are detected automatically and re-added through the normal pipeline as before. `collivind sync` keeps using that older per-project format.

## Team Sharing

Share memory with your team through any git repo — no server required:

```bash
collivind sync ~/team-repo/memory   # import teammates' knowledge, write yours back
cd ~/team-repo && git add memory && git commit -m "memory sync" && git push
```

Each sync imports records you don't have yet (exact duplicates are skipped; updated versions of known facts supersede your stale copies) and rewrites the file in stable order, so git diffs show exactly what knowledge was added — reviewable in PRs like any other change. Every memory keeps its `user_id`, and searches can be scoped per contributor (`user_id` filter on `collivind_search`).

## How It Works

Collivind stores memories as nodes in a knowledge graph with vector embeddings for semantic search.

**Memory categories:** fact, decision, pattern, error, architecture, preference, snippet, and summary (written only through `collivind_summarize`)

**Entity types:** project, file, service, concept, person, library, tool

**Relationships:** ABOUT, MENTIONS, SUPERSEDES, CONTRADICTS, RELATES_TO, CAUSED_BY, DEPENDS_ON, IMPLEMENTS, PART_OF, USES, and more

**Search** runs in two stages, all local (no LLM, no API key). The first stage combines vector similarity (70% weight) with graph proximity (30% weight) and temporal decay. The second stage takes a pool of 50 candidates, fused by reciprocal rank from the dense top 50 and a keyword (BM25) top 50, plus the best graph-linked memories, and rescores each against the query with a small cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, about 90 MB, downloaded once on first use). Its score takes the vector similarity's place in the same formula, so graph links and recency still count. If the reranker is not loaded yet, fails or is turned off, search returns the first-stage ranking unchanged; it never fails or blocks a prompt. The same happens when no candidate reaches the model's relevance boundary (logit 0): its order among texts it finds irrelevant is noise. Recall (the prompt hook and `collivind_get_context`) also drops memories scored below `recall_min_score`. Explicit search (`collivind search`, `collivind_search`) never does, because an explicit search that returns nothing is more surprising than one weak match. Keyword search uses SQLite FTS5 in embedded mode and a Neo4j full-text index in docker mode; both are built automatically for memories that already exist.

**Summary memories** group the detail memories of one session (everything saved with its session id; the group exists before any summary text does) or of a named topic. The agent writes the summary, never collivind: when a session has accumulated enough new memories, the checkpoint's `collivind_batch_add` response says `summary_due`, the agent reads the members with `collivind_summarize` and passes back a list of self-contained dated facts. The summary is an ordinary searchable memory (category `summary`) linked to each member by a `SUMMARIZES` edge; writing it again for new members supersedes the old version, keeping the version chain. At recall, the best summaries for the query are ranked against each other by their best fact line; those whose best line the reranker judges relevant come first, followed by their most relevant members and the best other memories, those two merged by score. An off-topic summary is never returned this way, and a prompt unrelated to anything stored still recalls nothing. With no summaries in the store, search is unchanged. Routing adds about 150 ms to a prompt's recall (measured on a laptop, embedded mode). Recall goes summary to details only: attaching a detail's summary when the detail ranks high did not measurably help offline (p = 0.20) and is not done. `export`, `import` and `migrate` carry summaries and their links; the older one-memory-per-line format (`sync`) skips them.

**Deduplication and supersession** keep the store current: resubmitting identical knowledge is rejected, and storing an updated version of existing knowledge retires the stale memory (it leaves search results but stays in the version history) instead of keeping old and new side by side. **Conflicts are resolved on write**: when a new memory is close to a live one about the same subject, `collivind_add_memory`/`collivind_batch_add` return it under `possible_conflicts` with an instruction. The agent checks the code, docs or the user's words, retires the stale or wrong one with `collivind_invalidate` (history kept), and asks the user one short question only when it cannot tell. Similarity alone no longer writes CONTRADICTS edges; `collivind contradictions review` triages the ones older versions wrote (dry run by default, `--apply` re-labels or invalidates, never deletes a memory).

## Benchmarks

**LongMemEval-S session retrieval, Recall@5: 98.0%** (500/500 questions,
standard split, raw mode — local embeddings, no LLM, no reranking, no API
keys). Fully reproducible: methodology, per-question results, and honest
caveats in [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md).
**LoCoMo turn-level retrieval through the full engine: strict hit@5 72.4%,
hit@10 77.5%** with the 0.19 hybrid keyword + cross-encoder rerank stage
(0.18: 57.6% / 67.2%; 1,536 non-adversarial questions, a much harder,
turn-level task, same doc).
**Summary memories (0.20)** lift LoCoMo QA through the engine from 75.7% to
83.7% at about 1,800 words of context (300 questions, Claude Sonnet 5.5
reader and judge, p = 0.0002), and the prompt hook's default 800-token recall
from 74.3% to 81.3%; multi-hop did not improve measurably. Summaries cost one
agent call per session at save time.
We publish
no side-by-side numbers against other memory tools — different projects
measure different things on different splits.

## Architecture

```
Claude Code → MCP Server (JSON-RPC stdio) → Core Library → Storage Backends
```

Three deployment modes share the same core library — only the storage layer changes:

| Mode | Vector Store | Graph Store | Embeddings |
|------|-------------|-------------|------------|
| `docker` | Qdrant container | Neo4j container | sentence-transformers container |
| `embedded` | SQLite + numpy | SQLite | In-process model |
| `remote` | External Qdrant | External Neo4j | External service |

## Configuration

Config lives at `~/.collivind/config.toml`:

```toml
[collivind]
user_id = "local"
data_dir = "~/.collivind"
mode = "docker"  # "docker", "embedded", or "remote"

[search]
default_limit = 10
vector_weight = 0.7
graph_weight = 0.3
similarity_threshold = 0.3
rerank = true             # cross-encoder second stage; false = the pre-0.19 ranking exactly
rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # embedded mode (docker: COLLIVIND_RERANK_MODEL on the embeddings service)
rerank_pool = 50          # candidates rescored per search
lexical = true            # add keyword (BM25) candidates to the rerank pool
recall_min_score = 0.00002 # recall drops memories the reranker scores below this (0 = off); explicit search never
summary_routes = 3        # top summaries returned with their best members; 0 = summaries are ordinary results

[hooks]
save_interval = 15        # responses between capture checkpoints, per session
summary_after = 8         # ask for a session summary after this many uncovered memories; 0 = never
enable_precompact = true
enable_stop = true
quiet_capture = true      # save without announcing routine checkpoints
recall_limit = 20         # memories recalled per prompt (1-50)
recall_max_tokens = 2400  # token budget for them (100-20000); `hook user-prompt --max-tokens` overrides
```

See the [design spec](docs/superpowers/specs/2026-04-29-collivind-design.md) for the full configuration reference.

## Development

```bash
git clone https://github.com/yourusername/collivind.git
cd collivind
uv sync --dev
uv run pytest tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT
