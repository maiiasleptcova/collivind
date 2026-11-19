# Collivind

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
> **Resident model:** hooks are short-lived processes, so the first embed starts a small background process that keeps the model loaded (`~/.collivind/embed.sock`, owner-only); later hooks embed in milliseconds instead of reloading the model every prompt. It exits after 30 idle minutes (`daemon_idle_timeout` under `[embeddings]`). Manage it with `collivind embed-daemon start|stop|status`; turn it off with `daemon = false` or `COLLIVIND_EMBED_DAEMON=0`.
>
> **Upgrading from ≤ 0.1.x:** older versions kept vectors in a single-process Qdrant store (`~/.collivind/qdrant_data`). It is migrated automatically on first use and left in place as a backup; restart any agent sessions still running the old version once after upgrading. If you later downgrade, memories written after the upgrade live only in the new store — carry them over with `collivind export` / `import`.

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

### Web UI

```
collivind web
```

Opens a local browser UI on `127.0.0.1:8765` for searching, reading, editing
and deleting memories. It reads `~/.collivind/config.toml`, so it shows exactly
the store your hooks and MCP tools use, and says which mode that is in the
header. Each memory carries a **Details** disclosure with the rest of its
record — project, source, confidence, tags, version chain, validity window,
session and id. Loopback only, no auth — it exposes the whole store to anything
that can reach the port.

### Agent support

| Agent | Memory tools (MCP) | Automatic recall | Periodic capture | Compaction capture |
|---|---|---|---|---|
| Claude Code | yes | yes — SessionStart + UserPromptSubmit | yes — Stop hook | yes — PreCompact marks it, SessionStart (`compact`) asks |
| Codex | yes | yes — after you trust the hooks | yes — Stop hook, after you trust it | yes — PreCompact marks it, SessionStart (`compact`) asks |
| opencode | yes | **no** — no prompt hook to inject into | yes — plugin on `session.idle` | yes — plugin on `session.compacted` |
| GitHub Copilot | yes | **no** — Copilot has no hook system | **no** | **no** |

`collivind init` registers the MCP server for every agent it finds installed,
plus the hooks (Claude Code, Codex) or the capture plugin (opencode,
`~/.config/opencode/plugins/collivind-capture.js`). Copilot gets the memory
tools only, so you search and save explicitly.

**What "capture" means, precisely.** A hook cannot write memories itself —
extraction needs the agent's model. So a checkpoint *asks* the agent to call
`collivind_batch_add` with a checkpoint id, and only a successful call carrying
that id confirms it. Until then the checkpoint stays pending and is re-asked a
few responses later; it is never counted as saved because a reminder was shown.
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
- **UserPromptSubmit** — searches memory with every prompt you type and injects the relevant context (budgeted, silent when nothing matches)
- **Stop** — every `save_interval` responses (default 15, per session), asks the agent to checkpoint
- **PreCompact** — marks a compaction checkpoint due (see *Compaction* above)

Registered for Claude Code (via the plugin, or `~/.claude/settings.json`) and for Codex in `~/.codex/hooks.json` when `~/.codex` exists. If any hook finds the store down it says so in the session instead of recalling nothing.

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
| `collivind_find_contradictions` | Detect conflicting memories |
| `collivind_get_version_chain` | Full supersession history of a memory |
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
collivind export -p proj -o mem.jsonl  # Backup / portability
collivind import mem.jsonl  # Re-import (deduplication applies)
collivind migrate --from embedded --to docker  # Copy everything between modes, ids kept
collivind sync ./shared-memory  # Two-way team sync via a git-tracked dir
collivind reset             # Wipe all data (with confirmation)
collivind docker up|down|logs  # Container management (docker mode)
collivind hook install      # Register Claude Code hooks
```

Memory verbs take `--json` for scripting and agent use.

## Team Sharing

Share memory with your team through any git repo — no server required:

```bash
collivind sync ~/team-repo/memory   # import teammates' knowledge, write yours back
cd ~/team-repo && git add memory && git commit -m "memory sync" && git push
```

Each sync imports records you don't have yet (exact duplicates are skipped; updated versions of known facts supersede your stale copies) and rewrites the file in stable order, so git diffs show exactly what knowledge was added — reviewable in PRs like any other change. Every memory keeps its `user_id`, and searches can be scoped per contributor (`user_id` filter on `collivind_search`).

## How It Works

Collivind stores memories as nodes in a knowledge graph with vector embeddings for semantic search.

**Memory categories:** fact, decision, pattern, error, architecture, preference, snippet

**Entity types:** project, file, service, concept, person, library, tool

**Relationships:** ABOUT, MENTIONS, SUPERSEDES, CONTRADICTS, RELATES_TO, CAUSED_BY, DEPENDS_ON, IMPLEMENTS, PART_OF, USES, and more

**Search** combines vector similarity (70% weight) with graph proximity (30% weight) for results that understand both meaning and structure.

**Deduplication and supersession** keep the store current: resubmitting identical knowledge is rejected, and storing an updated version of existing knowledge retires the stale memory (it leaves search results but stays in the version history) instead of keeping old and new side by side. **Contradiction detection** links genuinely conflicting memories with CONTRADICTS relationships.

## Benchmarks

**LongMemEval-S session retrieval, Recall@5: 98.0%** (500/500 questions,
standard split, raw mode — local embeddings, no LLM, no reranking, no API
keys). Fully reproducible: methodology, per-question results, and honest
caveats in [benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md). We publish
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

[hooks]
save_interval = 15        # responses between capture checkpoints, per session
enable_precompact = true
enable_stop = true
quiet_capture = true      # save without announcing routine checkpoints
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
