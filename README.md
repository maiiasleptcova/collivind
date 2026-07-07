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

This uses in-process Qdrant, SQLite for the graph, and loads the embedding model directly — zero Docker dependency.

### npm (alternative)

```bash
npx collivind-memory init
```

## Claude Code Integration

Register Collivind as an MCP server:

```bash
claude mcp add --global collivind -- python3 -m collivind.mcp.server
```

`collivind init` does this automatically.

### Hooks

Collivind uses Claude Code hooks for automatic memory capture and recall:

- **SessionStart hook** — injects a compact index (~100 tokens) of your project's stored knowledge at the start of every session
- **UserPromptSubmit hook** — searches memory with every prompt you type and injects the relevant context (budgeted, silent when nothing matches), so the agent starts building with what it already learned
- **Stop hook** — every 15 responses, prompts Claude to extract and store knowledge
- **PreCompact hook** — saves all session knowledge before context compression

The recall hooks (SessionStart, UserPromptSubmit) are also registered for Codex CLI when `~/.codex` exists.

### In-session commands

`collivind init` also installs manual slash commands, so you can save and recall memory mid-session without leaving your agent:

- `/mem-save <text>` — store knowledge (or, with no text, extract it from the current conversation)
- `/mem-recall <topic>` — pull relevant stored knowledge into the session

Installed for Claude Code (`~/.claude/commands`) and Codex (`~/.codex/prompts`) automatically; for VS Code Copilot run `collivind commands install --tool copilot` inside a repo (writes `.github/prompts/`).

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
collivind status            # Health check
collivind search "query"    # Search memories from terminal
collivind add "content" -c decision -t db,infra  # Store a memory
collivind get <id>          # Show one memory
collivind context "query"   # Formatted context block (pipe into prompts)
collivind update <id> --content "..."  # Update a memory (re-embeds)
collivind invalidate <id> -r outdated  # Mark outdated, keep history
collivind forget <id>       # Delete permanently (with confirmation)
collivind export -p proj -o mem.jsonl  # Backup / portability
collivind import mem.jsonl  # Re-import (deduplication applies)
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

**LongMemEval-S session retrieval, Recall@5: 91.8%** (500/500 questions, raw
mode — local embeddings, no LLM, no reranking, no API keys). Fully
reproducible: methodology, per-question results, and honest caveats in
[benchmarks/BENCHMARKS.md](benchmarks/BENCHMARKS.md). We publish no
side-by-side numbers against other memory tools — different projects measure
different things on different splits.

## Architecture

```
Claude Code → MCP Server (JSON-RPC stdio) → Core Library → Storage Backends
```

Three deployment modes share the same core library — only the storage layer changes:

| Mode | Vector Store | Graph Store | Embeddings |
|------|-------------|-------------|------------|
| `docker` | Qdrant container | Neo4j container | sentence-transformers container |
| `embedded` | In-process Qdrant | SQLite | In-process model |
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
save_interval = 15
enable_precompact = true
enable_stop = true
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
