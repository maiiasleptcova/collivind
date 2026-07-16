# Collivind — Session Handover

## What is Collivind

Open-source, graph-based memory layer for AI coding assistants. Stores knowledge from coding sessions (facts, decisions, patterns, errors, architecture choices) in graph + vector search + embeddings. Runs entirely local — no LLM API keys. Three deployment modes: **docker** (Qdrant + Neo4j + sentence-transformers containers), **embedded** (SQLite vector + graph stores + local model, no Docker, multi-process safe), **remote** (external services). Designed with pluggable backends so an enterprise version (separate repo) can swap storage via config.

## What was done

### Session 1 (2026-04-29): Features 1–8

1. **Feature 1: Project Skeleton + Config + CLI + Docker** — git repo, pyproject.toml, click CLI, docker-compose templating
2. **Feature 2: Storage Interfaces + Qdrant** — data models, VectorStore/GraphStore/EmbeddingProvider ABCs, QdrantVectorStore, HttpEmbeddingProvider
3. **Feature 3: Neo4j Graph Store** — Neo4jGraphStore with Cypher queries, schema constraints
4. **Feature 4: Memory Manager + Dedup** — MemoryManager orchestrator, Deduplicator engine
5. **Feature 5: Hybrid Search Engine** — SearchEngine (0.7 vector + 0.3 graph), temporal filtering
6. **Feature 6: MCP Server + Core Tools** — JSON-RPC stdio server, 6 core tools
7. **Feature 7: Graph Tools + Batch Ops** — get_entity, get_timeline, batch_add (10 tools total)
8. **Feature 8: Hook Integration** — Stop + PreCompact hooks, state tracking

### Session 2 (2026-04-30): Features 9, 10, 12 + Embedded Mode

9. **Feature 9: CLI Search + Reset + Docker Mgmt** — `search`, `reset`, `docker` (up/down/logs) CLI commands
10. **Feature 10: Contradiction Detection** — find_contradictions in SearchEngine, auto-detection in add_memory pipeline, MCP tool
11. **Feature 12: Performance + Robustness** — retry logic with exponential backoff on all storage backends, graceful degradation (MCP server starts even if backends fail), StorageUnavailableError
12. **Embedded Mode** (cross-cutting):
    - `storage/qdrant_embedded.py` — EmbeddedQdrantStore (in-process Qdrant, `QdrantClient(path=...)`; since replaced — the class is now a shim over the SQLite-backed `storage/vector_sqlite.py`)
    - `storage/graph_sqlite.py` — SqliteGraphStore (SQLite with adjacency tables, full GraphStore impl)
    - `storage/embedding_local.py` — LocalEmbeddingProvider (lazy-loads sentence-transformers in-process)
    - `storage/factory.py` — backend factory selecting implementations based on `config.mode`
    - `config.py` — added `mode` field ("docker", "embedded", "remote")
    - `mcp/server.py` — uses factory instead of hardcoded backends
    - `cli/commands/init.py` — mode-aware initialization (embedded skips Docker, creates local dirs)
    - `cli/commands/status.py` — mode-aware health reporting
    - `cli/commands/reset.py` — uses factory for mode-agnostic reset
    - `pyproject.toml` — `[embedded]` optional dependency group (sentence-transformers)
    - `models/relationship.py` — added `id` field to RelationshipEdge

**Test state:** 36 passed, 6 skipped (integration tests needing Docker). All unit + MCP tests green.

## Current repo state

```
collivind/
  pyproject.toml
  .gitignore
  LICENSE
  README.md
  HANDOVER.md
  docs/superpowers/specs/2026-04-29-collivind-design.md
  src/collivind/
    __init__.py, __main__.py, version.py, config.py, exceptions.py
    cli/                  # click CLI (init, status, hook, search, reset, docker)
    docker/               # Compose templating and HTTP health checks
    models/               # dataclasses (Memory, Entity, Relationship, Session)
    storage/              # SQLite vector + graph, Qdrant (remote), Neo4j, Embedding (local + HTTP), factory
    engine/               # MemoryManager, SearchEngine, GraphEngine, Deduplicator, Extractor
    mcp/                  # Stdio MCP Server, CollivindTools, Hooks
  tests/
    unit/                 # test_config, test_factory, test_graph_sqlite, test_graph_engine, test_hooks,
                          # test_memory_manager, test_models, test_search_engine
    integration/          # test_embedding_service, test_full_pipeline, test_neo4j_store, test_qdrant_store
    mcp/                  # test_server, test_tools
```

## What to do next

### Immediate next step

**Feature 11: npm Package + Docs + OSS Polish**

Deliverables:
- `README.md` — comprehensive with install instructions for all 3 modes, usage examples, architecture overview
- `CONTRIBUTING.md` — dev setup, testing, PR guidelines
- `CLAUDE.md` — project conventions for AI assistants
- `npm/` — wrapper package for `npx collivind` install
- `.github/workflows/ci.yml` — pytest + ruff + mypy
- `.github/workflows/release.yml` — PyPI publish on tag
- `.pre-commit-config.yaml` — ruff, mypy hooks

### After Feature 11

The project is essentially complete. All 12 features from the spec are implemented. Feature 11 is the final polish for public release.

## Key architecture details

**Three deployment modes:**
```
docker:    Claude Code → MCP Server → Core Library → Docker (Qdrant/Neo4j/embeddings)
embedded:  Claude Code → MCP Server → Core Library → In-process (SQLite/sentence-transformers)
remote:    Claude Code → MCP Server → Core Library → External services
```

**Storage factory pattern:** `config.mode` selects backends at startup via `storage/factory.py`.

**Data model:**
- 3 node types: Memory (7 categories), Entity (7 types), Session
- 14 relationship types (ABOUT, MENTIONS, SUPERSEDES, CONTRADICTS, etc.)
- Scoping: User > Project > Session
- Hybrid search: 0.7 vector + 0.3 graph proximity

**Config:** TOML at `~/.collivind/config.toml`, mode defaults to "docker"

## Implementation Preferences

- **Build backend:** uv with hatchling
- **Python version:** 3.11+
- **Docker interaction:** subprocess calling `docker compose` directly (no Python docker SDK)
- **Embedded optional dep:** `pip install collivind-memory[embedded]` for sentence-transformers
