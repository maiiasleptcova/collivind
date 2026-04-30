# Collivind — Session Handover

## What is Collivind

Open-source, graph-based memory layer for AI coding assistants. Stores knowledge from coding sessions (facts, decisions, patterns, errors, architecture choices) in Neo4j (graph) + Qdrant (vector search) + sentence-transformers (embeddings). Runs entirely local via Docker — no LLM API keys. Designed with pluggable backends so an enterprise version (separate repo) can swap Docker for cloud services via config.

## What was done this session (2026-04-29)

The project skeleton has been created and we have successfully implemented the first 8 features outlined in the design spec:

1. **Feature 1: Project Skeleton + Config + CLI + Docker**
   - Initialized git repo, `pyproject.toml` with `uv` and `hatchling`.
   - Setup `click` CLI with `init` and `status` commands.
   - Built `docker-compose.yml` templating for Qdrant, Neo4j, and the embedding FastAPI server. 
2. **Feature 2: Storage Interfaces + Qdrant Implementation**
   - Created core data models (`MemoryNode`, `EntityNode`, `RelationshipEdge`) using standard python `dataclasses`.
   - Built `VectorStore`, `GraphStore`, and `EmbeddingProvider` ABC interfaces.
   - Built `QdrantVectorStore` and `HttpEmbeddingProvider`.
3. **Feature 3: Neo4j Graph Store Implementation**
   - Implemented `Neo4jGraphStore` translating data models into robust Cypher queries (`CREATE`, `MERGE`, `MATCH`, `DETACH DELETE`).
   - Integrated schema uniqueness constraints.
4. **Feature 4: Memory Manager + Deduplication**
   - Created `MemoryManager` orchestrator and `Deduplicator` engine mapping vector responses into graph updates efficiently and resolving similarities.
5. **Feature 5: Search Engine (Hybrid Search)**
   - Built `SearchEngine` scoring logic bridging vector score (0.7) and graph shared-entity proximity (0.3).
   - Filtered out temporal (`valid_to`) expired nodes.
6. **Feature 6: MCP Server + Core Tools**
   - Mapped out the standard `tools/list` and `tools/call` JSON-RPC stdio protocol into `MCPServer`.
   - Setup `collivind_add_memory`, `collivind_search`, `collivind_get_context`, `collivind_invalidate`, `collivind_forget`, and `collivind_status`.
7. **Feature 7: Graph Tools + Batch Operations**
   - Added `collivind_get_entity`, `collivind_get_timeline`, and `collivind_batch_add`.
8. **Feature 8: Hook Integration (Stop + PreCompact)**
   - Wired `collivind hook stop` and `collivind hook precompact` CLI commands.
   - Implemented state tracking in `~/.collivind/hook_state.json`.

All `tests/unit` and `tests/integration` pass successfully (`pytest tests/`).

## Current repo state

```
collivind/
  pyproject.toml
  .gitignore
  LICENSE
  README.md
  HANDOVER.md                     # this file
  src/collivind/
    __init__.py, __main__.py, version.py, config.py, exceptions.py
    cli/                  # click CLI (init, status, hook)
    docker/               # Compose templating and HTTP health checks
    models/               # dataclasses (Memory, Entity, Relationship, Session)
    storage/              # Qdrant, Neo4j, and Embedding logic
    engine/               # MemoryManager, SearchEngine, Deduplicator
    mcp/                  # Stdio Server, CollivindTools
  tests/
    unit/                 # Fast mocks for logic
    integration/          # Architecture pipelines
    mcp/                  # Server standard I/O checks
```

## What to do next

### Immediate next step

We are ready to start on **Feature 9: CLI Search + Reset + Docker Mgmt**. 
You can find the feature breakdown in the design spec (`docs/superpowers/specs/2026-04-29-collivind-design.md`).

Feature 9 deliverables:
- `cli/commands/search.py`: Pretty print search results.
- `cli/commands/reset.py`: `docker compose down -v` to clear everything.
- `cli/commands/start.py` / `stop.py`: basic docker management.

### After Feature 9

Continue sequentially through the remaining features. Each feature is a self-contained unit. 

| # | Feature | Key milestone |
|---|---|---|
| 9 | CLI Search + Reset + Docker Mgmt | Complete CLI |
| 10 | Contradiction Detection | Memory quality |
| 11 | npm Package + Docs + OSS Polish | **Public release ready** |
| 12 | Performance + Robustness | Production-grade |

## Key architecture details (quick reference)

**Layered architecture:**
```
Claude Code → MCP Server (thin shell) → Core Library → Storage Interfaces → Docker (Qdrant/Neo4j/embeddings)
```

**Data model:**
- 3 node types: Memory (7 categories), Entity (7 types), Session
- 12+ relationship types
- Scoping: User > Project (default) > Session
- Hybrid search: 0.7 vector + 0.3 graph proximity

**Config:** TOML at `~/.collivind/config.toml`, env var overrides

## Implementation Preferences

**Build backend:** Use **uv** with **hatchling**. `pyproject.toml` uses `[build-system] requires = ["hatchling"]`. 
**Python version:** **Python 3.11+ only**. 
**Docker interaction:** Use **subprocess** calling `docker compose` directly. Do NOT add the Python docker SDK.
**npm wrapper timing:** Save for **Feature 11**. 

Good luck!
