# Collivind Design Spec

Open-source, graph-based memory layer for AI coding assistants. Stores knowledge from coding sessions (facts, decisions, patterns, errors, architecture choices) in a graph database with vector embeddings for semantic search. Runs entirely locally via Docker — no LLM API keys required.

## Problem

AI coding assistants (Claude Code, Copilot, Codex) lose all context between sessions. Developers repeatedly re-explain architecture, past decisions, and project conventions. In teams, knowledge stays siloed — an ML engineer's decisions are invisible to backend engineers working on the same system.

## Solution

Collivind is an MCP server that gives AI coding assistants persistent, structured memory. It:
- Extracts knowledge automatically from coding sessions (using the AI assistant itself, not a separate LLM)
- Stores memories as graph nodes with typed relationships in Neo4j
- Enables semantic search via vector embeddings in Qdrant
- Runs fully local via Docker (Qdrant + Neo4j + sentence-transformers)
- Designed with pluggable backends so an enterprise version can swap Docker for cloud services via config

## Scope

This spec covers the **open-source repo only**. The enterprise version (sign-up, UI, dashboard, team memory sharing, department-level scoping) is a separate repo that will import collivind-core as a dependency.

## Future Consideration

The scoping model should eventually support a "department" or "team" scope between user and project, enabling executives to see progress across ML engineering, backend, MLOps, and frontend simultaneously. The entity graph naturally supports this — shared entities (services, concepts, libraries) connect cross-department knowledge. Not a requirement for v1.

## Deployment Modes

Collivind supports three deployment modes to work in any environment:

**`docker` (default)** — Qdrant, Neo4j, and sentence-transformers run in Docker containers. Best for local development with Docker Desktop available.

**`embedded`** — Qdrant runs in-process (embedded Rust engine via qdrant-client's local mode), sentence-transformers loads the model directly in Python, and Neo4j is replaced by NetworkX + SQLite for graph storage. Zero Docker dependency. Install via `pip install collivind-memory[embedded]`. Adds ~2GB for PyTorch but works everywhere — devcontainers, Codespaces, CI/CD, cloud IDEs, any environment without Docker.

**`remote`** — Connects to externally hosted Qdrant/Neo4j/embeddings services via config. No Docker, no local model. Just point config at existing endpoints. This is also the path the enterprise version uses.

Mode is set in config:
```toml
[collivind]
mode = "docker"     # default
# mode = "embedded" # in-process, no Docker
# mode = "remote"   # external services, no Docker
```

`collivind init` adapts to the mode:
- `docker`: pulls images, starts containers, runs health checks
- `embedded`: downloads the embedding model on first run, creates local Qdrant storage dir, initializes SQLite graph DB
- `remote`: validates connectivity to external endpoints

---

## Architecture

**Approach B — Layered Core + MCP Shell**

```
AI Coding Assistants
  Claude Code  |  Copilot (future)  |  Codex (future)
       |
       | MCP Protocol (stdio)
       v
MCP Server Layer (thin shell)
  MCP Tool Handlers  |  Claude Code Skill (extraction hooks)
       |
       | Python API calls
       v
Core Library (collivind-core)
  Memory Manager  |  Graph Engine  |  Embedding Service  |  Search & Retrieval
       |
       | Storage Interfaces (pluggable)
       v
Storage Backends
  VectorStore (ABC)  |  GraphStore (ABC)  |  EmbeddingProvider (ABC)
       |
       v
Docker Containers (local) / Cloud APIs (enterprise)
  Qdrant  |  Neo4j  |  sentence-transformers
```

The core library defines abstract storage interfaces (`VectorStore`, `GraphStore`, `EmbeddingProvider`). The open-source version implements these with Docker-based Qdrant, Neo4j, and a sentence-transformers HTTP service. The enterprise version swaps implementations via config — same core, different backends.

---

## Data Model

### Node Types

**Memory Nodes** — extracted knowledge units with one category:

| Category | Purpose | Example |
|---|---|---|
| fact | How something works | "The auth service uses JWT with RS256" |
| decision | A choice and its rationale | "Chose PostgreSQL over MongoDB for relational needs" |
| pattern | Recurring coding convention | "All API routes use /api/v1/ prefix with snake_case" |
| error | Bug and its resolution | "CORS error fixed by adding allowed_origins to FastAPI middleware" |
| architecture | System design element | "Event-driven architecture using Redis pub/sub between services" |
| preference | Coding style preference | "Prefers dataclasses over Pydantic for internal DTOs" |
| snippet | Code worth remembering | "The retry decorator with exponential backoff" |

Memory node properties:
- `id` (UUID), `content` (text), `summary` (one-line)
- `category` (one of 7 above)
- `confidence` (0.0-1.0, default 1.0)
- `valid_from` (datetime), `valid_to` (datetime | null, null = still valid)
- `project_id`, `session_id`, `user_id` (defaults to "local")
- `source` (hook_stop | hook_precompact | periodic | manual)
- `superseded_by` (id | null)
- `tags` (string[])
- `created_at`, `updated_at`

**Entity Nodes** — things memories reference:

| Type | Examples |
|---|---|
| project | "collivind", "cybergine" |
| file | "src/auth/middleware.py" |
| service | "PostgreSQL", "Redis", "Qdrant" |
| concept | "authentication", "caching" |
| person | "Eugene", "Alice" |
| library | "FastAPI", "sentence-transformers" |
| tool | "Docker", "pytest", "Claude Code" |

Entity node properties:
- `id` (normalized: lowercase, underscored), `name` (display name)
- `type` (one of 7 above)
- `properties` (arbitrary metadata dict)
- `created_at`, `updated_at`

**Session Nodes** — coding session records:
- `id` (UUID), `project_id`, `started_at`, `ended_at`, `summary`, `memory_count`

### Relationship Types

Memory to Entity:
- `ABOUT` — memory is primarily about this entity
- `MENTIONS` — memory references this entity

Memory to Memory:
- `SUPERSEDES` — newer replaces older (with reason property)
- `CONTRADICTS` — memories conflict (with resolution property)
- `RELATES_TO` — general thematic link (with strength float)
- `CAUSED_BY` — this memory resulted from the other
- `DEPENDS_ON` — this decision depends on the other

Entity to Entity:
- `IMPLEMENTS` — file implements concept
- `DEPENDS_ON` — service depends on another
- `PART_OF` — file is part of project
- `USES` — project uses library
- `RELATED_TO` — general link

Session edges:
- `PRODUCED` — session created this memory
- `WORKED_ON` — session touched this entity

All relationship edges carry: `created_at`, `confidence`, `source`.

### Vector Embeddings (Qdrant)

Every Memory node gets a 384-dimension vector embedding (all-MiniLM-L6-v2) stored in Qdrant, synced by UUID.

Qdrant point payload: `content`, `summary`, `category`, `project_id`, `user_id`, `valid_from`, `valid_to`, `tags`, `created_at`.

Entity nodes optionally get embeddings for semantic entity search.

### Scoping

Three levels:
1. **User** — all memories across all projects (broadest)
2. **Project** — memories for one codebase (default search scope)
3. **Session** — single coding session (narrowest)

Search defaults to project scope, can widen to user scope for cross-project knowledge.

---

## MCP Tools

14 tools in 5 categories:

### Store (3 tools)

`collivind_add_memory` — Store a single extracted fact/decision/pattern/error.
- Params: `content` (str), `category` (enum), `entities[]` ({name, type, relation}), `relationships[]` ({source, target, type}), `tags[]`, `confidence` (float)
- Pipeline: deduplicate -> embed -> create Neo4j node -> upsert Qdrant vector -> create/merge entities -> create relationships

`collivind_add_memories_batch` — Store multiple memories at once (session-end).
- Params: `memories[]` (array of add_memory params)

`collivind_invalidate` — Mark a memory as outdated/superseded.
- Params: `memory_id`, `reason`, `superseded_by`

### Search (3 tools)

`collivind_search` — Semantic + graph hybrid search.
- Params: `query`, `category`, `project_id`, `limit`, `include_graph`
- Pipeline: embed query -> Qdrant top-K -> graph expansion via shared entities -> re-rank (0.7 vector + 0.3 graph) -> return

`collivind_get_context` — Get relevant context as formatted text for Claude.
- Params: `query`, `max_tokens`

`collivind_get_project_summary` — Overview of key knowledge for a project.
- Params: `project_id`

### Graph (3 tools)

`collivind_relate` — Create a relationship between two memories or entities.
- Params: `source_id`, `target_id`, `rel_type`, `properties`

`collivind_traverse` — Walk the graph from a node.
- Params: `start_id`, `rel_types[]`, `direction`, `max_depth`

`collivind_find_entity` — Find all memories about a named entity.
- Params: `entity_name`, `entity_type`, `limit`

### Context (3 tools)

`collivind_session_start` — Register a new coding session.
- Params: `project_id`, `session_id`

`collivind_session_end` — End session, return extraction prompt for Claude.
- Params: `session_id`

`collivind_timeline` — Chronological view of project knowledge.
- Params: `project_id`, `entity`, `limit`

### Manage (2 tools)

`collivind_status` — Health check + stats (memory count, entity count, graph edges).

`collivind_forget` — Delete a specific memory permanently.
- Params: `memory_id`

---

## Claude Code Integration

### MCP Server Registration

`collivind init` registers the server globally:
```
claude mcp add --global collivind -- python3 -m collivind.mcp.server
```

### Hooks

**Stop hook** — periodic save every 15 messages. The hook counts human messages since last save. When threshold is reached, returns a block response with structured extraction instructions telling Claude to:
1. Identify noteworthy facts, decisions, patterns, errors, architecture choices
2. Call `collivind_add_memory` with content, category, entities, relationships, tags, confidence
3. Focus on knowledge valuable in future sessions, skip trivial/obvious facts
4. Continue the conversation after saving

**PreCompact hook** — urgent save-all before context compression. Same extraction instructions but more aggressive ("save ALL session content before context is lost").

Hook state tracked in `~/.collivind/hook_state/` (last save message count per session).

### Extraction Schema

The extraction prompt tells Claude exactly what to extract:
- Write self-contained content (makes sense without conversation context)
- Classify into the 7 categories
- Identify all entities with types
- Identify relationships between entities
- Add relevant tags for cross-cutting search

---

## Docker Setup

Three containers via Docker Compose at `~/.collivind/docker-compose.yml`:

**Qdrant** (qdrant/qdrant:v1.12.6)
- Ports: 6333 (HTTP), 6334 (gRPC)
- Volume: `collivind-qdrant-data`
- Memory limit: 512M

**Neo4j** (neo4j:5.26-community)
- Ports: 7474 (browser), 7687 (bolt)
- Volume: `collivind-neo4j-data`
- Auth: neo4j/collivind_local
- Memory: 256M heap initial, 512M heap max, 128M pagecache
- Memory limit: 768M

**Embedding Service** (custom, built from Dockerfile.embeddings)
- Port: 8090
- Model: all-MiniLM-L6-v2
- Volume: `collivind-model-cache`
- Memory limit: 512M
- Minimal FastAPI app with `/embed`, `/embed_batch`, `/health` endpoints

Total local resource usage: ~1.8GB RAM.

---

## Installation Flow

### pip (primary)

```bash
pip install collivind-memory
collivind init
```

### npm (alternative)

```bash
npx collivind-memory init
```

The npm package (a thin wrapper in this same repo under `npm/`) checks for Python, pip-installs the Python package, and runs `collivind init`.

### `collivind init` sequence

1. Check Docker is installed and running
2. Create `~/.collivind/` directory
3. Generate `~/.collivind/config.toml` with defaults
4. Generate `~/.collivind/docker-compose.yml`
5. Generate embedding service files (Dockerfile + server.py)
6. Build embedding service image
7. Pull Qdrant and Neo4j images
8. Start all containers
9. Wait for health checks (with timeout + progress)
10. Initialize Qdrant collection (384-dim, cosine distance)
11. Initialize Neo4j constraints/indexes
12. Register MCP server with Claude Code
13. Register hooks in `~/.claude/settings.json`
14. Print success summary

### CLI commands

- `collivind init` — full setup
- `collivind status` — health dashboard
- `collivind search "query"` — CLI search
- `collivind reset` — wipe all data (with confirmation)
- `collivind docker up|down|logs` — container management

---

## Configuration

TOML config at `~/.collivind/config.toml`:

```toml
[collivind]
user_id = "local"
data_dir = "~/.collivind"

[docker]
compose_project = "collivind"
auto_start = true

[qdrant]
host = "localhost"
port = 6333
collection_name = "collivind_memories"

[neo4j]
uri = "bolt://localhost:7687"
user = "neo4j"
password = "collivind_local"
database = "neo4j"

[embeddings]
service_url = "http://localhost:8090"
model = "all-MiniLM-L6-v2"
dimension = 384

[search]
default_limit = 10
vector_weight = 0.7
graph_weight = 0.3
similarity_threshold = 0.3
dedup_threshold = 0.92

[hooks]
save_interval = 15
enable_precompact = true
enable_stop = true
```

Priority: env vars (`COLLIVIND_QDRANT_HOST`) > config file > defaults.

Enterprise swaps this config to point at cloud-hosted services — same core library, different backends.

---

## Project Structure

```
collivind/
  src/collivind/
    __init__.py
    __main__.py
    version.py
    config.py
    exceptions.py
    models/
      __init__.py
      memory.py
      entity.py
      relationship.py
      session.py
      search.py
    storage/
      __init__.py
      interfaces.py           # VectorStore, GraphStore, EmbeddingProvider ABCs
      qdrant_store.py
      neo4j_store.py
      embedding_service.py
    engine/
      __init__.py
      memory_manager.py       # Core orchestrator
      graph_engine.py
      search_engine.py
      dedup.py
      extractor.py
    mcp/
      __init__.py
      server.py               # MCP JSON-RPC stdio server
      tools.py
      hooks.py
    cli/
      __init__.py
      main.py
      commands/
        __init__.py
        init.py
        status.py
        reset.py
        search.py
        docker.py
    docker/
      __init__.py
      compose.py
      health.py
      templates/
        docker-compose.yml.j2
        Dockerfile.embeddings
        embedding_server.py
  tests/
    conftest.py
    unit/
      test_config.py
      test_models.py
      test_memory_manager.py
      test_graph_engine.py
      test_search_engine.py
      test_dedup.py
    integration/
      test_qdrant_store.py
      test_neo4j_store.py
      test_embedding_service.py
      test_full_pipeline.py
    mcp/
      test_server.py
      test_tools.py
      test_hooks.py
  pyproject.toml
  README.md
  LICENSE                     # MIT
  CLAUDE.md
  .gitignore
  .pre-commit-config.yaml
```

---

## Storage Interfaces

All backends implement these ABCs (in `storage/interfaces.py`):

**VectorStore**: `initialize()`, `upsert(id, vector, payload)`, `search(vector, limit, filters, threshold)`, `delete(id)`, `health_check()`, `close()`

**GraphStore**: `initialize()`, `create_memory(MemoryCreate)`, `get_memory(id)`, `update_memory(id, **updates)`, `delete_memory(id)`, `create_entity(name, type, properties)`, `create_relationship(source, target, type, properties)`, `get_neighbors(node_id, rel_types, direction, depth)`, `find_related_memories(entity_name, limit)`, `get_timeline(project_id, entity, limit)`, `invalidate_memory(id, superseded_by, reason)`, `health_check()`, `close()`

**EmbeddingProvider**: `embed(text)`, `embed_batch(texts)`, `health_check()`, `dimension` (property)

---

## Core Engine

**MemoryManager** — primary API that MCP tools call:
- `add_memory()` — full pipeline: deduplicate -> embed -> create graph node -> upsert vector -> create entities -> create relationships
- `add_memories_batch()` — batch version for session-end summaries
- `search()` — hybrid semantic + graph search with re-ranking
- `get_context()` — formatted context string for Claude (token-limited)
- `invalidate()` — mark memory as outdated
- `get_project_summary()` — key facts, decisions, patterns for a project

**SearchEngine** — hybrid search pipeline:
- Vector search (embed query, Qdrant top-K)
- Graph expansion (connected memories via shared entities, 1-hop)
- Re-ranking (0.7 vector similarity + 0.3 graph proximity)
- Temporal filtering (exclude expired memories)
- Contradiction detection (find conflicting memories)
- Duplicate detection (embedding similarity > 0.92)

**GraphEngine** — relationship management and traversal:
- Neighbor discovery, path finding
- Timeline queries
- Related memory discovery via entity links

---

## Feature Breakdown

### Feature 1: Project Skeleton + Config + CLI + Docker
Installable package with `collivind init` and `collivind status` that set up Docker containers.

Files: `pyproject.toml`, `src/collivind/__init__.py`, `__main__.py`, `version.py`, `config.py`, `exceptions.py`, `cli/main.py`, `cli/commands/init.py`, `cli/commands/status.py`, `docker/compose.py`, `docker/health.py`, `docker/templates/*`, `tests/unit/test_config.py`

Acceptance: `pip install -e .` works. `collivind init` starts 3 Docker containers. `collivind status` shows all healthy.

### Feature 2: Storage Interfaces + Qdrant Implementation
Abstract storage interfaces and working Qdrant vector store.

Files: `models/memory.py`, `models/entity.py`, `models/relationship.py`, `models/search.py`, `storage/interfaces.py`, `storage/qdrant_store.py`, `storage/embedding_service.py`, `tests/unit/test_models.py`, `tests/integration/test_qdrant_store.py`, `tests/integration/test_embedding_service.py`

Acceptance: Can embed text via Docker embedding service, store vectors in Qdrant, retrieve by similarity search. All through the abstract interface.

### Feature 3: Neo4j Graph Store Implementation
Working Neo4j graph store with memory/entity CRUD and relationship management.

Files: `storage/neo4j_store.py`, `models/session.py`, `tests/integration/test_neo4j_store.py`

Acceptance: Can create memory nodes, entity nodes, relationships, traverse the graph. Temporal validity filtering works.

### Feature 4: Memory Manager + Deduplication
Core orchestrator tying vector and graph stores together.

Files: `engine/memory_manager.py`, `engine/dedup.py`, `tests/unit/test_memory_manager.py`, `tests/integration/test_full_pipeline.py`

Acceptance: `add_memory` creates embeddings, stores in both Qdrant and Neo4j, links entities. Near-duplicates detected and rejected/merged.

### Feature 5: Search Engine (Hybrid Search)
Combine vector similarity with graph relationships for better retrieval.

Files: `engine/search_engine.py`, `engine/graph_engine.py`, `tests/unit/test_search_engine.py`, `tests/unit/test_graph_engine.py`

Acceptance: Searching for "authentication" returns direct vector matches AND memories connected via shared entities. Results re-ranked by combined score.

### Feature 6: MCP Server + Core Tools
Working MCP server that Claude Code can connect to.

Files: `mcp/server.py`, `mcp/tools.py`, `tests/mcp/test_server.py`, `tests/mcp/test_tools.py`

Initial tools: `collivind_status`, `collivind_add_memory`, `collivind_search`, `collivind_get_context`, `collivind_invalidate`, `collivind_forget`

Acceptance: `claude mcp add collivind -- python3 -m collivind.mcp.server` works. Claude Code can store and search memories.

### Feature 7: Graph Tools + Batch Operations
Full graph manipulation tools and batch memory storage.

Files: `mcp/tools.py` (extended), `engine/extractor.py`, `tests/mcp/test_tools.py` (extended)

Additional tools: `collivind_add_memories_batch`, `collivind_relate`, `collivind_traverse`, `collivind_find_entity`, `collivind_timeline`, `collivind_get_project_summary`, `collivind_session_start`, `collivind_session_end`

Acceptance: All 14 MCP tools work. `collivind_session_end` returns structured extraction prompt.

### Feature 8: Hook Integration (Stop + PreCompact)
Automatic memory extraction triggered by Claude Code hooks.

Files: `mcp/hooks.py`, `tests/mcp/test_hooks.py`

Acceptance: After 15 user messages, Claude is automatically prompted to extract and save memories. Before context compaction, Claude saves all important knowledge.

### Feature 9: CLI Search + Reset + Docker Management
Complete CLI for debugging, testing, and management.

Files: `cli/commands/search.py`, `cli/commands/reset.py`, `cli/commands/docker.py`

Acceptance: Can search from terminal, reset data, manage containers.

### Feature 10: Contradiction Detection + Memory Quality
Detect and handle conflicting or outdated memories.

Files: `engine/search_engine.py` (extended), `tests/unit/test_search_engine.py` (extended)

Acceptance: Storing conflicting memories creates CONTRADICTS relationship. Invalidation with superseded_by chains properly. Old unreferenced memories get lower ranking.

### Feature 11: npm Package + Docs + OSS Polish
Public release readiness.

Files: `README.md`, `LICENSE`, `CLAUDE.md`, `CONTRIBUTING.md`, `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `.pre-commit-config.yaml`, npm wrapper package

Acceptance: Developer can read README, run `pip install collivind-memory && collivind init`, and have working memory in Claude Code within 5 minutes.

### Feature 12: Performance + Robustness
Production-grade reliability.

Files: connection pooling, retry logic, graceful degradation, batch embeddings, Qdrant indexing, Neo4j query optimization, pagination, `tests/integration/test_resilience.py`

Acceptance: MCP server starts gracefully even if Docker is down. 1000 memories stores without timeouts. Search across 10k memories returns in under 500ms.

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Language | Python 3.11+ | Best ML/embedding ecosystem, MCP SDK available |
| Vector DB | Qdrant (Docker) | Purpose-built, great Python client, horizontal scaling for enterprise |
| Graph DB | Neo4j Community (Docker) | Industry standard, Cypher query language, mature Python driver |
| Embeddings | all-MiniLM-L6-v2 via sentence-transformers (Docker) | 384 dims, fast, ~80MB, good quality for code/text |
| MCP Protocol | Manual JSON-RPC stdio | Minimal deps, proven pattern (mempalace uses same), no web framework needed |
| CLI | Click | Standard Python CLI framework |
| Config | TOML | Python stdlib (3.11+), simple, no security footguns |
| Package | pip + optional npm wrapper | pip for Python ecosystem, npm for JS/TS developers |
| License | MIT | Maximum adoption for OSS |

## Design Decisions

**Why manual JSON-RPC instead of FastMCP?** FastMCP pulls in pydantic-settings, uvicorn, starlette, sse-starlette. For a stdio-only MCP server, the JSON-RPC loop is ~100 lines. Keeps dependency tree minimal.

**Why separate embedding Docker container?** sentence-transformers pulls ~2GB of dependencies (PyTorch). Keeping it in Docker means `pip install collivind-memory` stays lightweight (~50MB).

**Why TOML over YAML?** TOML is Python stdlib (3.11+), simpler, no arbitrary code execution risk, matches pyproject.toml convention.

**Why leverage the AI assistant for extraction instead of a local LLM?** The coding assistant already has an LLM running. Using it to extract memories means zero extra compute cost and the extraction quality matches the LLM quality the user is already paying for.

**Why async storage interfaces with sync MCP server?** Storage interfaces are async (Qdrant/Neo4j both have async clients). MCP server bridges with asyncio.run(). Keeps storage non-blocking while avoiding fully async MCP complexity.
