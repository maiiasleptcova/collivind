# Collivind — AI Assistant Guide

## What This Is

Graph-based memory layer for AI coding assistants. Python package with MCP server integration.

## Build & Test

```bash
uv sync --dev              # Install deps
uv run pytest tests/ -v    # Run all tests
uv run pytest tests/unit/  # Unit tests only (no Docker)
uv run ruff check src/     # Lint
```

## Architecture

Layered: CLI / MCP Server → Engine (MemoryManager, SearchEngine) → Storage Interfaces → Backends

Three deployment modes selected by `config.mode`: `docker` (Qdrant + Neo4j + HTTP embeddings), `embedded` (SQLite vector + graph stores + local model, multi-process safe), `remote` (external services). Backend selection happens in `storage/factory.py`.

## Key Files

- `storage/interfaces.py` — ABCs defining the storage contract
- `storage/factory.py` — creates backends based on config mode
- `engine/memory_manager.py` — central orchestrator for all operations
- `mcp/server.py` — JSON-RPC stdio MCP server
- `mcp/tools.py` — tool definitions and handlers
- `config.py` — TOML config loading with dataclass defaults

## Conventions

- Python 3.11+, dataclasses (not pydantic), standard library where possible
- Build: uv + hatchling
- Tests: pytest, mocks for unit tests, real backends for integration
- Docker interaction via subprocess (no Python docker SDK)
- Immutable data patterns — return new objects, don't mutate
- Storage backends must implement the ABCs in `storage/interfaces.py`
