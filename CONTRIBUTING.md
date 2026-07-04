# Contributing to Collivind

## Setup

```bash
git clone https://github.com/yourusername/collivind.git
cd collivind
uv sync --dev
```

For embedded mode development:

```bash
uv sync --dev --extra embedded
```

## Running Tests

```bash
# Unit tests (no Docker required)
uv run pytest tests/unit/ -v

# All tests (requires Docker services running)
collivind init
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=collivind --cov-report=term-missing
```

## Code Quality

```bash
# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run mypy src/collivind/
```

## Project Structure

```
src/collivind/
  config.py           # Configuration loading (TOML)
  exceptions.py       # Custom exception hierarchy
  cli/                # Click CLI commands
  docker/             # Docker Compose management
  models/             # Data models (dataclasses)
  storage/            # Storage backends (Qdrant, Neo4j, SQLite, embeddings)
    interfaces.py     # ABC contracts — all backends implement these
    factory.py        # Mode-based backend selection
  engine/             # Business logic (MemoryManager, SearchEngine, Dedup)
  mcp/                # MCP server (JSON-RPC stdio)
tests/
  unit/               # Fast, no external deps
  integration/        # Require Docker services
  mcp/                # MCP protocol tests
```

## Adding a Storage Backend

1. Implement the relevant ABC from `storage/interfaces.py` (VectorStore, GraphStore, or EmbeddingProvider)
2. Add a factory branch in `storage/factory.py`
3. Add a mode value in `config.py`
4. Write tests in `tests/unit/`

## Pull Requests

- One feature per PR
- Include tests (80%+ coverage target)
- Run `ruff check` and `pytest` before submitting
- Use conventional commit messages: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`
