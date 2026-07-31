"""Regression: #8 added a health probe to create_all_backends, and the SQLite
graph store's probe queries `memories`. On a data dir that has never been
initialized that table does not exist, so the whole mode was reported
unavailable instead of simply empty."""

import tempfile

from collivind.config import CollivindConfig
from collivind.storage.factory import create_all_backends
from collivind.storage.graph_sqlite import SqliteGraphStore


def test_fresh_data_dir_is_usable_without_running_init():
    """A never-initialized store is empty, not broken."""
    config = CollivindConfig(mode="embedded", data_dir=tempfile.mkdtemp())
    vector, graph, embeddings = create_all_backends(config)
    assert graph.health_check()["status"] == "ok"
    assert graph.get_timeline("default", limit=1) == []


def test_graph_store_creates_its_schema_on_construction():
    store = SqliteGraphStore(data_dir=tempfile.mkdtemp())
    tables = {r[0] for r in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "memories" in tables


def test_construction_is_idempotent_and_preserves_data():
    """Re-opening an existing store must not wipe it."""
    from collivind.models import MemoryCategory, MemoryCreate

    data_dir = tempfile.mkdtemp()
    first = SqliteGraphStore(data_dir=data_dir)
    first.initialize()
    created = first.create_memory(MemoryCreate(content="keep me", summary="keep me", category=MemoryCategory.FACT))
    first.close()

    second = SqliteGraphStore(data_dir=data_dir)
    assert second.get_memory(created.id) is not None, "re-opening must not drop existing rows"
