"""Invalidation must reach the vector index, not only the graph.

With the graph down, search rebuilds hits from vector payloads alone (#39).
A memory invalidated only in the graph kept a payload claiming it was live,
so degraded search kept returning it. These run the real SQLite stores and
then kill the graph, the way it happens in production.
"""

from unittest.mock import MagicMock

import pytest

from collivind.config import CollivindConfig, QdrantConfig
from collivind.engine.memory_manager import MemoryManager
from collivind.exceptions import CollivindError
from collivind.models import MemoryCategory, MemoryCreate, SearchQuery
from collivind.storage.graph_sqlite import SqliteGraphStore
from collivind.storage.vector_sqlite import SqliteVectorStore


@pytest.fixture
def mgr(tmp_path):
    embed = MagicMock()
    embed.embed.side_effect = lambda text: [1.0, float(len(text) % 7), 0.5]
    embed.embed_query.return_value = [1.0, 1.0, 0.5]
    vectors = SqliteVectorStore(str(tmp_path), QdrantConfig(), dimension=3)
    graph = SqliteGraphStore(str(tmp_path))
    config = CollivindConfig()
    config.search.similarity_threshold = 0.0
    config.search.dedup_threshold = 1.01  # keep every add distinct
    return MemoryManager(vectors, graph, embed, config)


def _add(mgr, content):
    return mgr.add_memory(MemoryCreate(content=content, summary=content, category=MemoryCategory.FACT))


def _degraded_ids(mgr):
    for method in ("get_memory", "find_related_memories", "get_neighbors"):
        setattr(mgr.graph_store, method, MagicMock(side_effect=CollivindError("graph down")))
    results, degraded = mgr.search_engine.search_with_status(SearchQuery(query="anything", limit=10))
    assert degraded
    return {r.memory.id for r in results}


def test_invalidated_memory_is_not_served_by_degraded_search(mgr):
    keep, gone = _add(mgr, "keep this fact"), _add(mgr, "this one gets invalidated")

    mgr.invalidate(gone.id, "", "wrong")

    assert _degraded_ids(mgr) == {keep.id}


def test_updating_an_invalidated_memory_does_not_resurrect_its_vector(mgr):
    keep, gone = _add(mgr, "keep this fact"), _add(mgr, "this one gets invalidated")
    mgr.invalidate(gone.id, "", "wrong")

    mgr.update_memory(gone.id, tags=["late-edit"])

    # the live index serves current knowledge only; history stays in the graph
    assert {i for i, _, _ in mgr.vector_store.iter_all()} == {keep.id}


def test_forgotten_memory_is_not_served_by_degraded_search(mgr):
    keep, gone = _add(mgr, "keep this fact"), _add(mgr, "this one gets forgotten")

    mgr.forget(gone.id)

    assert _degraded_ids(mgr) == {keep.id}
