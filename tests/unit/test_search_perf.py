"""Batched graph expansion: same results as the per-item path, and no N+1.

Seeds come from benchmarks/search_bench.py, so the test and the published
numbers exercise the same data shape (Zipf-skewed entities, some far past
the 20-per-entity expansion limit).
"""

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

from collivind.engine import search_engine as search_engine_module
from collivind.models import MemoryCategory, MemoryNode, SearchQuery
from collivind.models.relationship import RelationshipEdge, RelType
from collivind.storage.interfaces import GraphStore

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "search_bench.py"
_spec = importlib.util.spec_from_file_location("search_bench", _BENCH)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

MEMORIES = 1500


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    data_dir = str(tmp_path_factory.mktemp("perf"))
    graph, vector, embedder, texts = bench.seed_stores(data_dir, MEMORIES)
    legacy = bench.LegacyGraphStore(data_dir)  # same rows, per-item lookups
    yield {"graph": graph, "legacy": legacy, "vector": vector, "embedder": embedder, "texts": texts}
    for store in (graph, legacy, vector):
        store.close()


@pytest.fixture
def frozen_now(monkeypatch):
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(search_engine_module, "datetime", _Frozen)


def _ranked(engine, query, limit=10):
    return [
        (r.memory.id, r.score, r.vector_score, r.graph_score, r.related_entities)
        for r in engine.search(SearchQuery(query=query, limit=limit, include_graph=True))
    ]


def test_batched_search_ranks_exactly_like_the_per_item_path(seeded, frozen_now):
    batched = bench.build_engine(seeded["graph"], seeded["vector"], seeded["embedder"])
    legacy = bench.build_engine(seeded["legacy"], seeded["vector"], seeded["embedder"])

    expanded_somewhere = False
    for query in bench.make_queries(seeded["texts"], 40):
        for limit in (5, 10):
            assert _ranked(batched, query, limit) == _ranked(legacy, query, limit), query
        expanded_somewhere |= any(r[3] > 0 for r in _ranked(batched, query))
    assert expanded_somewhere, "no query exercised graph expansion; the comparison proves nothing"


def test_batched_expansion_matches_per_item_expansion(seeded):
    batched = bench.build_engine(seeded["graph"], seeded["vector"], seeded["embedder"]).graph_engine
    legacy = bench.build_engine(seeded["legacy"], seeded["vector"], seeded["embedder"]).graph_engine
    seeds = [m.id for m in list(seeded["graph"].iter_memories())[:60:2]]

    new, old = batched.get_expanded_memories(seeds), legacy.get_expanded_memories(seeds)

    assert list(new) == list(old)  # insertion order feeds the stable score sort
    assert {k: (v["memory"], set(v["shared_entities"])) for k, v in new.items()} == {
        k: (v["memory"], set(v["shared_entities"])) for k, v in old.items()
    }


def test_store_batches_match_the_per_item_methods(tmp_path):
    graph = bench.SqliteGraphStore(str(tmp_path))
    a, b = (MemoryNode(id=i, content=i, summary=i, category=MemoryCategory.FACT) for i in ("mem-a", "mem-b"))
    graph.restore_memory(a)
    graph.restore_memory(b)
    graph.create_entity(bench.EntityNode(name="Fast API", type=bench.EntityType.CONCEPT))
    for i, (src, dst, rel) in enumerate(
        [
            ("mem-a", "fast_api", RelType.ABOUT),
            ("mem-a", "fast_api", RelType.MENTIONS),  # twice to one entity: takes two LIMIT slots
            ("mem-b", "fast_api", RelType.ABOUT),
            ("mem-a", "mem-b", RelType.RELATES_TO),
        ]
    ):
        graph.restore_relationship(RelationshipEdge(id=f"r{i}", source_id=src, target_id=dst, type=rel))

    ids = ["mem-a", "mem-b", "fast_api", "missing", "mem-a"]
    for direction in ("OUT", "IN", "BOTH"):
        for rel_types in ([], ["ABOUT", "MENTIONS"]):
            assert graph.get_neighbors_batch(ids, rel_types, direction) == GraphStore.get_neighbors_batch(
                graph, ids, rel_types, direction
            )
    names = ["Fast API", "fast-api", "fast api", "nobody"]
    for limit in (1, 2, 10):
        assert graph.find_related_memories_batch(names, limit) == GraphStore.find_related_memories_batch(
            graph, names, limit
        )
    graph.close()


def test_graph_statements_per_search_do_not_grow_with_seeds_or_entities(seeded):
    """The N+1 regression guard: deterministic, so it cannot flake.

    The per-item path ran one get_neighbors per seed plus one
    find_related_memories per entity per seed (100+ statements here). What
    is left is one get_memory per vector hit (limit * 3) plus two batches.
    """
    engine = bench.build_engine(seeded["graph"], seeded["vector"], seeded["embedder"])
    query = bench.make_queries(seeded["texts"], 1)[0]
    limit = 10

    batched = bench.count_statements(seeded["graph"], engine, query)
    legacy_engine = bench.build_engine(seeded["legacy"], seeded["vector"], seeded["embedder"])
    per_item = bench.count_statements(seeded["legacy"], legacy_engine, query)

    assert batched <= limit * 3 + 2
    assert per_item > batched * 2, "fixture too small to show the N+1; the guard would be toothless"


@pytest.mark.perf
def test_search_latency_has_headroom_under_the_500ms_target(seeded):
    # generous: the 10k benchmark p95 is ~50 ms; this is 1.5k on a shared CI box
    engine = bench.build_engine(seeded["graph"], seeded["vector"], seeded["embedder"])
    stats = bench.run_queries(engine, bench.make_queries(seeded["texts"], 30))
    assert stats["p50_ms"] < 250, stats
