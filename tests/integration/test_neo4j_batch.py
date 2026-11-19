"""Neo4j batched lookups and index use, against a DISPOSABLE Neo4j only.

The test wipes the database, so it never falls back to the default
localhost config. Point it at a throwaway container explicitly:

    docker run -d --name collivind-test-neo4j -p 17687:7687 -e NEO4J_AUTH=neo4j/testpass123 neo4j:5.26-community
    COLLIVIND_TEST_NEO4J_URI=bolt://localhost:17687 COLLIVIND_TEST_NEO4J_PASSWORD=testpass123 \\
        uv run pytest tests/integration/test_neo4j_batch.py
"""

import importlib.util
import os
from pathlib import Path

import pytest

from collivind.config import Neo4jConfig
from collivind.models import EntityCreate, EntityType, MemoryCategory, MemoryCreate, RelationshipCreate, RelType
from collivind.models.relationship import RelationshipEdge
from collivind.storage.interfaces import GraphStore
from collivind.storage.neo4j_store import Neo4jGraphStore

URI = os.environ.get("COLLIVIND_TEST_NEO4J_URI")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URI, reason="set COLLIVIND_TEST_NEO4J_URI to a disposable Neo4j"),
]

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks" / "search_bench.py"
_spec = importlib.util.spec_from_file_location("search_bench", _BENCH)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


class _NoVectors:
    def upsert(self, *args):
        pass


@pytest.fixture(scope="module")
def store():
    s = Neo4jGraphStore(
        Neo4jConfig(
            uri=URI,
            user=os.environ.get("COLLIVIND_TEST_NEO4J_USER", "neo4j"),
            password=os.environ.get("COLLIVIND_TEST_NEO4J_PASSWORD", ""),
        )
    )
    s.initialize()
    s.clear_all()
    bench.seed(s, _NoVectors(), bench.TokenHashEmbedder(), 300)
    yield s
    s.clear_all()
    s.close()


def _seed_ids(store, n=30):
    return [m.id for m in list(store.iter_memories())[:n]]


def test_neighbors_batch_matches_per_item(store):
    ids = _seed_ids(store) + ["missing"]
    for direction in ("OUT", "IN", "BOTH"):
        new = store.get_neighbors_batch(ids, ["ABOUT", "MENTIONS"], direction)
        old = GraphStore.get_neighbors_batch(store, ids, ["ABOUT", "MENTIONS"], direction)
        assert list(new) == list(old)
        for nid in ids:
            key = lambda n: (n["id"], n["relationships"][0].get("id"))  # noqa: E731
            assert sorted(new[nid], key=key) == sorted(old[nid], key=key)


def test_related_batch_matches_per_item(store):
    names = [f"Ent {e}" for e in range(40)] + ["ent-1", "nobody"]
    # past the limit Neo4j picks which memories (no ORDER BY on either path),
    # so compare whole sets with a limit above every degree, sizes below it
    full_new = store.find_related_memories_batch(names, limit=10_000)
    full_old = GraphStore.find_related_memories_batch(store, names, limit=10_000)
    assert {k: sorted(m.id for m in v) for k, v in full_new.items()} == {
        k: sorted(m.id for m in v) for k, v in full_old.items()
    }
    assert any(len(v) > 20 for v in full_new.values()), "no hot entity: the limit is not exercised"
    capped = store.find_related_memories_batch(names, limit=20)
    for name, mems in capped.items():
        assert len(mems) == min(20, len(full_new[name]))
        assert {m.id for m in mems} <= {m.id for m in full_new[name]}


def test_write_and_lookup_queries_use_the_id_indexes(store):
    """No AllNodesScan: the unlabeled `MATCH (n {id: ...})` this replaced scanned every node."""
    mem = store.create_memory(MemoryCreate(content="idx probe", summary="idx", category=MemoryCategory.FACT))
    ent = store.create_entity(EntityCreate(name="Index Probe", type=EntityType.CONCEPT))

    recorded = []
    real_session = store.driver.session

    class _Recording:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

        def run(self, query, **params):
            recorded.append((query, params))
            return self._inner.run(query, **params)

    store.driver.session = lambda **kw: _Recording(real_session(**kw))
    try:
        store.create_relationship(RelationshipCreate(source_id=mem.id, target_id=ent.id, type=RelType.ABOUT))
        store.restore_relationship(
            RelationshipEdge(id="probe-edge", source_id=mem.id, target_id=ent.id, type=RelType.MENTIONS)
        )
        store.get_neighbors(mem.id, ["ABOUT", "MENTIONS"])
        store.get_neighbors_batch([mem.id], ["ABOUT", "MENTIONS"])
        store.find_related_memories_batch(["Index Probe"], limit=20)
    finally:
        store.driver.session = real_session

    assert {n["id"] for n in store.get_neighbors(mem.id, ["ABOUT", "MENTIONS"])} == {ent.id}
    with real_session(database=store.config.database) as session:
        for query, params in recorded:
            plan = session.run("EXPLAIN " + query, **params).consume().plan
            ops = []
            stack = [plan]
            while stack:
                node = stack.pop()
                ops.append(node.get("operatorType", ""))
                stack.extend(node.get("children", []))
            assert not any(op.startswith("AllNodesScan") for op in ops), (query, ops)
