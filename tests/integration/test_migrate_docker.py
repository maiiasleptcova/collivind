"""embedded -> docker -> embedded through real Qdrant + Neo4j.

Opt-in and never against a live install: set MIGRATE_TEST_QDRANT_PORT,
MIGRATE_TEST_NEO4J_URI and MIGRATE_TEST_NEO4J_PASSWORD to a THROWAWAY stack on
non-default ports. The test wipes that Neo4j database and Qdrant collection.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from test_migrate import DIM, HashEmbedder, _graph_snapshot, _stores, _vectors, populated  # noqa: E402,F401

from collivind.config import CollivindConfig, Neo4jConfig, QdrantConfig  # noqa: E402
from collivind.engine.memory_manager import MemoryManager  # noqa: E402
from collivind.engine.migrate import Stores, migrate  # noqa: E402
from collivind.models import EntityCreate, MemoryCreate  # noqa: E402
from collivind.models.entity import EntityType  # noqa: E402
from collivind.models.memory import MemoryCategory  # noqa: E402
from collivind.storage.neo4j_store import Neo4jGraphStore  # noqa: E402
from collivind.storage.qdrant_store import QdrantVectorStore  # noqa: E402

QDRANT_PORT = os.environ.get("MIGRATE_TEST_QDRANT_PORT")
NEO4J_URI = os.environ.get("MIGRATE_TEST_NEO4J_URI")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (QDRANT_PORT and NEO4J_URI), reason="needs a throwaway Qdrant+Neo4j (see module doc)"),
]


@pytest.fixture
def docker_stores():
    if QDRANT_PORT == "6333" or NEO4J_URI.endswith(":7687"):
        pytest.fail("refusing default ports: they are where a live collivind install listens")
    graph = Neo4jGraphStore(Neo4jConfig(uri=NEO4J_URI, password=os.environ["MIGRATE_TEST_NEO4J_PASSWORD"]))
    graph.clear_all()
    qcfg = QdrantConfig(port=int(QDRANT_PORT), collection_name="migrate_test")
    vector = QdrantVectorStore(qcfg, DIM)
    vector.delete_collection()
    vector.initialize()
    stores = Stores(vector=vector, graph=graph, embedder=None)
    yield stores
    graph.clear_all()
    vector.delete_collection()
    graph.close()
    vector.close()


def test_embedded_to_docker_and_back_is_lossless(populated, docker_stores, tmp_path):  # noqa: F811
    src, ids = populated

    first = migrate(src, docker_stores, reembed=False, dry_run=False)
    again = migrate(src, docker_stores, reembed=False, dry_run=False)
    assert first.ok and again.ok
    assert {c.kind: c.target for c in again.counts} == {c.kind: c.source for c in first.counts}

    assert [m.id for m in docker_stores.graph.get_version_chain(ids["new"])] == [ids["old"], ids["new"]]
    assert {m.id for m in docker_stores.graph.find_related_memories("Kafka")} == {ids["new"]}

    back = _stores(tmp_path / "back")
    assert migrate(docker_stores, back, reembed=False, dry_run=False).ok

    mem_s, ent_s, rel_s = _graph_snapshot(src)
    mem_b, ent_b, rel_b = _graph_snapshot(back)
    assert mem_b == mem_s
    assert ent_b == ent_s
    assert set(rel_b) == {r for r in rel_s if r[2] != "gone"}
    src_vecs, back_vecs = _vectors(src), _vectors(back)
    assert set(back_vecs) == set(src_vecs)
    for vid, (vec, payload) in src_vecs.items():
        assert back_vecs[vid][1] == payload
        assert abs(float(back_vecs[vid][0] @ vec) - 1.0) < 1e-5


def test_docker_native_edges_without_ids_migrate_idempotently(docker_stores, tmp_path):
    """Edges written by Neo4j's own create_relationship/invalidate carry no id."""
    mgr = MemoryManager(docker_stores.vector, docker_stores.graph, HashEmbedder(), CollivindConfig())
    mgr.search_engine.find_contradictions = lambda memory: []

    def add(content):
        return mgr.add_memory(
            MemoryCreate(content=content, summary=content, category=MemoryCategory.FACT),
            entities=[EntityCreate(name="Neo4j", type=EntityType.SERVICE)],
        )

    old, new = add("graph lives in sqlite"), add("graph lives in neo4j")
    mgr.invalidate(old.id, new.id, "moved")

    dst = _stores(tmp_path / "dst")
    first = migrate(docker_stores, dst, reembed=False, dry_run=False)
    edges_once = _graph_snapshot(dst)[2]
    second = migrate(docker_stores, dst, reembed=False, dry_run=False)

    assert first.ok and second.ok
    assert _graph_snapshot(dst)[2] == edges_once
    counts = {c.kind: c.target for c in second.counts}
    assert counts["relationships"] == 3  # 2 ABOUT + 1 SUPERSEDES
    supersedes = [r for r in dst.graph.iter_relationships() if r.type.value == "SUPERSEDES"]
    assert [(r.source_id, r.target_id, r.properties.get("reason")) for r in supersedes] == [(new.id, old.id, "moved")]
    assert [m.id for m in dst.graph.get_version_chain(new.id)] == [old.id, new.id]
