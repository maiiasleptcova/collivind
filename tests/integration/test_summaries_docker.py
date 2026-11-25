"""Summary memories on the docker backends (Neo4j + Qdrant), and embedded <-> docker migration.

Opt-in and never against a live install: point MIGRATE_TEST_QDRANT_PORT,
MIGRATE_TEST_NEO4J_URI and MIGRATE_TEST_NEO4J_PASSWORD at a THROWAWAY stack on
non-default ports. The test wipes that Neo4j database and Qdrant collection.

    docker run -d --rm --name cv-n -p 27687:7687 -e NEO4J_AUTH=neo4j/testpass123 neo4j:5.26-community
    docker run -d --rm --name cv-q -p 26333:6333 qdrant/qdrant
    MIGRATE_TEST_QDRANT_PORT=26333 MIGRATE_TEST_NEO4J_URI=bolt://localhost:27687 \\
        MIGRATE_TEST_NEO4J_PASSWORD=testpass123 uv run pytest tests/integration/test_summaries_docker.py
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from test_summaries import SUMMARY, WordEmbedder, _add, _manager, _summary_state, _trip_session  # noqa: E402

from collivind.config import CollivindConfig, Neo4jConfig, QdrantConfig  # noqa: E402
from collivind.engine import summaries  # noqa: E402
from collivind.engine.memory_manager import MemoryManager  # noqa: E402
from collivind.engine.migrate import Stores, migrate  # noqa: E402
from collivind.models import MemoryCategory, SearchQuery  # noqa: E402
from collivind.storage.neo4j_store import Neo4jGraphStore  # noqa: E402
from collivind.storage.qdrant_store import QdrantVectorStore  # noqa: E402

QDRANT_PORT = os.environ.get("MIGRATE_TEST_QDRANT_PORT")
NEO4J_URI = os.environ.get("MIGRATE_TEST_NEO4J_URI")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (QDRANT_PORT and NEO4J_URI), reason="needs a throwaway Qdrant+Neo4j (see module doc)"),
]


@pytest.fixture
def docker_mgr():
    if QDRANT_PORT == "6333" or NEO4J_URI.endswith(":7687"):
        pytest.fail("refusing default ports: they are where a live collivind install listens")
    graph = Neo4jGraphStore(Neo4jConfig(uri=NEO4J_URI, password=os.environ["MIGRATE_TEST_NEO4J_PASSWORD"]))
    graph.clear_all()
    vector = QdrantVectorStore(
        QdrantConfig(port=int(QDRANT_PORT), collection_name="summaries_test"), WordEmbedder().dimension
    )
    vector.delete_collection()
    vector.initialize()
    graph.initialize()
    yield MemoryManager(vector, graph, WordEmbedder(), CollivindConfig())
    graph.clear_all()
    vector.delete_collection()
    graph.close()
    vector.close()


def test_write_rewrite_and_route_on_neo4j_and_qdrant(docker_mgr):
    mgr = docker_mgr
    ids = _trip_session(mgr)
    group = summaries.Group("session", "s1")
    assert {m.id for m in mgr.graph_store.find_memories("p", session_id="s1")} == set(ids)
    first = summaries.save_summary(mgr, "p", group, SUMMARY)
    late = _add(mgr, "Alice will adopt a cat after the trip").id
    assert [m.id for m in summaries.unsummarized(mgr, "p", group)] == [late]
    second = summaries.save_summary(mgr, "p", group, SUMMARY + "\n- cat")
    assert second.previous_version_id == first.id and second.category == MemoryCategory.SUMMARY
    assert [s.id for s in mgr.graph_store.find_memories("p", category="summary")] == [second.id]
    assert {m.id for m in summaries.members_of(mgr, [second.id])[second.id]} == set(ids) | {late}
    assert [s.id for s in summaries.summaries_of(mgr, late)] == [second.id]
    hits = mgr.search(SearchQuery(query="Will Alice bring her dog to Paris?", project_id="p", limit=4))
    assert hits[0].memory.id == second.id and ids[2] in [h.memory.id for h in hits[1:3]]


def test_migrate_embedded_to_docker_and_back_keeps_summaries(docker_mgr, tmp_path):
    src = _manager(tmp_path / "src")
    _trip_session(src)
    group = summaries.Group("session", "s1")
    summaries.save_summary(src, "p", group, SUMMARY)
    _add(src, "Alice will adopt a cat after the trip")
    summaries.save_summary(src, "p", group, SUMMARY + "\n- cat")
    before = _summary_state(src)

    def stores(m):
        return Stores(m.vector_store, m.graph_store, m.embedding_provider)

    assert migrate(stores(src), stores(docker_mgr), reembed=False, dry_run=False).ok
    assert _summary_state(docker_mgr) == before
    back = _manager(tmp_path / "back")
    assert migrate(stores(docker_mgr), stores(back), reembed=False, dry_run=False).ok
    assert _summary_state(back) == before
    assert len(back.get_version_chain(summaries.live_summary(back, "p", group).id)) == 2
