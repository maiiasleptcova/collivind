"""The entity merge scenario against a DISPOSABLE Neo4j (the test wipes it).

    docker run -d --name collivind-test-neo4j -p 17687:7687 -e NEO4J_AUTH=neo4j/testpass123 neo4j:5.26-community
    COLLIVIND_TEST_NEO4J_URI=bolt://localhost:17687 COLLIVIND_TEST_NEO4J_PASSWORD=testpass123 \\
        uv run pytest tests/integration/test_entity_merge_neo4j.py
"""

import os

import pytest
from entity_merge_scenario import GONE, KEEP, OTHERS, build, edge_set, expected

from collivind.config import Neo4jConfig
from collivind.engine.entity_duplicates import find_duplicates
from collivind.engine.entity_merge import apply_merge, plan_merge
from collivind.models import EntityCreate, EntityType, RelationshipCreate, RelType
from collivind.storage.neo4j_store import Neo4jGraphStore

URI = os.environ.get("COLLIVIND_TEST_NEO4J_URI")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URI, reason="set COLLIVIND_TEST_NEO4J_URI to a disposable Neo4j"),
]


@pytest.fixture
def graph():
    s = Neo4jGraphStore(
        Neo4jConfig(
            uri=URI,
            user=os.environ.get("COLLIVIND_TEST_NEO4J_USER", "neo4j"),
            password=os.environ.get("COLLIVIND_TEST_NEO4J_PASSWORD", ""),
        )
    )
    s.initialize()
    s.clear_all()
    yield s
    s.clear_all()
    s.close()


def test_merge_on_neo4j(graph, tmp_path):
    mem = build(graph)
    before = edge_set(graph)
    plan = plan_merge(graph, KEEP, OTHERS)
    assert edge_set(graph) == before  # planning reads only
    apply_merge(graph, plan, tmp_path / "b.jsonl")
    assert edge_set(graph) == sorted(expected(mem))
    assert not GONE & {e.id for e in graph.iter_entities()}
    assert graph.get_entity(KEEP).properties["aliases"] == list(OTHERS)
    (uses,) = graph.edges_of("flask")
    assert (uses.source_id, uses.properties, uses.confidence) == ("cybergine", {"note": "web framework"}, 0.7)


def test_neo4j_delete_entity_refuses_while_edges_remain(graph):
    build(graph)
    with pytest.raises(ValueError, match="still has"):
        graph.delete_entity("v5_flask")


def test_duplicates_read_neo4j(graph):
    mem = build(graph)
    for name in ("PR 774", "PR #774"):
        graph.create_entity(EntityCreate(name=name, type=EntityType.CONCEPT))
    graph.create_relationship(
        RelationshipCreate(source_id=mem["m4"], target_id="cybergine_v5_flask", type=RelType.ABOUT)
    )
    graph.create_relationship(RelationshipCreate(source_id=mem["m4"], target_id="v5_flask", type=RelType.ABOUT))
    pairs = {frozenset((p["a"], p["b"])): p for g in find_duplicates(graph) for p in g["pairs"]}
    assert pairs[frozenset(("PR 774", "PR #774"))]["evidence"] == [{"reason": "same-name"}]
    contains = pairs[frozenset(("v5_flask", "Cybergine v5_flask"))]
    assert (contains["evidence"], contains["shared_memories"]) == ([{"reason": "contains"}], 1)
