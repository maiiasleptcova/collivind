"""Typed edges with meanings: agents can state how entities fit together, and readers see it in words."""

from unittest.mock import MagicMock

import pytest

from collivind.config import CollivindConfig
from collivind.engine.dedup import Neighbour
from collivind.engine.memory_manager import LINKABLE, MemoryManager
from collivind.models import MemoryCreate, RelationshipCreate, RelType
from collivind.models.memory import MemoryCategory
from collivind.models.relations import (
    ENTITY_RELATIONS,
    MAX_NOTE,
    MEANINGS,
    EntityRelation,
    parse_relationship,
    phrase,
)
from collivind.storage.graph_sqlite import SqliteGraphStore
from collivind.web.api import MemoryAPI, route
from collivind.web.graph import build_graph


@pytest.fixture
def manager(tmp_path):
    vectors, embedder = MagicMock(), MagicMock()
    vectors.search.return_value = []
    embedder.embed.return_value = [0.1]
    return MemoryManager(vectors, SqliteGraphStore(str(tmp_path / "graph.db")), embedder, CollivindConfig())


def test_every_linkable_and_rel_type_has_a_meaning():
    for types in LINKABLE.values():
        assert set(types) <= set(MEANINGS)
    assert {r.value for r in RelType} <= set(MEANINGS)


def test_phrases_read_from_either_end():
    assert phrase("USES") == "uses" and phrase("USES", outgoing=False) == "is used by"
    assert phrase("PART_OF", outgoing=False) == "contains"


def test_parse_named_entity_relation_and_memory_edge():
    rel = parse_relationship({"source": "Neo4j", "type": "PART_OF", "target": "docker stack", "note": " graph  store "})
    assert rel == EntityRelation("Neo4j", "PART_OF", "docker stack", "graph store")
    edge = parse_relationship({"target_id": "m1", "type": "RESOLVES", "note": "the fix"})
    assert isinstance(edge, RelationshipCreate) and edge.properties == {"note": "the fix"}


@pytest.mark.parametrize(
    "raw",
    [
        {"source": "a", "type": "ABOUT", "target": "b"},  # not an entity-entity type
        {"source": "a", "type": "USES"},  # missing target
        {"source": "a", "type": "USES", "target": "b", "note": "x" * (MAX_NOTE + 1)},
        {"source": "a", "type": "SOMETHING", "target": "b"},
    ],
)
def test_parse_rejects(raw):
    with pytest.raises(ValueError):
        parse_relationship(raw)


def _save_with(manager, relationships):
    return manager.batch_add_memories(
        [
            {
                "content": "collivind's docker mode stores graph data in Neo4j",
                "summary": "docker mode uses Neo4j",
                "category": "architecture",
                "entities": [{"name": "collivind", "type": "project"}, {"name": "Neo4j", "type": "service"}],
                "relationships": relationships,
            }
        ]
    )


def test_a_save_writes_entity_to_entity_edges_and_get_entity_explains_them(manager):
    _save_with(
        manager,
        [
            {"source": "collivind", "type": "USES", "target": "Neo4j", "note": "graph store in docker mode"},
            {"source": "Neo4j", "type": "PART_OF", "target": "docker stack"},  # unknown name: created
        ],
    )
    neo = manager.get_entity("Neo4j")
    rels = {(r["phrase"], r["other_id"], r["note"]) for r in neo["relations"]}
    assert rels == {("is used by", "collivind", "graph store in docker mode"), ("is part of", "docker_stack", "")}
    assert manager.get_entity("docker stack")["entity"]["type"] == "concept"


def test_note_on_an_existing_link_is_kept_without_duplicating_it(manager):
    m = manager.add_memory(MemoryCreate(content="c", summary="s", category=MemoryCategory.FACT)).id
    n = manager.add_memory(MemoryCreate(content="d", summary="t", category=MemoryCategory.FACT)).id
    manager.link(m, n, RelType.RESOLVES)
    manager.link(m, n, RelType.RESOLVES, note="the decision that fixed it")
    edges = [e for e in manager.graph_store.iter_relationships() if e.type == RelType.RESOLVES]
    assert len(edges) == 1 and edges[0].properties["note"] == "the decision that fixed it"


def test_graph_payload_carries_note_and_origin_and_entity_hops(manager):
    _save_with(manager, [{"source": "Neo4j", "type": "PART_OF", "target": "docker stack", "note": "one of 4"}])
    g = build_graph(manager.graph_store)
    part = next(e for e in g["edges"] if e["type"] == "PART_OF")
    assert part["note"] == "one of 4" and part["origin"] == "extraction"
    # 'docker stack' has no memory of its own but is one entity hop from Neo4j
    assert "docker_stack" in {n["id"] for n in g["nodes"]}


def test_status_serves_meanings_and_link_route_takes_a_note(manager):
    api = MemoryAPI(manager, mode="embedded")
    _, status = route(api, "GET", "/api/status", {}, {})
    assert status["meanings"]["USES"]["inverse"] == "is used by"
    assert set(status["linkable"]["entity:entity"]) == set(ENTITY_RELATIONS)
    _save_with(manager, [])
    code, _ = route(
        api,
        "POST",
        "/api/links",
        {},
        {"source_id": "neo4j", "target_id": "collivind", "type": "INTERACTS_WITH", "note": "bolt"},
    )
    assert code == 201
    assert any(r["note"] == "bolt" for r in manager.get_entity("Neo4j")["relations"])


def _item(content, category="error", relationships=()):
    return {"content": content, "summary": content, "category": category, "relationships": list(relationships)}


def test_a_fix_links_to_the_error_saved_in_the_same_batch(manager):
    """The checkpoint saves an error and its fix in one call; target_index is how the fix points back."""
    ids = manager.batch_add_memories(
        [
            _item("docker exec fails: no space left on device"),
            _item("the colima vm disk was full", "fact"),
            _item(
                "resized the colima disk to 80 GB",
                "decision",
                [{"target_index": 0, "type": "RESOLVES", "note": "more disk"}],
            ),
        ]
    )
    manager.link(ids[0], ids[1], RelType.CAUSED_BY)
    edges = {(r.source_id, r.type.value, r.target_id) for r in manager.graph_store.iter_relationships()}
    assert (ids[2], "RESOLVES", ids[0]) in edges and (ids[0], "CAUSED_BY", ids[1]) in edges
    assert not any(t.startswith("batch-index:") for _, _, t in edges)


@pytest.mark.parametrize("index", [0, 1, -1, "0", True])
def test_target_index_must_point_earlier_in_the_batch(manager, index):
    """Itself, a later item, or a non-integer is refused before anything is stored."""
    with pytest.raises(ValueError, match="target_index"):
        # the fix comes first, so 0 is itself and 1 is a later item
        manager.batch_add_memories(
            [_item("its fix", "decision", [{"target_index": index, "type": "RESOLVES"}]), _item("an error")]
        )
    assert list(manager.graph_store.iter_memories()) == []


def test_target_index_outside_a_batch_is_refused():
    with pytest.raises(ValueError, match="batch"):
        parse_relationship({"target_index": 0, "type": "RESOLVES"})


def test_a_duplicate_fix_still_records_its_link(manager):
    """The fix text was saved before; the RESOLVES edge stated now must not vanish with the duplicate."""
    fix = manager.batch_add_memories([_item("resized the colima disk to 80 GB", "decision")])[0]
    manager.deduplicator.neighbours = lambda vector, project: (
        [Neighbour(fix, 0.99, {"content": "resized the colima disk to 80 GB"})] if vector == "dup" else []
    )
    manager.embedding_provider.embed.side_effect = lambda text: "dup" if "colima" in text else [0.1]
    ids = manager.batch_add_memories(
        [
            _item("docker exec fails: no space left on device"),
            _item("resized the colima disk to 80 GB", "decision", [{"target_index": 0, "type": "RESOLVES"}]),
        ]
    )
    assert ids[1] == fix
    edges = {(r.source_id, r.type.value, r.target_id) for r in manager.graph_store.iter_relationships()}
    assert (fix, "RESOLVES", ids[0]) in edges


def test_a_target_id_cannot_pose_as_a_batch_reference():
    with pytest.raises(ValueError, match="target_index"):
        parse_relationship({"target_id": "batch-index:0", "type": "RESOLVES"}, in_batch=True)
