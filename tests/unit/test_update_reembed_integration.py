"""update_memory's re-embed, against a real graph store.

The mock-based tests in test_memory_verbs.py cannot catch a wrong lookup: a
MagicMock returns its configured neighbours whatever node id, relationship
types or direction it is handed. Pointing `get_neighbors` at the wrong node
therefore left the whole suite green while the entity segment vanished from
every re-embedded vector. These run the real SqliteGraphStore so the query
itself has to be right.
"""

from unittest.mock import MagicMock

import pytest

from collivind.config import CollivindConfig
from collivind.engine.memory_manager import MemoryManager
from collivind.models import EntityCreate, MemoryCreate
from collivind.models.entity import EntityType
from collivind.models.memory import MemoryCategory
from collivind.storage.graph_sqlite import SqliteGraphStore


@pytest.fixture
def manager(tmp_path):
    """A real graph store; the vector side stays mocked so we can read back
    exactly what text was embedded."""
    graph = SqliteGraphStore(str(tmp_path / "graph.db"))
    vectors, embedder = MagicMock(), MagicMock()
    vectors.search.return_value = []
    embedder.embed.return_value = [0.1]
    return MemoryManager(vectors, graph, embedder, CollivindConfig()), embedder


def _add(manager, **kw):
    return manager.add_memory(
        MemoryCreate(
            content=kw.get("content", "we moved the index onto managed hardware"),
            summary=kw.get("summary", "index moved"),
            category=MemoryCategory.FACT,
            tags=kw.get("tags", ["infra"]),
        ),
        entities=[
            EntityCreate(name="Qdrant", type=EntityType.SERVICE),
            EntityCreate(name="data-integrity", type=EntityType.CONCEPT),
        ],
    )


def test_reembed_finds_the_linked_entities_through_a_real_store(manager):
    """Sabotage this by querying the wrong node, the wrong relationship type or
    the wrong direction and this test fails — the mocked ones do not."""
    mgr, embedder = manager
    node = _add(mgr)
    embedder.embed.reset_mock()

    mgr.update_memory(node.id, tags=["infra", "migration"])

    text = embedder.embed.call_args.args[0]
    assert "migration" in text, text
    # SQLite's get_neighbors returns only the slugged id, so the names come back
    # lower-cased with hyphens flattened. Asserting on the slugged forms keeps
    # the test honest about what this backend can actually recover.
    assert "qdrant" in text.lower(), f"the entity segment was dropped: {text}"
    assert "data integrity" in text.lower(), f"the entity segment was dropped: {text}"


def test_reembed_also_follows_mentions_edges(manager):
    """add_memory only ever creates ABOUT edges, so a fixture built from it
    cannot tell whether MENTIONS is in the rel-type list. Wire one directly."""
    from collivind.models import RelationshipCreate, RelType

    mgr, embedder = manager
    node = _add(mgr)
    ent = mgr.graph_store.create_entity(EntityCreate(name="Redis", type=EntityType.SERVICE))
    mgr.graph_store.create_relationship(
        RelationshipCreate(source_id=node.id, target_id=ent.id, type=RelType.MENTIONS, confidence=1.0)
    )
    embedder.embed.reset_mock()

    mgr.update_memory(node.id, tags=["infra", "migration"])

    text = embedder.embed.call_args.args[0]
    assert "redis" in text.lower(), f"MENTIONS edges were not followed: {text}"


def test_a_failing_entity_lookup_leaves_the_vector_alone(manager):
    """A degraded vector is worse than a stale one: without the entity segment
    the memory stops matching entity-bearing queries, and nothing says so."""
    mgr, embedder = manager
    node = _add(mgr)
    embedder.embed.reset_mock()
    mgr.vector_store.upsert.reset_mock()
    mgr.graph_store.get_neighbors = MagicMock(side_effect=RuntimeError("graph down"))

    result = mgr.update_memory(node.id, tags=["infra", "migration"])

    assert result.tags == ["infra", "migration"], "the graph edit must still land"
    embedder.embed.assert_not_called()
    mgr.vector_store.upsert.assert_not_called()


def test_blanking_content_is_refused_at_the_engine(manager):
    """update_memory writes in place with no version chain, so an accepted
    empty string destroys the memory with nothing to recover it from."""
    mgr, _ = manager
    node = _add(mgr)

    with pytest.raises(ValueError):
        mgr.update_memory(node.id, content="   ")
    assert mgr.graph_store.get_memory(node.id).content == node.content


def test_a_blank_summary_is_allowed(manager):
    """create() itself produces blank summaries (`add "x" -s "   "` stores ""),
    and the web editor posts summary on every save — so refusing it here threw
    away the user's whole edit and left such a memory unsavable from the UI."""
    mgr, _ = manager
    node = _add(mgr)

    result = mgr.update_memory(node.id, summary="  ", content="a real edit that must land")

    assert result.content == "a real edit that must land"


@pytest.mark.parametrize("bad", [99, -5, float("nan"), float("inf"), True, "0.5"])
def test_create_refuses_a_bad_confidence_on_every_surface(bad):
    """MemoryCreate is the one thing every create path builds — CLI add, the
    MCP add/batch tools, extraction, import, the web POST — so the check lives
    there rather than in whichever caller someone remembered."""
    with pytest.raises(ValueError):
        MemoryCreate(content="c", summary="s", category=MemoryCategory.FACT, confidence=bad)
