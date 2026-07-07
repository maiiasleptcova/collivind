from unittest.mock import MagicMock

from collivind.config import CollivindConfig
from collivind.engine.dedup import Deduplicator
from collivind.engine.memory_manager import MemoryManager
from collivind.models import (
    EntityCreate,
    EntityType,
    MemoryCategory,
    MemoryCreate,
    MemoryNode,
)


def test_dedup_match_exact():
    vs = MagicMock()
    from collivind.config import SearchConfig

    config = SearchConfig()
    dedup = Deduplicator(vs, config)

    vs.search.return_value = [{"id": "dup-1", "score": 0.99}]
    match = dedup.find_duplicate_detailed([0.1], "proj")

    assert match is not None
    assert match.is_exact is True
    assert match.memory_id == "dup-1"


def test_dedup_match_near():
    vs = MagicMock()
    from collivind.config import SearchConfig

    config = SearchConfig()
    dedup = Deduplicator(vs, config)

    vs.search.return_value = [{"id": "dup-1", "score": 0.93}]
    match = dedup.find_duplicate_detailed([0.1], "proj")

    assert match is not None
    assert match.is_exact is False


def test_dedup_no_match():
    vs = MagicMock()
    from collivind.config import SearchConfig

    config = SearchConfig()
    dedup = Deduplicator(vs, config)

    vs.search.return_value = [{"id": "m-1", "score": 0.5}]
    match = dedup.find_duplicate_detailed([0.1], "proj")

    assert match is None


def test_exact_duplicate_rejected_without_merge():
    vs = MagicMock()
    vs.search.return_value = [{"id": "dup-1", "score": 0.99}]

    existing = MemoryNode(id="dup-1", content="Test", summary="Test", category=MemoryCategory.FACT, tags=["tag1"])
    gs = MagicMock()
    gs.get_memory.return_value = existing

    ep = MagicMock()
    ep.embed.return_value = [0.1]

    config = CollivindConfig()
    manager = MemoryManager(vs, gs, ep, config)

    mem_create = MemoryCreate(content="Test", summary="Test", category=MemoryCategory.FACT, tags=["tag1", "tag2"])
    result = manager.add_memory(mem_create, entities=[EntityCreate(name="NewEntity", type=EntityType.CONCEPT)])

    assert result is existing
    gs.update_memory.assert_not_called()
    gs.create_entity.assert_not_called()
    gs.create_memory.assert_not_called()


def _supersede_manager(existing, new_node):
    """Manager whose vector search reports a near-duplicate of `existing`."""
    vs, gs, ep = MagicMock(), MagicMock(), MagicMock()
    vs.search.return_value = [{"id": existing.id, "score": 0.95}]
    gs.get_memory.side_effect = lambda mid: {existing.id: existing, new_node.id: new_node}.get(mid)
    gs.create_memory.return_value = new_node
    ep.embed.return_value = [0.1]
    manager = MemoryManager(vs, gs, ep, CollivindConfig())
    manager.search_engine.find_contradictions = MagicMock(return_value=[])
    return manager, vs, gs


def test_near_duplicate_supersedes_stale_memory():
    existing = MemoryNode(
        id="dup-1", content="Postgres 15 is our DB", summary="db", category=MemoryCategory.FACT, tags=["db"]
    )
    new_node = MemoryNode(
        id="new-1", content="Postgres 16 is our DB", summary="db", category=MemoryCategory.FACT, tags=["db"]
    )
    manager, vs, gs = _supersede_manager(existing, new_node)

    result = manager.add_memory(
        MemoryCreate(content="Postgres 16 is our DB", summary="db", category=MemoryCategory.FACT, tags=["db"])
    )

    assert result.id == "new-1"
    gs.create_memory.assert_called_once()
    gs.invalidate_memory.assert_called_once_with("dup-1", "new-1", "superseded_by_update")
    vs.delete.assert_called_once_with("dup-1")  # stale vector leaves the live index
    vs.upsert.assert_called_once()


def test_supersede_inherits_stale_tags():
    existing = MemoryNode(id="dup-1", content="old", summary="old", category=MemoryCategory.FACT, tags=["legacy"])
    new_node = MemoryNode(id="new-1", content="new", summary="new", category=MemoryCategory.FACT, tags=["fresh"])
    manager, _, gs = _supersede_manager(existing, new_node)

    manager.add_memory(MemoryCreate(content="new", summary="new", category=MemoryCategory.FACT, tags=["fresh"]))

    gs.update_memory.assert_called_once_with("new-1", tags=["legacy", "fresh"])
