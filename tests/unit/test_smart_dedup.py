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


def test_merge_adds_new_tags():
    vs = MagicMock()
    vs.search.return_value = [{"id": "dup-1", "score": 0.99}]

    existing = MemoryNode(
        id="dup-1", content="Test", summary="Test",
        category=MemoryCategory.FACT, tags=["tag1"]
    )
    gs = MagicMock()
    gs.get_memory.return_value = existing

    ep = MagicMock()
    ep.embed.return_value = [0.1]

    entity_mock = MagicMock()
    entity_mock.id = "ent-1"
    gs.create_entity.return_value = entity_mock

    config = CollivindConfig()
    manager = MemoryManager(vs, gs, ep, config)

    mem_create = MemoryCreate(
        content="Test", summary="Test",
        category=MemoryCategory.FACT, tags=["tag1", "tag2"]
    )
    ent = EntityCreate(name="NewEntity", type=EntityType.CONCEPT)

    manager.add_memory(mem_create, entities=[ent])

    gs.update_memory.assert_called_once()
    gs.create_entity.assert_called_once()
    assert gs.create_relationship.call_count == 1


def test_merge_no_update_when_same_tags():
    vs = MagicMock()
    vs.search.return_value = [{"id": "dup-1", "score": 0.99}]

    existing = MemoryNode(
        id="dup-1", content="Test", summary="Test",
        category=MemoryCategory.FACT, tags=["tag1"]
    )
    gs = MagicMock()
    gs.get_memory.return_value = existing

    ep = MagicMock()
    ep.embed.return_value = [0.1]

    config = CollivindConfig()
    manager = MemoryManager(vs, gs, ep, config)

    mem_create = MemoryCreate(
        content="Test", summary="Test",
        category=MemoryCategory.FACT, tags=["tag1"]
    )

    manager.add_memory(mem_create)
    gs.update_memory.assert_not_called()
