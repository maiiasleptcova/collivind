from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from collivind.config import SearchConfig
from collivind.engine.search_engine import SearchEngine
from collivind.models import MemoryCategory, MemoryNode, SearchQuery


def _make_engine_with_memories(memories):
    config = SearchConfig()
    vs = MagicMock()
    gs = MagicMock()
    ep = MagicMock()
    ep.embed.return_value = [0.1, 0.2]

    vs.search.return_value = [
        {"id": m.id, "score": 0.8} for m in memories
    ]
    gs.get_memory.side_effect = lambda mid: next(
        (m for m in memories if m.id == mid), None
    )

    engine = SearchEngine(vs, gs, ep, config)
    engine.graph_engine = MagicMock()
    engine.graph_engine.get_expanded_memories.return_value = {}
    return engine


def test_filter_by_tags():
    now = datetime.now(timezone.utc)
    m1 = MemoryNode(id="m1", content="A", summary="s",
                     category=MemoryCategory.FACT, tags=["backend"], created_at=now)
    m2 = MemoryNode(id="m2", content="B", summary="s",
                     category=MemoryCategory.FACT, tags=["frontend"], created_at=now)

    engine = _make_engine_with_memories([m1, m2])
    results = engine.search(SearchQuery(query="test", tags=["backend"], limit=10))

    assert len(results) == 1
    assert results[0].memory.id == "m1"


def test_filter_by_project_id():
    now = datetime.now(timezone.utc)
    m1 = MemoryNode(id="m1", content="A", summary="s",
                     category=MemoryCategory.FACT, project_id="alpha", created_at=now)
    m2 = MemoryNode(id="m2", content="B", summary="s",
                     category=MemoryCategory.FACT, project_id="beta", created_at=now)

    engine = _make_engine_with_memories([m1, m2])
    results = engine.search(SearchQuery(query="test", project_id="alpha", limit=10))

    assert len(results) == 1
    assert results[0].memory.id == "m1"
    engine.vector_store.search.assert_called_once()
    assert engine.vector_store.search.call_args.kwargs["filters"] == {"project_id": "alpha"}


def test_filter_by_date_from():
    now = datetime.now(timezone.utc)
    old = MemoryNode(id="old", content="A", summary="s",
                      category=MemoryCategory.FACT, created_at=now - timedelta(days=30))
    recent = MemoryNode(id="recent", content="B", summary="s",
                         category=MemoryCategory.FACT, created_at=now)

    engine = _make_engine_with_memories([old, recent])
    results = engine.search(SearchQuery(
        query="test", date_from=now - timedelta(days=1), limit=10
    ))

    assert len(results) == 1
    assert results[0].memory.id == "recent"


def test_filter_by_date_to():
    now = datetime.now(timezone.utc)
    old = MemoryNode(id="old", content="A", summary="s",
                      category=MemoryCategory.FACT, created_at=now - timedelta(days=30))
    recent = MemoryNode(id="recent", content="B", summary="s",
                         category=MemoryCategory.FACT, created_at=now)

    engine = _make_engine_with_memories([old, recent])
    results = engine.search(SearchQuery(
        query="test", date_to=now - timedelta(days=1), limit=10
    ))

    assert len(results) == 1
    assert results[0].memory.id == "old"


def test_filter_by_entity_names():
    now = datetime.now(timezone.utc)
    m1 = MemoryNode(id="m1", content="Uses PostgreSQL", summary="s",
                     category=MemoryCategory.FACT, created_at=now)

    config = SearchConfig()
    vs = MagicMock()
    gs = MagicMock()
    ep = MagicMock()
    ep.embed.return_value = [0.1, 0.2]
    vs.search.return_value = []
    gs.get_memory.return_value = None
    gs.find_related_memories.return_value = [m1]

    engine = SearchEngine(vs, gs, ep, config)
    engine.graph_engine = MagicMock()
    engine.graph_engine.get_expanded_memories.return_value = {}

    results = engine.search(SearchQuery(
        query="database", entity_names=["PostgreSQL"], limit=10
    ))

    assert len(results) == 1
    assert results[0].memory.id == "m1"


def test_combined_filters():
    now = datetime.now(timezone.utc)
    m1 = MemoryNode(id="m1", content="A", summary="s",
                     category=MemoryCategory.DECISION, tags=["backend"],
                     created_at=now)
    m2 = MemoryNode(id="m2", content="B", summary="s",
                     category=MemoryCategory.FACT, tags=["backend"],
                     created_at=now)
    m3 = MemoryNode(id="m3", content="C", summary="s",
                     category=MemoryCategory.DECISION, tags=["frontend"],
                     created_at=now)

    engine = _make_engine_with_memories([m1, m2, m3])
    results = engine.search(SearchQuery(
        query="test", category="decision", tags=["backend"], limit=10
    ))

    assert len(results) == 1
    assert results[0].memory.id == "m1"
