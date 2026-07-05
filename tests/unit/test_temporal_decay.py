from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from collivind.config import SearchConfig
from collivind.engine.search_engine import SearchEngine
from collivind.models import MemoryCategory, MemoryNode, SearchQuery


def _make_engine(config=None):
    config = config or SearchConfig()
    vs = MagicMock()
    gs = MagicMock()
    ep = MagicMock()
    ep.embed.return_value = [0.1, 0.2]
    engine = SearchEngine(vs, gs, ep, config)
    engine.graph_engine = MagicMock()
    engine.graph_engine.get_expanded_memories.return_value = {}
    return engine, vs, gs


def test_recent_memory_no_decay():
    engine, _, _ = _make_engine()
    mem = MemoryNode(
        content="fresh",
        summary="s",
        category=MemoryCategory.FACT,
        created_at=datetime.now(timezone.utc),
    )
    decay = engine._compute_temporal_decay(mem, datetime.now(timezone.utc))
    assert decay == pytest.approx(1.0, abs=0.01)


def test_old_memory_decays():
    engine, _, _ = _make_engine()
    mem = MemoryNode(
        content="old",
        summary="s",
        category=MemoryCategory.FACT,
        created_at=datetime.now(timezone.utc) - timedelta(days=365),
    )
    decay = engine._compute_temporal_decay(mem, datetime.now(timezone.utc))
    assert decay < 1.0
    assert decay >= 1.0 - engine.config.temporal_decay_max


def test_decay_never_below_floor():
    config = SearchConfig(temporal_decay_rate=1.0, temporal_decay_max=0.5)
    engine, _, _ = _make_engine(config)
    mem = MemoryNode(
        content="ancient",
        summary="s",
        category=MemoryCategory.FACT,
        created_at=datetime.now(timezone.utc) - timedelta(days=10000),
    )
    decay = engine._compute_temporal_decay(mem, datetime.now(timezone.utc))
    assert decay >= 0.5


def test_decay_affects_final_score():
    engine, vs, gs = _make_engine()

    now = datetime.now(timezone.utc)
    recent = MemoryNode(
        id="recent",
        content="A",
        summary="s",
        category=MemoryCategory.FACT,
        created_at=now,
    )
    old = MemoryNode(
        id="old",
        content="B",
        summary="s",
        category=MemoryCategory.FACT,
        created_at=now - timedelta(days=365),
    )

    vs.search.return_value = [
        {"id": "recent", "score": 0.7},
        {"id": "old", "score": 0.7},
    ]
    gs.get_memory.side_effect = lambda mid: recent if mid == "recent" else old

    results = engine.search(SearchQuery(query="test", limit=10))
    assert len(results) == 2
    assert results[0].memory.id == "recent"
    assert results[0].score > results[1].score
