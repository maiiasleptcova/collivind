from unittest.mock import MagicMock

import pytest

from collivind.config import SearchConfig
from collivind.engine.search_engine import SearchEngine
from collivind.models import MemoryCategory, MemoryNode, SearchQuery


def test_search_engine_hybrid_scoring():
    vector_store = MagicMock()
    # Mock vector results
    vector_store.search.return_value = [
        {"id": "mem-1", "score": 0.8},
        {"id": "mem-2", "score": 0.6}
    ]
    
    graph_store = MagicMock()
    
    def mock_get_memory(mem_id):
        return MemoryNode(content=f"Content {mem_id}", summary="Test", category=MemoryCategory.FACT, id=mem_id)
        
    graph_store.get_memory.side_effect = mock_get_memory
    
    embedding_provider = MagicMock()
    embedding_provider.embed.return_value = [0.1, 0.2]
    
    config = SearchConfig(vector_weight=0.7, graph_weight=0.3)
    engine = SearchEngine(vector_store, graph_store, embedding_provider, config)
    
    # Mock graph engine to add graph score to mem-2
    engine.graph_engine = MagicMock()
    engine.graph_engine.get_expanded_memories.return_value = {
        "mem-2": {"memory": mock_get_memory("mem-2"), "shared_entities": ["ent-1", "ent-2"]}
    }
    
    query = SearchQuery(query="test", limit=10)
    results = engine.search(query)
    
    assert len(results) == 2
    # mem-1 score: (0.8 * 0.7) + (0 * 0.3) = 0.56
    # mem-2 score: (0.6 * 0.7) + (0.2 * 0.3) = 0.42 + 0.06 = 0.48
    assert results[0].memory.id == "mem-1"
    assert results[1].memory.id == "mem-2"
    assert results[0].score == pytest.approx(0.56)
    assert results[1].score == pytest.approx(0.48)


def test_find_contradictions():
    """Test that find_contradictions identifies similar memories with same category but different content."""
    from unittest.mock import MagicMock

    from collivind.config import SearchConfig

    vector_store = MagicMock()
    graph_store = MagicMock()
    embedding_provider = MagicMock()
    config = SearchConfig()

    engine = SearchEngine(vector_store, graph_store, embedding_provider, config)

    target = MemoryNode(
        content="We use PostgreSQL for the database",
        summary="DB choice",
        category=MemoryCategory.DECISION,
        project_id="test"
    )

    existing = MemoryNode(
        content="We use MongoDB for the database",
        summary="DB choice",
        category=MemoryCategory.DECISION,
        project_id="test"
    )

    embedding_provider.embed.return_value = [0.1] * 384
    vector_store.search.return_value = [{"id": existing.id, "score": 0.85}]
    graph_store.get_memory.return_value = existing

    results = engine.find_contradictions(target)
    assert len(results) == 1
    assert results[0].memory.id == existing.id
    assert results[0].vector_score == 0.85


def test_find_contradictions_skips_same_content():
    """Test that identical content is not flagged as contradiction."""
    from unittest.mock import MagicMock

    from collivind.config import SearchConfig

    vector_store = MagicMock()
    graph_store = MagicMock()
    embedding_provider = MagicMock()
    config = SearchConfig()

    engine = SearchEngine(vector_store, graph_store, embedding_provider, config)

    target = MemoryNode(
        content="We use PostgreSQL for the database",
        summary="DB choice",
        category=MemoryCategory.DECISION,
        project_id="test"
    )

    # Same content as target -- should not be flagged
    existing = MemoryNode(
        content="We use PostgreSQL for the database",
        summary="DB choice",
        category=MemoryCategory.DECISION,
        project_id="test"
    )

    embedding_provider.embed.return_value = [0.1] * 384
    vector_store.search.return_value = [{"id": existing.id, "score": 0.95}]
    graph_store.get_memory.return_value = existing

    results = engine.find_contradictions(target)
    assert len(results) == 0


def test_find_contradictions_skips_invalidated():
    """Test that already-invalidated memories are skipped."""
    from datetime import datetime, timezone
    from unittest.mock import MagicMock

    from collivind.config import SearchConfig

    vector_store = MagicMock()
    graph_store = MagicMock()
    embedding_provider = MagicMock()
    config = SearchConfig()

    engine = SearchEngine(vector_store, graph_store, embedding_provider, config)

    target = MemoryNode(
        content="We use PostgreSQL for the database",
        summary="DB choice",
        category=MemoryCategory.DECISION,
        project_id="test"
    )

    # This memory has been invalidated (valid_to is set)
    existing = MemoryNode(
        content="We use MongoDB for the database",
        summary="DB choice",
        category=MemoryCategory.DECISION,
        project_id="test",
        valid_to=datetime.now(timezone.utc)
    )

    embedding_provider.embed.return_value = [0.1] * 384
    vector_store.search.return_value = [{"id": existing.id, "score": 0.85}]
    graph_store.get_memory.return_value = existing

    results = engine.find_contradictions(target)
    assert len(results) == 0


def test_find_contradictions_skips_different_category():
    """Test that memories with different categories are not flagged."""
    from unittest.mock import MagicMock

    from collivind.config import SearchConfig

    vector_store = MagicMock()
    graph_store = MagicMock()
    embedding_provider = MagicMock()
    config = SearchConfig()

    engine = SearchEngine(vector_store, graph_store, embedding_provider, config)

    target = MemoryNode(
        content="We use PostgreSQL for the database",
        summary="DB choice",
        category=MemoryCategory.DECISION,
        project_id="test"
    )

    # Different category -- should not be flagged
    existing = MemoryNode(
        content="PostgreSQL is a relational database",
        summary="DB info",
        category=MemoryCategory.FACT,
        project_id="test"
    )

    embedding_provider.embed.return_value = [0.1] * 384
    vector_store.search.return_value = [{"id": existing.id, "score": 0.85}]
    graph_store.get_memory.return_value = existing

    results = engine.find_contradictions(target)
    assert len(results) == 0
