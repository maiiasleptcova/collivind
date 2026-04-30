import pytest
from unittest.mock import MagicMock
from collivind.config import SearchConfig
from collivind.engine.search_engine import SearchEngine
from collivind.models import SearchQuery, MemoryNode, MemoryCategory

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
