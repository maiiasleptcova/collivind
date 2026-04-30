import pytest
from unittest.mock import MagicMock

from collivind.config import CollivindConfig
from collivind.models import MemoryCreate, MemoryCategory, MemoryNode, EntityCreate, EntityType, RelationshipCreate, RelType
from collivind.engine.memory_manager import MemoryManager

def test_add_memory_no_duplicate():
    # Setup mocks
    vector_store = MagicMock()
    vector_store.search.return_value = [] # No duplicates
    
    graph_store = MagicMock()
    mock_memory = MemoryNode(content="Test", summary="Test", category=MemoryCategory.FACT, id="mem-1")
    graph_store.create_memory.return_value = mock_memory
    
    mock_entity = MagicMock()
    mock_entity.id = "ent-1"
    graph_store.create_entity.return_value = mock_entity
    
    embedding_provider = MagicMock()
    embedding_provider.embed.return_value = [0.1, 0.2, 0.3]
    
    config = CollivindConfig()
    manager = MemoryManager(vector_store, graph_store, embedding_provider, config)
    
    # Run add_memory
    mem_create = MemoryCreate(content="Test", summary="Test", category=MemoryCategory.FACT)
    ent_create = EntityCreate(name="Entity1", type=EntityType.CONCEPT)
    rel_create = RelationshipCreate(source_id="mem-1", target_id="ent-1", type=RelType.RELATES_TO)
    
    result = manager.add_memory(mem_create, entities=[ent_create], relationships=[rel_create])
    
    # Asserts
    assert result == mock_memory
    # embed is called twice: once for dedup, once for contradiction detection
    assert embedding_provider.embed.call_count == 2
    # search is called twice: once for dedup, once for contradiction detection
    assert vector_store.search.call_count == 2
    graph_store.create_memory.assert_called_once()
    vector_store.upsert.assert_called_once_with("mem-1", [0.1, 0.2, 0.3], mock_memory.to_dict())
    graph_store.create_entity.assert_called_once()
    # 2 relationships created: 1 automatic ABOUT, 1 manual RELATES_TO
    # (no contradictions found since vector_store.search returns [])
    assert graph_store.create_relationship.call_count == 2

def test_add_memory_with_duplicate():
    vector_store = MagicMock()
    # Mock finding a duplicate
    vector_store.search.return_value = [{"id": "dup-1", "score": 0.95}]
    
    graph_store = MagicMock()
    mock_memory = MemoryNode(content="Dup", summary="Dup", category=MemoryCategory.FACT, id="dup-1")
    graph_store.get_memory.return_value = mock_memory
    
    embedding_provider = MagicMock()
    embedding_provider.embed.return_value = [0.1, 0.2, 0.3]
    
    config = CollivindConfig()
    manager = MemoryManager(vector_store, graph_store, embedding_provider, config)
    
    mem_create = MemoryCreate(content="Dup", summary="Dup", category=MemoryCategory.FACT)
    
    result = manager.add_memory(mem_create)
    
    assert result == mock_memory
    vector_store.search.assert_called_once()
    graph_store.create_memory.assert_not_called()
    vector_store.upsert.assert_not_called()
