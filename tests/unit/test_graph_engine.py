import pytest
from unittest.mock import MagicMock
from collivind.engine.graph_engine import GraphEngine
from collivind.models import MemoryNode, MemoryCategory

def test_graph_engine_expansion():
    graph_store = MagicMock()
    
    # Mock get_neighbors for a memory
    graph_store.get_neighbors.return_value = [
        {"node": {"name": "TestEntity"}}
    ]
    
    # Mock find_related_memories for an entity
    mock_mem = MemoryNode(content="Related", summary="Test", category=MemoryCategory.FACT, id="mem-2")
    graph_store.find_related_memories.return_value = [mock_mem]
    
    engine = GraphEngine(graph_store)
    expanded = engine.get_expanded_memories(["mem-1"])
    
    assert "mem-2" in expanded
    assert "TestEntity" in expanded["mem-2"]["shared_entities"]
    assert expanded["mem-2"]["memory"] == mock_mem
