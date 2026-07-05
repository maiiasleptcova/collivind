from unittest.mock import MagicMock

from collivind.engine.graph_engine import GraphEngine
from collivind.models import MemoryCategory, MemoryNode


def test_graph_engine_expansion():
    graph_store = MagicMock()

    graph_store.get_neighbors.return_value = [{"id": "test_entity", "rel_type": "ABOUT", "direction": "OUT"}]

    mock_mem = MemoryNode(content="Related", summary="Test", category=MemoryCategory.FACT, id="mem-2")
    graph_store.find_related_memories.return_value = [mock_mem]

    engine = GraphEngine(graph_store)
    expanded = engine.get_expanded_memories(["mem-1"])

    assert "mem-2" in expanded
    assert "test entity" in expanded["mem-2"]["shared_entities"]
    assert expanded["mem-2"]["memory"] == mock_mem


def test_graph_engine_accepts_neo4j_neighbor_shape():
    graph_store = MagicMock()
    graph_store.get_neighbors.return_value = [{"node": {"id": "fastapi", "name": "FastAPI"}, "relationships": []}]
    mock_mem = MemoryNode(content="Related", summary="Test", category=MemoryCategory.FACT, id="mem-2")
    graph_store.find_related_memories.return_value = [mock_mem]

    engine = GraphEngine(graph_store)
    expanded = engine.get_expanded_memories(["mem-1"])

    assert "mem-2" in expanded
    assert "FastAPI" in expanded["mem-2"]["shared_entities"]
