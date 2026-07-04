from datetime import datetime

from collivind.models.entity import EntityNode, EntityType
from collivind.models.memory import MemoryCategory, MemoryNode
from collivind.models.relationship import RelationshipEdge, RelType


def test_memory_node_creation():
    node = MemoryNode(
        content="Test content",
        summary="Test",
        category=MemoryCategory.FACT
    )
    assert node.id is not None
    assert node.content == "Test content"
    assert node.category == MemoryCategory.FACT
    assert isinstance(node.created_at, datetime)
    assert node.confidence == 1.0

def test_entity_node_creation():
    node = EntityNode(
        name="PostgreSQL",
        type=EntityType.SERVICE
    )
    assert node.id == "postgresql"
    assert node.name == "PostgreSQL"
    assert node.type == EntityType.SERVICE

def test_relationship_edge_creation():
    edge = RelationshipEdge(
        source_id="mem-1",
        target_id="ent-1",
        type=RelType.ABOUT
    )
    assert edge.source_id == "mem-1"
    assert edge.target_id == "ent-1"
    assert edge.type == RelType.ABOUT
