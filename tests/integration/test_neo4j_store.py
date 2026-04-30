import pytest
from collivind.config import Neo4jConfig
from collivind.storage.neo4j_store import Neo4jGraphStore
from collivind.models import MemoryCreate, MemoryCategory, EntityCreate, EntityType, RelationshipCreate, RelType

@pytest.fixture
def store():
    config = Neo4jConfig() # uses localhost defaults
    s = Neo4jGraphStore(config)
    yield s
    s.close()

@pytest.mark.skip(reason="Requires running Neo4j container")
def test_neo4j_lifecycle(store):
    store.initialize()
    
    # Check health
    health = store.health_check()
    assert health["status"] == "ok"
    
    # Create memory
    mem_create = MemoryCreate(
        content="Neo4j is a graph database.",
        summary="Neo4j overview",
        category=MemoryCategory.FACT,
        project_id="test_project"
    )
    memory = store.create_memory(mem_create)
    assert memory.id is not None
    
    # Get memory
    fetched = store.get_memory(memory.id)
    assert fetched is not None
    assert fetched.content == "Neo4j is a graph database."
    
    # Update memory
    updated = store.update_memory(memory.id, summary="Updated summary")
    assert updated.summary == "Updated summary"
    
    # Create entity
    ent_create = EntityCreate(
        name="Neo4j",
        type=EntityType.SERVICE,
        properties={"language": "Java"}
    )
    entity = store.create_entity(ent_create)
    assert entity.id == "neo4j"
    
    # Create relationship
    rel_create = RelationshipCreate(
        source_id=memory.id,
        target_id=entity.id,
        type=RelType.ABOUT,
        confidence=0.9
    )
    edge = store.create_relationship(rel_create)
    assert edge.source_id == memory.id
    assert edge.target_id == entity.id
    assert edge.type == RelType.ABOUT
    
    # Graph traversal (get related memories)
    related = store.find_related_memories("Neo4j")
    assert len(related) > 0
    assert related[0].id == memory.id
    
    # Timeline
    timeline = store.get_timeline("test_project")
    assert len(timeline) > 0
    
    # Delete memory
    store.delete_memory(memory.id)
    assert store.get_memory(memory.id) is None
