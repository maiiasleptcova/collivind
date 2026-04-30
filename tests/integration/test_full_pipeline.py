import pytest
from collivind.config import CollivindConfig
from collivind.storage.qdrant_store import QdrantVectorStore
from collivind.storage.neo4j_store import Neo4jGraphStore
from collivind.storage.embedding_service import HttpEmbeddingProvider
from collivind.engine.memory_manager import MemoryManager
from collivind.models import MemoryCreate, MemoryCategory, EntityCreate, EntityType

@pytest.fixture
def manager():
    config = CollivindConfig()
    config.qdrant.collection_name = "test_integration"
    
    vector_store = QdrantVectorStore(config.qdrant, config.embeddings.dimension)
    graph_store = Neo4jGraphStore(config.neo4j)
    embedding_provider = HttpEmbeddingProvider(config.embeddings)
    
    m = MemoryManager(vector_store, graph_store, embedding_provider, config)
    yield m
    vector_store.close()
    graph_store.close()

@pytest.mark.skip(reason="Requires running Docker containers")
def test_full_pipeline(manager):
    # This test hits all backends: embeddings -> dedup -> neo4j -> qdrant -> neo4j relations
    mem_create = MemoryCreate(
        content="Collivind uses Qdrant and Neo4j.",
        summary="Collivind Architecture",
        category=MemoryCategory.ARCHITECTURE,
        project_id="test_pipeline"
    )
    ent_qdrant = EntityCreate(name="Qdrant", type=EntityType.SERVICE)
    ent_neo4j = EntityCreate(name="Neo4j", type=EntityType.SERVICE)
    
    # 1. Add memory
    memory = manager.add_memory(mem_create, entities=[ent_qdrant, ent_neo4j])
    assert memory.id is not None
    
    # 2. Check it was stored in graph
    fetched = manager.graph_store.get_memory(memory.id)
    assert fetched.content == memory.content
    
    # 3. Check duplicate rejection (exact same content)
    dup = manager.add_memory(mem_create)
    assert dup.id == memory.id # Should return the same memory due to high threshold
    
    # Cleanup
    manager.graph_store.delete_memory(memory.id)
    manager.vector_store.delete(memory.id)
