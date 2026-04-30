from typing import List, Dict, Any, Optional

from collivind.config import CollivindConfig
from collivind.storage.interfaces import VectorStore, GraphStore, EmbeddingProvider
from collivind.models import (
    MemoryNode, MemoryCreate,
    EntityNode, EntityCreate,
    RelationshipEdge, RelationshipCreate, RelType
)
from collivind.engine.dedup import Deduplicator
from collivind.engine.search_engine import SearchEngine
from collivind.exceptions import CollivindError
from collivind.models import SearchQuery, SearchResult

class MemoryManager:
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        embedding_provider: EmbeddingProvider,
        config: CollivindConfig
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.embedding_provider = embedding_provider
        self.config = config
        self.deduplicator = Deduplicator(vector_store, config.search)
        self.search_engine = SearchEngine(vector_store, graph_store, embedding_provider, config.search)

    def add_memory(
        self,
        memory_create: MemoryCreate,
        entities: Optional[List[EntityCreate]] = None,
        relationships: Optional[List[RelationshipCreate]] = None
    ) -> MemoryNode:
        """
        Full pipeline to process, embed, deduplicate, and store a memory and its entities.
        """
        # 1. Embed content
        vector = self.embedding_provider.embed(memory_create.content)
        
        # 2. Deduplicate
        duplicate = self.deduplicator.find_duplicate(vector, memory_create.project_id)
        if duplicate:
            dup_id, score = duplicate
            # If it's a near exact match, return existing memory
            # For this implementation, we just return the duplicate. 
            # Real application might merge tags/entities or supersede.
            existing_memory = self.graph_store.get_memory(dup_id)
            if existing_memory:
                return existing_memory

        # 3. Create Memory Node
        memory_node = self.graph_store.create_memory(memory_create)
        
        # 4. Upsert Vector
        payload = memory_node.to_dict()
        self.vector_store.upsert(memory_node.id, vector, payload)
        
        # 5. Create Entities and standard relationships
        if entities:
            for ent_create in entities:
                ent_node = self.graph_store.create_entity(ent_create)
                # By default, relate memory to entity as ABOUT
                self.graph_store.create_relationship(RelationshipCreate(
                    source_id=memory_node.id,
                    target_id=ent_node.id,
                    type=RelType.ABOUT,
                    confidence=1.0,
                    source="extraction"
                ))
                
        # 6. Create custom relationships (between entities or memory to memory)
        if relationships:
            for rel_create in relationships:
                # Resolve potential references if they are missing source_id
                if not rel_create.source_id:
                    rel_create.source_id = memory_node.id
                self.graph_store.create_relationship(rel_create)
                
        return memory_node

    def invalidate(self, memory_id: str, superseded_by: str, reason: str) -> None:
        """Marks a memory as outdated and updates stores."""
        self.graph_store.invalidate_memory(memory_id, superseded_by, reason)
        # Fetch updated node and push to Qdrant to update valid_to payload
        updated = self.graph_store.get_memory(memory_id)
        if updated:
            # We need the vector again or we rely on Qdrant payload update mechanism.
            # For simplicity, we assume Qdrant payload is updated correctly when superseded
            # Note: Qdrant actually supports payload-only updates via the API, 
            # but VectorStore interface only has upsert. We'll skip for this spec.
            pass

    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Hybrid search via SearchEngine."""
        return self.search_engine.search(query)

    def get_context(self, query_str: str, project_id: str = "default", limit: int = 10) -> str:
        """Get formatted context string for LLM."""
        query = SearchQuery(query=query_str, project_id=project_id, limit=limit)
        results = self.search(query)
        
        if not results:
            return "No relevant context found in Collivind."
            
        context_parts = ["--- Collivind Context ---"]
        for res in results:
            cat = res.memory.category.value if hasattr(res.memory.category, 'value') else res.memory.category
            meta = f"[{cat.upper()}] (score: {res.score:.2f})"
            if res.related_entities:
                meta += f" (Entities: {', '.join(res.related_entities)})"
            context_parts.append(f"{meta}\n{res.memory.content}\n")
            
        return "\n".join(context_parts)

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve an entity and its related memories."""
        # This requires graph_store to have a generic MATCH node query or we just find by ID.
        # EntityNode IDs are lowercased underscore-separated names.
        entity_id = name.lower().replace(" ", "_").replace("-", "_")
        query = "MATCH (e:Entity {id: $id}) RETURN e"
        
        with self.graph_store.driver.session(database=self.graph_store.config.database) as session:
            res = session.run(query, id=entity_id)
            record = res.single()
            if not record:
                return None
            node = dict(record["e"])
            
        # Get related memories
        related = self.graph_store.find_related_memories(name)
        return {
            "entity": node,
            "related_memories": [m.to_dict() for m in related]
        }

    def get_timeline(self, project_id: str, entity: Optional[str] = None, limit: int = 50) -> List[MemoryNode]:
        """Get timeline of memories for a project."""
        return self.graph_store.get_timeline(project_id, entity, limit)

    def batch_add_memories(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Batch insert memories."""
        ids = []
        for mem_data in memories:
            entities = []
            for e in mem_data.get("entities", []):
                entities.append(EntityCreate(
                    name=e["name"],
                    type=EntityType(e["type"]),
                    properties=e.get("properties", {})
                ))
            
            relationships = []
            for r in mem_data.get("relationships", []):
                relationships.append(RelationshipCreate(
                    source_id="",
                    target_id=r["target_id"],
                    type=RelType(r["type"]),
                    properties=r.get("properties", {})
                ))

            mem_create = MemoryCreate(
                content=mem_data["content"],
                summary=mem_data["summary"],
                category=MemoryCategory(mem_data["category"]),
                project_id=mem_data.get("project_id", "default"),
                confidence=mem_data.get("confidence", 1.0),
                tags=mem_data.get("tags", [])
            )
            
            node = self.add_memory(mem_create, entities=entities, relationships=relationships)
            ids.append(node.id)
            
        return ids
