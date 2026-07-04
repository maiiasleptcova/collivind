from typing import Any, Dict, List, Optional

from collivind.config import CollivindConfig
from collivind.engine.dedup import Deduplicator
from collivind.engine.enrichment import build_enriched_text
from collivind.engine.search_engine import SearchEngine
from collivind.models import (
    EntityCreate,
    MemoryCreate,
    MemoryNode,
    RelationshipCreate,
    RelType,
    SearchQuery,
    SearchResult,
)
from collivind.models.entity import EntityType
from collivind.models.memory import MemoryCategory, MemorySource
from collivind.storage.interfaces import EmbeddingProvider, GraphStore, VectorStore


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

    def _merge_tags(self, existing: MemoryNode, new_tags: List[str]) -> List[str]:
        merged = list(existing.tags)
        for tag in new_tags:
            if tag not in merged:
                merged.append(tag)
        return merged

    def _merge_into_existing(
        self,
        existing: MemoryNode,
        memory_create: MemoryCreate,
        entities: Optional[List[EntityCreate]],
    ) -> MemoryNode:
        merged_tags = self._merge_tags(existing, memory_create.tags)
        if merged_tags != existing.tags:
            self.graph_store.update_memory(existing.id, tags=merged_tags)

        if entities:
            for ent_create in entities:
                ent_node = self.graph_store.create_entity(ent_create)
                self.graph_store.create_relationship(RelationshipCreate(
                    source_id=existing.id,
                    target_id=ent_node.id,
                    type=RelType.ABOUT,
                    confidence=1.0,
                    source="merge"
                ))

        return self.graph_store.get_memory(existing.id) or existing

    def add_memory(
        self,
        memory_create: MemoryCreate,
        entities: Optional[List[EntityCreate]] = None,
        relationships: Optional[List[RelationshipCreate]] = None
    ) -> MemoryNode:
        entity_names = [e.name for e in entities] if entities else None
        enriched = build_enriched_text(memory_create, entity_names=entity_names)
        vector = self.embedding_provider.embed(enriched)

        dup_match = self.deduplicator.find_duplicate_detailed(vector, memory_create.project_id)
        if dup_match:
            existing = self.graph_store.get_memory(dup_match.memory_id)
            if existing:
                if dup_match.is_exact:
                    # identical content resubmitted — reject, keep existing as-is
                    return existing
                return self._merge_into_existing(existing, memory_create, entities)

        memory_node = self.graph_store.create_memory(memory_create)

        payload = memory_node.to_dict()
        self.vector_store.upsert(memory_node.id, vector, payload)

        if entities:
            for ent_create in entities:
                ent_node = self.graph_store.create_entity(ent_create)
                self.graph_store.create_relationship(RelationshipCreate(
                    source_id=memory_node.id,
                    target_id=ent_node.id,
                    type=RelType.ABOUT,
                    confidence=1.0,
                    source="extraction"
                ))

        if relationships:
            for rel_create in relationships:
                if not rel_create.source_id:
                    rel_create.source_id = memory_node.id
                self.graph_store.create_relationship(rel_create)

        contradictions = self.search_engine.find_contradictions(memory_node)
        for contra in contradictions:
            self.graph_store.create_relationship(RelationshipCreate(
                source_id=memory_node.id,
                target_id=contra.memory.id,
                type=RelType.CONTRADICTS,
                confidence=contra.vector_score,
                source="auto_detection"
            ))

        return memory_node

    def invalidate(self, memory_id: str, superseded_by: str, reason: str) -> None:
        self.graph_store.invalidate_memory(memory_id, superseded_by, reason)

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
        entity = self.graph_store.get_entity(name)
        if not entity:
            return None

        related = self.graph_store.find_related_memories(name)
        return {
            "entity": {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type.value if hasattr(entity.type, "value") else entity.type,
                "properties": entity.properties,
                "created_at": entity.created_at.isoformat() if entity.created_at else None,
                "updated_at": entity.updated_at.isoformat() if entity.updated_at else None,
            },
            "related_memories": [m.to_dict() for m in related]
        }

    def get_timeline(self, project_id: str, entity: Optional[str] = None, limit: int = 50) -> List[MemoryNode]:
        """Get timeline of memories for a project."""
        return self.graph_store.get_timeline(project_id, entity, limit)

    def get_version_chain(self, memory_id: str) -> List[MemoryNode]:
        """Get the full version history of a memory."""
        return self.graph_store.get_version_chain(memory_id)

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
                session_id=mem_data.get("session_id"),
                user_id=mem_data.get("user_id", "local"),
                source=MemorySource(mem_data.get("source", "manual")),
                confidence=mem_data.get("confidence", 1.0),
                tags=mem_data.get("tags", [])
            )
            
            node = self.add_memory(mem_create, entities=entities, relationships=relationships)
            ids.append(node.id)
            
        return ids
