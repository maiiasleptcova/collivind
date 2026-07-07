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
        config: CollivindConfig,
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

    def add_memory(
        self,
        memory_create: MemoryCreate,
        entities: Optional[List[EntityCreate]] = None,
        relationships: Optional[List[RelationshipCreate]] = None,
    ) -> MemoryNode:
        entity_names = [e.name for e in entities] if entities else None
        enriched = build_enriched_text(memory_create, entity_names=entity_names)
        vector = self.embedding_provider.embed(enriched)

        stale = None
        dup_match = self.deduplicator.find_duplicate_detailed(vector, memory_create.project_id)
        if dup_match:
            existing = self.graph_store.get_memory(dup_match.memory_id)
            if existing:
                if dup_match.is_exact:
                    # identical content resubmitted — reject, keep existing as-is
                    return existing
                # near-duplicate with different content: the incoming memory is
                # the fresh version — store it and retire the stale one below,
                # instead of keeping old and new side by side
                stale = existing

        memory_node = self.graph_store.create_memory(memory_create)

        if stale:
            inherited = self._merge_tags(stale, memory_create.tags)
            if inherited != memory_create.tags:
                self.graph_store.update_memory(memory_node.id, tags=inherited)
            self.graph_store.invalidate_memory(stale.id, memory_node.id, "superseded_by_update")
            # history stays in the graph (version chain); the live vector index
            # must only serve current knowledge
            self.vector_store.delete(stale.id)
            memory_node = self.graph_store.get_memory(memory_node.id) or memory_node

        payload = memory_node.to_dict()
        self.vector_store.upsert(memory_node.id, vector, payload)

        if entities:
            for ent_create in entities:
                ent_node = self.graph_store.create_entity(ent_create)
                self.graph_store.create_relationship(
                    RelationshipCreate(
                        source_id=memory_node.id,
                        target_id=ent_node.id,
                        type=RelType.ABOUT,
                        confidence=1.0,
                        source="extraction",
                    )
                )

        if relationships:
            for rel_create in relationships:
                if not rel_create.source_id:
                    rel_create.source_id = memory_node.id
                self.graph_store.create_relationship(rel_create)

        contradictions = self.search_engine.find_contradictions(memory_node)
        for contra in contradictions:
            self.graph_store.create_relationship(
                RelationshipCreate(
                    source_id=memory_node.id,
                    target_id=contra.memory.id,
                    type=RelType.CONTRADICTS,
                    confidence=contra.vector_score,
                    source="auto_detection",
                )
            )

        return memory_node

    def invalidate(self, memory_id: str, superseded_by: str, reason: str) -> None:
        self.graph_store.invalidate_memory(memory_id, superseded_by, reason)

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[List[str]] = None,
        confidence: Optional[float] = None,
    ) -> Optional[MemoryNode]:
        """Update fields on a memory; re-embeds when the text changes."""
        existing = self.graph_store.get_memory(memory_id)
        if not existing:
            return None

        updates = {
            k: v
            for k, v in {"content": content, "summary": summary, "tags": tags, "confidence": confidence}.items()
            if v is not None
        }
        if not updates:
            return existing

        updated = self.graph_store.update_memory(memory_id, **updates)
        if updated and (content is not None or summary is not None):
            recreate = MemoryCreate(
                content=updated.content,
                summary=updated.summary,
                category=updated.category,
                project_id=updated.project_id,
                tags=updated.tags,
            )
            vector = self.embedding_provider.embed(build_enriched_text(recreate))
            self.vector_store.upsert(updated.id, vector, updated.to_dict())
        return updated

    def forget(self, memory_id: str) -> bool:
        """Permanently delete a memory from graph and vector stores."""
        if not self.graph_store.get_memory(memory_id):
            return False
        self.graph_store.delete_memory(memory_id)
        self.vector_store.delete(memory_id)
        return True

    def export_memories(self, project_id: str = "default", limit: int = 100_000) -> List[Dict[str, Any]]:
        """All memories of a project as plain dicts (newest first).

        v1 exports memory nodes only; entity links are rebuilt on import
        by re-running extraction, not preserved.
        """
        return [m.to_dict() for m in self.graph_store.get_timeline(project_id, limit=limit)]

    def import_memories(self, records: List[Dict[str, Any]]) -> int:
        """Re-add exported records through the normal pipeline (dedup applies).

        Returns the number of records processed.
        """
        for rec in records:
            self.add_memory(
                MemoryCreate(
                    content=rec["content"],
                    summary=rec.get("summary", rec["content"][:120]),
                    category=MemoryCategory(rec.get("category", "fact")),
                    project_id=rec.get("project_id", "default"),
                    user_id=rec.get("user_id", "local"),
                    confidence=rec.get("confidence", 1.0),
                    tags=rec.get("tags") or [],
                )
            )
        return len(records)

    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Hybrid search via SearchEngine."""
        return self.search_engine.search(query)

    def get_context(
        self,
        query_str: str,
        project_id: str = "default",
        limit: int = 10,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get formatted context string for LLM.

        max_tokens greedily packs results in score order using a len/4
        token estimate. Importance-aware budgeting lives in collivind-pro.
        """
        query = SearchQuery(query=query_str, project_id=project_id, limit=limit)
        results = self.search(query)

        if not results:
            return "No relevant context found in Collivind."

        header = "--- Collivind Context ---"
        context_parts = [header]
        budget = max_tokens - len(header) // 4 if max_tokens else None
        for res in results:
            cat = res.memory.category.value if hasattr(res.memory.category, "value") else res.memory.category
            meta = f"[{cat.upper()}] (score: {res.score:.2f})"
            if res.related_entities:
                meta += f" (Entities: {', '.join(res.related_entities)})"
            part = f"{meta}\n{res.memory.content}\n"
            if budget is not None:
                cost = len(part) // 4
                if cost > budget:
                    break
                budget -= cost
            context_parts.append(part)

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
            "related_memories": [m.to_dict() for m in related],
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
                entities.append(
                    EntityCreate(name=e["name"], type=EntityType(e["type"]), properties=e.get("properties", {}))
                )

            relationships = []
            for r in mem_data.get("relationships", []):
                relationships.append(
                    RelationshipCreate(
                        source_id="",
                        target_id=r["target_id"],
                        type=RelType(r["type"]),
                        properties=r.get("properties", {}),
                    )
                )

            mem_create = MemoryCreate(
                content=mem_data["content"],
                summary=mem_data["summary"],
                category=MemoryCategory(mem_data["category"]),
                project_id=mem_data.get("project_id", "default"),
                session_id=mem_data.get("session_id"),
                user_id=mem_data.get("user_id", "local"),
                source=MemorySource(mem_data.get("source", "manual")),
                confidence=mem_data.get("confidence", 1.0),
                tags=mem_data.get("tags", []),
            )

            node = self.add_memory(mem_create, entities=entities, relationships=relationships)
            ids.append(node.id)

        return ids
