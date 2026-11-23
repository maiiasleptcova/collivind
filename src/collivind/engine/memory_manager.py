import logging
from typing import Any, Dict, List, Optional

from collivind.config import CollivindConfig
from collivind.engine.dedup import Deduplicator
from collivind.engine.enrichment import build_enriched_text
from collivind.engine.search_engine import SearchEngine
from collivind.exceptions import BatchPartiallyStoredError, GraphStoreUnavailableError
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
from collivind.models.memory import MemoryCategory, MemorySource, validate_confidence
from collivind.storage.interfaces import EmbeddingProvider, GraphStore, VectorStore

logger = logging.getLogger(__name__)

DEGRADED_NOTE = (
    "[DEGRADED] The graph store is unavailable: these are vector-only results with no "
    "entity or relationship enrichment, and new memories cannot be saved until it is back. "
    "Run `collivind status`."
)


# Which edge types a person may draw between which kinds of node. SUPERSEDES
# is written only by versioning, and session edges only by capture.
LINKABLE = {
    ("memory", "memory"): ("RELATES_TO", "CONTRADICTS", "CAUSED_BY", "DEPENDS_ON"),
    ("memory", "entity"): ("ABOUT", "MENTIONS"),
    ("entity", "entity"): ("RELATED_TO", "DEPENDS_ON", "PART_OF", "USES", "IMPLEMENTS"),
}


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
            existing = self._graph_write(self.graph_store.get_memory, dup_match.memory_id)
            if existing:
                if dup_match.is_exact:
                    # identical content resubmitted — reject, keep existing as-is
                    return existing
                # near-duplicate with different content: the incoming memory is
                # the fresh version — store it and retire the stale one below,
                # instead of keeping old and new side by side
                stale = existing

        memory_node = self._graph_write(self.graph_store.create_memory, memory_create)

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

    @staticmethod
    def _graph_write(fn, *args, outcome: str = "the memory was NOT saved"):
        """Run a write-path graph call; a failure says the graph is down (#39).

        Wraps only the calls before anything is written, so a dead store
        stops the write cleanly with nothing half-stored in either index.
        """
        try:
            return fn(*args)
        except Exception as e:
            raise GraphStoreUnavailableError(e, outcome) from e

    def invalidate(self, memory_id: str, superseded_by: str, reason: str) -> None:
        self._graph_write(
            self.graph_store.invalidate_memory, memory_id, superseded_by, reason, outcome="nothing was invalidated"
        )
        # Same rule as a superseded-by-update memory in add_memory: history
        # stays in the graph, the vector index serves only live knowledge.
        # Otherwise its payload still claims valid_to=None and vector-only
        # search (graph down) keeps returning it.
        self.vector_store.delete(memory_id)

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        category: Optional[MemoryCategory] = None,
    ) -> Optional[MemoryNode]:
        """Update fields on a memory; re-embeds when the embedded text changes.

        Raises ValueError on a bad confidence or on blanking content/summary.
        The guards live here rather than in a caller because every surface
        routes through this method: with the range check in the HTTP layer
        only, `collivind update --confidence nan` reached the store and read
        back as NULL, and `--content ""` wiped a memory in place with no
        version chain to recover it from.
        """
        if confidence is not None:
            confidence = validate_confidence(confidence)
        # Only content. A blank summary is a state create() itself produces
        # (`add "x" -s "   "` stores ""), and the web editor posts summary on
        # every save — so guarding it here discarded the user's whole edit and
        # left any blank-summary memory permanently unsavable from the UI.
        if content is not None and not str(content).strip():
            raise ValueError("content cannot be blank; forget the memory instead")

        existing = self._graph_write(self.graph_store.get_memory, memory_id, outcome="nothing was updated")
        if not existing:
            return None

        updates = {
            k: v
            for k, v in {
                "content": content,
                "summary": summary,
                "tags": tags,
                "confidence": confidence,
                "category": category,
            }.items()
            if v is not None
        }
        if not updates:
            return existing

        updated = self.graph_store.update_memory(memory_id, **updates)

        # Confidence is the only editable field outside the embedded text, so a
        # confidence-only edit skips the graph lookup and the round-trip below.
        #
        # Deliberately keyed on "a field was supplied" rather than on the
        # enriched text differing. The stored vector is not always what the
        # current fields imply — add_memory embeds before the dedup tag-merge
        # and before MENTIONS edges exist — so a text-equality gate would look
        # at an unchanged memory and skip a re-embed that was in fact overdue.
        # Re-embedding an unchanged memory costs one round-trip; skipping one
        # that was needed leaves a permanently wrong index.
        # An invalidated memory has no vector (see invalidate); editing its
        # history must not put it back into the live index.
        live = updated is not None and not updated.valid_to and not updated.superseded_by
        if updated is not None and live and any(f is not None for f in (content, summary, category, tags)):
            entity_names, ok = self._entity_names(memory_id)
            if ok:
                after = build_enriched_text(self._as_create(updated), entity_names=entity_names)
                self.vector_store.upsert(updated.id, self.embedding_provider.embed(after), updated.to_dict())
        return updated

    @staticmethod
    def _as_create(node: MemoryNode) -> MemoryCreate:
        """The subset of a stored memory that build_enriched_text reads."""
        return MemoryCreate(
            content=node.content,
            summary=node.summary,
            category=node.category,
            project_id=node.project_id,
            tags=node.tags,
        )

    def _entity_names(self, memory_id: str) -> tuple[Optional[List[str]], bool]:
        """Entity names linked to a memory, and whether the lookup succeeded.

        add_memory feeds entity names into the embedding; without them here a
        re-embed would quietly strip that whole segment out of the vector. So
        a failed lookup returns ok=False and the caller leaves the existing
        vector alone: a vector that is stale on tags beats one silently
        missing its entity signal, and the next edit retries.

        ponytail: the SQLite store returns only the slugged id (no name), so a
        name is reconstructed from it and loses hyphens and case —
        "data-integrity" comes back as "data integrity". Neo4j returns the real
        name. Give GraphStore an entity-by-id lookup if that drift ever matters.
        """
        try:
            neighbors = self.graph_store.get_neighbors(
                memory_id, [RelType.ABOUT.value, RelType.MENTIONS.value], direction="OUT"
            )
        except Exception:
            logger.warning("entity lookup failed for %s; leaving its vector as-is", memory_id, exc_info=True)
            return None, False
        names = [n.get("name") or str(n.get("id") or "").replace("_", " ") for n in neighbors]
        return ([n for n in names if n] or None), True

    def forget(self, memory_id: str) -> bool:
        """Permanently delete a memory from graph and vector stores."""
        if not self._graph_write(self.graph_store.get_memory, memory_id, outcome="nothing was deleted"):
            return False
        self.graph_store.delete_memory(memory_id)
        self.vector_store.delete(memory_id)
        return True

    def get_memory(self, memory_id: str) -> Optional[MemoryNode]:
        return self.graph_store.get_memory(memory_id)

    def _node_kind(self, node_id: str) -> Optional[str]:
        if self.graph_store.get_memory(node_id):
            return "memory"
        entity = self.graph_store.get_entity(node_id)
        # get_entity slugifies its argument, so it must round-trip to this id
        return "entity" if entity is not None and entity.id == node_id else None

    def link(self, source_id: str, target_id: str, rel_type: RelType) -> RelationshipCreate:
        """Create a manual edge after checking both ends exist and the type fits them.

        Idempotent: when the triple already exists it is left as it is (an
        extracted edge keeps its confidence and source), so a second click
        does not stack duplicate rows in SQLite. Linking does not re-embed;
        the next content edit picks up new entity names.
        """
        if source_id == target_id:
            raise ValueError("a node cannot link to itself")
        source_kind, target_kind = self._node_kind(source_id), self._node_kind(target_id)
        if source_kind is None or target_kind is None:
            raise LookupError(f"no memory or entity {source_id if source_kind is None else target_id!r}")
        allowed = LINKABLE.get((source_kind, target_kind), ())
        if rel_type.value not in allowed:
            choices = ", ".join(allowed) or "none"
            raise ValueError(f"{rel_type.value} cannot link {source_kind} -> {target_kind}; use one of {choices}")
        edge = RelationshipCreate(source_id=source_id, target_id=target_id, type=rel_type, source="manual")
        existing = self._graph_write(
            self.graph_store.get_neighbors, source_id, [rel_type.value], "OUT", outcome="nothing was linked"
        )
        if any(n.get("id") == target_id for n in existing):
            return edge
        self.graph_store.create_relationship(edge)
        return edge

    def unlink(self, source_id: str, target_id: str, rel_type: RelType) -> int:
        """Remove edges matching the triple. Version edges are history, not links."""
        if rel_type == RelType.SUPERSEDES:
            raise ValueError("SUPERSEDES is the version history and cannot be unlinked")
        return self._graph_write(
            self.graph_store.delete_relationship, source_id, target_id, rel_type.value, outcome="nothing was unlinked"
        )

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
                    # An export written before confidence was validated can
                    # carry null (a NaN stored by an older build reads back as
                    # NULL). MemoryCreate now rejects that, and there is no
                    # per-record handling here — one bad row would abort the
                    # loop after earlier rows had already been committed.
                    confidence=1.0 if rec.get("confidence") is None else rec["confidence"],
                    tags=rec.get("tags") or [],
                )
            )
        return len(records)

    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Hybrid search via SearchEngine."""
        return self.search_engine.search(query)

    def search_with_status(self, query: SearchQuery) -> tuple[List[SearchResult], bool]:
        """Hybrid search plus whether it degraded to vector-only (graph down)."""
        return self.search_engine.search_with_status(query)

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
        results, degraded = self.search_with_status(query)

        if not results:
            return "No relevant context found in Collivind." + (f"\n{DEGRADED_NOTE}" if degraded else "")

        header = "--- Collivind Context ---" + (f"\n{DEGRADED_NOTE}" if degraded else "")
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
        """Batch insert memories, validating every item before writing any (#37)."""
        prepared = []
        for i, mem_data in enumerate(memories):
            try:
                prepared.append(self._prepare_batch_item(mem_data))
            except (KeyError, ValueError, TypeError) as e:
                detail = f"missing field {e}" if isinstance(e, KeyError) else str(e)
                raise ValueError(f"memory[{i}]: {detail} — nothing in the batch was stored") from e
        return self.store_batch(prepared)

    def store_batch(self, prepared: List[tuple]) -> List[str]:
        """Store already-validated (MemoryCreate, entities, relationships) items.

        A backend failure part-way raises BatchPartiallyStoredError carrying
        the ids that did land, rather than discarding them.
        """
        ids: List[str] = []
        for i, (mem_create, entities, relationships) in enumerate(prepared):
            try:
                node = self.add_memory(mem_create, entities=entities, relationships=relationships)
            except Exception as e:
                if not ids:
                    raise
                raise BatchPartiallyStoredError(ids, i, e) from e
            ids.append(node.id)
        return ids

    @staticmethod
    def _prepare_batch_item(mem_data: Dict[str, Any]) -> tuple:
        entities = [
            EntityCreate(name=e["name"], type=EntityType(e["type"]), properties=e.get("properties", {}))
            for e in mem_data.get("entities", [])
        ]
        relationships = [
            RelationshipCreate(
                source_id="",
                target_id=r["target_id"],
                type=RelType(r["type"]),
                properties=r.get("properties", {}),
            )
            for r in mem_data.get("relationships", [])
        ]
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
        return mem_create, entities, relationships
