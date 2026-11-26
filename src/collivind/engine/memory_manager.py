import logging
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from collivind.config import CollivindConfig
from collivind.engine.conflicts import MAX_CANDIDATES, is_live
from collivind.engine.dedup import Deduplicator, Neighbour
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
from collivind.models.entity import ALIASES_KEY, EntityType, entity_slug
from collivind.models.memory import MemoryCategory, MemorySource, validate_confidence
from collivind.models.relations import (
    BATCH_REF,
    ENTITY_RELATIONS,
    NOTE_KEY,
    EntityRelation,
    clean_note,
    parse_relationship,
    phrase,
)
from collivind.storage.interfaces import EmbeddingProvider, GraphStore, VectorStore

logger = logging.getLogger(__name__)

# How long a process trusts its alias table before reading it again, so an
# alias set by another process (a merge from the CLI) reaches a running server.
ALIAS_TTL_SECONDS = 60.0

DEGRADED_NOTE = (
    "[DEGRADED] The graph store is unavailable: these are vector-only results with no "
    "entity or relationship enrichment, and new memories cannot be saved until it is back. "
    "Run `collivind status`."
)


def _batch_index(rel: Any) -> Optional[int]:
    """The batch position a relationship's target_index points at, or None for an ordinary edge."""
    if isinstance(rel, RelationshipCreate) and rel.target_id.startswith(BATCH_REF):
        return int(rel.target_id[len(BATCH_REF) :])
    return None


# Which edge types a person may draw between which kinds of node. SUPERSEDES
# is written only by versioning, and session edges only by capture.
LINKABLE = {
    ("memory", "memory"): ("RELATES_TO", "CAUSED_BY", "RESOLVES", "DEPENDS_ON", "CONTRADICTS"),
    ("memory", "entity"): ("ABOUT", "MENTIONS"),
    ("entity", "entity"): ENTITY_RELATIONS,
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
        self.deduplicator = Deduplicator(vector_store, config.search, embedding_provider)
        self.search_engine = SearchEngine(vector_store, graph_store, embedding_provider, config.search)
        self._aliases: Dict[str, str] = {}
        self._aliases_read_at = float("-inf")

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
        relationships: Optional[Sequence[Union[RelationshipCreate, EntityRelation]]] = None,
    ) -> MemoryNode:
        return self.add_memory_checked(memory_create, entities, relationships)[0]

    def add_memory_checked(
        self,
        memory_create: MemoryCreate,
        entities: Optional[List[EntityCreate]] = None,
        relationships: Optional[Sequence[Union[RelationshipCreate, EntityRelation]]] = None,
    ) -> Tuple[MemoryNode, List[SearchResult]]:
        """Store a memory and return it with the live memories it may conflict with.

        Only a resubmission (the same text once normalised) is rejected.
        Nothing is retired here: close neighbours and likely duplicates come
        back as conflict candidates, and the caller (an agent, via the MCP
        response) decides whether one is an older version, a real conflict or
        just a neighbour. See engine/dedup.py for why.
        """
        if memory_create.category == MemoryCategory.SUMMARY:
            raise ValueError("summaries are written with collivind_summarize (`collivind summarize`), not add")
        entity_names = [e.name for e in entities] if entities else None
        enriched = build_enriched_text(memory_create, entity_names=entity_names)
        vector = self.embedding_provider.embed(enriched)

        neighbours = self.deduplicator.neighbours(vector, memory_create.project_id)
        same = self.deduplicator.exact(memory_create.content, neighbours)
        if same:
            existing = self._graph_write(self.graph_store.get_memory, same.memory_id)
            if existing:
                # identical content resubmitted — keep existing as-is, but the
                # links the save states are new knowledge ("this fix resolves
                # that error") and must not vanish with the duplicate text
                self._write_relationships(existing.id, relationships)
                return existing, []

        memory_node = self._graph_write(self.graph_store.create_memory, memory_create)
        self.vector_store.upsert(memory_node.id, vector, memory_node.to_dict())

        if entities:
            for ent_create in entities:
                canonical = self.canonical_entity_name(ent_create.name)
                if canonical != ent_create.name:
                    ent_create = EntityCreate(name=canonical, type=ent_create.type, properties=ent_create.properties)
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

        self._write_relationships(memory_node.id, relationships)

        # No CONTRADICTS edge from similarity: on a real store it flagged
        # thousands of compatible pairs (see engine/conflicts.py).
        try:
            conflicts = self._possible_conflicts(memory_create, memory_node, vector, neighbours)
        except Exception:
            # the memory is stored; a failed lookup must not turn that into an error
            logger.warning("conflict check failed for %s", memory_node.id, exc_info=True)
            conflicts = []
        return memory_node, conflicts

    def _possible_conflicts(
        self, memory_create: MemoryCreate, node: MemoryNode, vector: List[float], neighbours: List[Neighbour]
    ) -> List[SearchResult]:
        """Likely duplicates first (any category), then same-category conflict candidates, at most MAX_CANDIDATES."""
        found: List[SearchResult] = []
        for n in self.deduplicator.likely_duplicates(memory_create, neighbours):
            existing = self.graph_store.get_memory(n.memory_id)
            if existing is not None and is_live(existing):
                found.append(SearchResult(memory=existing, score=n.score, vector_score=n.score, graph_score=0.0))
        seen = {r.memory.id for r in found}
        found += [c for c in self.search_engine.conflict_candidates(node, vector) if c.memory.id not in seen]
        return found[:MAX_CANDIDATES]

    def _write_relationships(
        self, memory_id: str, relationships: Optional[Sequence[Union[RelationshipCreate, EntityRelation]]]
    ) -> None:
        for rel in relationships or []:
            if isinstance(rel, EntityRelation):
                self._relate_entities(rel)
            else:
                self.graph_store.create_relationship(replace(rel, source_id=rel.source_id or memory_id))

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
        # str enum: neo4j hands the category back as a plain string
        if category is not None and (category == MemoryCategory.SUMMARY) != (
            existing.category == MemoryCategory.SUMMARY
        ):
            if existing.category == MemoryCategory.SUMMARY:
                raise ValueError("a summary cannot be reclassified; forget it instead")
            raise ValueError("a memory cannot be turned into a summary; use collivind_summarize")
        if tags is not None and existing.category == MemoryCategory.SUMMARY:
            # the group tag is how a rewrite finds this summary; keep it whatever the edit
            group_tags = [t for t in existing.tags if t.startswith("summary-of:")]
            tags = group_tags + [t for t in tags if t not in group_tags]

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

    def canonical_entity_name(self, name: str) -> str:
        """The entity a name means: an alias listed on another entity wins over the name itself."""
        now = time.monotonic()
        if now - self._aliases_read_at > ALIAS_TTL_SECONDS:
            # ponytail: one scan of all entities per minute; index aliases in the store if that gets slow
            self._aliases = {
                entity_slug(alias): e.name
                for e in self.graph_store.iter_entities()
                for alias in (e.properties or {}).get(ALIASES_KEY) or []
                if isinstance(alias, str) and alias.strip()
            }
            self._aliases_read_at = now
        return self._aliases.get(entity_slug(name), name)

    def _relate_entities(self, rel: EntityRelation) -> None:
        """Write '<source> <type> <target>' between two entities named by an agent."""
        ends = []
        for name in (self.canonical_entity_name(rel.source), self.canonical_entity_name(rel.target)):
            entity = self.graph_store.get_entity(name) or self.graph_store.create_entity(
                EntityCreate(name=name, type=EntityType.CONCEPT)
            )
            ends.append(entity.id)
        if ends[0] == ends[1]:
            return
        self.graph_store.create_relationship(
            RelationshipCreate(
                source_id=ends[0],
                target_id=ends[1],
                type=RelType(rel.type),
                properties={NOTE_KEY: rel.note} if rel.note else {},
                source="extraction",
            )
        )

    def edges_explained(self, node_id: str) -> List[Dict[str, Any]]:
        """A node's edges as readable sentences, for agents walking the graph."""
        out = []
        for e in self.graph_store.edges_of(node_id):
            outgoing = e.source_id == node_id
            other = e.target_id if outgoing else e.source_id
            rel = e.type.value if hasattr(e.type, "value") else str(e.type)
            out.append(
                {
                    "type": rel,
                    "direction": "out" if outgoing else "in",
                    "phrase": phrase(rel, outgoing),
                    "other_id": other,
                    "note": (e.properties or {}).get(NOTE_KEY, ""),
                    "source": e.source,
                }
            )
        return out

    def link(self, source_id: str, target_id: str, rel_type: RelType, note: str = "") -> RelationshipCreate:
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
        note = clean_note(note)
        edge = RelationshipCreate(
            source_id=source_id,
            target_id=target_id,
            type=rel_type,
            source="manual",
            properties={NOTE_KEY: note} if note else {},
        )
        existing = self._graph_write(
            self.graph_store.get_neighbors, source_id, [rel_type.value], "OUT", outcome="nothing was linked"
        )
        if any(n.get("id") == target_id for n in existing):
            if not note:
                return edge
            # a note on an existing edge: rewrite it (SQLite would otherwise stack a duplicate row)
            self.graph_store.delete_relationship(source_id, target_id, rel_type.value)
        self.graph_store.create_relationship(edge)
        return edge

    def link_topic(self, name: str, entity_type: EntityType, memory_ids: List[str]) -> Dict[str, Any]:
        """Make (or reuse) entity `name` and link every given memory ABOUT it.

        Turns a set of search results into one graph node, so a subject the
        extractor never named (a client site, an incident) can be opened as
        a whole. Checks every id before writing anything.
        """
        name = name.strip()
        if not name:
            raise ValueError("topic name is required")
        name = self.canonical_entity_name(name)
        ids = list(dict.fromkeys(memory_ids))
        missing = [i for i in ids if self._node_kind(i) != "memory"]
        if missing:
            raise LookupError(f"no memory {missing[0]!r}")
        entity = self._graph_write(
            self.graph_store.create_entity, EntityCreate(name=name, type=entity_type), outcome="nothing was linked"
        )
        for memory_id in ids:
            self.link(memory_id, entity.id, RelType.ABOUT)
        return {"entity_id": entity.id, "name": entity.name, "type": entity_type.value, "linked": len(ids)}

    def find_conflicts(self, memory_id: str) -> Optional[List[SearchResult]]:
        """Conflict candidates for a stored memory, scored on the vector it was stored with.

        None when the memory does not exist.
        """
        memory = self.graph_store.get_memory(memory_id)
        if memory is None:
            return None
        entity_names, _ = self._entity_names(memory_id)
        vector = self.embedding_provider.embed(build_enriched_text(self._as_create(memory), entity_names=entity_names))
        return self.search_engine.conflict_candidates(memory, vector)

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
            if rec.get("category") == MemoryCategory.SUMMARY.value:
                # v1 carries no edges, so a summary would come back with no
                # members; the v2 bundle (`collivind export`) keeps them.
                continue
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
        query = SearchQuery(
            query=query_str,
            project_id=project_id,
            limit=limit,
            min_rerank_score=self.config.search.recall_min_score,
        )
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
                    # a summary that does not fit gives way to the details after it
                    if res.memory.category == MemoryCategory.SUMMARY:
                        continue
                    break
                budget -= cost
            context_parts.append(part)

        return "\n".join(context_parts)

    def get_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve an entity and its related memories."""
        name = self.canonical_entity_name(name)
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
            # how this entity fits with others: "uses Neo4j", "is part of collivind"
            "relations": [r for r in self.edges_explained(entity.id) if r["type"] in ENTITY_RELATIONS],
        }

    def get_timeline(self, project_id: str, entity: Optional[str] = None, limit: int = 50) -> List[MemoryNode]:
        """Get timeline of memories for a project."""
        if entity:
            entity = self.canonical_entity_name(entity)
        return self.graph_store.get_timeline(project_id, entity, limit)

    def get_version_chain(self, memory_id: str) -> List[MemoryNode]:
        """Get the full version history of a memory."""
        return self.graph_store.get_version_chain(memory_id)

    def batch_add_memories(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Batch insert memories, validating every item before writing any (#37)."""
        return self.store_batch(self.prepare_batch(memories))

    def batch_add_checked(self, memories: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, List[SearchResult]]]:
        """batch_add_memories plus each new memory's conflict candidates (see add_memory_checked)."""
        return self.store_batch_checked(self.prepare_batch(memories))

    def prepare_batch(self, memories: List[Dict[str, Any]]) -> List[tuple]:
        """Validate every item up front; one bad item stores nothing (#37)."""
        prepared = []
        for i, mem_data in enumerate(memories):
            try:
                item = self._prepare_batch_item(mem_data)
                for rel in item[2]:
                    ref = _batch_index(rel)
                    if ref is not None and ref >= i:
                        raise ValueError(f"target_index {ref} must point to an earlier memory in this batch")
                prepared.append(item)
            except (KeyError, ValueError, TypeError) as e:
                detail = f"missing field {e}" if isinstance(e, KeyError) else str(e)
                raise ValueError(f"memory[{i}]: {detail} — nothing in the batch was stored") from e
        return prepared

    def store_batch(self, prepared: List[tuple]) -> List[str]:
        return self.store_batch_checked(prepared)[0]

    def store_batch_checked(self, prepared: List[tuple]) -> Tuple[List[str], Dict[str, List[SearchResult]]]:
        """Store already-validated (MemoryCreate, entities, relationships) items.

        Returns the stored ids and, per id, its conflict candidates (see
        add_memory_checked). A backend failure part-way raises
        BatchPartiallyStoredError carrying the ids that did land, rather
        than discarding them.
        """
        ids: List[str] = []
        conflicts: Dict[str, List[SearchResult]] = {}
        for i, (mem_create, entities, relationships) in enumerate(prepared):
            # an edge to an earlier memory of this batch: it has an id now
            relationships = [
                replace(r, target_id=ids[ref]) if (ref := _batch_index(r)) is not None else r for r in relationships
            ]
            try:
                node, found = self.add_memory_checked(mem_create, entities=entities, relationships=relationships)
            except Exception as e:
                if not ids:
                    raise
                raise BatchPartiallyStoredError(ids, i, e) from e
            ids.append(node.id)
            # a later item in the same batch is not a conflict of an earlier one's making
            found = [c for c in found if c.memory.id not in ids]
            if found:
                conflicts[node.id] = found
        return ids, conflicts

    @staticmethod
    def _prepare_batch_item(mem_data: Dict[str, Any]) -> tuple:
        entities = [
            EntityCreate(name=e["name"], type=EntityType(e["type"]), properties=e.get("properties", {}))
            for e in mem_data.get("entities", [])
        ]
        relationships = [parse_relationship(r, in_batch=True) for r in mem_data.get("relationships", [])]
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
