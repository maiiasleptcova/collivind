"""Copy every memory, entity, edge and vector from one store set to another.

Unlike `collivind import`, which re-adds content through the dedup pipeline
and mints new ids, this restores records verbatim by id: entity links,
memory<->memory edges and version chains survive, and re-running is a no-op
(every write is an upsert). The source is only ever read.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from collivind.engine.enrichment import build_enriched_text
from collivind.engine.memory_manager import MemoryManager
from collivind.models.memory import MemoryNode
from collivind.models.relationship import RelationshipEdge, RelType
from collivind.storage.interfaces import EmbeddingProvider, GraphStore, VectorStore

_ENTITY_LINKS = (RelType.ABOUT, RelType.MENTIONS)


@dataclass(frozen=True)
class Stores:
    vector: VectorStore
    graph: GraphStore
    embedder: EmbeddingProvider


@dataclass(frozen=True)
class KindCount:
    kind: str
    source: int
    target: int
    missing: int  # source ids absent from the target


@dataclass(frozen=True)
class MigrationReport:
    counts: Tuple[KindCount, ...]
    dangling: Tuple[RelationshipEdge, ...]  # edges whose endpoint is in neither node set
    reembedded: bool
    dry_run: bool

    @property
    def ok(self) -> bool:
        return all(c.missing == 0 for c in self.counts)


def _target_ids(dst: Stores) -> Dict[str, Set[str]]:
    return {
        "memories": {m.id for m in dst.graph.iter_memories()},
        "entities": {e.id for e in dst.graph.iter_entities()},
        "relationships": {r.id for r in dst.graph.iter_relationships()},
        "vectors": {vid for vid, _, _ in dst.vector.iter_all()},
    }


def _retired(memories: Iterable[MemoryNode]) -> Set[str]:
    return {m.id for m in memories if m.valid_to or m.superseded_by}


def migrate(src: Stores, dst: Stores, *, reembed: bool, dry_run: bool) -> MigrationReport:
    memories = list(src.graph.iter_memories())
    entities = list(src.graph.iter_entities())
    edges = list(src.graph.iter_relationships())
    # The live vector index serves only current knowledge (invalidate() drops
    # the vector), so a vector whose memory is retired on either side — on
    # the target since the last run, or in a source written before 0.11.1 —
    # is neither copied nor counted.
    retired = _retired(memories) | _retired(dst.graph.iter_memories())
    vectors = [v for v in src.vector.iter_all() if v[0] not in retired]

    node_ids = {m.id for m in memories} | {e.id for e in entities}

    def attached(edge: RelationshipEdge) -> bool:
        return edge.source_id in node_ids and edge.target_id in node_ids

    live_edges = [e for e in edges if attached(e)]
    dangling = tuple(e for e in edges if not attached(e))

    if not dry_run:
        dst.graph.initialize()
        dst.vector.initialize()
        # Copy only what the target lacks: a record already there may have been
        # edited or invalidated since the last run, and the source must not
        # silently revert it. (A record *deleted* from the target since then
        # cannot be told apart from a new one and is copied again.)
        have = _target_ids(dst)
        for mem in memories:
            if mem.id not in have["memories"]:
                dst.graph.restore_memory(mem)
        for ent in entities:
            if ent.id not in have["entities"]:
                dst.graph.restore_entity(ent)
        for edge in live_edges:
            if edge.id not in have["relationships"]:
                dst.graph.restore_relationship(edge)
        texts = _embedding_texts(memories, entities, live_edges) if reembed else {}
        for vid, vector, payload in vectors:
            if vid in have["vectors"]:
                continue
            if reembed:
                vector = dst.embedder.embed(texts.get(vid) or _payload_text(payload))
            dst.vector.upsert(vid, vector, payload)

    source_ids = {
        "memories": {m.id for m in memories},
        "entities": {e.id for e in entities},
        "relationships": {e.id for e in live_edges},
        "vectors": {vid for vid, _, _ in vectors},
    }
    target_ids = _target_ids(dst)
    counts = tuple(
        KindCount(kind, len(ids), len(target_ids[kind]), len(ids - target_ids[kind]))
        for kind, ids in source_ids.items()
    )
    return MigrationReport(counts=counts, dangling=dangling, reembedded=reembed, dry_run=dry_run)


def _embedding_texts(memories, entities, edges: List[RelationshipEdge]) -> Dict[str, str]:
    """The enriched text add_memory embedded: content + summary + category + entity names + tags."""
    names = {e.id: e.name for e in entities}
    linked: Dict[str, List[str]] = {}
    for edge in edges:
        if edge.type in _ENTITY_LINKS and edge.target_id in names:
            linked.setdefault(edge.source_id, []).append(names[edge.target_id])
    return {
        m.id: build_enriched_text(MemoryManager._as_create(m), entity_names=linked.get(m.id) or None) for m in memories
    }


def _payload_text(payload: dict) -> str:
    """Fallback for a vector with no graph node: its payload is the memory snapshot."""
    return " | ".join(str(payload[k]) for k in ("content", "summary") if payload.get(k)) or ""


DOCKER_SERVICE_MODEL = "BAAI/bge-small-en-v1.5"  # hard-coded in docker/templates/embedding_server.py


def embedding_space(provider: str, model: str, dimension: int) -> Tuple[str, Optional[str], int]:
    """Identity of the vector space a provider writes into.

    `local` and the docker `http` service both run sentence-transformers on the
    configured model (the bundled service image hard-codes the default,
    BAAI/bge-small-en-v1.5), so their vectors are interchangeable.
    """
    family = "sentence-transformers" if provider in ("local", "http") else provider
    if provider == "http":
        # The bundled service ignores config.embeddings.model; comparing the
        # configured name would call MiniLM vectors "the same space" as bge.
        model = DOCKER_SERVICE_MODEL
    return family, model, dimension
