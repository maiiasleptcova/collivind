"""The graph the explorer draws: memories, entities and the edges between them.

Built from the store's iterators (the same ones export and migrate use), so
docker, embedded and remote modes all produce the same shape. Nodes are kept
slim (no content) because the browser holds all of them at once; the side
panel fetches a memory in full when it is opened.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from collivind.engine.projects import is_meta_entity
from collivind.models.memory import MemoryNode
from collivind.models.relations import NOTE_KEY
from collivind.models.relationship import RelType
from collivind.storage.interfaces import GraphStore

# Beyond this many memories a force layout in the browser stops being
# readable long before it stops being fast; newest win and the payload says so.
DEFAULT_LIMIT = 3000
MAX_LIMIT = 10000
LABEL_LEN = 140


def _enum(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _memory_node(m: MemoryNode, members: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    extra = {"members": (members or {}).get(m.id, 0)} if _enum(m.category) == "summary" else {}
    return {
        **extra,
        "id": m.id,
        "kind": "memory",
        "label": (m.summary or m.content or "")[:LABEL_LEN],
        "category": _enum(m.category),
        "project_id": m.project_id,
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "superseded": bool(m.superseded_by or m.valid_to),
        "confidence": m.confidence,
        "version": m.version,
        "tags": m.tags or [],
    }


# An entity more memories than this are about is a hub ("Docker", the project
# itself): drawn, with its count, but a walk never passes through it. Through
# one, 2 hops around a memory on a real store went from 4 nodes to 308.
# ponytail: one fixed threshold; per-store percentile if stores vary widely.
HUB_DEGREE = 30


def _reach(
    seeds: Set[str], pairs: List[Tuple[str, str]], depth: int, stop: Optional[Set[str]] = None
) -> Dict[str, int]:
    """Hop distance from the nearest seed, for every node within `depth` hops.

    Nodes in `stop` are reached but never walked through.
    """
    adj: Dict[str, Set[str]] = {}
    for s, t in pairs:
        adj.setdefault(s, set()).add(t)
        adj.setdefault(t, set()).add(s)
    dist = {s: 0 for s in seeds}
    frontier = list(seeds)
    for hop in range(1, depth + 1):
        nxt = []
        for node in frontier:
            if stop and node in stop:
                continue
            for other in adj.get(node, ()):
                if other not in dist:
                    dist[other] = hop
                    nxt.append(other)
        frontier = nxt
    return dist


@dataclass(frozen=True)
class Loaded:
    """One read of the store, in the shapes every graph view starts from."""

    memories: List[MemoryNode]  # newest first
    entities: Dict[str, Any]
    raw: List[Tuple[str, str, str]]  # (source, type, target), the allowed types only
    derived: List[Tuple[str, str, str]]  # SUPERSEDES from the memory rows
    detail: Dict[Tuple[str, str, str], Tuple[str, str]]  # edge -> (note, origin)
    members: Dict[str, int]  # summary id -> live member count

    def memory_degree(self) -> Dict[str, int]:
        """How many memories each entity is linked to, over the allowed types."""
        ids = {m.id for m in self.memories}
        seen: Dict[str, Set[str]] = {}
        for s, _, t in self.raw:
            if s in ids and t in self.entities:
                seen.setdefault(t, set()).add(s)
            elif t in ids and s in self.entities:
                seen.setdefault(s, set()).add(t)
        return {e: len(m) for e, m in seen.items()}

    def edges(self, nodes_ok: Set[str]) -> List[Dict[str, Any]]:
        seen: Set[Tuple[str, str, str]] = set()
        out: List[Dict[str, Any]] = []
        for is_derived, group in ((False, self.raw), (True, self.derived)):
            for s, rel, t in group:
                key = (s, rel, t)
                if key in seen or s == t or s not in nodes_ok or t not in nodes_ok:
                    continue
                seen.add(key)
                note, origin = self.detail.get(key, ("", "versioning"))
                out.append(
                    {"source": s, "target": t, "type": rel, "derived": is_derived, "note": note, "origin": origin}
                )
        return out

    def entity_node(self, entity_id: str, degree: Dict[str, int], hubs: Optional[Set[str]] = None) -> Dict[str, Any]:
        e = self.entities[entity_id]
        node = {"id": e.id, "kind": "entity", "label": e.name, "type": _enum(e.type), "count": degree.get(e.id, 0)}
        return {**node, "hub": True} if hubs and e.id in hubs else node


def load(graph: GraphStore, project: Optional[str] = None, types: Optional[Set[str]] = None) -> Loaded:
    memories = [m for m in graph.iter_memories() if project is None or m.project_id == project]
    memories.sort(key=lambda m: m.created_at.isoformat() if m.created_at else "", reverse=True)
    entities = {e.id: e for e in graph.iter_entities() if not is_meta_entity(e)}

    def allowed(rel: str) -> bool:
        return types is None or rel in types

    every_rel = list(graph.iter_relationships())
    rels = [r for r in every_rel if allowed(_enum(r.type))]
    # a summary's member count, whatever link types are drawn: live members only
    live = {m.id for m in memories if not (m.superseded_by or m.valid_to)}
    members: Dict[str, int] = {}
    for r in every_rel:
        if _enum(r.type) == RelType.SUMMARIZES.value and r.target_id in live:
            members[r.source_id] = members.get(r.source_id, 0) + 1
    # SQLite records supersession only in the memory row; Neo4j also writes a
    # SUPERSEDES edge. Derive it from the row so both modes draw the chain.
    derived = (
        [(m.superseded_by, RelType.SUPERSEDES.value, m.id) for m in memories if m.superseded_by]
        if allowed(RelType.SUPERSEDES.value)
        else []
    )
    return Loaded(
        memories=memories,
        entities=entities,
        raw=[(r.source_id, _enum(r.type), r.target_id) for r in rels],
        derived=derived,
        # what the panel and the edge labels show: the note, and who drew it
        detail={
            (r.source_id, _enum(r.type), r.target_id): ((r.properties or {}).get(NOTE_KEY, ""), r.source) for r in rels
        },
        members=members,
    )


def build_graph(
    graph: GraphStore,
    project: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    types: Optional[Set[str]] = None,
    seeds: Optional[Set[str]] = None,
    depth: int = 1,
) -> Dict[str, Any]:
    """The newest `limit` memories, or with `seeds`, the nodes within `depth` hops of them.

    `types` keeps only those relationship types, for drawing and for the walk:
    a neighbourhood reached through CONTRADICTS would be most of the store.
    Around seeds, hub entities are reached but not walked through (see HUB_DEGREE).
    """
    data = load(graph, project, types)
    memories, entities, raw, derived = data.memories, data.entities, data.raw, data.derived
    degree = data.memory_degree()
    hubs: Set[str] = set()

    if seeds:
        known = {m.id for m in memories} | set(entities)
        hubs = {e for e, n in degree.items() if n > HUB_DEGREE}
        pairs = [(s, t) for s, _, t in raw + derived if s in known and t in known]
        dist = _reach(seeds & known, pairs, depth, stop=hubs)
        # nearest first, so the cap drops the far rim, never the seeds
        reached = sorted(dist, key=lambda i: dist[i])[:limit]
        reached_set = set(reached)
        in_reach = [m for m in memories if m.id in dist]
        kept = [m for m in in_reach if m.id in reached_set]
        memory_ids = {m.id for m in kept}
        linked_entities = {i for i in reached_set if i in entities}
        total, truncated = len(in_reach), len(dist) > len(reached)
    else:
        kept = memories[:limit]
        memory_ids = {m.id for m in kept}
        # An entity is drawn only when a kept memory reaches it, so a project
        # filter or the cap does not leave a cloud of orphan entities behind.
        touching = [(s, t) for s, _, t in raw if s in memory_ids or t in memory_ids]
        linked_entities = {end for pair in touching for end in pair if end in entities}
        # and one entity-to-entity hop further: "Neo4j is part of the docker stack"
        # should show even when no loaded memory is about the docker stack itself
        linked_entities |= {
            end
            for s, _, t in raw
            if s in entities and t in entities and (s in linked_entities or t in linked_entities)
            for end in (s, t)
        }
        total, truncated = len(memories), len(memories) > len(kept)

    nodes_ok = memory_ids | linked_entities
    nodes = [_memory_node(m, data.members) for m in kept] + [
        data.entity_node(i, degree, hubs) for i in sorted(linked_entities)
    ]
    return {
        "nodes": nodes,
        "edges": data.edges(nodes_ok),
        "total_memories": total,
        "truncated": truncated,
        "limit": limit,
    }
