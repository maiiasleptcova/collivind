"""The graph the explorer draws: memories, entities and the edges between them.

Built from the store's iterators (the same ones export and migrate use), so
docker, embedded and remote modes all produce the same shape. Nodes are kept
slim (no content) because the browser holds all of them at once; the side
panel fetches a memory in full when it is opened.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from collivind.engine.projects import is_meta_entity
from collivind.models.memory import MemoryNode
from collivind.models.relationship import RelType
from collivind.storage.interfaces import GraphStore

# Beyond this many memories a force layout in the browser stops being
# readable long before it stops being fast; newest win and the payload says so.
DEFAULT_LIMIT = 3000
MAX_LIMIT = 10000
LABEL_LEN = 140


def _enum(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _memory_node(m: MemoryNode) -> Dict[str, Any]:
    return {
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


def build_graph(graph: GraphStore, project: Optional[str] = None, limit: int = DEFAULT_LIMIT) -> Dict[str, Any]:
    memories = [m for m in graph.iter_memories() if project is None or m.project_id == project]
    memories.sort(key=lambda m: m.created_at.isoformat() if m.created_at else "", reverse=True)
    kept = memories[:limit]
    memory_ids = {m.id for m in kept}

    entities = {e.id: e for e in graph.iter_entities() if not is_meta_entity(e)}
    seen: Set[Tuple[str, str, str]] = set()
    edges: List[Dict[str, Any]] = []

    def add_edge(source: str, rel: str, target: str, derived: bool = False) -> None:
        key = (source, rel, target)
        if key in seen or source == target:
            return
        seen.add(key)
        edges.append({"source": source, "target": target, "type": rel, "derived": derived})

    raw = [(r.source_id, _enum(r.type), r.target_id) for r in graph.iter_relationships()]
    # An entity is drawn only when a kept memory reaches it, so a project
    # filter or the cap does not leave a cloud of orphan entities behind.
    touching = [(s, t) for s, _, t in raw if s in memory_ids or t in memory_ids]
    linked_entities = {end for pair in touching for end in pair if end in entities}
    nodes_ok = memory_ids | linked_entities
    for s, rel, t in raw:
        if s in nodes_ok and t in nodes_ok:
            add_edge(s, rel, t)
    # SQLite records supersession only in the memory row; Neo4j also writes a
    # SUPERSEDES edge. Derive it from the row so both modes draw the chain.
    for m in kept:
        if m.superseded_by and m.superseded_by in memory_ids:
            add_edge(m.superseded_by, RelType.SUPERSEDES.value, m.id, derived=True)

    nodes = [_memory_node(m) for m in kept] + [
        {"id": e.id, "kind": "entity", "label": e.name, "type": _enum(e.type)}
        for e in sorted((entities[i] for i in linked_entities), key=lambda e: e.id)
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "total_memories": len(memories),
        "truncated": len(memories) > len(kept),
        "limit": limit,
    }
