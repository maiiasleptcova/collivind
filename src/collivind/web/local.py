"""The local graph around one node, and the cause-and-fix chain through a memory.

Measured on a real store (2026-11-25, 3,080 memories): 2 hops around a
memory went from 4 nodes to 308, all reached through entities like "Docker"
that hundreds of memories are about. Sharing such a hub says almost nothing
about how two memories relate, so the local graph:
- reaches a hub (an entity with more than HUB_DEGREE memories) but does not
  walk through it, unless the reader opens it (`expand`);
- ranks what it reaches: a typed memory-to-memory edge (RESOLVES, CAUSED_BY,
  SUPERSEDES ...) outranks a shared entity, and a shared entity counts for
  less the more memories share it (1/sqrt of its degree), so an opened hub
  lists first the memories that also share something else with the centre;
- keeps the centre and its direct neighbours, then the best of the rest up to
  `limit`, and says how many it left out (`more`).
Ranking is structural only: no embedding round-trip per click.
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from collivind.models.relationship import RelType
from collivind.storage.interfaces import GraphStore
from collivind.web.graph import HUB_DEGREE, _memory_node, load

LOCAL_LIMIT = 60
MAX_LOCAL_LIMIT = 500
# how much one edge of a type says about the relation between its ends
WEIGHT = {
    RelType.RESOLVES.value: 6.0,
    RelType.CAUSED_BY.value: 6.0,
    RelType.SUPERSEDES.value: 4.0,
    RelType.DEPENDS_ON_MEM.value: 3.0,
    RelType.RELATES_TO.value: 2.0,
    RelType.CONTRADICTS.value: 2.0,
}
CHAIN_TYPES = (RelType.CAUSED_BY.value, RelType.RESOLVES.value, RelType.SUPERSEDES.value)
MAX_CHAIN = 12


def neighbourhood(
    graph: GraphStore,
    center: str,
    depth: int = 1,
    limit: int = LOCAL_LIMIT,
    types: Optional[Set[str]] = None,
    expand: Optional[Set[str]] = None,
    project: Optional[str] = None,
) -> Dict[str, Any]:
    """Up to `limit` nodes within `depth` hops of `center`, best first (see the module doc)."""
    data = load(graph, project, types)
    memory_by_id = {m.id: m for m in data.memories}
    known = set(memory_by_id) | set(data.entities)
    if center not in known:
        return {"nodes": [], "edges": [], "total_memories": 0, "truncated": False, "limit": limit, "more": 0}
    degree = data.memory_degree()
    hubs = {e for e, n in degree.items() if n > HUB_DEGREE} - (expand or set())

    adj: Dict[str, List[Tuple[str, str]]] = {}
    for s, rel, t in data.raw + data.derived:
        if s in known and t in known and s != t:
            adj.setdefault(s, []).append((t, rel))
            adj.setdefault(t, []).append((s, rel))

    dist = {center: 0}
    score: Dict[str, float] = {}
    frontier = [center]
    for hop in range(1, depth + 1):
        nxt: List[str] = []
        for node in frontier:
            # the centre is walked even when it is a hub: its members are what was asked for
            if node != center and node in hubs:
                continue
            spread = 1 / math.sqrt(max(1, degree.get(node, 1))) if node in data.entities else 1.0
            for other, rel in adj.get(node, ()):
                if other not in dist:
                    dist[other] = hop
                    nxt.append(other)
                if dist[other] == hop:
                    score[other] = score.get(other, 0.0) + WEIGHT.get(rel, 1.0) * spread / hop
        frontier = nxt
    # An opened hub lends its members even at the edge of the reach: opening
    # "Docker" at 1 hop is a request to see what is about Docker. Its members
    # that are also tied to something already shown rank first.
    shown = set(dist)

    def tie(node: str, rel: str) -> float:
        return WEIGHT.get(rel, 1.0) / (math.sqrt(max(1, degree.get(node, 1))) if node in data.entities else 1.0)

    for hub in sorted((expand or set()) & shown):
        if dist[hub] < depth:
            continue  # already walked through above
        for other, rel in adj.get(hub, ()):
            if other in shown:
                continue
            if other not in dist:
                dist[other] = dist[hub] + 1
                score[other] = sum(tie(n, r) for n, r in adj.get(other, ()) if n in shown and n != hub)
            score[other] += tie(hub, rel) / dist[other]

    def newest(i: str) -> str:
        m = memory_by_id.get(i)
        return m.created_at.isoformat() if m and m.created_at else ""

    # newest first among equals, then nearest first, then best first (stable sorts)
    ranked = sorted((i for i in dist if i != center), key=newest, reverse=True)
    ranked.sort(key=lambda i: -score.get(i, 0.0))
    ranked.sort(key=lambda i: min(dist[i], 2))  # the centre's own neighbours always come first
    kept = [center, *ranked[: max(0, limit - 1)]]
    kept_set = set(kept)
    nodes = [
        (
            {**_memory_node(memory_by_id[i], data.members), "hop": dist[i]}
            if i in memory_by_id
            else {**data.entity_node(i, degree, hubs), "hop": dist[i], "expanded": i in (expand or set())}
        )
        for i in kept
    ]
    return {
        "nodes": nodes,
        "edges": data.edges(kept_set),
        "total_memories": sum(1 for i in dist if i in memory_by_id),
        "truncated": len(dist) > len(kept),
        "limit": limit,
        "more": len(dist) - len(kept),
        "center": center,
    }


def chain(graph: GraphStore, memory_id: str) -> Optional[Dict[str, Any]]:
    """The memories linked to this one through CAUSED_BY, RESOLVES and SUPERSEDES, nearest first.

    Followed in both directions and transitively, so from an error you reach
    its cause, the fix that resolves it, and the fix's later versions. None
    when the memory does not exist.
    """
    data = load(graph, types=set(CHAIN_TYPES))
    memory_by_id = {m.id: m for m in data.memories}
    if memory_id not in memory_by_id:
        return None
    links = [(s, rel, t) for s, rel, t in data.raw + data.derived if s in memory_by_id and t in memory_by_id]
    seen = {memory_id}
    order: List[Tuple[str, str, str]] = []
    frontier = [memory_id]
    while frontier and len(seen) < MAX_CHAIN:
        nxt = []
        for node in frontier:
            for s, rel, t in links:
                if node not in (s, t):
                    continue
                other = t if s == node else s
                if other in seen or len(seen) >= MAX_CHAIN:
                    continue
                seen.add(other)
                nxt.append(other)
                order.append((s, rel, t))
        frontier = nxt
    notes = {k: v[0] for k, v in data.detail.items()}
    return {
        "links": [{"source": s, "type": rel, "target": t, "note": notes.get((s, rel, t), "")} for s, rel, t in order],
        "memories": {i: _memory_node(memory_by_id[i], data.members) for i in seen},
    }
