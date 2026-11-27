"""Merge duplicate entities into one: re-point every edge, keep the old names as aliases.

Everything goes through the GraphStore ABC (edges_of, restore_relationship,
delete_relationship, restore_entity, delete_entity), so SQLite and Neo4j run
the same logic. Nothing here is atomic across calls; instead every step is
re-runnable. Order: backup, aliases on the kept entity (so writes that land
mid-merge already resolve to it), new edges, old edges deleted, old entities
deleted last (delete_entity refuses while an edge remains).
"""

import json
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from collivind.engine.projects import is_meta_entity
from collivind.engine.summaries import GROUP_TAG
from collivind.models.entity import ALIASES_KEY, EntityNode, entity_slug
from collivind.models.relationship import RelationshipEdge, RelType, stable_edge_id
from collivind.storage.interfaces import GraphStore

Triple = Tuple[str, str, str]


def _rel(edge: RelationshipEdge) -> str:
    return RelType(edge.type).value


def _triple(edge: RelationshipEdge) -> Triple:
    return (edge.source_id, _rel(edge), edge.target_id)


@dataclass(frozen=True)
class MergePlan:
    keep: EntityNode
    others: Tuple[EntityNode, ...]
    removed: Tuple[RelationshipEdge, ...]  # every edge touching an entity being merged away
    moved: Tuple[RelationshipEdge, ...]  # their replacements, re-pointed to the kept entity
    dropped: Tuple[Tuple[RelationshipEdge, str], ...]  # (old edge, "self-loop" | "duplicate")
    aliases: Tuple[str, ...]  # the kept entity's aliases after the merge
    topic_summaries: Tuple[str, ...]  # ids of summaries tagged with a merged entity's topic key

    def counts(self) -> Dict[str, int]:
        memory_edges = sum(1 for e in self.moved if _rel(e) in (RelType.ABOUT.value, RelType.MENTIONS.value))
        return {
            "edges_removed": len(self.removed),
            "edges_moved": len(self.moved),
            "memory_edges_moved": memory_edges,
            "entity_edges_moved": len(self.moved) - memory_edges,
            "self_loops_dropped": sum(1 for _, why in self.dropped if why == "self-loop"),
            "duplicates_dropped": sum(1 for _, why in self.dropped if why == "duplicate"),
        }


def _resolve(graph: GraphStore, name: str) -> EntityNode:
    """The stored entity a name slugs to. Deliberately not alias-aware: after the
    aliases are written an alias lookup would map the other name onto the kept one."""
    entity = graph.get_entity(name)
    if entity is None:
        raise LookupError(f"no entity named {name!r}")
    if is_meta_entity(entity):
        raise ValueError(f"{name!r} is project metadata, not an entity")
    return entity


def plan_merge(graph: GraphStore, keep_name: str, other_names: Sequence[str]) -> MergePlan:
    """What a merge would change. Reads only."""
    keep = _resolve(graph, keep_name)
    already = {entity_slug(a) for a in (keep.properties or {}).get(ALIASES_KEY) or [] if isinstance(a, str)}
    others: Dict[str, EntityNode] = {}
    for name in other_names:
        if graph.get_entity(name) is None and entity_slug(name) in already:
            continue  # merged by an earlier (possibly interrupted) run
        entity = _resolve(graph, name)
        if entity.id == keep.id:
            raise ValueError(f"{name!r} is the entity being kept")
        others.setdefault(entity.id, entity)
    if not others:
        raise ValueError("nothing to merge: name at least one entity that still exists apart from the kept one")

    gone = set(others)
    taken: Set[Triple] = {_triple(e) for e in graph.edges_of(keep.id)}
    removed: Dict[Tuple[str, ...], RelationshipEdge] = {}
    for oid in others:
        for edge in graph.edges_of(oid):  # an edge between two merged entities shows up twice
            removed.setdefault((edge.id, *_triple(edge)), edge)

    moved: List[RelationshipEdge] = []
    dropped: List[Tuple[RelationshipEdge, str]] = []
    for edge in removed.values():
        source = keep.id if edge.source_id in gone else edge.source_id
        target = keep.id if edge.target_id in gone else edge.target_id
        new = (source, _rel(edge), target)
        if source == target:
            dropped.append((edge, "self-loop"))
        elif new in taken:
            dropped.append((edge, "duplicate"))
        else:
            taken.add(new)
            # a deterministic id makes a re-run after a crash find this edge as "taken"
            moved.append(replace(edge, id=stable_edge_id(*new), source_id=source, target_id=target))

    keep_aliases = (keep.properties or {}).get(ALIASES_KEY) or []
    names = [*keep_aliases]
    for e in others.values():
        names += [e.name, *((e.properties or {}).get(ALIASES_KEY) or [])]
    aliases = tuple(
        dict.fromkeys(n.strip() for n in names if isinstance(n, str) and n.strip() and n.strip() != keep.name)
    )
    topic_tags = {f"{GROUP_TAG}topic:{oid}" for oid in others}
    # ponytail: full memory scan to find topic summaries; fine at 10k memories, index tags if it is not
    summaries = tuple(m.id for m in graph.iter_memories() if topic_tags & set(m.tags or []))
    return MergePlan(
        keep, tuple(others.values()), tuple(removed.values()), tuple(moved), tuple(dropped), aliases, summaries
    )


def write_backup(plan: MergePlan, path: Path) -> None:
    """Every row the merge touches, as bundle-style lines, fsynced before any write.

    `entity` and `relationship` lines are the pre-merge state (restorable with
    the bundle parser); `created` lines are the edges the merge will add, so an
    undo knows what to delete.
    """
    from collivind.engine.bundle import _edge_dict, _entity_dict  # bundle imports MemoryManager, which imports us

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as out:  # never overwrite an earlier backup
        header = {
            "kind": "entity_merge",
            "at": datetime.now(timezone.utc).isoformat(),
            "keep": plan.keep.id,
            "others": [e.id for e in plan.others],
        }
        out.write(json.dumps(header) + "\n")
        for entity in (plan.keep, *plan.others):
            out.write(json.dumps({"kind": "entity", **_entity_dict(entity)}) + "\n")
        for edge in plan.removed:
            out.write(json.dumps({"kind": "relationship", **_edge_dict(edge)}) + "\n")
        for edge in plan.moved:
            out.write(json.dumps({"kind": "created", **_edge_dict(edge)}) + "\n")
        out.flush()
        os.fsync(out.fileno())


def apply_merge(graph: GraphStore, plan: MergePlan, backup: Path) -> Dict[str, int]:
    """Carry out a plan from plan_merge. Interrupted? Plan and apply again: it picks up where it stopped."""
    write_backup(plan, backup)
    merged_props: Dict = {}
    for e in plan.others:
        merged_props.update(e.properties or {})
    keep_props = {**merged_props, **(plan.keep.properties or {}), ALIASES_KEY: list(plan.aliases)}
    graph.restore_entity(replace(plan.keep, properties=keep_props, updated_at=datetime.now(timezone.utc)))
    for edge in plan.moved:
        graph.restore_relationship(edge)
    for source, rel, target in dict.fromkeys(_triple(e) for e in plan.removed):
        graph.delete_relationship(source, target, rel)
    for e in plan.others:
        graph.delete_entity(e.id)
    return {**plan.counts(), "entities_removed": len(plan.others)}
