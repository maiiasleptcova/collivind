"""Backup bundle (export format v2): every memory, entity, edge and vector, by id.

JSONL. Line 1 is a header; then memory, entity, relationship and vector lines,
each tagged with "kind". Vector lines always carry id + payload, so import
knows exactly which memories were searchable (superseded versions are not);
the float array itself is optional and re-embedded when absent.

Import validates the whole bundle (parse, per-kind counts against the header)
before the first write, then adds only what the target lacks via
`migrate.restore`, so a re-import never reverts an edit made since.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List, Optional, Set, TextIO, Tuple

from collivind.engine.migrate import MigrationReport, Stores, restore
from collivind.models.entity import EntityNode, EntityType
from collivind.models.memory import MemoryNode
from collivind.models.relationship import RelationshipEdge, RelType
from collivind.version import __version__

FORMAT = "collivind-backup"
FORMAT_VERSION = 2
KINDS = ("memory", "entity", "relationship", "vector")

Space = Tuple[str, Optional[str], int]


class BundleError(ValueError):
    """The bundle is malformed, truncated or from a newer format."""


@dataclass(frozen=True)
class Bundle:
    header: dict
    memories: List[MemoryNode]
    entities: List[EntityNode]
    relationships: List[RelationshipEdge]
    vectors: List[Tuple[str, Optional[List[float]], dict]]

    @property
    def space(self) -> Space:
        e = self.header["embedding"]
        return e["family"], e["model"], e["dimension"]

    @property
    def has_vectors(self) -> bool:
        return all(vec is not None for _, vec, _ in self.vectors)


def is_bundle_header(record: object) -> bool:
    return isinstance(record, dict) and record.get("format") == FORMAT


# ---- serialisation -------------------------------------------------------


def _iso(when: Optional[datetime]) -> Optional[str]:
    return when.isoformat() if when else None


def _when(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value else None  # absent stays absent, not "now"


def _entity_dict(e: EntityNode) -> dict:
    return {
        "id": e.id,
        "name": e.name,
        "type": EntityType(e.type).value,
        "properties": e.properties,
        "created_at": _iso(e.created_at),
        "updated_at": _iso(e.updated_at),
    }


def _entity(d: dict) -> EntityNode:
    return EntityNode(
        id=d["id"],
        name=d["name"],
        type=EntityType(d["type"]),
        properties=dict(d.get("properties") or {}),
        created_at=_when(d.get("created_at")),
        updated_at=_when(d.get("updated_at")),
    )


def _edge_dict(r: RelationshipEdge) -> dict:
    return {
        "id": r.id,
        "source_id": r.source_id,
        "target_id": r.target_id,
        "type": RelType(r.type).value,
        "confidence": r.confidence,
        "source": r.source,
        "properties": r.properties,
        "created_at": _iso(r.created_at),
    }


def _edge(d: dict) -> RelationshipEdge:
    return RelationshipEdge(
        id=d["id"],
        source_id=d["source_id"],
        target_id=d["target_id"],
        type=RelType(d["type"]),
        confidence=d.get("confidence", 1.0),
        source=d.get("source") or "manual",
        properties=dict(d.get("properties") or {}),
        created_at=_when(d.get("created_at")),
    )


# ---- export --------------------------------------------------------------


def _selection(src: Stores, project: Optional[str]) -> Tuple[Dict[str, Set[str]], int]:
    """Ids to export per kind, plus how many edges touching them were left out (dangling or cross-project)."""
    memory_ids = {m.id for m in src.graph.iter_memories() if project is None or m.project_id == project}
    entity_ids = {e.id for e in src.graph.iter_entities()}
    edges = [(r.id, r.source_id, r.target_id) for r in src.graph.iter_relationships()]
    if project is not None:
        # Entities have no project: keep those a selected memory links to.
        linked = {end for _, s, t in edges if s in memory_ids or t in memory_ids for end in (s, t)}
        entity_ids &= linked
    nodes = memory_ids | entity_ids
    edge_ids = {rid for rid, s, t in edges if s in nodes and t in nodes}
    touching = sum(1 for _, s, t in edges if s in nodes or t in nodes)
    vector_ids = {vid for vid, _, _ in src.vector.iter_all() if project is None or vid in memory_ids}
    ids = {"memory": memory_ids, "entity": entity_ids, "relationship": edge_ids, "vector": vector_ids}
    return ids, touching - len(edge_ids)


def export_bundle(
    src: Stores, out: TextIO, *, space: Space, project: Optional[str] = None, vectors: bool = False
) -> Tuple[Dict[str, int], int]:
    """Stream a v2 bundle to `out`. Returns (count per kind, edges skipped).

    Two passes over the stores: ids first (so the header can carry counts),
    then the records. Only id sets are held in memory.
    """
    ids, skipped = _selection(src, project)
    counts = {kind: len(ids[kind]) for kind in KINDS}
    family, model, dimension = space
    # Allow-list only: nothing from config (it holds API keys and passwords).
    header = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "collivind_version": __version__,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "project": project,
        "embedding": {"family": family, "model": model, "dimension": dimension},
        "vectors_included": vectors,
        "counts": counts,
    }
    out.write(json.dumps(header) + "\n")

    written = {kind: 0 for kind in KINDS}

    def emit(kind: str, record: dict) -> None:
        out.write(json.dumps({"kind": kind, **record}) + "\n")
        written[kind] += 1

    for m in src.graph.iter_memories():
        if m.id in ids["memory"]:
            emit("memory", m.to_dict())
    for e in src.graph.iter_entities():
        if e.id in ids["entity"]:
            emit("entity", _entity_dict(e))
    for r in src.graph.iter_relationships():
        if r.id in ids["relationship"]:
            emit("relationship", _edge_dict(r))
    for vid, vec, payload in src.vector.iter_all():
        if vid in ids["vector"]:
            emit("vector", {"id": vid, "payload": payload, "vector": [float(x) for x in vec] if vectors else None})
    if written != counts:
        # Something wrote to the store between the two passes; the header now
        # lies and import would (rightly) reject the file.
        raise RuntimeError(f"store changed during export (header {counts}, wrote {written}); run export again")
    return counts, skipped


# ---- import --------------------------------------------------------------


def _parse(kind: str, rec: dict):
    if kind == "memory":
        return MemoryNode.from_dict(rec)
    if kind == "entity":
        return _entity(rec)
    if kind == "relationship":
        return _edge(rec)
    vec = rec.get("vector")
    if not isinstance(rec.get("payload"), dict) or not (vec is None or isinstance(vec, list)):
        raise ValueError("vector needs a payload object and a list or null vector")
    return rec["id"], vec, rec["payload"]


def read_bundle(lines: Iterable[str]) -> Bundle:
    """Parse and fully validate a bundle. Raises BundleError; never partial.

    ponytail: holds every record in memory (as `migrate` does); stream the
    restore if backups outgrow RAM.
    """
    numbered: Iterator[Tuple[int, str]] = ((n, line) for n, line in enumerate(lines, 1) if line.strip())
    try:
        _, first = next(numbered)
    except StopIteration:
        raise BundleError("bundle is empty") from None
    header = _json(1, first)
    if not is_bundle_header(header):
        raise BundleError("first line is not a collivind backup header")
    version = header.get("format_version")
    if not isinstance(version, int) or version > FORMAT_VERSION:
        raise BundleError(f"bundle format {version!r} is newer than this collivind supports; upgrade collivind")
    try:
        expected = {kind: int(header["counts"][kind]) for kind in KINDS}
        emb = header["embedding"]
        emb["family"], emb["model"], int(emb["dimension"])  # noqa: B018 - presence check
    except (KeyError, TypeError, ValueError) as e:
        raise BundleError(f"bundle header is incomplete: {e}") from None

    parsed: Dict[str, list] = {kind: [] for kind in KINDS}
    for n, line in numbered:
        rec = _json(n, line)
        kind = rec.pop("kind", None) if isinstance(rec, dict) else None
        if kind not in parsed:
            raise BundleError(f"line {n}: unknown record kind {kind!r}")
        try:
            parsed[kind].append(_parse(kind, rec))
        except (KeyError, TypeError, ValueError) as e:
            raise BundleError(f"line {n}: bad {kind} record: {e!r}") from None

    got = {kind: len(parsed[kind]) for kind in KINDS}
    if got != expected:
        raise BundleError(f"bundle is truncated or altered: header promises {expected}, file holds {got}")
    return Bundle(header, parsed["memory"], parsed["entity"], parsed["relationship"], parsed["vector"])


def _json(n: int, line: str):
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        raise BundleError(f"line {n} is not valid JSON (truncated file?): {e}") from None


def import_bundle(bundle: Bundle, dst: Stores, *, space: Space, dry_run: bool) -> MigrationReport:
    """Restore by id, adding only what `dst` lacks. Re-embeds when the bundle
    has no vectors or they come from another embedding space."""
    reembed = not bundle.has_vectors or tuple(bundle.space) != tuple(space)
    return restore(
        bundle.memories,
        bundle.entities,
        bundle.relationships,
        bundle.vectors,
        dst,
        reembed=reembed,
        dry_run=dry_run,
    )
