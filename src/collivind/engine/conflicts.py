"""Conflict candidates on write, and triage of the old auto-detected CONTRADICTS backlog.

Measured on a real 2,970-memory store (2026-11-23): the old detector wrote an
edge for every same-category memory at cosine >= 0.75, which gave 5,560
CONTRADICTS edges. A 40-pair sample held 0 genuine contradictions, 3 newer
versions of the same fact and 37 compatible or unrelated pairs. Updates
(0.86-0.89) and unrelated pairs (up to 0.89) overlap in cosine, and word
overlap does not separate them either. So similarity alone never decides that
something is a contradiction. On write the server returns a few gated
candidates and the agent judges them (see CONFLICT_INSTRUCTION). The review
command only re-labels what the new gate would not have surfaced, and it
retires only one of two identical memories (same_memory), the same rule
add_memory applies (see engine/dedup.py).
"""

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Set, Union

from collivind.models import MemoryCategory, MemoryNode, RelationshipCreate, RelType

if TYPE_CHECKING:
    from collivind.engine.memory_manager import MemoryManager
    from collivind.models import MemoryCreate
    from collivind.storage.interfaces import GraphStore

# Cosine (enriched text vs enriched text) a stored memory must reach before a
# new one is checked against it. ponytail: tuned on one store; if agents see too
# many candidates, raise it, and if real conflicts slip through, lower it.
CONFLICT_THRESHOLD = 0.85
MAX_CANDIDATES = 3
MIN_WORD_OVERLAP = 0.15

CONFLICT_INSTRUCTION = (
    "These existing memories are close to the one just saved. Check each one. "
    "If it states the same fact as the new memory (said differently, or with other detail around it), "
    "or the new memory is a newer state of it, call collivind_invalidate on the existing one with "
    "superseded_by set to the memory just saved; keep both only if the existing one holds detail the "
    "new one lacks. If it is a different fact that only looks similar, ignore it. If they genuinely "
    "disagree, check the source of truth (the code, docs, git history, the user's own words) and "
    "invalidate the wrong one the same way. If you still cannot tell, ask the user one short question, "
    "invalidate the wrong one after they answer, and save their answer as a memory. If nobody can be "
    "asked now, call collivind_mark_conflict on the pair so it is flagged for review. Do not use "
    "collivind_forget for this, because invalidating keeps the history."
)

_STOP = frozenset(
    "the and for with that this from was were are has have had not but its into after before when then "
    "than also only been being which while what their there they them about over under more most very "
    "just can will would should could each other same such these those your you our out off per via any "
    "all one two".split()
)
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:t[\d:.]+(?:z|[+-]\d{2}:\d{2})?)?\b")
_WORD = re.compile(r"[a-z][a-z0-9_]{2,}")
_NUMBER = re.compile(r"\b\d{2,}\b")
_HEX = re.compile(r"\b[0-9a-f]{7,40}\b")
_REF = re.compile(r"(?:\bpr\s*#?|\bissue\s*#?|\bpull/|#)(\d+)\b")


def _words(text: str) -> Set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _anchors(text: str) -> Set[str]:
    """Identifiers that pin a memory to one subject: PR/issue numbers, ports, SHAs. Dates are left out."""
    text = _DATE.sub(" ", text.lower())
    hexes = {h for h in _HEX.findall(text) if re.search(r"\d", h) and re.search(r"[a-f]", h)}
    return set(_NUMBER.findall(text)) | hexes


def same_subject(a: str, b: str) -> bool:
    """Cheap lexical check that two similar memories talk about the same thing.

    Two memories about different PRs (776 vs 773) embed almost identically,
    so when both name a PR or issue they must name the same one. Any other
    shared identifier (a SHA, a port) is enough; otherwise a real overlap in
    wording is required. Other numbers never veto: "timeout is 30s" vs
    "timeout is 60s" is exactly the conflict to catch.
    """
    ra, rb = set(_REF.findall(a.lower())), set(_REF.findall(b.lower()))
    if ra and rb:
        return bool(ra & rb)
    if _anchors(a) & _anchors(b):
        return True
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / max(1, len(wa | wb)) >= MIN_WORD_OVERLAP


def same_memory(a: Union[MemoryNode, "MemoryCreate"], b: Union[MemoryNode, "MemoryCreate"]) -> bool:
    """A resubmission: the same content, summary, category and tags; only outer whitespace may differ.

    Nothing looser. Case and inner whitespace matter in code (Button.tsx vs
    button.tsx, indentation), and the same status line tagged pr-212 and
    pr-213 is two facts.
    """
    return (
        a.content.strip() == b.content.strip()
        and (a.summary or "").strip() == (b.summary or "").strip()
        and MemoryCategory(a.category) == MemoryCategory(b.category)
        and set(a.tags or []) == set(b.tags or [])
    )


def is_live(memory: Optional[MemoryNode]) -> bool:
    return memory is not None and memory.valid_to is None and not memory.superseded_by


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return dot / norm if norm else 0.0


# Review buckets. Only REVIEW keeps its CONTRADICTS edge.
RESOLVED = "resolved"  # an endpoint is gone or already invalidated: the edge is history
UNRELATED = "unrelated"  # below the new gate: re-label to RELATES_TO
DUPLICATE = "duplicate"  # identical memories (same_memory): retire the older one
REVIEW = "review"  # passes the gate: a person or agent has to read it


@dataclass(frozen=True)
class EdgeVerdict:
    source_id: str
    target_id: str
    bucket: str
    score: Optional[float]
    reason: str


def classify_edge(
    a: Optional[MemoryNode], b: Optional[MemoryNode], score: Optional[float], dedup_threshold: float
) -> tuple[str, str]:
    """Bucket one auto-detected CONTRADICTS edge. Returns (bucket, reason)."""
    if not is_live(a) or not is_live(b):
        return RESOLVED, "one side is already invalidated or deleted"
    assert a is not None and b is not None
    if same_memory(a, b):
        return DUPLICATE, "identical content, summary, category and tags"
    if score is None:
        return REVIEW, "no live vector to compare"
    if score < CONFLICT_THRESHOLD:
        return UNRELATED, f"cosine {score:.3f} < {CONFLICT_THRESHOLD}"
    if not same_subject(a.content, b.content):
        return UNRELATED, f"cosine {score:.3f} but no shared identifier or wording"
    if score >= dedup_threshold:
        return REVIEW, f"cosine {score:.3f} >= dedup threshold {dedup_threshold}, same subject: likely duplicate"
    return REVIEW, f"cosine {score:.3f}, same subject"


def classify_edges(
    edges: Iterable[tuple[str, str]],
    memories: Dict[str, Optional[MemoryNode]],
    vectors: Dict[str, List[float]],
    dedup_threshold: float,
) -> List[EdgeVerdict]:
    verdicts = []
    for src, tgt in edges:
        va, vb = vectors.get(src), vectors.get(tgt)
        score = cosine(va, vb) if va is not None and vb is not None else None
        bucket, reason = classify_edge(memories.get(src), memories.get(tgt), score, dedup_threshold)
        verdicts.append(EdgeVerdict(src, tgt, bucket, score, reason))
    return verdicts


REVIEW_SOURCE = "contradictions_review"


def review_backlog(
    manager: "MemoryManager", project_id: Optional[str] = None, apply: bool = False
) -> List[EdgeVerdict]:
    """Bucket every auto-detected CONTRADICTS edge and, with apply, act on the buckets.

    Only edges the old detector wrote (source "auto_detection") are touched;
    a CONTRADICTS edge a person drew is left alone. With apply:
    RESOLVED and UNRELATED edges are re-labelled RELATES_TO (source
    "contradictions_review", so they can be found and reverted); for a
    DUPLICATE the older memory is invalidated with the newer one as its
    successor (history kept, vector dropped) and the edge removed; REVIEW
    edges stay as they are.
    """
    graph, vectors_store = manager.graph_store, manager.vector_store
    edges = [e for e in graph.iter_relationships() if e.type == RelType.CONTRADICTS and e.source == "auto_detection"]
    ids = {i for e in edges for i in (e.source_id, e.target_id)}
    memories: Dict[str, Optional[MemoryNode]] = {i: graph.get_memory(i) for i in ids}
    if project_id is not None:
        edges = [
            e
            for e in edges
            if any(m is not None and m.project_id == project_id for m in (memories[e.source_id], memories[e.target_id]))
        ]
    # ponytail: holds every live vector in memory (~3k x 768 floats is fine);
    # page through ids if stores grow to hundreds of thousands
    vectors = {vid: vec for vid, vec, _ in vectors_store.iter_all() if vid in ids}
    verdicts = classify_edges(
        [(e.source_id, e.target_id) for e in edges], memories, vectors, manager.config.search.dedup_threshold
    )
    if apply:
        _apply(manager, verdicts, memories)
    return verdicts


def _apply(manager: "MemoryManager", verdicts: List[EdgeVerdict], memories: Dict[str, Optional[MemoryNode]]) -> None:
    graph = manager.graph_store
    retired: Set[str] = set()
    for v in verdicts:
        if v.bucket == REVIEW:
            continue
        if v.bucket == DUPLICATE and not {v.source_id, v.target_id} & retired:
            older, newer = sorted((memories[v.source_id], memories[v.target_id]), key=_created)
            assert older is not None and newer is not None
            # invalidate overwrites the successor's previous_version_id; never
            # cut an existing version chain (or one set earlier in this run)
            current = graph.get_memory(newer.id)
            if current is not None and not current.previous_version_id:
                graph.delete_relationship(v.source_id, v.target_id, RelType.CONTRADICTS.value)
                manager.invalidate(older.id, newer.id, "duplicate found by contradictions review")
                retired.add(older.id)
                continue
        graph.delete_relationship(v.source_id, v.target_id, RelType.CONTRADICTS.value)
        graph.create_relationship(
            RelationshipCreate(
                source_id=v.source_id,
                target_id=v.target_id,
                type=RelType.RELATES_TO,
                confidence=v.score if v.score is not None else 0.0,
                source=REVIEW_SOURCE,
                properties={"relabelled_from": RelType.CONTRADICTS.value, "bucket": v.bucket},
            )
        )


def _created(m: Optional[MemoryNode]) -> str:
    return m.created_at.isoformat() if m is not None and m.created_at else ""


# Before 0.22 add_memory retired a live memory whenever a new one landed within
# cosine 0.92 of it, in the same call that stored the new one.
AUTO_SUPERSEDE_WINDOW = 2.0  # seconds between the successor's creation and the retirement


def auto_superseded(graph: "GraphStore", project_id: Optional[str] = None) -> List[tuple[MemoryNode, MemoryNode]]:
    """(retired, successor) pairs that look retired by similarity alone, oldest first. Read only.

    ponytail: neither store records who retired a memory, so this goes by
    timing: retired within AUTO_SUPERSEDE_WINDOW of the successor's creation
    means the same add call did it. An agent's collivind_invalidate that fast
    would be listed too, and a `contradictions review --apply` retirement is not.
    """
    nodes = {m.id: m for m in graph.iter_memories()}
    pairs = []
    for m in nodes.values():
        new = nodes.get(m.superseded_by or "")
        if new is None or m.valid_to is None or new.created_at is None or m.category == MemoryCategory.SUMMARY:
            continue
        if project_id is not None and m.project_id != project_id:
            continue
        if abs((_aware(m.valid_to) - _aware(new.created_at)).total_seconds()) <= AUTO_SUPERSEDE_WINDOW:
            pairs.append((m, new))
    return sorted(pairs, key=lambda p: _created(p[0]))


def _aware(when: datetime) -> datetime:
    # legacy rows may hold naive timestamps; everything collivind writes is UTC
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)
