"""Find entities that are probably one thing under several names. Reads only; never merges.

Each candidate pair carries its evidence:
- "same-name": names equal once case, spaces and punctuation are ignored ("PR 774" / "PR #774").
- "basename": a path and its bare last part. They must share a memory, unless
  the bare name is a file name and the path is the only one ending in it (so
  `__init__.py` does not pair with every package, nor ~/Documents/cybergine with Cybergine).
- "contains": one name's words appear in order inside the other's ("Cybergine" /
  "Cybergine v5_flask"), sharing at least one memory. Never into a path or file name.
- "embedding": cosine similarity of the names at or above a threshold, with the
  same digits (so "PR 774" / "PR 775" never pair). Not for paths or file names:
  sibling files ("email_fetch.py" / "email_mime.py") embed close together.
Strong evidence (same-name, basename) is joined into groups with union-find.
Weak evidence (contains, embedding) stays a pair of its own: chained, it would
pull every "Cybergine PR 774" into one "Cybergine" group. Strong groups list first.
"""

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from collivind.engine.projects import is_meta_entity
from collivind.models.entity import EntityNode
from collivind.models.relationship import RelType
from collivind.storage.interfaces import EmbeddingProvider, GraphStore

logger = logging.getLogger(__name__)

MEMORY_EDGES = (RelType.ABOUT.value, RelType.MENTIONS.value)
MIN_CONTAINED = 4  # "api" inside "api gateway" is too weak a signal on its own
EMBED_BATCH = 256


def _words(name: str) -> Tuple[str, ...]:
    return tuple(re.findall(r"[0-9a-z]+", name.casefold()))


def _norm(name: str) -> str:
    return "".join(_words(name))


def _digits(name: str) -> Tuple[str, ...]:
    return tuple(re.findall(r"\d+", name))


def _pathlike(name: str) -> bool:
    return "/" in name or re.search(r"\.[A-Za-z0-9]{1,5}$", name) is not None


STRONG = ("same-name", "basename")


def _contains(short: Tuple[str, ...], long: Tuple[str, ...]) -> bool:
    n = len(short)
    return 0 < n < len(long) and any(long[i : i + n] == short for i in range(len(long) - n + 1))


@dataclass(frozen=True)
class _Stats:
    memories: Dict[str, Set[str]]  # entity id -> memory ids linked ABOUT/MENTIONS
    edges: Dict[str, int]  # entity id -> every edge touching it


def _stats(graph: GraphStore, ids: Set[str]) -> _Stats:
    memories: Dict[str, Set[str]] = defaultdict(set)
    edges: Dict[str, int] = defaultdict(int)
    for e in graph.iter_relationships():  # one pass: find_related_memories caps at a limit
        rel = RelType(e.type).value
        for end in {e.source_id, e.target_id} & ids:
            edges[end] += 1
        if rel in MEMORY_EDGES and e.target_id in ids:
            memories[e.target_id].add(e.source_id)
    return _Stats(memories, edges)


def _embedding_pairs(
    entities: List[EntityNode], embedder: EmbeddingProvider, threshold: float
) -> List[Tuple[str, str, float]]:
    names = [e.name for e in entities]
    vectors: List[List[float]] = []
    for i in range(0, len(names), EMBED_BATCH):
        vectors += embedder.embed_batch(names[i : i + EMBED_BATCH])
    m = np.asarray(vectors, dtype=np.float32)
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    out = []
    for start in range(0, len(m), 1024):  # blocks keep the similarity matrix small
        sims = m[start : start + 1024] @ m.T
        for i, j in zip(*np.nonzero(sims >= threshold)):
            a, b = start + int(i), int(j)
            if a < b and _digits(names[a]) == _digits(names[b]):
                out.append((entities[a].id, entities[b].id, float(sims[i, j])))
    return out


def find_duplicates(
    graph: GraphStore,
    embedder: Optional[EmbeddingProvider] = None,
    *,
    project: Optional[str] = None,
    threshold: float = 0.92,
) -> List[dict]:
    """Candidate groups, largest memory count first. Pass embedder=None to skip name embeddings."""
    entities = [e for e in graph.iter_entities() if not is_meta_entity(e)]
    stats = _stats(graph, {e.id for e in entities})
    mems = stats.memories
    if project is not None:
        in_project = {m.id for m in graph.iter_memories() if m.project_id == project}
        mems = {k: v & in_project for k, v in mems.items()}
        entities = [e for e in entities if mems.get(e.id)]
    by_id = {e.id: e for e in entities}

    def shared(a: str, b: str) -> int:
        return len(mems.get(a, set()) & mems.get(b, set()))

    evidence: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

    def note(a: str, b: str, reason: str, **detail) -> None:
        if a != b:
            evidence[(min(a, b), max(a, b))].append({"reason": reason, **detail})

    by_norm: Dict[str, List[str]] = defaultdict(list)
    by_word: Dict[str, List[str]] = defaultdict(list)
    for e in entities:
        if _norm(e.name):
            by_norm[_norm(e.name)].append(e.id)
        for w in set(_words(e.name)):
            by_word[w].append(e.id)
    for ids in by_norm.values():
        for b in ids[1:]:
            note(ids[0], b, "same-name")

    def base(name: str) -> str:
        return _norm(name.rstrip("/").rsplit("/", 1)[-1])

    paths = [e for e in entities if "/" in e.name.strip("/")]
    paths_per_base = Counter(base(e.name) for e in paths)
    for e in paths:
        for b in by_norm.get(base(e.name), []):
            if "/" in by_id[b].name.strip("/") or e.name.startswith("@"):  # @scope/pkg is a package, not a path
                continue
            is_file = re.search(r"\.[A-Za-z0-9]{1,5}$", by_id[b].name) is not None
            if shared(e.id, b) or (is_file and paths_per_base[base(e.name)] == 1):
                note(e.id, b, "basename")
    for e in entities:
        words = _words(e.name)
        if len("".join(words)) < MIN_CONTAINED:
            continue
        for b in by_word.get(words[0], []):
            other = by_id[b].name
            if not _pathlike(other) and _contains(words, _words(other)) and shared(e.id, b):
                note(e.id, b, "contains")
    plain = [e for e in entities if not _pathlike(e.name)]
    if embedder is not None and plain:
        try:
            for a, b, sim in _embedding_pairs(plain, embedder, threshold):
                note(a, b, "embedding", similarity=round(sim, 3))
        except Exception as e:  # the cheap reasons still stand without the embedder
            logger.warning("name embeddings skipped: %s", e)

    parent = {e.id: e.id for e in entities}

    def root(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def strong(pair: Tuple[str, str]) -> bool:
        return any(e["reason"] in STRONG for e in evidence[pair])

    for a, b in filter(strong, evidence):
        parent[root(a)] = root(b)
    groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for pair in evidence:
        groups[root(pair[0]) if strong(pair) else "|".join(pair)].append(pair)

    out = []
    for pairs in groups.values():
        ids = sorted({i for p in pairs for i in p}, key=lambda i: (-len(mems.get(i, ())), i))
        members = [
            {
                "id": i,
                "name": by_id[i].name,
                "type": getattr(by_id[i].type, "value", by_id[i].type),
                "memories": len(mems.get(i, ())),
                "edges": stats.edges.get(i, 0),
            }
            for i in ids
        ]
        out.append(
            {
                "members": members,
                "pairs": [
                    {
                        "a": by_id[a].name,
                        "b": by_id[b].name,
                        "shared_memories": shared(a, b),
                        "evidence": evidence[(a, b)],
                    }
                    for a, b in sorted(pairs)
                ],
                "memories": len(set().union(*(mems.get(i, set()) for i in ids))),
                "strength": "strong" if any(map(strong, pairs)) else "weak",
                "suggested_keep": members[0]["name"],
            }
        )
    # strong groups by size; weak pairs by shared memories, so a hub's many pairs do not crowd the top
    return sorted(
        out,
        key=lambda g: (
            (0, 0, -g["memories"])
            if g["strength"] == "strong"
            else (1, -max(p["shared_memories"] for p in g["pairs"]), -g["memories"])
        ),
    )


def merge_command(group: dict) -> str:
    """The dry-run command that would merge a group into its suggested entity."""
    quote = lambda s: "'" + s.replace("'", "'\\''") + "'"  # noqa: E731
    others = [m["name"] for m in group["members"] if m["name"] != group["suggested_keep"]]
    return "collivind entities merge " + " ".join(quote(n) for n in [group["suggested_keep"], *others])
