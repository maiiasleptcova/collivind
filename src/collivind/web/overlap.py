"""Memories that may say the same thing as the open one, for the explorer's panel.

The write-time conflict check (engine/conflicts.py) compares whole contents at
cosine >= 0.85 and asks for a shared identifier or wording. On a real store it
missed the pair that motivated this panel: "Docker VM disk ... enlarged to
77GB on 2026-11-13" and "... enlarged to 76GB on 2026-11-13". Each wraps the
one shared fact in different detail, so their contents sit at cosine 0.77,
while their summaries, where agents put the fact itself, sit at 0.90.

So a candidate is scored on both: the mean of the stored-text cosine (the
vector it was saved with) and the summary-to-summary cosine. Measured with
bge-small on agent-style memories (31 memories, six paraphrased pairs plus
same-topic neighbours): neither cosine alone separates pairs from neighbours
(pairs 0.69-0.93 on text and 0.81-0.91 on summaries; neighbours up to 0.89
and 0.90), the mean at 0.85 caught 5 of 6 pairs and let through one
neighbour, two "no space left" docker errors from different incidents.
- `likely duplicate`: same category and mean >= the store's dedup threshold
  (the score add_memory itself would merge at).
- `overlapping`: same category, mean >= OVERLAP, and saved the same day or
  sharing an entity. Said with caution: they share a fact, one may not hold
  everything the other does.
At most MAX_ROWS rows. Cost per open memory: two embeddings, two vector
searches and one batch embedding of up to 2 x POOL short summaries.
"""

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from collivind.engine.conflicts import is_live
from collivind.engine.enrichment import build_enriched_text
from collivind.models.memory import MemoryCategory, MemoryNode

if TYPE_CHECKING:
    from collivind.engine.memory_manager import MemoryManager

OVERLAP = 0.85
POOL = 30  # candidates per search
FLOOR = 2 * OVERLAP - 1  # 0.7: below it even an identical summary cannot lift the mean to OVERLAP
MAX_ROWS = 3


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _summary(m: MemoryNode) -> str:
    return (m.summary or m.content[:200]).strip()


def _day(m: MemoryNode) -> str:
    return m.created_at.date().isoformat() if m.created_at else ""


def overlapping(manager: "MemoryManager", memory_id: str) -> Optional[List[Dict[str, Any]]]:
    """Up to MAX_ROWS live memories that likely duplicate or overlap `memory_id`, best first. None if it is unknown."""
    memory = manager.graph_store.get_memory(memory_id)
    if memory is None:
        return None
    if memory.category == MemoryCategory.SUMMARY:
        return []
    names, _ = manager._entity_names(memory_id)
    embed = manager.embedding_provider
    text_vec = embed.embed(build_enriched_text(manager._as_create(memory), entity_names=names))
    summary_vec = embed.embed(_summary(memory))
    where = {"project_id": memory.project_id}

    # the stored-text score comes from the first search; a candidate only the
    # summary search found is scored against its stored text by that search
    text_score: Dict[str, float] = {}
    for vec in (text_vec, summary_vec):
        for hit in manager.vector_store.search(vector=vec, limit=POOL, filters=where, threshold=FLOOR):
            text_score.setdefault(hit["id"], hit["score"])
    candidates = [
        m
        for m in (manager.graph_store.get_memory(i) for i in text_score if i != memory_id)
        if m is not None and is_live(m) and m.category == memory.category and m.content != memory.content
    ]
    if not candidates:
        return []
    summary_scores = [_cos(summary_vec, v) for v in embed.embed_batch([_summary(m) for m in candidates])]

    mine = set(names or [])
    likely_at = manager.config.search.dedup_threshold
    rows = []
    for m, s in zip(candidates, summary_scores):
        score = (text_score[m.id] + s) / 2
        if score < OVERLAP:
            continue
        likely = score >= likely_at
        if not likely and _day(m) != _day(memory) and not (mine & set(manager._entity_names(m.id)[0] or [])):
            continue
        rows.append((score, m, likely, s, text_score[m.id]))
    rows.sort(key=lambda r: -r[0])
    return [
        {
            "memory": m,
            "score": round(score, 3),
            "likely": likely,
            "summary_similarity": round(s, 3),
            "text_similarity": round(t, 3),
        }  # fmt: skip
        for score, m, likely, s, t in rows[:MAX_ROWS]
    ]
