"""Write-time duplicate handling.

The rule: a new memory is rejected only when its text, normalised, is
identical to a live memory's in the same project. Nothing is ever retired on
similarity. A close neighbour is stored next to the new memory and returned
to the agent under possible_conflicts, and the agent decides whether one
supersedes the other (collivind_invalidate keeps the version chain).

Why: on LoCoMo the old rule (retire the older memory at cosine >= 0.92) took
343 of 5,882 turns out of search and cost about 4 points of hit@5, because
short similar sentences are usually different facts. Similarity cannot tell
"PR 212 CI passed" from "PR 212 CI failed", or a reworded decision from a
neighbouring one, so the agent judges, as it already does for conflicts.

A neighbour counts as a likely duplicate, surfaced even when it is in another
category, when it is about the same subject (conflicts.same_subject) and
either its stored (enriched) vector is within `dedup_threshold` or, when both
memories have their own summary, the summaries are within SUMMARY_THRESHOLD.
The summary check catches the same fact wrapped in different detail, whose
content cosine can be as low as 0.77.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from collivind.config import SearchConfig
from collivind.engine.conflicts import cosine, normalise, same_subject
from collivind.models import MemoryCreate
from collivind.storage.interfaces import EmbeddingProvider, VectorStore

# Stored-vector cosine below which a neighbour is not looked at at all.
# ponytail: tuned on tests/fixtures/dedup_pairs.json; lower it if restatements
# with very different wording slip through, at one summary embed per neighbour.
NEIGHBOUR_FLOOR = 0.75
SUMMARY_THRESHOLD = 0.90
MAX_NEIGHBOURS = 8


@dataclass(frozen=True)
class Neighbour:
    memory_id: str
    score: float
    payload: Dict[str, Any]


class Deduplicator:
    def __init__(self, vector_store: VectorStore, config: SearchConfig, embedder: Optional[EmbeddingProvider] = None):
        self.vector_store = vector_store
        self.config = config
        self.embedder = embedder

    def neighbours(self, vector: List[float], project_id: str) -> List[Neighbour]:
        """Live memories of the project close to `vector`, summaries left out, closest first."""
        floor = min(NEIGHBOUR_FLOOR, self.config.dedup_threshold)
        hits = self.vector_store.search(
            vector=vector, limit=MAX_NEIGHBOURS, filters={"project_id": project_id}, threshold=floor
        )
        return [
            Neighbour(h["id"], h["score"], h.get("payload") or {})
            for h in hits
            # a summary is never a duplicate of a detail memory, nor the reverse
            if (h.get("payload") or {}).get("category") != "summary"
        ]

    @staticmethod
    def exact(content: str, neighbours: List[Neighbour]) -> Optional[Neighbour]:
        """The neighbour whose text is the same as `content` once normalised: a resubmission."""
        key = normalise(content)
        return next((n for n in neighbours if normalise(n.payload.get("content", "")) == key), None)

    def likely_duplicates(self, memory: MemoryCreate, neighbours: List[Neighbour]) -> List[Neighbour]:
        """Neighbours that are probably the same fact: returned to the agent, never retired here."""
        subject = [n for n in neighbours if same_subject(n.payload.get("content", ""), memory.content)]
        likely = [n for n in subject if n.score >= self.config.dedup_threshold]
        rest = [n for n in subject if n.score < self.config.dedup_threshold and _own_summary(n.payload)]
        if rest and self.embedder is not None and memory.summary and memory.summary != memory.content:
            mine, *theirs = self.embedder.embed_batch([memory.summary] + [n.payload["summary"] for n in rest])
            likely += [n for n, v in zip(rest, theirs) if cosine(mine, v) >= SUMMARY_THRESHOLD]
        return likely


def _own_summary(payload: Dict[str, Any]) -> bool:
    summary = payload.get("summary")
    return bool(summary) and summary != payload.get("content")
