from dataclasses import dataclass
from typing import List, Optional, Tuple

from collivind.config import SearchConfig
from collivind.storage.interfaces import VectorStore


@dataclass
class DuplicateMatch:
    memory_id: str
    score: float
    is_exact: bool


class Deduplicator:
    EXACT_THRESHOLD = 0.98

    def __init__(self, vector_store: VectorStore, config: SearchConfig):
        self.vector_store = vector_store
        self.config = config

    def find_duplicate(self, vector: List[float], project_id: str) -> Optional[Tuple[str, float]]:
        match = self.find_duplicate_detailed(vector, project_id)
        if match:
            return (match.memory_id, match.score)
        return None

    def find_duplicate_detailed(self, vector: List[float], project_id: str) -> Optional[DuplicateMatch]:
        results = self.vector_store.search(
            vector=vector,
            limit=1,
            filters={"project_id": project_id},
            threshold=self.config.dedup_threshold,
        )

        if results and results[0]["score"] >= self.config.dedup_threshold:
            return DuplicateMatch(
                memory_id=results[0]["id"],
                score=results[0]["score"],
                is_exact=results[0]["score"] >= self.EXACT_THRESHOLD,
            )
        return None
