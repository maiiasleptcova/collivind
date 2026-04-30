from typing import List, Optional, Tuple

from collivind.config import SearchConfig
from collivind.storage.interfaces import VectorStore

class Deduplicator:
    def __init__(self, vector_store: VectorStore, config: SearchConfig):
        self.vector_store = vector_store
        self.config = config

    def find_duplicate(self, vector: List[float], project_id: str) -> Optional[Tuple[str, float]]:
        """
        Check if a highly similar memory already exists.
        Returns a tuple of (duplicate_id, score) if found, else None.
        """
        results = self.vector_store.search(
            vector=vector,
            limit=1,
            filters={"project_id": project_id},
            threshold=self.config.dedup_threshold
        )
        
        if results and results[0]["score"] >= self.config.dedup_threshold:
            return (results[0]["id"], results[0]["score"])
            
        return None
