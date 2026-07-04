import logging
import time
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from collivind.config import QdrantConfig
from collivind.exceptions import CollivindError
from collivind.storage.interfaces import VectorStore

logger = logging.getLogger(__name__)


def _retry(fn, max_retries=3, base_delay=0.5):
    """Retry with exponential backoff."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
    raise last_error


class QdrantVectorStore(VectorStore):
    def __init__(self, config: QdrantConfig, dimension: int):
        self.config = config
        self.dimension = dimension
        if config.url and config.api_key:
            self.client = QdrantClient(url=config.url, api_key=config.api_key)
        elif config.url:
            self.client = QdrantClient(url=config.url)
        else:
            self.client = QdrantClient(host=config.host, port=config.port)

    def initialize(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections()
            if self.config.collection_name not in [c.name for c in collections.collections]:
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.dimension,
                        distance=qmodels.Distance.COSINE
                    )
                )
        except UnexpectedResponse as e:
            raise CollivindError(f"Qdrant initialization failed: {e}")

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.config.collection_name)

    def upsert(self, id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        def _do_upsert():
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=[
                    qmodels.PointStruct(
                        id=id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )

        try:
            _retry(_do_upsert)
        except UnexpectedResponse as e:
            raise CollivindError(f"Qdrant upsert failed after retries: {e}")
        except Exception as e:
            raise CollivindError(f"Qdrant upsert failed after retries: {e}")

    def search(
        self, vector: List[float], limit: int = 10,
        filters: Optional[Dict[str, Any]] = None, threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        # Basic filter conversion (only handles simple exact matches for now)
        qdrant_filter = None
        if filters:
            must_conditions = []
            for k, v in filters.items():
                must_conditions.append(
                    qmodels.FieldCondition(
                        key=k,
                        match=qmodels.MatchValue(value=v)
                    )
                )
            qdrant_filter = qmodels.Filter(must=must_conditions)

        def _do_search():
            response = self.client.query_points(
                collection_name=self.config.collection_name,
                query=vector,
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=threshold,
            )
            return [
                {
                    "id": str(p.id),
                    "score": p.score,
                    "payload": p.payload
                }
                for p in response.points
            ]

        try:
            return _retry(_do_search)
        except UnexpectedResponse as e:
            raise CollivindError(f"Qdrant search failed after retries: {e}")
        except Exception as e:
            raise CollivindError(f"Qdrant search failed after retries: {e}")

    def delete(self, id: str) -> None:
        try:
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=qmodels.PointIdsList(
                    points=[id]
                )
            )
        except UnexpectedResponse as e:
            raise CollivindError(f"Qdrant delete failed: {e}")

    def health_check(self) -> Dict[str, Any]:
        try:
            collections = self.client.get_collections()
            return {"status": "ok", "collections": [c.name for c in collections.collections]}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close(self) -> None:
        self.client.close()
