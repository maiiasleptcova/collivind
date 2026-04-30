from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from collivind.config import QdrantConfig
from collivind.storage.interfaces import VectorStore
from collivind.exceptions import CollivindError

class QdrantVectorStore(VectorStore):
    def __init__(self, config: QdrantConfig, dimension: int):
        self.config = config
        self.dimension = dimension
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
        try:
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
        except UnexpectedResponse as e:
            raise CollivindError(f"Qdrant upsert failed: {e}")

    def search(self, vector: List[float], limit: int = 10, filters: Optional[Dict[str, Any]] = None, threshold: float = 0.3) -> List[Dict[str, Any]]:
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

        try:
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=vector,
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=threshold
            )
            return [
                {
                    "id": str(res.id),
                    "score": res.score,
                    "payload": res.payload
                }
                for res in results
            ]
        except UnexpectedResponse as e:
            raise CollivindError(f"Qdrant search failed: {e}")

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
