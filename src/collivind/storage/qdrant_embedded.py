import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from collivind.config import QdrantConfig
from collivind.exceptions import CollivindError
from collivind.storage.interfaces import VectorStore

logger = logging.getLogger(__name__)


class EmbeddedQdrantStore(VectorStore):
    """Qdrant running in-process using its embedded Rust engine. No Docker needed."""

    def __init__(self, data_dir: str, config: QdrantConfig, dimension: int):
        self.config = config
        self._dimension = dimension
        storage_path = str(Path(data_dir).expanduser() / "qdrant_data")
        self.client = QdrantClient(path=storage_path)

    def initialize(self) -> None:
        try:
            collections = self.client.get_collections()
            if self.config.collection_name not in [c.name for c in collections.collections]:
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self._dimension,
                        distance=qmodels.Distance.COSINE
                    )
                )
        except Exception as e:
            raise CollivindError(f"Embedded Qdrant initialization failed: {e}")

    def delete_collection(self) -> None:
        self.client.delete_collection(self.config.collection_name)

    def upsert(self, id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        try:
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=[
                    qmodels.PointStruct(id=id, vector=vector, payload=payload)
                ]
            )
        except Exception as e:
            raise CollivindError(f"Embedded Qdrant upsert failed: {e}")

    def search(
        self, vector: List[float], limit: int = 10,
        filters: Optional[Dict[str, Any]] = None, threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        qdrant_filter = None
        if filters:
            must_conditions = []
            for k, v in filters.items():
                must_conditions.append(
                    qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v))
                )
            qdrant_filter = qmodels.Filter(must=must_conditions)

        try:
            response = self.client.query_points(
                collection_name=self.config.collection_name,
                query=vector,
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=threshold,
            )
            return [
                {"id": str(p.id), "score": p.score, "payload": p.payload}
                for p in response.points
            ]
        except Exception as e:
            raise CollivindError(f"Embedded Qdrant search failed: {e}")

    def delete(self, id: str) -> None:
        try:
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=qmodels.PointIdsList(points=[id])
            )
        except Exception as e:
            raise CollivindError(f"Embedded Qdrant delete failed: {e}")

    def health_check(self) -> Dict[str, Any]:
        try:
            self.client.get_collections()
            return {"status": "ok", "message": "Embedded Qdrant is healthy"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close(self) -> None:
        self.client.close()
