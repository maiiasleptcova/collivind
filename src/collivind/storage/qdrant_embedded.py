import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from collivind.config import QdrantConfig
from collivind.exceptions import CollivindError
from collivind.storage.interfaces import VectorStore

logger = logging.getLogger(__name__)

LOCK_HINT = "already accessed by another instance"


class EmbeddedQdrantStore(VectorStore):
    """Qdrant running in-process using its embedded Rust engine. No Docker needed.

    The embedded engine is single-process: the first process locks the storage
    directory and every other collivind process (a second MCP server, a CLI
    command, a hook) is refused until it exits. See _connect for handling.
    """

    def __init__(self, data_dir: str, config: QdrantConfig, dimension: int):
        self.config = config
        self._dimension = dimension
        self._storage_path = Path(data_dir).expanduser() / "qdrant_data"
        self.client = self._connect()

    def _connect(self) -> QdrantClient:
        # brief retry survives overlap between short-lived hook/CLI processes;
        # a lock held by a long-running MCP server won't clear, so fail with
        # the holder's PID and recovery steps instead of a bare qdrant error
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return QdrantClient(path=str(self._storage_path))
            except Exception as e:
                if LOCK_HINT not in str(e):
                    raise
                last_error = e
                time.sleep(0.5 * (attempt + 1))
        raise CollivindError(self._lock_message()) from last_error

    def _lock_message(self) -> str:
        pids = self._lock_holders()
        holder = f" (held by PID{'s' if len(pids) > 1 else ''} {', '.join(pids)})" if pids else ""
        recovery = f"kill a stale process (kill {pids[0]})" if pids else "find the holder (lsof +D the storage path)"
        return (
            f"Embedded Qdrant storage at {self._storage_path} is locked by another collivind "
            f"process{holder} — usually another agent session's MCP server. Embedded mode "
            f"supports ONE process at a time. Recovery: close the other session, {recovery}, "
            'or set mode = "docker" or "remote" in ~/.collivind/config.toml for concurrent access.'
        )

    def _lock_holders(self) -> List[str]:
        try:
            out = subprocess.run(
                ["lsof", "-t", str(self._storage_path / ".lock")],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return [p for p in out.stdout.split() if p.strip()]
        except Exception:
            return []

    def initialize(self) -> None:
        try:
            collections = self.client.get_collections()
            if self.config.collection_name not in [c.name for c in collections.collections]:
                self.client.create_collection(
                    collection_name=self.config.collection_name,
                    vectors_config=qmodels.VectorParams(size=self._dimension, distance=qmodels.Distance.COSINE),
                )
        except Exception as e:
            raise CollivindError(f"Embedded Qdrant initialization failed: {e}")

    def delete_collection(self) -> None:
        self.client.delete_collection(self.config.collection_name)

    def upsert(self, id: str, vector: List[float], payload: Dict[str, Any]) -> None:
        try:
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=[qmodels.PointStruct(id=id, vector=vector, payload=payload)],
            )
        except Exception as e:
            raise CollivindError(f"Embedded Qdrant upsert failed: {e}")

    def search(
        self,
        vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        qdrant_filter = None
        if filters:
            must_conditions = []
            for k, v in filters.items():
                must_conditions.append(qmodels.FieldCondition(key=k, match=qmodels.MatchValue(value=v)))
            qdrant_filter = qmodels.Filter(must=must_conditions)

        try:
            response = self.client.query_points(
                collection_name=self.config.collection_name,
                query=vector,
                query_filter=qdrant_filter,
                limit=limit,
                score_threshold=threshold,
            )
            return [{"id": str(p.id), "score": p.score, "payload": p.payload} for p in response.points]
        except Exception as e:
            raise CollivindError(f"Embedded Qdrant search failed: {e}")

    def delete(self, id: str) -> None:
        try:
            self.client.delete(
                collection_name=self.config.collection_name, points_selector=qmodels.PointIdsList(points=[id])
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
