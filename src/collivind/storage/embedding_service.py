import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from collivind.config import EmbeddingsConfig
from collivind.exceptions import CollivindError
from collivind.storage.interfaces import EmbeddingProvider

logger = logging.getLogger(__name__)

# On the prompt path: one try, short, never the 30 s / 3-retry embed budget.
RERANK_TIMEOUT = 3.0
RERANK_BACKOFF = 60.0  # seconds without rerank after a failed call: a slow service must not tax every prompt


def _retry(fn, max_retries=3, base_delay=0.5):
    """Retry with exponential backoff."""
    last_error: Exception = RuntimeError("_retry needs max_retries >= 1")
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
    raise last_error


class HttpEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: EmbeddingsConfig, rerank: bool = False):
        self.config = config
        self._dimension = config.dimension
        self.client = httpx.Client(base_url=config.service_url, timeout=30.0)
        self._rerank = rerank
        self._rerank_missing = False  # an embeddings image built before 0.19 has no /rerank
        self._rerank_off_until = 0.0  # monotonic

    @property
    def can_rerank(self) -> bool:
        return self._rerank and not self._rerank_missing and time.monotonic() >= self._rerank_off_until

    def rerank(self, query: str, texts: List[str]) -> Optional[List[float]]:
        """POST /rerank on the embeddings service; None on any problem (search keeps its ranking)."""
        if not self.can_rerank or not texts:
            return None
        try:
            resp = self.client.post("/rerank", json={"query": query, "texts": texts}, timeout=RERANK_TIMEOUT)
            if resp.status_code == 404:
                self._rerank_missing = True
                logger.warning("embeddings service has no /rerank; rebuild it with `collivind docker up`")
                return None
            resp.raise_for_status()
            scores = resp.json()["scores"]
            if not isinstance(scores, list) or len(scores) != len(texts):
                return None
            return [float(x) for x in scores]
        except Exception as e:
            self._rerank_off_until = time.monotonic() + RERANK_BACKOFF
            logger.warning(f"reranker unavailable for {RERANK_BACKOFF:.0f}s, keeping first-stage ranking: {e}")
            return None

    def embed(self, text: str) -> List[float]:
        def _do_embed():
            resp = self.client.post("/embed", json={"text": text})
            resp.raise_for_status()
            return resp.json()["embedding"]

        try:
            return _retry(_do_embed)
        except Exception as e:
            raise CollivindError(f"Embedding failed after retries: {e}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        def _do_embed_batch():
            resp = self.client.post("/embed_batch", json={"texts": texts})
            resp.raise_for_status()
            return resp.json()["embeddings"]

        try:
            return _retry(_do_embed_batch)
        except Exception as e:
            raise CollivindError(f"Batch embedding failed after retries: {e}")

    def health_check(self) -> Dict[str, Any]:
        try:
            resp = self.client.get("/health", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                # Update dimension dynamically if service provides it
                if "dimension" in data:
                    self._dimension = data["dimension"]
                return {"status": "ok"}
            return {"status": "error", "message": f"Status {resp.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @property
    def dimension(self) -> int:
        return self._dimension
