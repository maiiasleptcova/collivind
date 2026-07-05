import logging
import time
from typing import Any, Dict, List

import httpx

from collivind.config import EmbeddingsConfig
from collivind.exceptions import CollivindError
from collivind.storage.interfaces import EmbeddingProvider

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
                delay = base_delay * (2**attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
    raise last_error


class HttpEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: EmbeddingsConfig):
        self.config = config
        self._dimension = config.dimension
        self.client = httpx.Client(base_url=config.service_url, timeout=30.0)

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
