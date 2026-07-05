import logging
import time
from typing import Any, Dict, List

import httpx

from collivind.config import EmbeddingsConfig
from collivind.exceptions import CollivindError
from collivind.storage.interfaces import EmbeddingProvider

logger = logging.getLogger(__name__)

OPENAI_DEFAULT_URL = "https://api.openai.com/v1"
OPENAI_DEFAULT_MODEL = "text-embedding-3-small"
OPENAI_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _retry(fn, max_retries=3, base_delay=1.0):
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


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embeddings via OpenAI-compatible API (works with OpenAI, Azure, Together, etc)."""

    def __init__(self, config: EmbeddingsConfig):
        self.config = config
        is_default_local_model = config.model in ("all-MiniLM-L6-v2", "")
        self._model = OPENAI_DEFAULT_MODEL if is_default_local_model else config.model
        base_url = config.base_url or OPENAI_DEFAULT_URL
        auto_dim = OPENAI_DIMENSIONS.get(self._model, 1536)
        uses_local_defaults = config.dimension == 384 and is_default_local_model
        self._dimension = auto_dim if uses_local_defaults else (config.dimension or auto_dim)

        if not config.api_key:
            raise CollivindError(
                "API key required for OpenAI embeddings. "
                "Set embeddings.api_key in ~/.collivind/config.toml "
                "or COLLIVIND_EMBEDDINGS_API_KEY env var."
            )

        self.client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=30.0,
        )

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        def _do_embed():
            resp = self.client.post(
                "/embeddings",
                json={"input": texts, "model": self._model},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            data.sort(key=lambda x: x["index"])
            return [d["embedding"] for d in data]

        try:
            return _retry(_do_embed)
        except Exception as e:
            raise CollivindError(f"OpenAI embedding failed: {e}")

    def health_check(self) -> Dict[str, Any]:
        try:
            resp = self.client.post(
                "/embeddings",
                json={"input": ["health check"], "model": self._model},
            )
            if resp.status_code == 200:
                return {"status": "ok", "model": self._model, "provider": "openai"}
            return {"status": "error", "message": f"Status {resp.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @property
    def dimension(self) -> int:
        return self._dimension
