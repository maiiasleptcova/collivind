import logging
import time
from typing import Any, Dict, List

import httpx

from collivind.config import EmbeddingsConfig
from collivind.exceptions import CollivindError
from collivind.storage.interfaces import EmbeddingProvider

logger = logging.getLogger(__name__)

OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "nomic-embed-text"


def _retry(fn, max_retries=3, base_delay=0.5):
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


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embeddings via Ollama's local API."""

    def __init__(self, config: EmbeddingsConfig):
        self.config = config
        is_default_local_model = config.model in ("all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "")
        self._model = OLLAMA_DEFAULT_MODEL if is_default_local_model else config.model
        base_url = config.service_url or OLLAMA_DEFAULT_URL
        self._dimension = 768 if (config.dimension == 384 and is_default_local_model) else (config.dimension or 768)
        self.client = httpx.Client(base_url=base_url, timeout=60.0)

    def embed(self, text: str) -> List[float]:
        def _do_embed():
            resp = self.client.post(
                "/api/embed",
                json={"model": self._model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
            if embeddings:
                self._dimension = len(embeddings[0])
                return embeddings[0]
            raise CollivindError("Empty embedding response from Ollama")

        try:
            return _retry(_do_embed)
        except CollivindError:
            raise
        except Exception as e:
            raise CollivindError(f"Ollama embedding failed: {e}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    def health_check(self) -> Dict[str, Any]:
        try:
            resp = self.client.get("/api/tags")
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                has_model = any(self._model in m for m in models)
                return {
                    "status": "ok" if has_model else "warning",
                    "model": self._model,
                    "available": has_model,
                    "provider": "ollama",
                }
            return {"status": "error", "message": f"Status {resp.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @property
    def dimension(self) -> int:
        return self._dimension
