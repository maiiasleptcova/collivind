import httpx
from typing import List, Dict, Any

from collivind.config import EmbeddingsConfig
from collivind.storage.interfaces import EmbeddingProvider
from collivind.exceptions import CollivindError

class HttpEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: EmbeddingsConfig):
        self.config = config
        self._dimension = config.dimension
        self.client = httpx.Client(base_url=config.service_url, timeout=30.0)

    def embed(self, text: str) -> List[float]:
        try:
            resp = self.client.post("/embed", json={"text": text})
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            raise CollivindError(f"Embedding failed: {e}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            resp = self.client.post("/embed_batch", json={"texts": texts})
            resp.raise_for_status()
            return resp.json()["embeddings"]
        except Exception as e:
            raise CollivindError(f"Batch embedding failed: {e}")

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
