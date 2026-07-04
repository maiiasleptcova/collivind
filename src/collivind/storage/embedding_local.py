import logging
from typing import Any, Dict, List

from collivind.config import EmbeddingsConfig
from collivind.exceptions import CollivindError
from collivind.storage.interfaces import EmbeddingProvider

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """Loads sentence-transformers model directly in-process. No Docker needed."""

    def __init__(self, config: EmbeddingsConfig):
        self.config = config
        self._model = None
        self._dimension = config.dimension

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise CollivindError(
                "sentence-transformers is required for embedded mode. "
                "Install with: pip install collivind-memory[embedded]"
            )
        logger.info(f"Loading embedding model: {self.config.model}")
        self._model = SentenceTransformer(self.config.model)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        self._load_model()
        try:
            return self._model.encode(text).tolist()
        except Exception as e:
            raise CollivindError(f"Local embedding failed: {e}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        try:
            return self._model.encode(texts).tolist()
        except Exception as e:
            raise CollivindError(f"Local batch embedding failed: {e}")

    def health_check(self) -> Dict[str, Any]:
        try:
            self._load_model()
            return {"status": "ok", "message": f"Model {self.config.model} loaded"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @property
    def dimension(self) -> int:
        return self._dimension
