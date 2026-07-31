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
            return self._model.encode(texts, batch_size=128).tolist()
        except Exception as e:
            raise CollivindError(f"Local batch embedding failed: {e}")

    def health_check(self) -> Dict[str, Any]:
        """Verify the model *can* load, without loading it.

        create_all_backends probes every backend on construction, so anything
        expensive here is paid by every code path — including SessionStart,
        which only reads the timeline and never embeds. Loading the model here
        cost 6.2s per invocation (#16). Importing sentence-transformers is the
        failure this catches; instantiating the model is not needed to catch
        it, and `embed()` still loads lazily on first use.
        """
        if self._model is not None:
            return {"status": "ok", "message": f"Model {self.config.model} loaded"}
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return {
                "status": "error",
                "message": (
                    "sentence-transformers is required for embedded mode. "
                    "Install with: pip install collivind-memory[embedded]"
                ),
            }
        return {"status": "ok", "message": f"Model {self.config.model} not yet loaded (loads on first use)"}

    @property
    def dimension(self) -> int:
        return self._dimension
