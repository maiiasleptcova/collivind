import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from collivind.config import EmbeddingsConfig
from collivind.exceptions import CollivindError
from collivind.storage import embed_daemon
from collivind.storage.interfaces import EmbeddingProvider

logger = logging.getLogger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):
    """Loads sentence-transformers model directly in-process. No Docker needed."""

    def __init__(self, config: EmbeddingsConfig, data_dir: Optional[str] = None):
        """data_dir enables the resident embedding daemon (#16); None keeps
        everything in-process (init's model pre-download, tests)."""
        self.config = config
        self._model: Any = None  # SentenceTransformer, imported lazily (optional extra)
        self._dimension = config.dimension
        self._socket_path = embed_daemon.socket_path_for(Path(data_dir)) if data_dir else None

    def _from_daemon(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Never raises: any daemon problem means "embed in-process"."""
        if self._model is not None or self._socket_path is None:
            return None
        if not embed_daemon.daemon_enabled(self.config.daemon):
            return None
        try:
            return embed_daemon.embed_via_daemon(
                self._socket_path, self.config.model, texts, self.config.daemon_idle_timeout
            )
        except Exception as e:
            logger.debug(f"embed daemon failed, embedding in-process: {e}")
            return None

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
        vectors = self._from_daemon([text])
        if vectors is not None:
            return vectors[0]
        self._load_model()
        try:
            return self._model.encode(text).tolist()
        except Exception as e:
            raise CollivindError(f"Local embedding failed: {e}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = self._from_daemon(texts)
        if vectors is not None:
            return vectors
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

        Not even an import: importing sentence-transformers pulls in torch,
        seconds on its own, which would undo the daemon's warm path.
        """
        if self._model is not None:
            return {"status": "ok", "message": f"Model {self.config.model} loaded"}
        if "sentence_transformers" not in sys.modules and importlib.util.find_spec("sentence_transformers") is None:
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
