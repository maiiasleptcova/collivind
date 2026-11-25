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

    def __init__(self, config: EmbeddingsConfig, data_dir: Optional[str] = None, rerank_model: str = ""):
        """data_dir enables the resident embedding daemon (#16); None keeps
        everything in-process (init's model pre-download, tests).
        rerank_model "" means no cross-encoder (rerank off)."""
        self.config = config
        self._model: Any = None  # SentenceTransformer, imported lazily (optional extra)
        self._dimension = config.dimension
        self._data_dir = Path(data_dir) if data_dir else None
        self.rerank_model = rerank_model
        self._reranker: Any = None  # CrossEncoder, in-process mode only
        self._reranker_failed = False

    def _daemon_owns_models(self) -> bool:
        """True when models belong in the daemon, not in this process."""
        if self._data_dir is None or not embed_daemon.daemon_enabled(self.config.daemon):
            return False
        return embed_daemon.daemon_usable(embed_daemon.socket_path_for(self._data_dir))

    @property
    def can_rerank(self) -> bool:
        return bool(self.rerank_model) and not self._reranker_failed

    def rerank(self, query: str, texts: List[str]) -> Optional[List[float]]:
        """Scores from the resident daemon, or in-process when there is no daemon.

        With the daemon on, a cross-encoder is never loaded here: hooks are
        short-lived and a load costs seconds, so a daemon that is not up yet
        (it is started here) means this search keeps its first-stage ranking.
        Never raises.
        """
        if not self.rerank_model or not texts:
            return None
        try:
            if self._daemon_owns_models():
                socket_path = embed_daemon.socket_path_for(self._data_dir)  # type: ignore[arg-type]
                return embed_daemon.rerank_via_daemon(
                    socket_path, self.config.model, self.rerank_model, query, texts, self.config.daemon_idle_timeout
                )
            if self._reranker is None:
                if self._reranker_failed:
                    return None
                try:
                    self._reranker = embed_daemon.load_cross_encoder(self.rerank_model)
                except Exception:
                    self._reranker_failed = True  # one failed load per process, not one per search
                    raise
            scores = self._reranker.predict([(query, t) for t in texts], batch_size=64, show_progress_bar=False)
            return [float(x) for x in scores]
        except Exception as e:
            logger.warning(f"reranker unavailable, keeping first-stage ranking: {e}")
            return None

    def _from_daemon(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Never raises: any daemon problem means "embed in-process"."""
        if self._model is not None or self._data_dir is None:
            return None
        if not embed_daemon.daemon_enabled(self.config.daemon):
            return None
        try:
            # resolved per call: a fallback dir under /tmp is re-checked
            # (owner, mode) before every connect/spawn, not once per process
            socket_path = embed_daemon.socket_path_for(self._data_dir)
            return embed_daemon.embed_via_daemon(
                socket_path, self.config.model, texts, self.config.daemon_idle_timeout, self.rerank_model
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
