"""A health check must be cheap.

`create_all_backends` probes every backend on construction (#8). When the
local provider's probe loaded the sentence-transformers model, every code
path paid for it — including SessionStart, which only reads the timeline and
never embeds anything. Measured at 6.2s. See #16.
"""

import sys
import types
from unittest.mock import patch

from collivind.config import EmbeddingsConfig
from collivind.storage.embedding_local import LocalEmbeddingProvider


def test_health_check_does_not_load_the_model():
    """Holds whether or not sentence-transformers is installed — CI runs
    `uv sync --dev`, without the `embedded` extra."""
    provider = LocalEmbeddingProvider(EmbeddingsConfig())
    with (
        patch.dict(sys.modules, {"sentence_transformers": types.ModuleType("sentence_transformers")}),
        patch.object(provider, "_load_model") as load,
    ):
        result = provider.health_check()
    load.assert_not_called()  # probing must not cost a model load
    assert result["status"] == "ok"
    assert provider._model is None


def test_health_check_never_loads_the_model_even_without_the_dependency():
    """The cost must be absent on both branches, not just the happy one."""
    provider = LocalEmbeddingProvider(EmbeddingsConfig())
    with patch.object(provider, "_load_model") as load:
        provider.health_check()
    load.assert_not_called()
    assert provider._model is None


def test_health_check_reports_error_when_dependency_missing():
    """The failure it exists to catch must still be caught."""
    provider = LocalEmbeddingProvider(EmbeddingsConfig())
    real_import = __import__

    def missing(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=missing):
        result = provider.health_check()
    assert result["status"] == "error"
    assert "sentence-transformers" in result["message"]


def test_embed_still_loads_the_model_lazily():
    provider = LocalEmbeddingProvider(EmbeddingsConfig())
    with patch.object(provider, "_load_model") as load:
        provider._model = type("M", (), {"encode": lambda self, t: type("A", (), {"tolist": lambda s: [0.1]})()})()
        provider.embed("some text")
    load.assert_called_once()
