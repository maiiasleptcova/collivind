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
    """The failure it exists to catch must still be caught — by a spec lookup,
    not an import: importing sentence-transformers pulls in torch."""
    provider = LocalEmbeddingProvider(EmbeddingsConfig())
    with (
        patch.dict(sys.modules),
        patch("importlib.util.find_spec", return_value=None),
    ):
        sys.modules.pop("sentence_transformers", None)
        result = provider.health_check()
    assert result["status"] == "error"
    assert "sentence-transformers" in result["message"]


def test_health_check_does_not_import_the_dependency():
    provider = LocalEmbeddingProvider(EmbeddingsConfig())
    with patch.dict(sys.modules):
        sys.modules.pop("sentence_transformers", None)
        provider.health_check()
        assert "sentence_transformers" not in sys.modules


def test_embed_still_loads_the_model_lazily():
    provider = LocalEmbeddingProvider(EmbeddingsConfig())
    with patch.object(provider, "_load_model") as load:
        provider._model = type("M", (), {"encode": lambda self, t: type("A", (), {"tolist": lambda s: [0.1]})()})()
        provider.embed("some text")
    load.assert_called_once()
