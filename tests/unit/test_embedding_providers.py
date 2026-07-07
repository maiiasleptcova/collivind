import pytest

from collivind.config import EmbeddingsConfig
from collivind.exceptions import CollivindError
from collivind.storage.embedding_openai import (
    OPENAI_DEFAULT_MODEL,
    OPENAI_DIMENSIONS,
    OpenAIEmbeddingProvider,
)


def test_openai_requires_api_key():
    config = EmbeddingsConfig(provider="openai", api_key="")
    with pytest.raises(CollivindError, match="API key required"):
        OpenAIEmbeddingProvider(config)


def test_openai_default_model():
    config = EmbeddingsConfig(provider="openai", api_key="sk-test")
    provider = OpenAIEmbeddingProvider(config)
    assert provider._model == OPENAI_DEFAULT_MODEL


def test_openai_custom_model():
    config = EmbeddingsConfig(
        provider="openai",
        api_key="sk-test",
        model="text-embedding-3-large",
        dimension=3072,
    )
    provider = OpenAIEmbeddingProvider(config)
    assert provider._model == "text-embedding-3-large"
    assert provider.dimension == 3072


def test_openai_custom_base_url():
    config = EmbeddingsConfig(
        provider="openai",
        api_key="sk-test",
        base_url="https://my-proxy.com/v1",
    )
    provider = OpenAIEmbeddingProvider(config)
    assert "my-proxy.com" in str(provider.client.base_url)


def test_openai_default_dimensions():
    for model, dim in OPENAI_DIMENSIONS.items():
        config = EmbeddingsConfig(
            provider="openai",
            api_key="sk-test",
            model=model,
            dimension=0,
        )
        provider = OpenAIEmbeddingProvider(config)
        assert provider.dimension == dim


def test_ollama_defaults():
    from collivind.storage.embedding_ollama import (
        OLLAMA_DEFAULT_MODEL,
        OllamaEmbeddingProvider,
    )

    config = EmbeddingsConfig(provider="ollama")
    provider = OllamaEmbeddingProvider(config)
    assert provider._model == OLLAMA_DEFAULT_MODEL
    assert provider.dimension == 768


def test_ollama_custom_model():
    from collivind.storage.embedding_ollama import OllamaEmbeddingProvider

    config = EmbeddingsConfig(
        provider="ollama",
        model="mxbai-embed-large",
        service_url="http://gpu-box:11434",
        dimension=1024,
    )
    provider = OllamaEmbeddingProvider(config)
    assert provider._model == "mxbai-embed-large"
    assert provider.dimension == 1024


def test_embed_query_applies_prefix_to_queries_only():
    from unittest.mock import MagicMock, patch

    from collivind.config import EmbeddingsConfig
    from collivind.storage.embedding_local import LocalEmbeddingProvider

    config = EmbeddingsConfig(provider="local", query_prefix="QUERY: ")
    provider = LocalEmbeddingProvider(config)
    model = MagicMock()
    model.encode.return_value.tolist.return_value = [0.1]
    with patch.object(provider, "_load_model"):
        provider._model = model
        provider.embed("a document")
        model.encode.assert_called_with("a document")
        provider.embed_query("a question")
        model.encode.assert_called_with("QUERY: a question")
