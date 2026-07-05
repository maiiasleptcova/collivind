import pytest

from collivind.config import EmbeddingsConfig
from collivind.storage.embedding_service import HttpEmbeddingProvider


@pytest.fixture
def provider():
    config = EmbeddingsConfig(service_url="http://localhost:8090", dimension=384)
    return HttpEmbeddingProvider(config)


@pytest.mark.skip(reason="Requires running docker container")
def test_embedding_service_health(provider):
    health = provider.health_check()
    assert health["status"] == "ok"


@pytest.mark.skip(reason="Requires running docker container")
def test_embed_single(provider):
    vector = provider.embed("hello world")
    assert len(vector) == provider.dimension
    assert isinstance(vector[0], float)


@pytest.mark.skip(reason="Requires running docker container")
def test_embed_batch(provider):
    vectors = provider.embed_batch(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == provider.dimension
