from unittest.mock import patch

import pytest

from collivind.config import CollivindConfig, EmbeddingsConfig, Neo4jConfig, QdrantConfig
from collivind.storage.factory import (
    create_all_backends,
    create_embedding_provider,
    create_graph_store,
    create_vector_store,
)

# --- Mode-based defaults ---


def test_create_vector_store_embedded():
    config = CollivindConfig(mode="embedded", data_dir="/tmp/collivind_test_factory")
    with patch("collivind.storage.qdrant_embedded.EmbeddedQdrantStore") as mock_cls:
        create_vector_store(config)
        mock_cls.assert_called_once_with(
            data_dir=config.data_dir,
            config=config.qdrant,
            dimension=config.embeddings.dimension,
        )


def test_create_vector_store_docker():
    config = CollivindConfig(mode="docker")
    with patch("collivind.storage.qdrant_store.QdrantVectorStore") as mock_cls:
        create_vector_store(config)
        mock_cls.assert_called_once_with(config.qdrant, config.embeddings.dimension)


def test_create_graph_store_embedded():
    config = CollivindConfig(mode="embedded", data_dir="/tmp/collivind_test_factory")
    with patch("collivind.storage.graph_sqlite.SqliteGraphStore") as mock_cls:
        create_graph_store(config)
        mock_cls.assert_called_once_with(data_dir=config.data_dir)


def test_create_graph_store_docker():
    config = CollivindConfig(mode="docker")
    with patch("collivind.storage.neo4j_store.Neo4jGraphStore") as mock_cls:
        create_graph_store(config)
        mock_cls.assert_called_once_with(config.neo4j)


def test_create_embedding_provider_embedded():
    config = CollivindConfig(mode="embedded")
    with patch("collivind.storage.embedding_local.LocalEmbeddingProvider") as mock_cls:
        create_embedding_provider(config)
        mock_cls.assert_called_once_with(config.embeddings)


def test_create_embedding_provider_docker():
    config = CollivindConfig(mode="docker")
    with patch("collivind.storage.embedding_service.HttpEmbeddingProvider") as mock_cls:
        create_embedding_provider(config)
        mock_cls.assert_called_once_with(config.embeddings)


def test_create_all_backends_returns_tuple():
    config = CollivindConfig(mode="embedded")
    with (
        patch("collivind.storage.qdrant_embedded.EmbeddedQdrantStore") as mv,
        patch("collivind.storage.graph_sqlite.SqliteGraphStore") as mg,
        patch("collivind.storage.embedding_local.LocalEmbeddingProvider") as me,
    ):
        for m in (mv, mg, me):
            m.return_value.health_check.return_value = {"status": "ok", "message": "ok"}
        v, g, e = create_all_backends(config)
        assert v == mv.return_value
        assert g == mg.return_value
        assert e == me.return_value


# --- Per-backend provider overrides ---


def test_qdrant_cloud_override():
    config = CollivindConfig(
        mode="embedded",
        qdrant=QdrantConfig(provider="cloud", url="https://xyz.cloud.qdrant.io", api_key="key123"),
    )
    with patch("collivind.storage.qdrant_store.QdrantVectorStore") as mock_cls:
        create_vector_store(config)
        mock_cls.assert_called_once_with(config.qdrant, config.embeddings.dimension)


def test_qdrant_local_override():
    config = CollivindConfig(
        mode="embedded",
        qdrant=QdrantConfig(provider="local", host="qdrant.myserver.com", port=6333),
    )
    with patch("collivind.storage.qdrant_store.QdrantVectorStore") as mock_cls:
        create_vector_store(config)
        mock_cls.assert_called_once()


def test_neo4j_override_on_embedded_mode():
    config = CollivindConfig(
        mode="embedded",
        neo4j=Neo4jConfig(provider="neo4j", uri="bolt://remote:7687"),
    )
    with patch("collivind.storage.neo4j_store.Neo4jGraphStore") as mock_cls:
        create_graph_store(config)
        mock_cls.assert_called_once_with(config.neo4j)


def test_sqlite_override_on_docker_mode():
    config = CollivindConfig(
        mode="docker",
        neo4j=Neo4jConfig(provider="sqlite"),
    )
    with patch("collivind.storage.graph_sqlite.SqliteGraphStore") as mock_cls:
        create_graph_store(config)
        mock_cls.assert_called_once()


def test_openai_embedding_override():
    config = CollivindConfig(
        mode="embedded",
        embeddings=EmbeddingsConfig(provider="openai", api_key="sk-test", model="text-embedding-3-small"),
    )
    with patch("collivind.storage.embedding_openai.OpenAIEmbeddingProvider") as mock_cls:
        create_embedding_provider(config)
        mock_cls.assert_called_once_with(config.embeddings)


def test_ollama_embedding_override():
    config = CollivindConfig(
        mode="embedded",
        embeddings=EmbeddingsConfig(provider="ollama", model="nomic-embed-text"),
    )
    with patch("collivind.storage.embedding_ollama.OllamaEmbeddingProvider") as mock_cls:
        create_embedding_provider(config)
        mock_cls.assert_called_once_with(config.embeddings)


def test_http_embedding_override():
    config = CollivindConfig(
        mode="embedded",
        embeddings=EmbeddingsConfig(provider="http", service_url="http://my-server:8090"),
    )
    with patch("collivind.storage.embedding_service.HttpEmbeddingProvider") as mock_cls:
        create_embedding_provider(config)
        mock_cls.assert_called_once_with(config.embeddings)


def test_mixed_backends():
    config = CollivindConfig(
        mode="embedded",
        qdrant=QdrantConfig(provider="cloud", url="https://cloud.qdrant.io", api_key="key"),
        neo4j=Neo4jConfig(provider="sqlite"),
        embeddings=EmbeddingsConfig(provider="openai", api_key="sk-test"),
    )
    with (
        patch("collivind.storage.qdrant_store.QdrantVectorStore") as mv,
        patch("collivind.storage.graph_sqlite.SqliteGraphStore") as mg,
        patch("collivind.storage.embedding_openai.OpenAIEmbeddingProvider") as me,
    ):
        for m in (mv, mg, me):
            m.return_value.health_check.return_value = {"status": "ok", "message": "ok"}
        v, g, e = create_all_backends(config)
        assert v == mv.return_value
        assert g == mg.return_value
        assert e == me.return_value


class TestModeValidation:
    """Issue #7: an unusable or misspelled mode must fail loudly, not quietly
    serve a different store."""

    def test_unknown_mode_raises_and_names_valid_modes(self):
        from collivind.storage.factory import UnknownModeError, _resolve_provider

        config = CollivindConfig()
        config.mode = "dcoker"  # typo
        with pytest.raises(UnknownModeError) as exc:
            _resolve_provider(config, "qdrant")
        msg = str(exc.value)
        assert "dcoker" in msg
        for mode in ("docker", "embedded", "remote"):
            assert mode in msg

    @pytest.mark.parametrize("mode", ["docker", "embedded", "remote"])
    def test_known_modes_resolve(self, mode):
        from collivind.storage.factory import _resolve_provider

        config = CollivindConfig()
        config.mode = mode
        assert _resolve_provider(config, "qdrant")

    def test_explicit_provider_still_wins_over_unknown_mode(self):
        """An explicit backend override is a deliberate choice — honour it."""
        from collivind.storage.factory import _resolve_provider

        config = CollivindConfig()
        config.mode = "nonsense"
        config.qdrant.provider = "embedded"
        assert _resolve_provider(config, "qdrant") == "embedded"


class TestModeAvailability:
    def test_unavailable_backend_raises_naming_the_mode(self, monkeypatch):
        """docker mode with Qdrant/Neo4j down must error, not build lazy
        clients that misbehave later."""
        from collivind.storage import factory
        from collivind.storage.factory import ModeUnavailableError

        class Dead:
            def health_check(self):
                return {"status": "error", "message": "connection refused"}

        class Alive:
            def health_check(self):
                return {"status": "ok", "message": "fine"}

        monkeypatch.setattr(factory, "create_vector_store", lambda c: Dead())
        monkeypatch.setattr(factory, "create_graph_store", lambda c: Alive())
        monkeypatch.setattr(factory, "create_embedding_provider", lambda c: Alive())

        config = CollivindConfig()
        config.mode = "docker"
        with pytest.raises(ModeUnavailableError) as exc:
            factory.create_all_backends(config)
        msg = str(exc.value)
        assert "docker" in msg and "connection refused" in msg

    def test_all_healthy_returns_backends(self, monkeypatch):
        from collivind.storage import factory

        class Alive:
            def health_check(self):
                return {"status": "ok", "message": "fine"}

        monkeypatch.setattr(factory, "create_vector_store", lambda c: Alive())
        monkeypatch.setattr(factory, "create_graph_store", lambda c: Alive())
        monkeypatch.setattr(factory, "create_embedding_provider", lambda c: Alive())

        config = CollivindConfig()
        config.mode = "embedded"
        assert len(factory.create_all_backends(config)) == 3

    def test_health_check_failure_is_not_fatal_by_itself(self, monkeypatch):
        """A backend whose health_check itself explodes must still produce the
        clear mode error, not an opaque AttributeError."""
        from collivind.storage import factory
        from collivind.storage.factory import ModeUnavailableError

        class Exploding:
            def health_check(self):
                raise RuntimeError("socket gone")

        monkeypatch.setattr(factory, "create_vector_store", lambda c: Exploding())
        monkeypatch.setattr(factory, "create_graph_store", lambda c: Exploding())
        monkeypatch.setattr(factory, "create_embedding_provider", lambda c: Exploding())

        config = CollivindConfig()
        config.mode = "docker"
        with pytest.raises(ModeUnavailableError):
            factory.create_all_backends(config)
