"""Factory for creating storage backends.

Each backend can be configured independently via its `provider` field,
or defaults are chosen based on the top-level `mode` setting.
"""

import logging

from collivind.config import CollivindConfig
from collivind.storage.interfaces import EmbeddingProvider, GraphStore, VectorStore

logger = logging.getLogger(__name__)

MODE_DEFAULTS = {
    "embedded": {"qdrant": "embedded", "graph": "sqlite", "embeddings": "local"},
    "docker": {"qdrant": "local", "graph": "neo4j", "embeddings": "http"},
    "remote": {"qdrant": "cloud", "graph": "neo4j", "embeddings": "openai"},
}


def _resolve_provider(config: CollivindConfig, backend: str) -> str:
    if backend == "qdrant":
        explicit = config.qdrant.provider
    elif backend == "graph":
        explicit = config.neo4j.provider
    else:
        explicit = config.embeddings.provider

    if explicit:
        return explicit

    defaults = MODE_DEFAULTS.get(config.mode, MODE_DEFAULTS["embedded"])
    return defaults[backend]


def create_vector_store(config: CollivindConfig) -> VectorStore:
    provider = _resolve_provider(config, "qdrant")
    logger.info(f"Vector store provider: {provider}")

    if provider == "embedded":
        from collivind.storage.qdrant_embedded import EmbeddedQdrantStore

        return EmbeddedQdrantStore(
            data_dir=config.data_dir,
            config=config.qdrant,
            dimension=config.embeddings.dimension,
        )
    else:
        from collivind.storage.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(config.qdrant, config.embeddings.dimension)


def create_graph_store(config: CollivindConfig) -> GraphStore:
    provider = _resolve_provider(config, "graph")
    logger.info(f"Graph store provider: {provider}")

    if provider == "sqlite":
        from collivind.storage.graph_sqlite import SqliteGraphStore

        return SqliteGraphStore(data_dir=config.data_dir)
    else:
        from collivind.storage.neo4j_store import Neo4jGraphStore

        return Neo4jGraphStore(config.neo4j)


def create_embedding_provider(config: CollivindConfig) -> EmbeddingProvider:
    provider = _resolve_provider(config, "embeddings")
    logger.info(f"Embedding provider: {provider}")

    if provider == "local":
        from collivind.storage.embedding_local import LocalEmbeddingProvider

        return LocalEmbeddingProvider(config.embeddings)
    elif provider == "openai":
        from collivind.storage.embedding_openai import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(config.embeddings)
    elif provider == "ollama":
        from collivind.storage.embedding_ollama import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider(config.embeddings)
    else:
        from collivind.storage.embedding_service import HttpEmbeddingProvider

        return HttpEmbeddingProvider(config.embeddings)


def create_all_backends(config: CollivindConfig) -> tuple[VectorStore, GraphStore, EmbeddingProvider]:
    logger.info(f"Creating backends (mode={config.mode})")
    vector_store = create_vector_store(config)
    graph_store = create_graph_store(config)
    embedding_provider = create_embedding_provider(config)
    return vector_store, graph_store, embedding_provider
