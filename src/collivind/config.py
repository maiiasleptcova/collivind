import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

from .exceptions import ConfigurationError


@dataclass
class DockerConfig:
    compose_project: str = "collivind"
    auto_start: bool = True

@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "collivind_memories"

@dataclass
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "collivind_local"
    database: str = "neo4j"

@dataclass
class EmbeddingsConfig:
    service_url: str = "http://localhost:8090"
    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384

@dataclass
class SearchConfig:
    default_limit: int = 10
    vector_weight: float = 0.7
    graph_weight: float = 0.3
    similarity_threshold: float = 0.3
    dedup_threshold: float = 0.92

@dataclass
class HooksConfig:
    save_interval: int = 15
    enable_precompact: bool = True
    enable_stop: bool = True

@dataclass
class CollivindConfig:
    user_id: str = "local"
    data_dir: str = "~/.collivind"
    docker: DockerConfig = field(default_factory=DockerConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    hooks: HooksConfig = field(default_factory=HooksConfig)

    @property
    def expanded_data_dir(self) -> Path:
        """Returns the expanded path for data_dir."""
        return Path(self.data_dir).expanduser()


def load_config() -> CollivindConfig:
    """
    Loads configuration from ~/.collivind/config.toml, falling back to defaults.
    """
    config_path = Path("~/.collivind/config.toml").expanduser()
    
    if not config_path.exists():
        return CollivindConfig()

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        raise ConfigurationError(f"Failed to parse config at {config_path}: {e}")

    # Build config merging defaults
    core_data = data.get("collivind", {})
    docker_data = data.get("docker", {})
    qdrant_data = data.get("qdrant", {})
    neo4j_data = data.get("neo4j", {})
    embeddings_data = data.get("embeddings", {})
    search_data = data.get("search", {})
    hooks_data = data.get("hooks", {})

    return CollivindConfig(
        user_id=core_data.get("user_id", "local"),
        data_dir=core_data.get("data_dir", "~/.collivind"),
        docker=DockerConfig(**docker_data),
        qdrant=QdrantConfig(**qdrant_data),
        neo4j=Neo4jConfig(**neo4j_data),
        embeddings=EmbeddingsConfig(**embeddings_data),
        search=SearchConfig(**search_data),
        hooks=HooksConfig(**hooks_data)
    )
