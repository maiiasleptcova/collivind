import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import ConfigurationError


@dataclass
class DockerConfig:
    compose_project: str = "collivind"
    auto_start: bool = True


@dataclass
class QdrantConfig:
    provider: str = ""  # "" = use mode default; "embedded", "local", "cloud"
    host: str = "localhost"
    port: int = 6333
    url: str = ""  # full URL for Qdrant Cloud (e.g. https://xyz.us-east.aws.cloud.qdrant.io)
    api_key: str = ""  # API key for Qdrant Cloud
    collection_name: str = "collivind_memories"


@dataclass
class Neo4jConfig:
    provider: str = ""  # "" = use mode default; "sqlite", "neo4j"
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "collivind_local"
    database: str = "neo4j"


@dataclass
class EmbeddingsConfig:
    provider: str = ""  # "" = use mode default; "local", "http", "openai", "ollama"
    service_url: str = "http://localhost:8090"
    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    api_key: str = ""  # API key for OpenAI or compatible services
    base_url: str = ""  # custom base URL for OpenAI-compatible APIs


@dataclass
class SearchConfig:
    default_limit: int = 10
    vector_weight: float = 0.7
    graph_weight: float = 0.3
    similarity_threshold: float = 0.3
    dedup_threshold: float = 0.92
    temporal_decay_rate: float = 0.01
    temporal_decay_max: float = 0.3


@dataclass
class HooksConfig:
    save_interval: int = 15
    enable_precompact: bool = True
    enable_stop: bool = True
    enable_session_start: bool = True


@dataclass
class CollivindConfig:
    user_id: str = "local"
    data_dir: str = "~/.collivind"
    mode: str = "docker"  # "docker", "embedded", or "remote"
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


DEFAULT_CONFIG_TEMPLATE = """\
# Environment variables (override config values):
#   COLLIVIND_API_KEY          — shared API key for all services
#   COLLIVIND_EMBEDDINGS_API_KEY — override for embeddings only
#   COLLIVIND_QDRANT_API_KEY   — override for Qdrant only
#   COLLIVIND_QDRANT_URL       — Qdrant connection URL
#   COLLIVIND_EMBEDDINGS_BASE_URL — custom embeddings endpoint
#   COLLIVIND_MODE             — override mode (embedded/docker/remote)

[collivind]
user_id = "local"
data_dir = "~/.collivind"
# mode selects defaults for all backends at once:
#   "embedded" = in-process Qdrant + SQLite + local sentence-transformers
#   "docker"   = Docker Qdrant + Neo4j + HTTP embedding service
#   "remote"   = external services (configure each section below)
# You can override individual backends below regardless of mode.
mode = "{mode}"

# --- Vector Store (Qdrant) ---
[qdrant]
# provider: "embedded" (in-process), "local" (Docker/localhost), "cloud" (Qdrant Cloud / remote)
# provider = "embedded"
# host = "localhost"          # for provider = "local"
# port = 6333                 # for provider = "local"
# url = ""                    # for provider = "cloud" (e.g. https://xyz.cloud.qdrant.io)
# api_key = ""                # for provider = "cloud"
collection_name = "collivind_memories"

# --- Graph Store ---
[neo4j]
# provider: "sqlite" (no Docker) or "neo4j" (Docker/remote)
# provider = "sqlite"
# uri = "bolt://localhost:7687"  # for provider = "neo4j"
# user = "neo4j"
# password = "collivind_local"
# database = "neo4j"

# --- Embeddings ---
[embeddings]
# provider: "local" (sentence-transformers), "http" (custom HTTP service),
#           "openai" (OpenAI API), "ollama" (Ollama local)
# provider = "local"
model = "all-MiniLM-L6-v2"
dimension = 384
# service_url = "http://localhost:8090"  # for provider = "http"
# api_key = ""                            # for provider = "openai"
# base_url = ""                           # custom OpenAI-compatible endpoint

[search]
default_limit = 10
vector_weight = 0.7
graph_weight = 0.3
similarity_threshold = 0.3
dedup_threshold = 0.92
temporal_decay_rate = 0.01
temporal_decay_max = 0.3

# --- Docker (docker mode only) ---
# [docker]
# compose_project = "collivind"
# auto_start = true

[hooks]
save_interval = 15
enable_precompact = true
enable_stop = true
enable_session_start = true
"""


def generate_default_config(path: Path, mode: str = "docker") -> None:
    """Write a default config.toml file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_CONFIG_TEMPLATE.format(mode=mode))


def _apply_env_overrides(config: "CollivindConfig") -> None:
    """Apply environment variable overrides. Priority: per-service env > COLLIVIND_API_KEY > config file."""
    shared_key = os.environ.get("COLLIVIND_API_KEY", "")

    embeddings_key = os.environ.get("COLLIVIND_EMBEDDINGS_API_KEY", "")
    if embeddings_key:
        config.embeddings.api_key = embeddings_key
    elif shared_key and not config.embeddings.api_key:
        config.embeddings.api_key = shared_key

    qdrant_key = os.environ.get("COLLIVIND_QDRANT_API_KEY", "")
    if qdrant_key:
        config.qdrant.api_key = qdrant_key
    elif shared_key and not config.qdrant.api_key:
        config.qdrant.api_key = shared_key

    qdrant_url = os.environ.get("COLLIVIND_QDRANT_URL", "")
    if qdrant_url:
        config.qdrant.url = qdrant_url

    embeddings_base = os.environ.get("COLLIVIND_EMBEDDINGS_BASE_URL", "")
    if embeddings_base:
        config.embeddings.base_url = embeddings_base

    mode = os.environ.get("COLLIVIND_MODE", "")
    if mode:
        config.mode = mode


def load_config() -> CollivindConfig:
    config_path = Path("~/.collivind/config.toml").expanduser()

    if not config_path.exists():
        config = CollivindConfig()
        _apply_env_overrides(config)
        return config

    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        raise ConfigurationError(f"Failed to parse config at {config_path}: {e}")

    core_data = data.get("collivind", {})
    docker_data = data.get("docker", {})
    qdrant_data = data.get("qdrant", {})
    neo4j_data = data.get("neo4j", {})
    embeddings_data = data.get("embeddings", {})
    search_data = data.get("search", {})
    hooks_data = data.get("hooks", {})

    config = CollivindConfig(
        user_id=core_data.get("user_id", "local"),
        data_dir=core_data.get("data_dir", "~/.collivind"),
        mode=core_data.get("mode", "docker"),
        docker=DockerConfig(**docker_data),
        qdrant=QdrantConfig(**qdrant_data),
        neo4j=Neo4jConfig(**neo4j_data),
        embeddings=EmbeddingsConfig(**embeddings_data),
        search=SearchConfig(**search_data),
        hooks=HooksConfig(**hooks_data),
    )
    _apply_env_overrides(config)
    return config
