import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import ConfigurationError

# Chosen from data (BENCHMARKS.md): just above the cross-encoder's floor
# (about 1.1e-5 to 1.4e-5, where it puts text unrelated to the query).
# LoCoMo costs nothing; a synthetic coding check keeps every target and 8 of
# 10 unrelated prompts recall nothing. Higher values cut more noise but drop
# indirect matches (LongMemEval preference questions) that sit at the floor.
RECALL_MIN_SCORE_DEFAULT = 0.00002
# Recall stopping rule, off by default (BENCHMARKS.md, "Recall precision").
# recall_window = 12 with recall_limit = 10 halved recall tokens on held-out
# LoCoMo but cost 3.4 points of QA accuracy (95% CI -6.7 to -0.1); the floor
# at 0.001 emptied half the unrelated coding prompts but dropped 1 in 8
# relevant memories on prompts written blind to the scores.
RECALL_WINDOW_DEFAULT = 0.0
RECALL_FALLBACK_MIN_SCORE_DEFAULT = 0.0


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
    model: str = "BAAI/bge-small-en-v1.5"
    dimension: int = 384
    # applied to queries only (never documents); clear it for models that
    # don't use retrieval prefixes
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    api_key: str = ""  # API key for OpenAI or compatible services
    base_url: str = ""  # custom base URL for OpenAI-compatible APIs
    # provider "local" only: keep the model resident in a background process
    # so short-lived hooks don't reload it every prompt (#16)
    daemon: bool = True
    daemon_idle_timeout: int = 1800  # seconds without requests before it exits


@dataclass
class SearchConfig:
    default_limit: int = 10
    vector_weight: float = 0.7
    graph_weight: float = 0.3
    similarity_threshold: float = 0.3
    # Cosine at or above which a close memory about the same subject is returned
    # to the agent as a likely duplicate (possible_conflicts), in any category.
    # Nothing is retired on similarity; only identical text is rejected.
    dedup_threshold: float = 0.92
    temporal_decay_rate: float = 0.01
    temporal_decay_max: float = 0.3
    # Second-stage ranking: a local cross-encoder rescores a pool of dense +
    # keyword candidates. rerank = false restores the dense + graph ranking
    # exactly, and the model is then never loaded.
    rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_pool: int = 50  # candidates scored per search (at least the requested limit)
    lexical: bool = True  # add keyword (BM25) candidates to the rerank pool
    # Recall (get_context, i.e. the prompt hook) drops memories whose
    # cross-encoder score is below this, so an unrelated prompt recalls
    # nothing. 0 disables. Explicit search never applies it.
    recall_min_score: float = RECALL_MIN_SCORE_DEFAULT
    # Recall only, after the cutoff; both 0 (off) by default. When some memory
    # clears the reranker's relevance boundary, drop memories more than
    # recall_window logits below the best one. When none does (the reranker
    # finds nothing relevant), keep only memories scoring at least
    # recall_fallback_min_score. Trades answer accuracy for fewer tokens and
    # fewer unrelated memories; see the constants above.
    recall_window: float = RECALL_WINDOW_DEFAULT
    recall_fallback_min_score: float = RECALL_FALLBACK_MIN_SCORE_DEFAULT
    # When a summary memory is among the reranked candidates, return the top
    # `summary_routes` summaries followed by their best member memories (and
    # the best other memories); 0 treats summaries as ordinary results.
    # Needs the reranker. 3 is the setting measured in BENCHMARKS.md.
    summary_routes: int = 3

    def __post_init__(self):
        # on the prompt path: a typo must fail at config load, not mid-search
        if (
            isinstance(self.rerank_pool, bool)
            or not isinstance(self.rerank_pool, int)
            or not 1 <= self.rerank_pool <= 200
        ):
            raise ConfigurationError(f"[search] rerank_pool must be an integer from 1 to 200, got {self.rerank_pool!r}")
        if (
            isinstance(self.summary_routes, bool)
            or not isinstance(self.summary_routes, int)
            or not 0 <= self.summary_routes <= 10
        ):
            raise ConfigurationError(
                f"[search] summary_routes must be an integer from 0 to 10, got {self.summary_routes!r}"
            )
        if (
            isinstance(self.recall_min_score, bool)
            or not isinstance(self.recall_min_score, (int, float))
            or not (0.0 <= self.recall_min_score < 1.0)
        ):
            raise ConfigurationError(
                f"[search] recall_min_score must be a number from 0 to below 1, got {self.recall_min_score!r}"
            )
        if (
            isinstance(self.recall_window, bool)
            or not isinstance(self.recall_window, (int, float))
            or not (0.0 <= self.recall_window <= 50.0)
        ):
            raise ConfigurationError(
                f"[search] recall_window must be a number from 0 to 50, got {self.recall_window!r}"
            )
        if (
            isinstance(self.recall_fallback_min_score, bool)
            or not isinstance(self.recall_fallback_min_score, (int, float))
            or not (0.0 <= self.recall_fallback_min_score < 1.0)
        ):
            raise ConfigurationError(
                "[search] recall_fallback_min_score must be a number from 0 to below 1, "
                f"got {self.recall_fallback_min_score!r}"
            )
        for name in ("rerank", "lexical"):
            if not isinstance(getattr(self, name), bool):
                raise ConfigurationError(f"[search] {name} must be true or false, got {getattr(self, name)!r}")


@dataclass
class HooksConfig:
    save_interval: int = 15
    enable_precompact: bool = True
    enable_stop: bool = True
    enable_session_start: bool = True
    enable_user_prompt: bool = True
    # Ask the agent to checkpoint without announcing or confirming routine
    # saves; failures are still reported (#35).
    quiet_capture: bool = True
    # memories the prompt hook recalls per prompt, packed into a token budget
    # (len/4 estimate); `hook user-prompt --max-tokens` overrides the budget.
    # 2400 fits about 20 coding-length memories (~120 tokens each); the recall
    # cutoff runs first. 10 halves the tokens but cost 3.4 points of LoCoMo QA
    # accuracy on held-out conversations (BENCHMARKS.md, "Recall precision").
    recall_limit: int = 20
    recall_max_tokens: int = 2400
    # Ask the agent for a session summary once a session holds this many
    # memories no summary covers yet; 0 never asks.
    summary_after: int = 8

    def __post_init__(self):
        if isinstance(self.summary_after, bool) or not isinstance(self.summary_after, int) or self.summary_after < 0:
            raise ConfigurationError(
                f"[hooks] summary_after must be a non-negative integer, got {self.summary_after!r}"
            )
        if (
            isinstance(self.recall_limit, bool)
            or not isinstance(self.recall_limit, int)
            or not 1 <= self.recall_limit <= 50
        ):
            raise ConfigurationError(f"[hooks] recall_limit must be an integer from 1 to 50, got {self.recall_limit!r}")
        if (
            isinstance(self.recall_max_tokens, bool)
            or not isinstance(self.recall_max_tokens, int)
            or not 100 <= self.recall_max_tokens <= 20000
        ):
            raise ConfigurationError(
                f"[hooks] recall_max_tokens must be an integer from 100 to 20000, got {self.recall_max_tokens!r}"
            )


@dataclass
class CollivindConfig:
    user_id: str = "local"
    data_dir: str = "~/.collivind"
    mode: str = "docker"  # "docker", "embedded", or "remote"
    check_updates: bool = True  # daily PyPI version check on CLI use
    docker: DockerConfig = field(default_factory=DockerConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    hooks: HooksConfig = field(default_factory=HooksConfig)

    @property
    def active_rerank_model(self) -> str:
        """The cross-encoder to load, or "" when rerank is off (nothing is loaded)."""
        return self.search.rerank_model if self.search.rerank else ""

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
#   COLLIVIND_EMBED_DAEMON=0   — keep the local model in-process (no daemon)

[collivind]
user_id = "local"
data_dir = "~/.collivind"
check_updates = true  # daily PyPI version check on CLI use
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
model = "BAAI/bge-small-en-v1.5"
dimension = 384
# query_prefix is applied to search queries only; clear it for models
# that don't use retrieval prefixes (e.g. all-MiniLM-L6-v2)
query_prefix = "Represent this sentence for searching relevant passages: "
# service_url = "http://localhost:8090"  # for provider = "http"
# api_key = ""                            # for provider = "openai"
# base_url = ""                           # custom OpenAI-compatible endpoint
# daemon = true               # provider "local": keep the model resident between hooks
# daemon_idle_timeout = 1800  # seconds idle before the resident model exits

[search]
default_limit = 10
vector_weight = 0.7
graph_weight = 0.3
similarity_threshold = 0.3
# likely-duplicate cosine: such a memory is returned to the agent as a possible
# conflict; only identical text is rejected, nothing is retired on similarity
dedup_threshold = 0.92
temporal_decay_rate = 0.01
temporal_decay_max = 0.3
# A local cross-encoder reranks dense + keyword candidates (no API key; the
# model, about 90 MB, downloads once). rerank = false gives the previous
# ranking exactly. lexical adds keyword (BM25) candidates to the rerank pool.
rerank = true
rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
rerank_pool = 50
lexical = true
# Recall (the prompt hook, collivind_get_context) drops memories the reranker
# scores below this (0..1), so an unrelated prompt recalls nothing. 0 = off.
# Explicit search never applies it.
recall_min_score = 0.00002
# Recall stopping rule, off by default: with a confident match, drop memories
# more than recall_window logits below the best (12 measured); with none, keep
# only memories scoring at least recall_fallback_min_score (0.001 measured).
# Fewer tokens and fewer unrelated memories, at some cost in answers.
recall_window = 0.0
recall_fallback_min_score = 0.0
# A summary among the candidates returns the top N summaries plus their most
# relevant member memories. 0 = treat summaries as ordinary results.
summary_routes = 3

# --- Docker (docker mode only) ---
# [docker]
# compose_project = "collivind"
# auto_start = true

[hooks]
save_interval = 15
enable_precompact = true
enable_stop = true
enable_session_start = true
enable_user_prompt = true
quiet_capture = true
recall_limit = 20  # memories recalled per prompt
recall_max_tokens = 2400  # token budget for those memories (about 20 coding-length memories)
summary_after = 8  # ask the agent for a session summary after this many new memories (0 = never)
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

    # Service addresses, so a container can reach its siblings by compose
    # service name without a config.toml of its own.
    if os.environ.get("COLLIVIND_QDRANT_HOST"):
        config.qdrant.host = os.environ["COLLIVIND_QDRANT_HOST"]
    if os.environ.get("COLLIVIND_NEO4J_URI"):
        config.neo4j.uri = os.environ["COLLIVIND_NEO4J_URI"]
    if os.environ.get("COLLIVIND_NEO4J_PASSWORD"):
        config.neo4j.password = os.environ["COLLIVIND_NEO4J_PASSWORD"]
    if os.environ.get("COLLIVIND_EMBEDDINGS_URL"):
        config.embeddings.service_url = os.environ["COLLIVIND_EMBEDDINGS_URL"]

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
        check_updates=core_data.get("check_updates", True),
        docker=DockerConfig(**docker_data),
        qdrant=QdrantConfig(**qdrant_data),
        neo4j=Neo4jConfig(**neo4j_data),
        embeddings=EmbeddingsConfig(**embeddings_data),
        search=SearchConfig(**search_data),
        hooks=HooksConfig(**hooks_data),
    )
    _apply_env_overrides(config)
    return config
