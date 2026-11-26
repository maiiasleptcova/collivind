"""Seeded search-latency benchmark: the "search under 500 ms at 10k memories" target.

Seeds N memories, each linked to a few entities, into the embedded SQLite
stores in a temp dir, then times SearchEngine.search with graph expansion on.
No embedding model is loaded: a token-hash embedder gives related texts
related vectors. The rerank arm adds the real cross-encoder (search.rerank,
loaded in-process) over the real pool: dense + keyword candidates, each text
padded to the model's 256-token window, so its cost is the worst case.

    uv run python benchmarks/search_bench.py                 # 10k memories, rerank off and on
    uv run python benchmarks/search_bench.py --no-rerank     # skip the cross-encoder arm
    uv run python benchmarks/search_bench.py --legacy        # per-item store lookups, see LegacyGraphStore
    uv run python benchmarks/search_bench.py --memories 2000 --queries 50

Everything is derived from --seed, so two runs see the same data and queries.
tests/unit/test_search_perf.py imports seed_stores/run_queries from here.
"""

import argparse
import hashlib
import random
import statistics
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from itertools import accumulate
from typing import Dict, List, Optional

import numpy as np

from collivind.config import CollivindConfig
from collivind.engine.search_engine import SearchEngine
from collivind.models import MemoryCategory, MemoryNode, SearchQuery
from collivind.models.entity import EntityNode, EntityType
from collivind.models.relationship import RelationshipEdge, RelType
from collivind.storage.graph_sqlite import SqliteGraphStore
from collivind.storage.interfaces import GraphStore
from collivind.storage.vector_sqlite import SqliteVectorStore

DIM = 384
VOCAB = 1500
WORDS_PER_MEMORY = 8
ENTITIES = 400
ENTITIES_PER_MEMORY = 3


class TokenHashEmbedder:
    """Sum of fixed random vectors per whitespace token, normalised.

    A pure per-text hash would make every pair orthogonal and nothing would
    clear the similarity threshold; summing token vectors means texts that
    share words score high, which is what makes seeds and expansion happen.
    """

    def __init__(self, dimension: int = DIM):
        self._dimension = dimension
        self._cache: Dict[str, np.ndarray] = {}

    def _token(self, tok: str) -> np.ndarray:
        vec = self._cache.get(tok)
        if vec is None:
            seed = int.from_bytes(hashlib.sha1(tok.encode()).digest()[:8], "big")
            vec = np.random.default_rng(seed).standard_normal(self._dimension)
            self._cache[tok] = vec
        return vec

    def embed(self, text: str) -> List[float]:
        total = np.zeros(self._dimension)
        for tok in text.split():
            total += self._token(tok)
        norm = np.linalg.norm(total) or 1.0
        return (total / norm).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed(text)

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def health_check(self):
        return {"status": "ok"}

    @property
    def dimension(self) -> int:
        return self._dimension


class LegacyGraphStore(SqliteGraphStore):
    """SqliteGraphStore with the batched lookups swapped for the per-item ABC defaults.

    One query per seed and per distinct entity: the reference the equivalence
    test compares against. It is not the whole pre-batching cost: that engine
    also re-queried an entity once per seed linking to it (144 statements per
    search at 10k against 113 here), so before-numbers come from origin/main.
    """

    get_neighbors_batch = GraphStore.get_neighbors_batch
    find_related_memories_batch = GraphStore.find_related_memories_batch


def _zipf_weights(n: int) -> List[float]:
    return list(accumulate(1.0 / (k + 1) for k in range(n)))


_WORD_WEIGHTS = _zipf_weights(VOCAB)
_ENTITY_WEIGHTS = _zipf_weights(ENTITIES)


def _zipf_pick(rng: random.Random, cum_weights: List[float]) -> int:
    # Zipf(s=1): a few hot words/entities (the top entity sits on ~15% of
    # memories, far past the 20-per-entity expansion limit) and a long tail
    return rng.choices(range(len(cum_weights)), cum_weights=cum_weights)[0]


def _memory_text(rng: random.Random) -> List[str]:
    return [f"w{_zipf_pick(rng, _WORD_WEIGHTS)}" for _ in range(WORDS_PER_MEMORY)]


def _seed_id(kind: str, i: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"collivind-bench/{kind}/{i}"))


def seed_stores(data_dir: str, memories: int, seed_value: int = 7, legacy: bool = False):
    """Fill fresh embedded stores in data_dir; return (graph, vector, embedder, texts)."""
    graph = (LegacyGraphStore if legacy else SqliteGraphStore)(data_dir)
    vector = SqliteVectorStore(data_dir, CollivindConfig().qdrant, DIM)
    # seeding only: a tmp dir does not need fsync per row
    graph.conn.execute("PRAGMA synchronous=OFF")
    vector.conn.execute("PRAGMA synchronous=OFF")
    embedder = TokenHashEmbedder()
    texts = seed(graph, vector, embedder, memories, seed_value)
    return graph, vector, embedder, texts


def seed(graph: GraphStore, vector, embedder: TokenHashEmbedder, memories: int, seed_value: int = 7) -> List[str]:
    """Write the deterministic data set into any GraphStore/VectorStore pair; return the texts."""
    rng = random.Random(seed_value)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for e in range(ENTITIES):
        graph.restore_entity(EntityNode(id=f"ent_{e}", name=f"Ent {e}", type=EntityType.CONCEPT, created_at=base))

    texts = []
    for i in range(memories):
        words = _memory_text(rng)
        text = " ".join(words)
        texts.append(text)
        created = base + timedelta(minutes=i)  # distinct: ORDER BY created_at has no ties
        node = MemoryNode(
            id=_seed_id("mem", i),
            content=text,
            summary=text[:40],
            category=MemoryCategory.FACT,
            created_at=created,
            updated_at=created,
            valid_from=created,
        )
        graph.restore_memory(node)
        vector.upsert(node.id, embedder.embed(text), node.to_dict())
        for e in sorted({_zipf_pick(rng, _ENTITY_WEIGHTS) for _ in range(ENTITIES_PER_MEMORY)}):
            graph.restore_relationship(
                RelationshipEdge(
                    id=_seed_id(f"rel/{i}", e),
                    source_id=node.id,
                    target_id=f"ent_{e}",
                    type=RelType.ABOUT,
                    created_at=created,
                )
            )
    return texts


def make_queries(texts: List[str], n: int, seed: int = 11) -> List[str]:
    rng = random.Random(seed)
    return [" ".join(rng.sample(rng.choice(texts).split(), 3)) for _ in range(n)]


def run_queries(engine: SearchEngine, queries: List[str], limit: int = 10, **recall: float) -> Dict[str, float]:
    """Time each search; the first one (vector cache load) is a warm-up, not counted.

    recall: SearchQuery recall fields (min_rerank_score, rerank_window, ...) to time the recall path.
    """
    engine.search(SearchQuery(query=queries[0], limit=limit, include_graph=True, **recall))
    timings = []
    hits = []
    for q in queries:
        start = time.perf_counter()
        results = engine.search(SearchQuery(query=q, limit=limit, include_graph=True, **recall))
        timings.append((time.perf_counter() - start) * 1000)
        hits.append(len(results))
    timings.sort()
    return {
        "p50_ms": statistics.median(timings),
        "p95_ms": timings[max(0, int(len(timings) * 0.95) - 1)],
        "max_ms": timings[-1],
        "mean_results": statistics.mean(hits),
    }


def count_statements(graph: SqliteGraphStore, engine: SearchEngine, query: str) -> int:
    """SQL statements the graph store runs for one search (the N+1 shows up here)."""
    count = 0

    def _tick(_sql):
        nonlocal count
        count += 1

    graph.conn.set_trace_callback(_tick)
    try:
        engine.search(SearchQuery(query=query, limit=10, include_graph=True))
    finally:
        graph.conn.set_trace_callback(None)
    return count


class RerankingHashEmbedder(TokenHashEmbedder):
    """TokenHashEmbedder plus the real cross-encoder, fed worst-case (window-filling) texts."""

    can_rerank = True

    def __init__(self, model: str, dimension: int = DIM):
        super().__init__(dimension)
        from collivind.storage.embed_daemon import load_cross_encoder

        self._reranker = load_cross_encoder(model)
        self.pool_sizes: List[int] = []

    def rerank(self, query: str, texts: List[str]) -> List[float]:
        self.pool_sizes.append(len(texts))
        padded = [(t + " ") * (1 + 1200 // (len(t) + 1)) for t in texts]  # >= 256 tokens: truncated to the window
        return [float(x) for x in self._reranker.predict([(query, t) for t in padded], batch_size=64)]


def build_engine(graph, vector, embedder, rerank: bool = False) -> SearchEngine:
    config = CollivindConfig().search
    config.rerank = rerank
    return SearchEngine(vector, graph, embedder, config)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--memories", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--legacy", action="store_true", help="per-item store lookups (the equivalence reference)")
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder arm")
    parser.add_argument("--rerank-model", default=CollivindConfig().search.rerank_model)
    parser.add_argument("--recall-window", type=float, default=0.0, help="also time the recall path with this window")
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="collivind-bench-") as tmp:
        start = time.perf_counter()
        graph, vector, embedder, texts = seed_stores(tmp, args.memories, args.seed, legacy=args.legacy)
        seeded_s = time.perf_counter() - start
        engine = build_engine(graph, vector, embedder)
        queries = make_queries(texts, args.queries)

        # sanity: without seeds and expansion the timing would measure nothing
        probe = SearchQuery(query=queries[0], limit=10, include_graph=True)
        enriched = engine.vector_store.search(
            embedder.embed(probe.query), limit=30, threshold=engine.config.similarity_threshold
        )
        expanded = engine.graph_engine.get_expanded_memories([r["id"] for r in enriched])
        stats = run_queries(engine, queries)
        statements = count_statements(graph, engine, queries[0])
        rerank_stats = window_stats = None
        if not args.no_rerank:
            reranking = RerankingHashEmbedder(args.rerank_model)
            reranking._cache = embedder._cache
            rerank_stats = run_queries(build_engine(graph, vector, reranking, rerank=True), queries)
            pool = statistics.mean(reranking.pool_sizes)
            if args.recall_window:
                window_stats = run_queries(
                    build_engine(graph, vector, reranking, rerank=True), queries,
                    min_rerank_score=0.00002, rerank_window=args.recall_window,
                )  # fmt: skip
        graph.close()
        vector.close()

    path = "legacy per-item" if args.legacy else "batched"
    print(f"memories={args.memories} queries={args.queries} path={path} seeded_in={seeded_s:.1f}s")
    print(f"probe: vector seeds={len(enriched)} expanded={len(expanded)} graph SQL statements/search={statements}")
    print(
        f"rerank off: p50={stats['p50_ms']:.1f}ms p95={stats['p95_ms']:.1f}ms max={stats['max_ms']:.1f}ms "
        f"mean_results={stats['mean_results']:.1f}"
    )
    if rerank_stats:
        print(
            f"rerank on:  p50={rerank_stats['p50_ms']:.1f}ms p95={rerank_stats['p95_ms']:.1f}ms "
            f"max={rerank_stats['max_ms']:.1f}ms mean_results={rerank_stats['mean_results']:.1f} "
            f"(in-process {args.rerank_model}, mean pool {pool:.0f} texts at 256 tokens)"
        )
    if window_stats:
        print(
            f"recall, window {args.recall_window:g}: p50={window_stats['p50_ms']:.1f}ms "
            f"p95={window_stats['p95_ms']:.1f}ms max={window_stats['max_ms']:.1f}ms "
            f"mean_results={window_stats['mean_results']:.1f}"
        )
    if not enriched or not expanded:
        print("WARNING: probe query had no vector seeds or no expansion; numbers are meaningless", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
