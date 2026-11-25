"""Keyword (FTS5) channel of the SQLite graph store, and the search engine's rerank stage around it."""

import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from collivind.config import QdrantConfig, SearchConfig
from collivind.engine.search_engine import SearchEngine, _rrf
from collivind.models import MemoryCategory, MemoryCreate, MemoryNode, SearchQuery
from collivind.storage.graph_sqlite import SqliteGraphStore
from collivind.storage.lexical import query_terms
from collivind.storage.vector_sqlite import SqliteVectorStore


def _add(graph, content, project="p", **kw):
    return graph.create_memory(
        MemoryCreate(
            content=content,
            summary=content[:20],
            category=kw.pop("category", MemoryCategory.FACT),
            project_id=project,
            **kw,
        )
    )


@pytest.fixture
def graph(tmp_path):
    store = SqliteGraphStore(str(tmp_path))
    yield store
    store.close()


def _contents(hits):
    return [m.content for m, _ in hits]


def test_query_terms_strip_syntax_and_stopwords():
    assert query_terms('Why did "postgres" fail?? (migration) AND NOT x*') == ["postgres", "fail", "migration", "x"]
    assert query_terms("the of and") == []
    assert len(query_terms(" ".join(f"w{i}" for i in range(100)))) == 32


def test_ranks_by_keywords_and_filters_project(graph):
    _add(graph, "postgres migration failed on tuesday")
    _add(graph, "redis cache warmed")
    _add(graph, "postgres in another project", project="other")
    assert _contents(graph.lexical_search("did the postgres migration fail?", 10, project_id="p")) == [
        "postgres migration failed on tuesday"
    ]
    assert len(graph.lexical_search("postgres", 10)) == 2  # no project: all projects


def test_superseded_invalidated_and_forgotten_never_come_back(graph):
    old = _add(graph, "deploy uses kubectl apply")
    new = _add(graph, "deploy uses helm upgrade")
    gone = _add(graph, "deploy uses ansible")
    expired = _add(graph, "deploy uses fabric")
    graph.invalidate_memory(old.id, new.id, "superseded_by_update")
    graph.delete_memory(gone.id)
    graph.update_memory(expired.id, valid_to=datetime.now(timezone.utc))
    assert _contents(graph.lexical_search("deploy", 10)) == ["deploy uses helm upgrade"]


def test_index_follows_content_edits_and_restores(graph):
    m = _add(graph, "alpha")
    graph.update_memory(m.id, content="beta")
    assert graph.lexical_search("alpha", 5) == []
    assert _contents(graph.lexical_search("beta", 5)) == ["beta"]
    graph.restore_memory(replace(graph.get_memory(m.id), content="gamma"))  # INSERT OR REPLACE
    assert _contents(graph.lexical_search("gamma", 5)) == ["gamma"]
    assert graph.lexical_search("beta", 5) == []
    assert graph.conn.execute("SELECT count(*) FROM memories_fts").fetchone()[0] == 1


def test_existing_store_is_backfilled_once(tmp_path):
    store = SqliteGraphStore(str(tmp_path))
    _add(store, "written before the upgrade")
    store.conn.executescript(
        "DROP TRIGGER memories_fts_ai; DROP TRIGGER memories_fts_ad; DROP TRIGGER memories_fts_au;"
        "DROP TABLE memories_fts;"
    )
    store.close()  # now shaped like a 0.18.0 store

    upgraded = SqliteGraphStore(str(tmp_path))
    assert _contents(upgraded.lexical_search("upgrade", 5)) == ["written before the upgrade"]
    again = SqliteGraphStore(str(tmp_path))  # a second open must not duplicate rows
    assert again.conn.execute("SELECT count(*) FROM memories_fts").fetchone()[0] == 1
    upgraded.close()
    again.close()


def test_no_fts5_means_no_keyword_candidates(tmp_path, monkeypatch):
    def boom(self):
        raise sqlite3.OperationalError("no such module: fts5")

    monkeypatch.setattr(SqliteGraphStore, "_has_fts_table", boom)
    store = SqliteGraphStore(str(tmp_path))
    _add(store, "postgres")
    assert store.lexical_search("postgres", 5) == []


def test_rrf_prefers_items_in_both_lists_and_dense_on_ties():
    assert _rrf([["a", "b", "c"], ["c", "d"]]) == ["c", "a", "b", "d"]
    assert _rrf([["x"], ["y"]]) == ["x", "y"]


# --- engine -----------------------------------------------------------------


class WordEmbedder:
    """Bag-of-words over a tiny vocabulary: deterministic and model-free."""

    VOCAB = ["deploy", "helm", "kubectl", "postgres", "redis", "cache", "migration", "<none>"]
    can_rerank = True

    def __init__(self, scores=None, fail=False):
        self.calls = []
        self.fail = fail

    def embed(self, text):
        words = text.lower().split()
        vec = [float(sum(w.startswith(v) for w in words)) for v in self.VOCAB]
        return vec if any(vec) else [0.0] * (len(self.VOCAB) - 1) + [1.0]  # no known word: its own axis

    embed_query = embed

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def rerank(self, query, texts):
        self.calls.append(texts)
        if self.fail:
            return None
        return [10.0 if "zebra" in t else -float(len(t)) for t in texts]  # "zebra" is the true answer

    def health_check(self):
        return {"status": "ok"}

    @property
    def dimension(self):
        return len(self.VOCAB)


@pytest.fixture
def stores(tmp_path):
    graph = SqliteGraphStore(str(tmp_path))
    vector = SqliteVectorStore(str(tmp_path), QdrantConfig(), len(WordEmbedder.VOCAB))
    yield graph, vector
    graph.close()
    vector.close()


def _store(graph, vector, embedder, content, **kw):
    node = _add(graph, content, **kw)
    vector.upsert(node.id, embedder.embed(content), node.to_dict())
    return node


def _engine(graph, vector, embedder, **cfg):
    return SearchEngine(vector, graph, embedder, replace(SearchConfig(), **cfg))


def test_keyword_only_hit_reaches_the_reranker_and_wins(stores):
    graph, vector = stores
    emb = WordEmbedder()
    for i in range(5):
        _store(graph, vector, emb, f"deploy with helm step {i}")
    # no vocabulary word: the dense channel cannot find it, the keyword channel can
    target = _store(graph, vector, emb, "zebra crossing notes for rollback")
    query = SearchQuery(query="deploy zebra rollback", project_id="p", limit=3)

    off = _engine(graph, vector, emb, rerank=False).search(query)
    assert target.id not in [r.memory.id for r in off]
    on = _engine(graph, vector, emb).search(query)
    assert on[0].memory.id == target.id and on[0].rerank_score is not None
    no_lexical = _engine(graph, vector, emb, lexical=False).search(query)
    assert target.id not in [r.memory.id for r in no_lexical]


def test_reranker_failure_serves_exactly_the_rerank_off_ranking(stores, monkeypatch):
    from collivind.engine import search_engine as module

    frozen = datetime(2026, 12, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(module, "datetime", type("F", (datetime,), {"now": classmethod(lambda cls, tz=None: frozen)}))
    graph, vector = stores
    emb = WordEmbedder()
    for i in range(12):
        _store(graph, vector, emb, f"deploy helm postgres {'redis ' * (i % 4)} {i}")
    query = SearchQuery(query="deploy postgres redis", project_id="p", limit=5)

    def ranked(engine):
        return [(r.memory.id, r.score, r.vector_score, r.graph_score) for r in engine.search(query)]

    off = ranked(_engine(graph, vector, emb, rerank=False))
    failing = WordEmbedder(fail=True)
    assert ranked(_engine(graph, vector, failing)) == off
    assert failing.calls, "the reranker was never asked"


def test_exploding_reranker_is_contained(stores):
    graph, vector = stores
    emb = WordEmbedder()
    _store(graph, vector, emb, "deploy helm")
    emb.rerank = lambda q, t: (_ for _ in ()).throw(RuntimeError("boom"))
    assert [
        r.memory.content for r in _engine(graph, vector, emb).search(SearchQuery(query="deploy", project_id="p"))
    ] == ["deploy helm"]


@pytest.mark.parametrize(
    "query_kw, stored_kw",
    [
        ({"category": "decision"}, {"category": MemoryCategory.FACT}),
        ({"tags": ["infra"]}, {"tags": ["frontend"]}),
        ({"session_id": "s1"}, {"session_id": "s2"}),
        ({"user_id": "alice"}, {"user_id": "bob"}),
        ({"project_id": "other"}, {}),
        ({"filters": {"source": "hook"}}, {}),
    ],
)
def test_search_filters_apply_to_keyword_hits(stores, query_kw, stored_kw):
    graph, vector = stores
    emb = WordEmbedder()
    # keyword-only (no vocabulary word), so only the lexical channel can surface it
    node = graph.create_memory(
        MemoryCreate(
            content="zebra notes", summary="z", project_id="p", **{"category": MemoryCategory.FACT, **stored_kw}
        )
    )
    q = SearchQuery(query="zebra", **{"project_id": "p", **query_kw})
    assert node.id not in [r.memory.id for r in _engine(graph, vector, emb).search(q)]
    # and the same memory does come back without the filter
    assert node.id in [
        r.memory.id for r in _engine(graph, vector, emb).search(SearchQuery(query="zebra", project_id="p"))
    ]


def test_date_and_valid_to_filters_apply_to_keyword_hits(stores):
    graph, vector = stores
    emb = WordEmbedder()
    node = _add(graph, "zebra notes")
    now = datetime.now(timezone.utc)
    engine = _engine(graph, vector, emb)
    assert node.id not in [
        r.memory.id
        for r in engine.search(SearchQuery(query="zebra", project_id="p", date_from=now + timedelta(days=1)))
    ]
    assert node.id not in [
        r.memory.id for r in engine.search(SearchQuery(query="zebra", project_id="p", date_to=now - timedelta(days=1)))
    ]
    graph.invalidate_memory(node.id, "nobody", "test")
    assert engine.search(SearchQuery(query="zebra", project_id="p")) == []


def test_graph_only_candidates_stay_in_the_pool(stores):
    graph, vector = stores
    emb = WordEmbedder()
    seed = _store(graph, vector, emb, "deploy helm")
    linked: MemoryNode = _add(graph, "unrelated words entirely")  # no vector, no keyword match
    from collivind.models import RelationshipCreate
    from collivind.models.entity import EntityCreate, EntityType
    from collivind.models.relationship import RelType

    graph.create_entity(EntityCreate(name="Infra", type=EntityType.CONCEPT))
    for m in (seed, linked):
        graph.create_relationship(RelationshipCreate(source_id=m.id, target_id="infra", type=RelType.ABOUT))
    emb.rerank = lambda q, texts: [1.0] * len(texts)  # clears the gate
    results = _engine(graph, vector, emb).search(SearchQuery(query="deploy", project_id="p"))
    by_id = {r.memory.id: r for r in results}
    assert linked.id in by_id and by_id[linked.id].graph_score > 0 and by_id[linked.id].rerank_score is not None


@pytest.mark.parametrize("bad", [[float("nan"), 1.0], [float("inf"), 1.0], ["x", 1.0]])
def test_non_finite_logits_fall_back_to_first_stage(stores, bad):
    graph, vector = stores
    emb = WordEmbedder()
    for text in ("deploy helm", "deploy kubectl"):
        _store(graph, vector, emb, text)
    query = SearchQuery(query="deploy", project_id="p")
    off = [r.memory.id for r in _engine(graph, vector, emb, rerank=False).search(query)]
    emb.rerank = lambda q, texts: bad
    results = _engine(graph, vector, emb).search(query)
    assert [r.memory.id for r in results] == off and all(r.rerank_score is None for r in results)


@pytest.mark.parametrize("pool", [0, 201, "50", True])
def test_rerank_pool_is_validated(pool):
    from collivind.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        SearchConfig(rerank_pool=pool)


def test_no_confident_candidate_serves_the_first_stage(stores):
    """All logits below the model's relevance boundary: its order is noise, keep dense + graph."""
    graph, vector = stores
    emb = WordEmbedder()
    for text in ("deploy helm", "deploy helm kubectl", "deploy postgres"):
        _store(graph, vector, emb, text)
    query = SearchQuery(query="deploy helm", project_id="p")
    off = [r.memory.id for r in _engine(graph, vector, emb, rerank=False).search(query)]
    emb.rerank = lambda q, texts: [-1.0 - i for i in range(len(texts))][::-1]  # reversed order, all negative
    gated = _engine(graph, vector, emb).search(query)
    assert [r.memory.id for r in gated] == off and all(r.rerank_score is None for r in gated)
    emb.rerank = lambda q, texts: [0.5] + [-3.0] * (len(texts) - 1)  # one confident candidate: rerank applies
    assert _engine(graph, vector, emb).search(query)[0].rerank_score is not None


def _scored(stores, logits_by_word):
    """Three memories; the reranker scores each by the first matching word."""
    graph, vector = stores
    emb = WordEmbedder()
    nodes = {w: _store(graph, vector, emb, f"deploy helm {w}") for w in logits_by_word}
    emb.rerank = lambda q, texts: [next(v for w, v in logits_by_word.items() if w in t) for t in texts]
    return graph, vector, emb, nodes


def test_recall_cutoff_drops_low_scores(stores):
    graph, vector, emb, nodes = _scored(stores, {"alpha": 4.0, "beta": 1.0, "gamma": -6.0})
    engine = _engine(graph, vector, emb)
    q = SearchQuery(query="deploy helm", project_id="p", limit=10)
    assert {r.memory.id for r in engine.search(q)} == {n.id for n in nodes.values()}  # no cutoff: all
    kept = engine.search(replace(q, min_rerank_score=0.5))
    assert [r.memory.id for r in kept] == [nodes["alpha"].id, nodes["beta"].id]


def test_recall_cutoff_can_return_nothing_and_also_filters_the_gated_order(stores):
    graph, vector, emb, nodes = _scored(stores, {"alpha": -1.0, "beta": -4.0, "gamma": -9.0})
    engine = _engine(graph, vector, emb)
    q = SearchQuery(query="deploy helm", project_id="p", limit=10)
    first_stage = [r.memory.id for r in _engine(graph, vector, emb, rerank=False).search(q)]
    assert [r.memory.id for r in engine.search(q)] == first_stage  # gated, no cutoff: first stage as is
    gated = engine.search(replace(q, min_rerank_score=0.01))  # sigmoid(-1)=0.27, (-4)=0.018, (-9)=0.0001
    assert [r.memory.id for r in gated] == [i for i in first_stage if i != nodes["gamma"].id]
    assert engine.search(replace(q, min_rerank_score=0.5)) == []


def test_cutoff_zero_or_none_is_the_no_cutoff_ranking(stores):
    graph, vector, emb, _ = _scored(stores, {"alpha": 4.0, "beta": 1.0, "gamma": -6.0})
    engine = _engine(graph, vector, emb)
    q = SearchQuery(query="deploy helm", project_id="p", limit=10)
    ranked = lambda query: [r.memory.id for r in engine.search(query)]  # noqa: E731
    assert ranked(replace(q, min_rerank_score=0.0)) == ranked(q) == ranked(replace(q, min_rerank_score=None))


def test_cutoff_never_applies_without_a_reranker(stores):
    graph, vector, emb, _ = _scored(stores, {"alpha": 4.0, "beta": 1.0, "gamma": -6.0})
    q = SearchQuery(query="deploy helm", project_id="p", limit=10, min_rerank_score=0.99)
    assert len(_engine(graph, vector, emb, rerank=False).search(q)) == 3


def test_recall_uses_the_cutoff_and_explicit_search_does_not(tmp_path):
    from unittest.mock import MagicMock

    from collivind.config import CollivindConfig
    from collivind.engine.memory_manager import MemoryManager

    config = CollivindConfig()
    config.search.recall_min_score = 0.3
    manager = MemoryManager(MagicMock(), MagicMock(), MagicMock(), config)
    manager.search_engine = MagicMock()
    manager.search_engine.search_with_status.return_value = ([], False)
    assert "No relevant context" in manager.get_context("q", project_id="p", limit=20)
    assert manager.search_engine.search_with_status.call_args.args[0].min_rerank_score == 0.3
    assert SearchQuery(query="q").min_rerank_score is None  # MCP/CLI search build plain queries


@pytest.mark.parametrize("bad", [-0.1, 1.0, "0.2", True])
def test_recall_min_score_is_validated(bad):
    from collivind.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        SearchConfig(recall_min_score=bad)
