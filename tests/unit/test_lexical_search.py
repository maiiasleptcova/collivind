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
    results = _engine(graph, vector, emb).search(SearchQuery(query="deploy", project_id="p"))
    by_id = {r.memory.id: r for r in results}
    assert linked.id in by_id and by_id[linked.id].graph_score > 0 and by_id[linked.id].rerank_score is not None
