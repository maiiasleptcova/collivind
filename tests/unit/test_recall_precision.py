"""Recall precision: the relative window and the no-confident-match floor (recall path only).

Real SQLite stores with a deterministic embedder and a scripted reranker.
"""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from test_lexical_search import WordEmbedder, _engine, _store, stores  # noqa: F401  (stores is a fixture)

from collivind.config import CollivindConfig, SearchConfig
from collivind.engine.memory_manager import MemoryManager
from collivind.exceptions import ConfigurationError
from collivind.models import SearchQuery


def _scored(stores, logits_by_word):  # noqa: F811
    graph, vector = stores
    emb = WordEmbedder()
    nodes = {w: _store(graph, vector, emb, f"deploy helm {w}") for w in logits_by_word}
    emb.rerank = lambda q, texts: [next(v for w, v in logits_by_word.items() if w in t) for t in texts]
    return graph, vector, emb, nodes


def _ids(results):
    return [r.memory.id for r in results]


Q = SearchQuery(query="deploy helm", project_id="p", limit=10, min_rerank_score=0.00002)


def test_window_keeps_results_close_to_the_best_match(stores):  # noqa: F811
    graph, vector, emb, nodes = _scored(stores, {"alpha": 4.0, "beta": 1.0, "gamma": -8.0})
    engine = _engine(graph, vector, emb)
    assert len(engine.search(Q)) == 3  # gamma passes the absolute cutoff (sigmoid(-8) = 3e-4)
    kept = engine.search(replace(Q, rerank_window=6.0))
    assert _ids(kept) == [nodes["alpha"].id, nodes["beta"].id]  # gamma is 12 logits below the best


def test_floor_applies_only_when_no_candidate_is_confident(stores):  # noqa: F811
    graph, vector, emb, nodes = _scored(stores, {"alpha": -1.0, "beta": -4.0, "gamma": -9.0})
    engine = _engine(graph, vector, emb)
    first_stage = _ids(_engine(graph, vector, emb, rerank=False).search(Q))
    floored = engine.search(replace(Q, fallback_min_score=0.01))  # sigmoid(-1)=0.27, (-4)=0.018, (-9)=1e-4
    assert _ids(floored) == [i for i in first_stage if i != nodes["gamma"].id]  # first-stage order kept
    assert engine.search(replace(Q, fallback_min_score=0.5)) == []  # nothing relevant: empty recall


def test_floor_does_not_touch_a_confident_ranking(stores):  # noqa: F811
    graph, vector, emb, _ = _scored(stores, {"alpha": 4.0, "beta": -3.0, "gamma": -4.0})
    engine = _engine(graph, vector, emb)
    assert len(engine.search(replace(Q, fallback_min_score=0.5))) == 3


@pytest.mark.parametrize(
    "logits", [{"alpha": 4.0, "beta": 1.0, "gamma": -8.0}, {"alpha": -1.0, "beta": -4.0, "gamma": -9.0}]
)
def test_off_reproduces_todays_recall_exactly(stores, logits):  # noqa: F811
    graph, vector, emb, _ = _scored(stores, logits)
    engine = _engine(graph, vector, emb)
    today = [(r.memory.id, round(r.score, 6), r.rerank_score) for r in engine.search(Q)]
    for off in (replace(Q, rerank_window=0.0, fallback_min_score=0.0), replace(Q, rerank_window=None)):
        assert [(r.memory.id, round(r.score, 6), r.rerank_score) for r in engine.search(off)] == today


def test_never_applies_without_a_reranker(stores):  # noqa: F811
    graph, vector, emb, _ = _scored(stores, {"alpha": -1.0, "beta": -4.0, "gamma": -9.0})
    q = replace(Q, rerank_window=0.5, fallback_min_score=0.9)
    assert len(_engine(graph, vector, emb, rerank=False).search(q)) == 3


def _manager(tmp_path, **search):
    from collivind.storage.graph_sqlite import SqliteGraphStore
    from collivind.storage.vector_sqlite import SqliteVectorStore

    emb = WordEmbedder()
    config = CollivindConfig(data_dir=str(tmp_path), mode="embedded", search=replace(SearchConfig(), **search))
    graph = SqliteGraphStore(str(tmp_path))
    vector = SqliteVectorStore(str(tmp_path), config.qdrant, len(WordEmbedder.VOCAB))
    return MemoryManager(vector, graph, emb, config), graph, vector, emb


def test_get_context_uses_the_recall_settings_and_search_does_not(tmp_path):
    manager, graph, vector, emb = _manager(tmp_path, recall_window=12.0, recall_fallback_min_score=0.001)
    for w in ("alpha", "beta", "gamma"):
        _store(graph, vector, emb, f"deploy helm {w}")
    emb.rerank = lambda q, texts: [-2.0 if "alpha" in t else -9.0 for t in texts]  # nothing confident
    ctx = manager.get_context("deploy helm", project_id="p", limit=10)
    assert "alpha" in ctx and "beta" not in ctx  # floor 0.001 keeps sigmoid(-2) = 0.12 only
    assert len(manager.search(SearchQuery(query="deploy helm", project_id="p"))) == 3  # explicit search: all
    emb.rerank = lambda q, texts: [-9.0] * len(texts)
    assert ctx.startswith("--- Collivind Context ---")
    assert manager.get_context("deploy helm", project_id="p") == "No relevant context found in Collivind."
    graph.close()
    vector.close()


def test_get_context_off_switch_reproduces_todays_output(tmp_path):
    manager, graph, vector, emb = _manager(tmp_path)  # the defaults are the off switch
    for w in ("alpha", "beta", "gamma"):
        _store(graph, vector, emb, f"deploy helm {w}")
    emb.rerank = lambda q, texts: [-2.0 if "alpha" in t else -9.0 for t in texts]
    ctx = manager.get_context("deploy helm", project_id="p", limit=10)
    assert all(w in ctx for w in ("alpha", "beta", "gamma"))  # only the 0.00002 cutoff, as before
    graph.close()
    vector.close()


def test_get_context_with_nothing_that_fits_the_budget_is_empty(tmp_path):
    manager, graph, vector, emb = _manager(tmp_path)
    _store(graph, vector, emb, "deploy helm " + "x" * 2000)
    emb.rerank = lambda q, texts: [5.0] * len(texts)
    assert manager.get_context("deploy helm", project_id="p", max_tokens=100) == (
        "No relevant context found in Collivind."
    )
    graph.close()
    vector.close()


def test_get_context_passes_settings_into_the_query():
    config = CollivindConfig()
    config.search.recall_window = 7.0
    config.search.recall_fallback_min_score = 0.002
    manager = MemoryManager(MagicMock(), MagicMock(), MagicMock(), config)
    manager.search_engine = MagicMock()
    manager.search_engine.search_with_status.return_value = ([], False)
    manager.get_context("q", project_id="p")
    query = manager.search_engine.search_with_status.call_args.args[0]
    assert (query.rerank_window, query.fallback_min_score) == (7.0, 0.002)
    assert SearchQuery(query="q").rerank_window is None and SearchQuery(query="q").fallback_min_score is None


@pytest.mark.parametrize("field,bad", [("recall_window", -1.0), ("recall_window", 51.0), ("recall_window", True),
                                       ("recall_fallback_min_score", 1.0), ("recall_fallback_min_score", -0.1),
                                       ("recall_fallback_min_score", "0.001")])  # fmt: skip
def test_recall_settings_are_validated(field, bad):
    with pytest.raises(ConfigurationError):
        SearchConfig(**{field: bad})


def test_defaults():
    config = CollivindConfig()
    assert (config.search.recall_window, config.search.recall_fallback_min_score) == (0.0, 0.0)  # off
    assert config.hooks.recall_limit == 20
