"""Unit tests for the SQLite-backed vector store.

Real SQLite against tmp dirs — no mocks except qdrant-client in the
migration/lock tests, and QdrantLocal as the parity oracle.
"""

import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

from collivind.config import QdrantConfig
from collivind.exceptions import CollivindError
from collivind.storage.vector_sqlite import SqliteVectorStore

DIM = 8

LOCK_ERROR = RuntimeError("Storage folder /x/qdrant_data is already accessed by another instance of Qdrant client.")


def unit(i: int, dim: int = DIM) -> list[float]:
    vec = [0.0] * dim
    vec[i] = 1.0
    return vec


def make_store(tmp_path, dimension: int = DIM) -> SqliteVectorStore:
    return SqliteVectorStore(data_dir=str(tmp_path), config=QdrantConfig(), dimension=dimension)


@pytest.fixture
def store(tmp_path):
    s = make_store(tmp_path)
    yield s
    s.close()


# --- CRUD and search semantics ---


def test_upsert_and_search_roundtrip(store):
    store.upsert("m1", unit(0), {"project_id": "proj", "kind": "fact"})
    results = store.search(unit(0))
    assert len(results) == 1
    assert results[0]["id"] == "m1"
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[0]["payload"] == {"project_id": "proj", "kind": "fact"}


def test_upsert_replaces_existing_id(store):
    store.upsert("m1", unit(0), {"v": 1})
    store.upsert("m1", unit(1), {"v": 2})
    results = store.search(unit(1))
    assert [r["id"] for r in results] == ["m1"]
    assert results[0]["payload"] == {"v": 2}
    assert store.search(unit(0), threshold=0.5) == []


def test_integer_ids_are_coerced_to_str(store):
    store.upsert(7, unit(0), {})
    results = store.search(unit(0))
    assert results[0]["id"] == "7"
    store.delete(7)
    assert store.search(unit(0)) == []


def test_search_empty_store_returns_empty(store):
    assert store.search(unit(0)) == []


def test_filters_none_and_empty_mean_no_filter(store):
    store.upsert("m1", unit(0), {"project_id": "proj"})
    assert store.search(unit(0), filters=None) == store.search(unit(0), filters={})


def test_filter_matches_flat_value(store):
    store.upsert("m1", unit(0), {"project_id": "a"})
    store.upsert("m2", unit(0), {"project_id": "b"})
    results = store.search(unit(0), filters={"project_id": "a"})
    assert [r["id"] for r in results] == ["m1"]


def test_filter_matches_inside_list_payload(store):
    """Qdrant MatchValue any-matches list payload values."""
    store.upsert("m1", unit(0), {"tags": ["python", "sqlite"]})
    assert [r["id"] for r in store.search(unit(0), filters={"tags": "python"})] == ["m1"]
    assert store.search(unit(0), filters={"tags": "rust"}) == []


def test_threshold_is_inclusive(store):
    """score >= threshold is kept — dedup's 0.92/0.98 band sits on this."""
    store.upsert("exact", unit(0), {})
    store.upsert("orthogonal", unit(1), {})
    # unit vectors: scores are exactly 1.0 and 0.0 in float32
    assert [r["id"] for r in store.search(unit(0), threshold=1.0)] == ["exact"]
    at_zero = store.search(unit(0), threshold=0.0)
    assert {r["id"] for r in at_zero} == {"exact", "orthogonal"}


def test_upsert_wrong_dimension_raises(store):
    with pytest.raises(CollivindError, match="dimension"):
        store.upsert("m1", [1.0] * (DIM + 1), {})


def test_zero_vector_scores_zero_not_nan(store):
    store.upsert("zero", [0.0] * DIM, {})
    store.upsert("one", unit(0), {})
    results = store.search(unit(0), threshold=0.0)
    scores = {r["id"]: r["score"] for r in results}
    assert scores["zero"] == pytest.approx(0.0)
    results = store.search([0.0] * DIM, threshold=0.0)
    assert all(r["score"] == pytest.approx(0.0) for r in results)


def test_delete_removes_vector(store):
    store.upsert("m1", unit(0), {})
    store.delete("m1")
    assert store.search(unit(0)) == []


def test_delete_collection_then_reinitialize(store):
    store.upsert("m1", unit(0), {})
    store.delete_collection()
    assert store.search(unit(0)) == []
    store.initialize()
    store.upsert("m2", unit(1), {})
    assert [r["id"] for r in store.search(unit(1))] == ["m2"]


def test_health_check_reports_count(store):
    store.upsert("m1", unit(0), {})
    health = store.health_check()
    assert health["status"] == "ok"
    assert "1" in health["message"]


def test_health_check_after_close_reports_error(store):
    store.close()
    assert store.health_check()["status"] == "error"


# --- Cross-instance freshness (read-your-writes) ---


def test_write_in_one_instance_visible_in_another(tmp_path):
    a, b = make_store(tmp_path), make_store(tmp_path)
    try:
        assert b.search(unit(0)) == []  # warm b's cache while empty
        a.upsert("from-a", unit(0), {"who": "a"})
        assert [r["id"] for r in b.search(unit(0))] == ["from-a"]
        b.upsert("from-b", unit(1), {"who": "b"})
        assert [r["id"] for r in a.search(unit(1))] == ["from-b"]
    finally:
        a.close()
        b.close()


def test_own_write_visible_with_warm_cache(store):
    store.upsert("m1", unit(0), {})
    assert len(store.search(unit(0))) == 1  # warm the cache
    store.upsert("m2", unit(1), {})
    assert [r["id"] for r in store.search(unit(1))] == ["m2"]


def test_own_delete_visible_with_warm_cache(store):
    store.upsert("m1", unit(0), {})
    assert len(store.search(unit(0))) == 1
    store.delete("m1")
    assert store.search(unit(0)) == []


def test_cross_instance_delete_visible(tmp_path):
    a, b = make_store(tmp_path), make_store(tmp_path)
    try:
        a.upsert("m1", unit(0), {})
        assert len(b.search(unit(0))) == 1
        a.delete("m1")
        assert b.search(unit(0)) == []
    finally:
        a.close()
        b.close()


# --- Score parity with QdrantLocal (the previous engine) ---


def test_score_parity_with_qdrant_local(tmp_path):
    """Same ordering and scores (±1e-5) as qdrant-client local mode.

    The executable proof that retrieval quality (LongMemEval R@5) and the
    dedup 0.92/0.98 threshold band cannot move.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    dim, n_vectors, n_queries = 384, 40, 3
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(n_vectors, dim)).astype(np.float32)
    queries = rng.normal(size=(n_queries, dim)).astype(np.float32)

    qdrant = QdrantClient(":memory:")
    qdrant.create_collection("parity", vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE))
    qdrant.upsert(
        "parity",
        points=[qmodels.PointStruct(id=i, vector=v.tolist(), payload={}) for i, v in enumerate(vectors)],
    )

    store = SqliteVectorStore(data_dir=str(tmp_path), config=QdrantConfig(collection_name="parity"), dimension=dim)
    for i, v in enumerate(vectors):
        store.upsert(i, v.tolist(), {})

    for q in queries:
        expected = qdrant.query_points("parity", query=q.tolist(), limit=10, score_threshold=-1.0).points
        got = store.search(q.tolist(), limit=10, threshold=-1.0)
        assert [r["id"] for r in got] == [str(p.id) for p in expected]
        for r, p in zip(got, expected):
            assert r["score"] == pytest.approx(p.score, abs=1e-5)

    store.close()
    qdrant.close()


# --- Migration from legacy embedded Qdrant data ---


def _make_legacy_qdrant(tmp_path, collections: dict, dim: int = DIM) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    client = QdrantClient(path=str(tmp_path / "qdrant_data"))
    for name, points in collections.items():
        client.create_collection(name, vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE))
        client.upsert(
            name,
            points=[qmodels.PointStruct(id=i, vector=v, payload=p) for i, (v, p) in enumerate(points)],
        )
    client.close()


def test_migrates_legacy_qdrant_data(tmp_path):
    _make_legacy_qdrant(tmp_path, {"collivind_memories": [(unit(0), {"project_id": "proj"}), (unit(1), {})]})

    store = make_store(tmp_path)
    results = store.search(unit(0))
    assert results[0]["id"] == "0"
    assert results[0]["payload"] == {"project_id": "proj"}
    assert len(store.search(unit(1))) >= 1
    store.close()

    # source data is never mutated: still present for rollback
    assert any((tmp_path / "qdrant_data").iterdir())


def test_second_construction_skips_migration(tmp_path):
    _make_legacy_qdrant(tmp_path, {"collivind_memories": [(unit(0), {})]})
    make_store(tmp_path).close()

    with patch("qdrant_client.QdrantClient", side_effect=AssertionError("must not re-open legacy store")):
        store = make_store(tmp_path)
        assert len(store.search(unit(0))) == 1
        store.close()


def test_migrates_all_collections_not_just_configured(tmp_path):
    _make_legacy_qdrant(
        tmp_path,
        {"collivind_memories": [(unit(0), {})], "renamed_collection": [(unit(1), {"old": True})]},
    )
    store = make_store(tmp_path)
    store.close()

    conn = sqlite3.connect(str(tmp_path / "collivind_vectors.db"))
    collections = {row[0] for row in conn.execute("SELECT DISTINCT collection FROM vectors")}
    conn.close()
    assert collections == {"collivind_memories", "renamed_collection"}


def test_fresh_install_never_imports_qdrant(tmp_path):
    with patch("qdrant_client.QdrantClient", side_effect=AssertionError("fresh install must not touch qdrant")):
        store = make_store(tmp_path)
        store.upsert("m1", unit(0), {})
        assert len(store.search(unit(0))) == 1
        store.close()

    # flag is set: later constructions skip migration even if qdrant_data appears
    (tmp_path / "qdrant_data").mkdir()
    with patch("qdrant_client.QdrantClient", side_effect=AssertionError("migration already flagged done")):
        make_store(tmp_path).close()


def test_locked_legacy_store_raises_diagnosable_error(tmp_path):
    (tmp_path / "qdrant_data").mkdir()
    with (
        patch("qdrant_client.QdrantClient", side_effect=LOCK_ERROR),
        patch("collivind.storage.vector_sqlite.time.sleep"),
        patch("collivind.storage.lockinfo.lock_holders", return_value=["12345"]),
    ):
        with pytest.raises(CollivindError) as err:
            make_store(tmp_path)
    message = str(err.value)
    assert "held by PID 12345" in message
    assert "kill 12345" in message
    assert "pre-upgrade" in message


def test_locked_legacy_store_without_lsof_still_explains(tmp_path):
    (tmp_path / "qdrant_data").mkdir()
    with (
        patch("qdrant_client.QdrantClient", side_effect=LOCK_ERROR),
        patch("collivind.storage.vector_sqlite.time.sleep"),
        patch("collivind.storage.lockinfo.lock_holders", return_value=[]),
    ):
        with pytest.raises(CollivindError) as err:
            make_store(tmp_path)
    assert "lsof" in str(err.value)


def test_transient_legacy_lock_recovers_on_retry(tmp_path):
    _make_legacy_qdrant(tmp_path, {"collivind_memories": [(unit(0), {})]})
    from qdrant_client import QdrantClient as RealClient

    calls = iter([LOCK_ERROR, None])

    def flaky(*args, **kwargs):
        effect = next(calls)
        if effect is not None:
            raise effect
        return RealClient(*args, **kwargs)

    with (
        patch("qdrant_client.QdrantClient", side_effect=flaky),
        patch("collivind.storage.vector_sqlite.time.sleep") as sleep,
    ):
        store = make_store(tmp_path)
    assert len(store.search(unit(0))) == 1
    sleep.assert_called_once()
    store.close()


def test_non_lock_migration_error_raises_immediately(tmp_path):
    (tmp_path / "qdrant_data").mkdir()
    with patch("qdrant_client.QdrantClient", side_effect=ValueError("disk corrupt")):
        with pytest.raises(ValueError):
            make_store(tmp_path)
