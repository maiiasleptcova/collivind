"""Issue #29: Qdrant's probe was satisfied that the SERVER answers, not that
our collection exists — so create_all_backends reported the mode healthy and
every subsequent search 404'd. Same defect class as the SQLite graph store
fixed in 0.5.2."""

from unittest.mock import MagicMock, patch

from collivind.config import QdrantConfig


def _collection(name):
    """`MagicMock(name=...)` names the mock itself; the attribute must be set
    afterwards or `.name` comes back as a child mock."""
    c = MagicMock()
    c.name = name
    return c


def _store(existing_collections):
    with patch("collivind.storage.qdrant_store.QdrantClient") as client_cls:
        client = client_cls.return_value
        client.get_collections.return_value = MagicMock(collections=[_collection(n) for n in existing_collections])
        from collivind.storage.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore(QdrantConfig(collection_name="collivind_memories"), dimension=384)
        return store, client


def test_collection_is_created_on_construction():
    """Any entry point must work against a running-but-uninitialized Qdrant."""
    store, client = _store(existing_collections=[])
    client.create_collection.assert_called_once()
    assert client.create_collection.call_args.kwargs["collection_name"] == "collivind_memories"


def test_existing_collection_is_not_recreated():
    """Re-opening must never touch stored vectors."""
    store, client = _store(existing_collections=["collivind_memories"])
    client.create_collection.assert_not_called()


def test_health_check_reports_a_missing_collection():
    """Reporting 'ok' while the collection is absent is what hid #29."""
    store, client = _store(existing_collections=["collivind_memories"])
    client.get_collections.return_value = MagicMock(collections=[])
    result = store.health_check()
    assert result["status"] == "error"
    assert "collivind_memories" in result["message"]


def test_health_check_ok_when_collection_present():
    store, client = _store(existing_collections=["collivind_memories"])
    assert store.health_check()["status"] == "ok"


def test_construction_survives_an_unreachable_server():
    """Construction must not explode when Qdrant is simply down — the mode
    probe in create_all_backends is what should report that."""
    with patch("collivind.storage.qdrant_store.QdrantClient") as client_cls:
        client_cls.return_value.get_collections.side_effect = ConnectionError("refused")
        from collivind.storage.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore(QdrantConfig(), dimension=384)
        assert store.health_check()["status"] == "error"
