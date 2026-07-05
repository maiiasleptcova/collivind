import uuid

import pytest

from collivind.config import QdrantConfig
from collivind.storage.qdrant_store import QdrantVectorStore


@pytest.fixture
def store():
    config = QdrantConfig(host="localhost", port=6333, collection_name="test_memories")
    s = QdrantVectorStore(config, dimension=4)  # small dimension for testing
    yield s
    # cleanup not strictly needed if we don't start container
    s.close()


@pytest.mark.skip(reason="Requires running Qdrant container")
def test_qdrant_lifecycle(store):
    store.initialize()

    # Check health
    health = store.health_check()
    assert health["status"] == "ok"
    assert "test_memories" in health["collections"]

    # Upsert
    test_id = str(uuid.uuid4())
    vector = [0.1, 0.2, 0.3, 0.4]
    payload = {"text": "test", "category": "fact"}
    store.upsert(test_id, vector, payload)

    # Search
    results = store.search(vector, limit=1)
    assert len(results) == 1
    assert results[0]["id"] == test_id
    assert results[0]["payload"]["text"] == "test"

    # Search with filter
    results = store.search(vector, limit=1, filters={"category": "fact"})
    assert len(results) == 1

    results = store.search(vector, limit=1, filters={"category": "wrong"})
    assert len(results) == 0

    # Delete
    store.delete(test_id)
    results = store.search(vector, limit=1)
    assert len(results) == 0
