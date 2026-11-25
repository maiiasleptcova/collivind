"""Neo4j full-text (keyword) channel. Needs a THROWAWAY Neo4j 5: it clears the database.

    docker run -d --rm --name cv-ft -p 17687:7687 -e NEO4J_AUTH=neo4j/testpass123 neo4j:5.26-community
    COLLIVIND_TEST_NEO4J_URI=bolt://localhost:17687 COLLIVIND_TEST_NEO4J_PASSWORD=testpass123 \\
        uv run pytest tests/integration/test_neo4j_lexical.py

Skipped unless COLLIVIND_TEST_NEO4J_URI is set, so it can never hit the default (live) store.
"""

import os
import time

import pytest

from collivind.config import Neo4jConfig
from collivind.models import MemoryCategory, MemoryCreate
from collivind.storage.neo4j_store import Neo4jGraphStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("COLLIVIND_TEST_NEO4J_URI"), reason="set COLLIVIND_TEST_NEO4J_URI"),
]


@pytest.fixture
def store():
    s = Neo4jGraphStore(
        Neo4jConfig(uri=os.environ["COLLIVIND_TEST_NEO4J_URI"], password=os.environ["COLLIVIND_TEST_NEO4J_PASSWORD"])
    )
    s.clear_all()
    s.driver.execute_query("DROP INDEX memory_content_fulltext IF EXISTS")
    yield s
    s.close()


def _add(store, content, project="p"):
    return store.create_memory(
        MemoryCreate(content=content, summary="s", category=MemoryCategory.FACT, project_id=project)
    )


def _search(store, query, project=None):
    for _ in range(20):  # a just-created index populates in the background
        hits = store.lexical_search(query, 10, project_id=project)
        if hits:
            return [m.content for m, _ in hits]
        time.sleep(0.25)
    return []


def test_pre_existing_memories_are_indexed_and_filtered(store):
    old = _add(store, "deploy uses kubectl apply")  # written before any index exists
    new = _add(store, "deploy uses helm upgrade")
    _add(store, "deploy elsewhere", project="other")
    store.invalidate_memory(old.id, new.id, "superseded_by_update")
    assert _search(store, "how do we deploy? (helm) AND NOT", project="p") == ["deploy uses helm upgrade"]
    assert sorted(_search(store, "deploy")) == ["deploy elsewhere", "deploy uses helm upgrade"]


def test_forgotten_memory_is_gone_and_bad_queries_are_empty(store):
    m = _add(store, "redis cache")
    assert _search(store, "redis") == ["redis cache"]
    store.delete_memory(m.id)
    assert store.lexical_search("redis", 10) == []
    assert store.lexical_search("??? ((( ", 10) == []
