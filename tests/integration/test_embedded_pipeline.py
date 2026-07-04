"""Integration test for the embedded mode pipeline.

Runs the full add->search->invalidate flow using in-process Qdrant + SQLite.
No Docker required - this test can run in CI.
"""
import shutil
import tempfile

import pytest

from collivind.config import CollivindConfig
from collivind.engine.memory_manager import MemoryManager
from collivind.models import SearchQuery
from collivind.models.entity import EntityCreate, EntityType
from collivind.models.memory import MemoryCategory, MemoryCreate, MemorySource
from collivind.storage.graph_sqlite import SqliteGraphStore
from collivind.storage.qdrant_embedded import EmbeddedQdrantStore


class FakeEmbeddingProvider:
    """Deterministic embeddings for testing without sentence-transformers."""

    def __init__(self, dimension=384):
        self._dimension = dimension

    def embed(self, text: str):
        vec = [0.0] * self._dimension
        for i, ch in enumerate(text.encode("utf-8")[: self._dimension]):
            vec[i] = ch / 255.0
        norm = max(sum(v * v for v in vec) ** 0.5, 1e-9)
        return [v / norm for v in vec]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def health_check(self):
        return {"status": "ok", "message": "Fake embeddings"}

    @property
    def dimension(self):
        return self._dimension


@pytest.fixture
def embedded_manager():
    tmpdir = tempfile.mkdtemp()
    config = CollivindConfig(mode="embedded", data_dir=tmpdir)
    config.search.similarity_threshold = 0.1

    graph = SqliteGraphStore(data_dir=tmpdir)
    graph.initialize()

    qdrant = EmbeddedQdrantStore(
        data_dir=tmpdir,
        config=config.qdrant,
        dimension=384,
    )
    qdrant.initialize()

    embedder = FakeEmbeddingProvider(dimension=384)

    manager = MemoryManager(
        vector_store=qdrant,
        graph_store=graph,
        embedding_provider=embedder,
        config=config,
    )
    yield manager

    qdrant.close()
    graph.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_add_and_search(embedded_manager):
    manager = embedded_manager

    mem_create = MemoryCreate(
        content="FastAPI is our web framework for the API layer",
        summary="FastAPI choice",
        category=MemoryCategory.ARCHITECTURE,
        project_id="test-project",
        user_id="local",
        source=MemorySource.MANUAL,
        tags=["backend"],
    )
    entities = [EntityCreate(name="FastAPI", type=EntityType.LIBRARY)]
    result = manager.add_memory(mem_create, entities=entities)
    assert result is not None
    assert result.id

    results = manager.search(
        SearchQuery(
            query="what web framework do we use",
            project_id="test-project",
            limit=5,
        )
    )
    assert len(results) >= 1
    assert "FastAPI" in results[0].memory.content


def test_add_multiple_and_search(embedded_manager):
    manager = embedded_manager

    for content, summary, cat, ents, tags in [
        (
            "We use PostgreSQL for the main database",
            "PostgreSQL choice",
            MemoryCategory.DECISION,
            [EntityCreate(name="PostgreSQL", type=EntityType.SERVICE)],
            ["database"],
        ),
        (
            "Redis is used for caching and rate limiting",
            "Redis caching",
            MemoryCategory.ARCHITECTURE,
            [EntityCreate(name="Redis", type=EntityType.SERVICE)],
            ["cache"],
        ),
        (
            "The CI pipeline runs on GitHub Actions",
            "CI setup",
            MemoryCategory.FACT,
            [EntityCreate(name="GitHub Actions", type=EntityType.TOOL)],
            ["ci"],
        ),
    ]:
        manager.add_memory(
            MemoryCreate(
                content=content,
                summary=summary,
                category=cat,
                project_id="proj",
                user_id="local",
                source=MemorySource.MANUAL,
                tags=tags,
            ),
            entities=ents,
        )

    results = manager.search(
        SearchQuery(query="database", project_id="proj", limit=5)
    )
    assert len(results) >= 1


def test_invalidate_memory(embedded_manager):
    manager = embedded_manager

    old = manager.add_memory(
        MemoryCreate(
            content="We deploy to Heroku",
            summary="Heroku deployment",
            category=MemoryCategory.FACT,
            project_id="proj",
            user_id="local",
            source=MemorySource.MANUAL,
            tags=[],
        )
    )
    new = manager.add_memory(
        MemoryCreate(
            content="We migrated from Heroku to AWS ECS",
            summary="AWS ECS migration",
            category=MemoryCategory.DECISION,
            project_id="proj",
            user_id="local",
            source=MemorySource.MANUAL,
            tags=[],
        )
    )

    manager.invalidate(old.id, reason="Migrated to AWS", superseded_by=new.id)

    old_mem = manager.graph_store.get_memory(old.id)
    assert old_mem.valid_to is not None
    assert old_mem.superseded_by == new.id


def test_batch_add(embedded_manager):
    manager = embedded_manager

    memories_data = [
        {
            "content": "Python 3.11 is the minimum version",
            "summary": "Python version",
            "category": "fact",
            "project_id": "proj",
        },
        {
            "content": "We use uv for dependency management",
            "summary": "uv tooling",
            "category": "preference",
            "project_id": "proj",
        },
    ]
    results = manager.batch_add_memories(memories_data)
    assert len(results) == 2
    assert all(r for r in results)


def test_get_timeline(embedded_manager):
    manager = embedded_manager

    contents = [
        "We chose PostgreSQL as our primary relational database for data storage",
        "The deployment pipeline uses GitHub Actions for continuous integration",
        "Authentication is handled by JWT tokens with RSA256 signing algorithm",
    ]
    for i, content in enumerate(contents):
        manager.add_memory(
            MemoryCreate(
                content=content,
                summary=f"Distinct fact {i}",
                category=MemoryCategory.FACT,
                project_id="timeline-proj",
                user_id="local",
                source=MemorySource.MANUAL,
                tags=[],
            )
        )

    timeline = manager.get_timeline("timeline-proj", limit=10)
    assert len(timeline) >= 2
