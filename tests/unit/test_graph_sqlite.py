import shutil
import tempfile

import pytest

from collivind.models.entity import EntityCreate, EntityType
from collivind.models.memory import MemoryCategory, MemoryCreate, MemorySource
from collivind.models.relationship import RelationshipCreate, RelType
from collivind.storage.graph_sqlite import SqliteGraphStore


@pytest.fixture
def store():
    tmpdir = tempfile.mkdtemp()
    s = SqliteGraphStore(data_dir=tmpdir)
    s.initialize()
    yield s
    s.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_create_and_get_memory(store):
    data = MemoryCreate(
        content="Python uses GIL for thread safety",
        summary="Python GIL",
        category=MemoryCategory.FACT,
        project_id="test",
        user_id="local",
        source=MemorySource.MANUAL,
        tags=["python"],
    )
    node = store.create_memory(data)
    assert node.id is not None
    assert node.content == data.content

    fetched = store.get_memory(node.id)
    assert fetched is not None
    assert fetched.content == data.content
    assert fetched.category == MemoryCategory.FACT
    assert fetched.tags == ["python"]


def test_get_memory_not_found(store):
    assert store.get_memory("nonexistent") is None


def test_delete_memory(store):
    data = MemoryCreate(
        content="temp", summary="temp", category=MemoryCategory.FACT,
        project_id="test", user_id="local", source=MemorySource.MANUAL, tags=[],
    )
    node = store.create_memory(data)
    store.delete_memory(node.id)
    assert store.get_memory(node.id) is None


def test_create_entity_idempotent(store):
    data = EntityCreate(name="FastAPI", type=EntityType.LIBRARY, properties={"version": "0.100"})
    e1 = store.create_entity(data)
    e2 = store.create_entity(data)
    assert e1.id == e2.id


def test_get_entity(store):
    data = EntityCreate(name="FastAPI", type=EntityType.LIBRARY, properties={"version": "0.100"})
    store.create_entity(data)

    fetched = store.get_entity("FastAPI")

    assert fetched is not None
    assert fetched.id == "fastapi"
    assert fetched.name == "FastAPI"
    assert fetched.properties == {"version": "0.100"}


def test_update_memory_serializes_tags(store):
    node = store.create_memory(MemoryCreate(
        content="tagged", summary="Tagged",
        category=MemoryCategory.FACT, project_id="test",
        user_id="local", source=MemorySource.MANUAL, tags=["old"],
    ))

    updated = store.update_memory(node.id, tags=["old", "new"])

    assert updated.tags == ["old", "new"]


def test_create_relationship_and_get_neighbors(store):
    mem = store.create_memory(MemoryCreate(
        content="We use FastAPI", summary="FastAPI usage",
        category=MemoryCategory.ARCHITECTURE, project_id="test",
        user_id="local", source=MemorySource.MANUAL, tags=[],
    ))
    entity = store.create_entity(EntityCreate(name="FastAPI", type=EntityType.LIBRARY))

    rel = store.create_relationship(RelationshipCreate(
        source_id=mem.id, target_id=entity.id, type=RelType.MENTIONS,
    ))
    assert rel.id is not None

    neighbors = store.get_neighbors(mem.id, rel_types=[], direction="OUT")
    assert len(neighbors) == 1
    assert neighbors[0]["id"] == entity.id


def test_find_related_memories(store):
    mem = store.create_memory(MemoryCreate(
        content="FastAPI is the web framework", summary="FastAPI choice",
        category=MemoryCategory.DECISION, project_id="test",
        user_id="local", source=MemorySource.MANUAL, tags=[],
    ))
    entity = store.create_entity(EntityCreate(name="FastAPI", type=EntityType.LIBRARY))
    store.create_relationship(RelationshipCreate(
        source_id=mem.id, target_id=entity.id, type=RelType.ABOUT,
    ))

    related = store.find_related_memories("FastAPI")
    assert len(related) == 1
    assert related[0].id == mem.id


def test_timeline(store):
    for i in range(3):
        store.create_memory(MemoryCreate(
            content=f"Memory {i}", summary=f"Mem {i}",
            category=MemoryCategory.FACT, project_id="proj1",
            user_id="local", source=MemorySource.MANUAL, tags=[],
        ))
    store.create_memory(MemoryCreate(
        content="Other project", summary="Other",
        category=MemoryCategory.FACT, project_id="proj2",
        user_id="local", source=MemorySource.MANUAL, tags=[],
    ))

    timeline = store.get_timeline("proj1")
    assert len(timeline) == 3


def test_invalidate_memory(store):
    m1 = store.create_memory(MemoryCreate(
        content="Old fact", summary="Old",
        category=MemoryCategory.FACT, project_id="test",
        user_id="local", source=MemorySource.MANUAL, tags=[],
    ))
    m2 = store.create_memory(MemoryCreate(
        content="New fact", summary="New",
        category=MemoryCategory.FACT, project_id="test",
        user_id="local", source=MemorySource.MANUAL, tags=[],
    ))

    store.invalidate_memory(m1.id, superseded_by=m2.id, reason="updated")

    old = store.get_memory(m1.id)
    assert old.valid_to is not None
    assert old.superseded_by == m2.id


def test_clear_all(store):
    store.create_memory(MemoryCreate(
        content="Will be cleared", summary="Clear me",
        category=MemoryCategory.FACT, project_id="test",
        user_id="local", source=MemorySource.MANUAL, tags=[],
    ))
    store.create_entity(EntityCreate(name="SomeLib", type=EntityType.LIBRARY))

    store.clear_all()

    health = store.health_check()
    assert health["status"] == "ok"
    assert "0 memories" in health["message"]


def test_health_check(store):
    health = store.health_check()
    assert health["status"] == "ok"
