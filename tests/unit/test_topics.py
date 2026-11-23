"""Search results -> one topic entity: the explorer's way to name a subject the extractor missed."""

from unittest.mock import MagicMock

import pytest

from collivind.config import CollivindConfig
from collivind.engine.memory_manager import MemoryManager
from collivind.models import MemoryCreate, RelType
from collivind.models.entity import EntityType
from collivind.models.memory import MemoryCategory
from collivind.storage.graph_sqlite import SqliteGraphStore
from collivind.web.api import MAX_TOPIC_MEMORIES, MemoryAPI, route


@pytest.fixture
def manager(tmp_path):
    vectors, embedder = MagicMock(), MagicMock()
    vectors.search.return_value = []
    embedder.embed.return_value = [0.1]
    return MemoryManager(vectors, SqliteGraphStore(str(tmp_path / "graph.db")), embedder, CollivindConfig())


def _add(mgr, text):
    return mgr.add_memory(MemoryCreate(content=text, summary=text, category=MemoryCategory.FACT, project_id="p")).id


def _about(mgr, entity_id):
    return sorted(e.source_id for e in mgr.graph_store.iter_relationships()
                  if e.type == RelType.ABOUT and e.target_id == entity_id)


def test_links_every_memory_and_is_idempotent(manager):
    ids = [_add(manager, "Harley scored 2 of 42"), _add(manager, "Harley hides emails behind Cloudflare")]
    first = manager.link_topic("Harley Street Skin", EntityType.CONCEPT, ids)
    again = manager.link_topic("Harley Street Skin", EntityType.CONCEPT, ids + ids[:1])
    assert first["entity_id"] == again["entity_id"] == "harley_street_skin"
    assert _about(manager, "harley_street_skin") == sorted(ids)  # no duplicate edges
    entity = manager.get_entity("Harley Street Skin")
    assert {m["id"] for m in entity["related_memories"]} == set(ids)


def test_unknown_memory_writes_nothing(manager):
    good = _add(manager, "a fact")
    with pytest.raises(LookupError):
        manager.link_topic("Topic", EntityType.CONCEPT, [good, "nope"])
    assert manager.graph_store.get_entity("Topic") is None


def _call(manager, body):
    return route(MemoryAPI(manager, mode="embedded"), "POST", "/api/topics", {}, body)


def test_route_creates_the_topic(manager):
    ids = [_add(manager, "x")]
    status, payload = _call(manager, {"name": "Client site", "type": "project", "memory_ids": ids})
    assert status == 201 and payload["linked"] == 1 and payload["type"] == "project"


@pytest.mark.parametrize(
    "body",
    [
        {"name": "", "memory_ids": ["a"]},
        {"name": "t", "memory_ids": []},
        {"name": "t", "memory_ids": "a"},
        {"name": "t", "memory_ids": ["a"] * (MAX_TOPIC_MEMORIES + 1)},
        {"name": "t", "type": "planet", "memory_ids": ["a"]},
    ],
)
def test_route_rejects_bad_bodies(manager, body):
    assert _call(manager, body)[0] == 400


def test_route_404s_an_unknown_memory(manager):
    assert _call(manager, {"name": "t", "memory_ids": ["missing"]})[0] == 404
