"""An entity's `aliases` property routes every name it lists to that entity."""

from unittest.mock import MagicMock

import pytest

from collivind.config import CollivindConfig
from collivind.engine.memory_manager import MemoryManager
from collivind.models import EntityCreate
from collivind.models.entity import EntityType
from collivind.storage.graph_sqlite import SqliteGraphStore


@pytest.fixture
def manager(tmp_path):
    vectors, embedder = MagicMock(), MagicMock()
    vectors.search.return_value = []
    embedder.embed.return_value = [0.1]
    graph = SqliteGraphStore(str(tmp_path / "graph.db"))
    graph.create_entity(
        EntityCreate(
            name="Cybergine", type=EntityType.PROJECT, properties={"aliases": ["v5_flask", "Cybergine v5_flask"]}
        )
    )
    return MemoryManager(vectors, graph, embedder, CollivindConfig())


def _save(manager, entity, relationships=()):
    manager.batch_add_memories(
        [
            {
                "content": f"{entity} runs its tests in cytm",
                "summary": "tests run in cytm",
                "category": "fact",
                "entities": [{"name": entity, "type": "project"}],
                "relationships": list(relationships),
            }
        ]
    )


def _entity_ids(manager):
    return {e.id for e in manager.graph_store.iter_entities()}


def test_a_save_naming_an_alias_lands_on_the_entity(manager):
    _save(manager, "cybergine-v5-flask", [{"source": "v5_flask", "type": "USES", "target": "cytm", "note": "tests"}])
    assert "v5_flask" not in _entity_ids(manager) and "cybergine_v5_flask" not in _entity_ids(manager)
    found = manager.get_entity("v5_flask")
    assert found["entity"]["id"] == "cybergine"
    assert len(found["related_memories"]) == 1
    assert ("uses", "cytm") in {(r["phrase"], r["other_id"]) for r in found["relations"]}


def test_an_alias_wins_over_a_stray_entity_with_that_id(manager):
    manager.graph_store.create_entity(EntityCreate(name="v5_flask", type=EntityType.PROJECT))
    _save(manager, "v5_flask")
    assert manager.get_entity("v5_flask")["entity"]["id"] == "cybergine"
    assert manager.graph_store.find_related_memories("v5_flask") == []


def test_saving_the_canonical_name_keeps_its_aliases(manager):
    _save(manager, "Cybergine")
    _save(manager, "Cybergine")
    assert manager.graph_store.get_entity("Cybergine").properties["aliases"] == ["v5_flask", "Cybergine v5_flask"]


def test_topics_and_unknown_names_are_untouched(manager):
    _save(manager, "MTBSloth")
    assert manager.get_entity("MTBSloth")["entity"]["id"] == "mtbsloth"
    assert manager.get_entity("nothing here") is None
