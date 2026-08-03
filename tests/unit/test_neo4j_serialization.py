import json
from unittest.mock import MagicMock, patch

from collivind.config import Neo4jConfig
from collivind.models import EntityCreate
from collivind.models.entity import EntityType


def _make_store():
    with patch("collivind.storage.neo4j_store.GraphDatabase") as gdb:
        from collivind.storage.neo4j_store import Neo4jGraphStore

        store = Neo4jGraphStore(Neo4jConfig())
        session = MagicMock()
        gdb.driver.return_value.session.return_value.__enter__.return_value = session
        return store, session


def test_create_entity_serializes_properties_as_json():
    store, session = _make_store()
    props = {"stars": 42, "nested": {"active": True, "note": None}}

    store.create_entity(EntityCreate(name="FastAPI", type=EntityType.LIBRARY, properties=props))

    sent = session.run.call_args.kwargs["props"]
    assert json.loads(sent) == props


def test_parse_properties_reads_json_and_legacy_repr():
    store, _ = _make_store()
    assert store._parse_properties('{"a": 1}') == {"a": 1}
    # legacy rows written by the old str(dict) serializer
    assert store._parse_properties("{'a': 1, 'b': True}") == {"a": 1, "b": True}
    assert store._parse_properties("garbage") == {}
    assert store._parse_properties(None) == {}


def test_update_memory_stamps_updated_at():
    """The SQLite store stamps this on every update. Without parity, an edit in
    docker mode leaves `updated_at` frozen at creation — and the web UI now
    shows that field, so the drift is visible."""
    store, session = _make_store()
    store.get_memory = lambda _id: None  # the return path is not under test

    store.update_memory("m-1", summary="sharper")

    sent = session.run.call_args.kwargs
    assert "updated_at" in sent, "an edit left updated_at untouched"
    assert "m.updated_at = $updated_at" in session.run.call_args.args[0]


def test_update_memory_serializes_a_category_enum():
    """The web UI and CLI can now reclassify a memory; the enum must reach
    Cypher as its string value, not as a Python object."""
    from collivind.models.memory import MemoryCategory

    store, session = _make_store()
    store.get_memory = lambda _id: None

    store.update_memory("m-1", category=MemoryCategory.DECISION)

    assert session.run.call_args.kwargs["category"] == "decision"
