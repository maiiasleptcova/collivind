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
