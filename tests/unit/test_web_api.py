"""The web UI is the only surface that can destroy memories with a click, so
its routing and guards are tested directly rather than through a browser."""

from unittest.mock import MagicMock

import pytest

from collivind.models import MemoryCategory, MemoryNode
from collivind.web.api import MemoryAPI, route


def _node(content="a stored fact", summary="a stored fact"):
    return MemoryNode(content=content, summary=summary, category=MemoryCategory.FACT)


@pytest.fixture
def api():
    manager = MagicMock()
    manager.get_timeline.return_value = [_node()]
    manager.search.return_value = []
    return MemoryAPI(manager, mode="docker")


def call(api, method, path, params=None, body=None):
    return route(api, method, path, params or {}, body or {})


class TestReading:
    def test_empty_query_browses_without_embedding(self, api):
        """Browsing must not pay a search round-trip (#16 makes that costly)."""
        status, payload = call(api, "GET", "/api/memories")
        assert status == 200 and payload["searched"] is False
        api.manager.get_timeline.assert_called_once()
        api.manager.search.assert_not_called()

    def test_query_searches(self, api):
        status, payload = call(api, "GET", "/api/memories", {"q": "storage lock"})
        assert status == 200 and payload["searched"] is True
        api.manager.search.assert_called_once()

    def test_category_is_serialised_as_a_string(self, api):
        _, payload = call(api, "GET", "/api/memories")
        assert payload["memories"][0]["category"] == "fact"

    def test_bad_limit_is_rejected(self, api):
        status, _ = call(api, "GET", "/api/memories", {"limit": "abc"})
        assert status == 400

    def test_status_reports_the_mode(self, api):
        """Principle 3: never let the reader guess which store they are in."""
        assert call(api, "GET", "/api/status")[1]["mode"] == "docker"


class TestWriting:
    def test_create_requires_content(self, api):
        status, payload = call(api, "POST", "/api/memories", body={"content": "   "})
        assert status == 400 and "content" in payload["error"]
        api.manager.add_memory.assert_not_called()

    def test_create_rejects_unknown_category(self, api):
        status, _ = call(api, "POST", "/api/memories", body={"content": "x", "category": "nonsense"})
        assert status == 400
        api.manager.add_memory.assert_not_called()

    def test_create_returns_201(self, api):
        api.manager.add_memory.return_value = _node()
        status, payload = call(api, "POST", "/api/memories", body={"content": "remember this"})
        assert status == 201 and payload["content"] == "a stored fact"

    def test_update_with_no_fields_is_rejected(self, api):
        status, _ = call(api, "PATCH", "/api/memories/abc", body={})
        assert status == 400
        api.manager.update_memory.assert_not_called()

    def test_update_missing_memory_is_404(self, api):
        api.manager.update_memory.return_value = None
        assert call(api, "PATCH", "/api/memories/ghost", body={"summary": "s"})[0] == 404

    def test_forget_missing_memory_is_404(self, api):
        api.manager.forget.return_value = False
        assert call(api, "DELETE", "/api/memories/ghost")[0] == 404

    def test_forget_reports_what_it_removed(self, api):
        api.manager.forget.return_value = True
        status, payload = call(api, "DELETE", "/api/memories/abc")
        assert status == 200 and payload["forgotten"] == "abc"


class TestRouting:
    def test_unknown_route_is_404(self, api):
        assert call(api, "GET", "/api/nope")[0] == 404

    def test_non_api_path_is_not_routed(self, api):
        assert call(api, "GET", "/index.html")[0] == 404

    def test_version_chain(self, api):
        api.manager.get_version_chain.return_value = [_node()]
        assert call(api, "GET", "/api/memories/abc/versions")[0] == 200

    def test_missing_entity_is_404(self, api):
        api.manager.get_entity.return_value = None
        assert call(api, "GET", "/api/entities/Qdrant")[0] == 404


class TestServerGuards:
    def test_static_traversal_is_refused(self, tmp_path, monkeypatch):
        """A URL must never reach outside the static dir."""
        from collivind.web import server

        handler_cls = server._make_handler(MemoryAPI(MagicMock()))
        handler = handler_cls.__new__(handler_cls)
        sent = {}
        handler._json = lambda status, payload: sent.update(status=status, payload=payload)
        handler._send = lambda *a: sent.update(status=200)
        handler._static("/../../../../etc/passwd")
        assert sent["status"] == 404

    def test_index_is_served_for_root(self):
        from collivind.web import server

        assert (server.STATIC / "index.html").is_file()
        assert (server.STATIC / "app.css").is_file()
        assert (server.STATIC / "app.js").is_file()
