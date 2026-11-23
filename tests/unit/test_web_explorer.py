"""The explorer endpoints (graph, links, projects) against a real SQLite store,
plus the request guard and the static assets over a real socket."""

import http.client
import json
import threading

import pytest
from test_backup import _cli
from test_migrate import _stores

from collivind.cli.commands.backup import export_cmd, import_cmd
from collivind.config import CollivindConfig
from collivind.engine.memory_manager import MemoryManager
from collivind.engine.projects import list_projects
from collivind.models import EntityCreate, MemoryCategory, MemoryCreate
from collivind.models.entity import EntityType
from collivind.web.api import MemoryAPI, route
from collivind.web.server import build_server, check_request


@pytest.fixture(autouse=True)
def _tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.fixture
def store(tmp_path):
    stores = _stores(tmp_path / "s")
    mgr = MemoryManager(stores.vector, stores.graph, stores.embedder, CollivindConfig())
    mgr.search_engine.find_contradictions = lambda memory: []

    def add(content, project="alpha", entities=()):
        return mgr.add_memory(
            MemoryCreate(content=content, summary=content[:30], category=MemoryCategory.DECISION, project_id=project),
            entities=[EntityCreate(name=n, type=EntityType.SERVICE) for n in entities],
        )

    old = add("we use postgres for the event log", entities=["Postgres"])
    new = add("we moved the event log to kafka", entities=["Kafka"])
    mgr.invalidate(old.id, new.id, "replaced")
    other = add("billing runs on stripe", project="my-app", entities=["Stripe"])
    return stores, MemoryAPI(mgr, mode="embedded"), {"old": old.id, "new": new.id, "other": other.id}


def call(api, method, path, params=None, body=None):
    return route(api, method, path, params or {}, {} if body is None else body)


class TestGraph:
    def test_shape(self, store):
        _, api, ids = store
        status, g = call(api, "GET", "/api/graph")
        assert status == 200
        assert set(g) == {"nodes", "edges", "total_memories", "truncated", "limit"}
        kinds = {n["id"]: n["kind"] for n in g["nodes"]}
        assert kinds[ids["new"]] == "memory" and kinds["kafka"] == "entity"
        mem = next(n for n in g["nodes"] if n["id"] == ids["old"])
        assert mem["superseded"] is True and "content" not in mem  # slim payload
        types = {(e["source"], e["type"], e["target"]) for e in g["edges"]}
        assert (ids["new"], "ABOUT", "kafka") in types
        # SQLite stores supersession only on the row; the graph still draws it.
        assert (ids["new"], "SUPERSEDES", ids["old"]) in types
        assert g["total_memories"] == 3 and g["truncated"] is False

    def test_project_filter_drops_unreached_entities(self, store):
        _, api, ids = store
        _, g = call(api, "GET", "/api/graph", {"project": "my-app"})
        assert {n["id"] for n in g["nodes"]} == {ids["other"], "stripe"}

    def test_cap_is_reported(self, store):
        _, api, _ = store
        _, g = call(api, "GET", "/api/graph", {"limit": "1"})
        assert g["truncated"] is True and len([n for n in g["nodes"] if n["kind"] == "memory"]) == 1

    @pytest.mark.parametrize("limit", ["0", "abc", "100001"])
    def test_bad_limit(self, store, limit):
        assert call(store[1], "GET", "/api/graph", {"limit": limit})[0] == 400

    def test_single_memory(self, store):
        _, api, ids = store
        status, m = call(api, "GET", f"/api/memories/{ids['new']}")
        assert status == 200 and m["content"] == "we moved the event log to kafka"
        assert call(api, "GET", "/api/memories/nope")[0] == 404


class TestLinks:
    def _edges(self, api):
        return {(e["source"], e["type"], e["target"]) for e in call(api, "GET", "/api/graph")[1]["edges"]}

    def test_link_and_unlink_memories(self, store):
        _, api, ids = store
        body = {"source_id": ids["new"], "target_id": ids["other"], "type": "RELATES_TO"}
        assert call(api, "POST", "/api/links", body=body)[0] == 201
        assert call(api, "POST", "/api/links", body=body)[0] == 201  # idempotent
        assert (ids["new"], "RELATES_TO", ids["other"]) in self._edges(api)
        status, payload = call(api, "DELETE", "/api/links", body=body)
        assert status == 200 and payload["removed"] == 1  # one row, not two
        assert (ids["new"], "RELATES_TO", ids["other"]) not in self._edges(api)
        assert call(api, "DELETE", "/api/links", body=body)[0] == 404

    def test_link_memory_to_entity(self, store):
        _, api, ids = store
        body = {"source_id": ids["other"], "target_id": "kafka", "type": "MENTIONS"}
        assert call(api, "POST", "/api/links", body=body)[0] == 201
        assert (ids["other"], "MENTIONS", "kafka") in self._edges(api)

    @pytest.mark.parametrize(
        "body, status",
        [
            ({"source_id": "X", "target_id": "kafka", "type": "ABOUT"}, 404),  # missing endpoint
            ({"source_id": "NEW", "target_id": "kafka", "type": "RELATES_TO"}, 400),  # wrong kinds
            ({"source_id": "NEW", "target_id": "OTHER", "type": "SUPERSEDES"}, 400),  # history only
            ({"source_id": "NEW", "target_id": "OTHER", "type": "BOGUS"}, 400),
            ({"source_id": "NEW", "target_id": "NEW", "type": "RELATES_TO"}, 400),
            ({"source_id": 5, "target_id": "OTHER", "type": "RELATES_TO"}, 400),
            ({"target_id": "OTHER", "type": "RELATES_TO"}, 400),
        ],
    )
    def test_link_validation(self, store, body, status):
        _, api, ids = store
        swap = {"NEW": ids["new"], "OTHER": ids["other"]}
        body = {k: swap.get(v, v) if isinstance(v, str) else v for k, v in body.items()}
        assert call(api, "POST", "/api/links", body=body)[0] == status

    def test_supersedes_cannot_be_unlinked(self, store):
        _, api, ids = store
        body = {"source_id": ids["new"], "target_id": ids["old"], "type": "SUPERSEDES"}
        assert call(api, "DELETE", "/api/links", body=body)[0] == 400

    def test_non_object_body(self, store):
        assert call(store[1], "POST", "/api/links", body=["x"])[0] == 400


class TestProjects:
    def test_counts(self, store):
        _, api, _ = store
        status, payload = call(api, "GET", "/api/projects")
        rows = {p["project_id"]: p for p in payload["projects"]}
        assert status == 200
        assert (rows["alpha"]["memories"], rows["alpha"]["live"]) == (2, 1)
        assert rows["my-app"]["display_name"] == "" and rows["my-app"]["tags"] == []

    def test_metadata_round_trip_survives_export_and_import(self, store, tmp_path):
        stores, api, _ = store
        meta = {
            "display_name": "My App",
            "description": "the <b>shop</b>",
            "color": "#AA3300",
            "repo": "https://github.com/x/my-app",
            "tags": ["web", "web", "shop"],
        }
        status, saved = call(api, "PATCH", "/api/project", {"id": "my-app"}, meta)
        assert status == 200 and saved["color"] == "#aa3300" and saved["tags"] == ["web", "shop"]
        # my_app slugs to the same id as my-app would; it must stay separate.
        call(api, "PATCH", "/api/project", {"id": "my_app"}, {"display_name": "Other"})
        # An extracted entity named after the project is untouched.
        stores.graph.create_entity(EntityCreate(name="my-app", type=EntityType.PROJECT, properties={"x": 1}))
        assert stores.graph.get_entity("my-app").properties == {"x": 1}

        # Partial update merges.
        call(api, "PATCH", "/api/project", {"id": "my-app"}, {"description": "shop"})
        rows = {p["project_id"]: p for p in call(api, "GET", "/api/projects")[1]["projects"]}
        assert rows["my-app"]["display_name"] == "My App" and rows["my-app"]["description"] == "shop"
        assert rows["my_app"]["display_name"] == "Other" and rows["my_app"]["memories"] == 0

        # A per-project export keeps the project's metadata even though it links to nothing.
        path = tmp_path / "b.jsonl"
        assert _cli(export_cmd, ["-o", path, "-p", "my-app"], stores).exit_code == 0
        dst = _stores(tmp_path / "dst")
        result = _cli(import_cmd, [path], dst)
        assert result.exit_code == 0, result.output
        restored = {p["project_id"]: p for p in list_projects(_stores(tmp_path / "dst").graph)}
        assert restored["my-app"]["display_name"] == "My App" and restored["my-app"]["color"] == "#aa3300"
        assert "my_app" not in restored

    def test_meta_entity_is_not_drawn(self, store):
        _, api, _ = store
        call(api, "PATCH", "/api/project", {"id": "alpha"}, {"display_name": "A"})
        nodes = call(api, "GET", "/api/graph")[1]["nodes"]
        assert not any(n["id"].startswith("project:") for n in nodes)

    @pytest.mark.parametrize(
        "body",
        [
            {"color": "red"},
            {"color": "#12345"},
            {"repo": "javascript:alert(1)"},
            {"repo": "data:text/html,x"},
            {"tags": "web"},
            {"tags": [""]},
            {"display_name": 3},
            {"display_name": "x" * 201},
            {"owner": "me"},
        ],
    )
    def test_meta_validation(self, store, body):
        assert call(store[1], "PATCH", "/api/project", {"id": "alpha"}, body)[0] == 400

    def test_repo_accepts_paths(self, store):
        for repo in ("/Users/me/code/app", "~/code/app", "C:\\code\\app", ""):
            assert call(store[1], "PATCH", "/api/project", {"id": "alpha"}, {"repo": repo})[0] == 200

    @pytest.mark.parametrize("pid", ["", "x" * 201])
    def test_bad_project_id(self, store, pid):
        assert call(store[1], "PATCH", "/api/project", {"id": pid}, {"display_name": "x"})[0] == 400


class TestRequestGuard:
    GOOD = {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765", "Content-Type": "application/json"}

    def test_same_origin_write_allowed(self):
        assert check_request("POST", self.GOOD) is None
        assert check_request("PATCH", {**self.GOOD, "Content-Type": "application/json; charset=utf-8"}) is None

    @pytest.mark.parametrize(
        "change",
        [
            {"Origin": "http://evil.example"},
            {"Origin": "http://127.0.0.1:3000"},  # another local port
            {"Origin": ""},
            {"Origin": "null"},
            {"Content-Type": "text/plain"},
            {"Content-Type": "application/x-www-form-urlencoded"},
        ],
    )
    @pytest.mark.parametrize("method", ["POST", "PATCH", "DELETE"])
    def test_write_refused(self, change, method):
        assert check_request(method, {**self.GOOD, **change}) is not None

    @pytest.mark.parametrize("host", ["evil.example:8765", "127.0.0.1.nip.io:8765", "192.168.1.5:8765", ""])
    def test_rebinding_host_refused_even_for_reads(self, host):
        assert check_request("GET", {"Host": host})[0] == 403

    @pytest.mark.parametrize("host", ["127.0.0.1:8765", "localhost:9000", "[::1]:8765", "LOCALHOST"])
    def test_loopback_reads_allowed(self, host):
        assert check_request("GET", {"Host": host}) is None


@pytest.fixture
def server(store):
    srv = build_server(store[1], port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def _request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, body=body, headers={"Host": f"127.0.0.1:{port}", **(headers or {})})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp, data


class TestOverTheWire:
    @pytest.mark.parametrize(
        "path, ctype",
        [
            ("/", "text/html"),
            ("/app.css", "text/css"),
            ("/app.js", "text/javascript"),
            ("/graph.js", "text/javascript"),
            ("/fonts/instrument-sans-latin-var.woff2", "font/woff2"),
            ("/fonts/OFL-instrument-sans.txt", "text/plain"),
        ],
    )
    def test_static_assets_are_served(self, server, path, ctype):
        resp, data = _request(server, "GET", path)
        assert resp.status == 200 and resp.getheader("Content-Type").startswith(ctype) and data
        assert "script-src 'self'" in resp.getheader("Content-Security-Policy")

    def test_index_references_only_served_assets(self, server):
        import re

        _, html = _request(server, "GET", "/")
        for ref in re.findall(r'(?:src|href)="(/[^"]+)"', html.decode()):
            assert _request(server, "GET", ref)[0].status == 200, ref

    def test_traversal_refused(self, server):
        assert _request(server, "GET", "/../server.py")[0].status == 404
        assert _request(server, "GET", "/%2e%2e/server.py")[0].status == 404

    def test_cross_origin_write_refused(self, server, store):
        _, _, ids = store
        body = json.dumps({"summary": "pwned"})
        resp, _ = _request(
            server,
            "PATCH",
            f"/api/memories/{ids['new']}",
            body,
            {"Origin": "http://evil.example", "Content-Type": "application/json"},
        )
        assert resp.status == 403
        _, data = _request(server, "GET", f"/api/memories/{ids['new']}")
        assert json.loads(data)["summary"] != "pwned"

    def test_same_origin_write_and_bad_bodies(self, server, store):
        _, _, ids = store
        headers = {"Origin": f"http://127.0.0.1:{server}", "Content-Type": "application/json"}
        resp, _ = _request(server, "PATCH", f"/api/memories/{ids['new']}", json.dumps({"summary": "ok"}), headers)
        assert resp.status == 200
        assert _request(server, "PATCH", f"/api/memories/{ids['new']}", "{nope", headers)[0].status == 400
        assert _request(server, "POST", "/api/links", "[1, 2]", headers)[0].status == 400

    def test_rebinding_read_refused(self, server):
        resp, _ = _request(server, "GET", "/api/graph", headers={"Host": "attacker.example"})
        assert resp.status == 403
