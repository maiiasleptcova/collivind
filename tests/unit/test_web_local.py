"""The explorer's local graph, cause-and-fix chain and duplicate handling, against a real SQLite store.

The local graph exists because on a real store 2 hops around a memory
reached 308 memories through hub entities like "Docker"; these tests pin the
rules that keep it readable: hubs are reached but not walked through, typed
memory links outrank shared entities, and the cap says what it left out.
"""

import hashlib
import math
import re

import pytest

from collivind.config import CollivindConfig, QdrantConfig
from collivind.engine.memory_manager import MemoryManager
from collivind.models import EntityCreate, MemoryCategory, MemoryCreate, RelType
from collivind.models.entity import EntityType
from collivind.storage.graph_sqlite import SqliteGraphStore
from collivind.storage.vector_sqlite import SqliteVectorStore
from collivind.web.api import MemoryAPI, route
from collivind.web.graph import HUB_DEGREE, build_graph
from collivind.web.local import chain, neighbourhood

DIM = 64


class WordEmbedder:
    """Bag of hashed words: texts that share most words embed close together, unlike a hash of the whole text."""

    def embed(self, text):
        v = [0.0] * DIM
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.sha256(w.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_query(self, text):
        return self.embed(text)

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def health_check(self):
        return {"status": "ok"}

    @property
    def dimension(self):
        return DIM


@pytest.fixture(autouse=True)
def _tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.fixture
def mgr(tmp_path):
    path = str(tmp_path / "store")
    manager = MemoryManager(
        SqliteVectorStore(path, QdrantConfig(), DIM), SqliteGraphStore(path), WordEmbedder(), CollivindConfig()
    )
    manager.search_engine.find_contradictions = lambda memory: []
    return manager


def _add(mgr, content, entities=(), category=MemoryCategory.FACT):
    return mgr.add_memory(
        MemoryCreate(content=content, summary=content[:60], category=category, project_id="p"),
        entities=[EntityCreate(name=n, type=EntityType.TOOL) for n in entities],
    )


@pytest.fixture
def hubbed(mgr):
    """A centre memory about a hub ("Docker", HUB_DEGREE + 10 others) and a small topic ("colima", 3 others)."""
    center = _add(mgr, "docker exec fails with no space left on device", ["Docker", "colima"], MemoryCategory.ERROR)
    crowd = [
        _add(mgr, f"unrelated docker note number {i} about ports {i * 7}", ["Docker"]) for i in range(HUB_DEGREE + 10)
    ]
    peers = [_add(mgr, f"colima peer {i} talks about vm sizing {i * 11}", ["colima"]) for i in range(3)]
    # one of the crowd also shares the small topic: the most related member of the hub
    mgr.link(crowd[5].id, "colima", RelType.ABOUT)
    fix = _add(mgr, "resized the colima disk to 80 GB", ["Docker"], MemoryCategory.DECISION)
    mgr.link(fix.id, center.id, RelType.RESOLVES, note="more disk")
    beyond = _add(mgr, "the fix was later rolled into the setup script")
    mgr.link(beyond.id, fix.id, RelType.RELATES_TO)
    return mgr, {
        "center": center.id,
        "crowd": [m.id for m in crowd],
        "peers": [m.id for m in peers],
        "fix": fix.id,
        "beyond": beyond.id,
    }


def _ids(g):
    return [n["id"] for n in g["nodes"]]


class TestHubs:
    def test_one_hop_shows_the_hub_collapsed_with_its_count(self, hubbed):
        mgr, ids = hubbed
        g = neighbourhood(mgr.graph_store, ids["center"], depth=1)
        docker = next(n for n in g["nodes"] if n["id"] == "docker")
        assert docker["hub"] is True and docker["count"] == HUB_DEGREE + 12
        assert "hub" not in next(n for n in g["nodes"] if n["id"] == "colima")
        assert set(_ids(g)) == {ids["center"], "docker", "colima", ids["fix"]}

    def test_two_hops_do_not_walk_through_the_hub(self, hubbed):
        mgr, ids = hubbed
        g = neighbourhood(mgr.graph_store, ids["center"], depth=2)
        reached = set(_ids(g))
        assert set(ids["peers"]) <= reached and ids["beyond"] in reached  # through colima and through the fix
        assert reached.isdisjoint(set(ids["crowd"]) - {ids["crowd"][5]})  # nothing only Docker connects
        assert g["more"] == 0 and len(g["nodes"]) < 15

    def test_the_old_walk_would_have_exploded(self, hubbed):
        """The search neighbourhood shares the hub rule; without it depth 2 is the whole crowd."""
        mgr, ids = hubbed
        g = build_graph(mgr.graph_store, seeds={ids["center"]}, depth=2)
        assert len(g["nodes"]) < 15
        assert next(n for n in g["nodes"] if n["id"] == "docker")["hub"] is True

    def test_a_hub_seed_is_not_expanded_either(self, hubbed):
        mgr, _ = hubbed
        g = build_graph(mgr.graph_store, seeds={"docker"}, depth=1)
        assert _ids(g) == ["docker"]

    def test_opening_a_hub_at_one_hop_lists_its_most_related_members_first(self, hubbed):
        mgr, ids = hubbed
        g = neighbourhood(mgr.graph_store, ids["center"], depth=1, expand={"docker"}, limit=6)
        nodes = _ids(g)
        assert nodes[0] == ids["center"]
        # the crowd member that also shares colima outranks the rest of the crowd
        assert ids["crowd"][5] in nodes
        assert g["more"] > 0 and not any(n.get("hub") for n in g["nodes"] if n["id"] == "docker")
        assert next(n for n in g["nodes"] if n["id"] == "docker")["expanded"] is True

    def test_a_hub_centre_is_capped_and_says_how_many_more(self, hubbed):
        mgr, _ = hubbed
        g = neighbourhood(mgr.graph_store, "docker", depth=1, limit=10)
        assert len(g["nodes"]) == 10 and _ids(g)[0] == "docker"
        assert g["more"] == HUB_DEGREE + 12 - 9 and g["truncated"] is True


class TestRanking:
    def test_typed_link_outranks_shared_entity(self, hubbed):
        """With room for one 2-hop node, the memory linked to the fix beats colima's peers."""
        mgr, ids = hubbed
        g = neighbourhood(mgr.graph_store, ids["center"], depth=2, limit=5)
        assert _ids(g)[:4] and set(_ids(g)[:4]) == {ids["center"], "docker", "colima", ids["fix"]}
        assert _ids(g)[4] == ids["beyond"]

    def test_types_filter_the_walk(self, hubbed):
        mgr, ids = hubbed
        g = neighbourhood(mgr.graph_store, ids["center"], depth=2, types={"RESOLVES"})
        assert set(_ids(g)) == {ids["center"], ids["fix"]}
        assert [e["type"] for e in g["edges"]] == ["RESOLVES"] and g["edges"][0]["note"] == "more disk"

    def test_unknown_centre_is_empty(self, mgr):
        g = neighbourhood(mgr.graph_store, "nope", depth=2)
        assert g["nodes"] == [] and g["more"] == 0


class TestChain:
    def test_error_reaches_cause_fix_and_the_fix_s_new_version(self, mgr):
        err = _add(mgr, "docker exec fails", category=MemoryCategory.ERROR)
        cause = _add(mgr, "the vm disk was full")
        fix = _add(mgr, "resized the disk to 80 GB", category=MemoryCategory.DECISION)
        fix2 = _add(mgr, "resized the disk to 100 GB after it filled again", category=MemoryCategory.DECISION)
        noise = _add(mgr, "unrelated: the cron runs at noon")
        mgr.link(err.id, cause.id, RelType.CAUSED_BY, note="disk full")
        mgr.link(fix.id, err.id, RelType.RESOLVES)
        mgr.link(noise.id, err.id, RelType.RELATES_TO)
        mgr.invalidate(fix.id, fix2.id, "bigger disk")

        found = chain(mgr.graph_store, err.id)
        links = [(link["source"], link["type"], link["target"]) for link in found["links"]]
        assert links[:2] == [(err.id, "CAUSED_BY", cause.id), (fix.id, "RESOLVES", err.id)]
        assert links[2] == (fix2.id, "SUPERSEDES", fix.id)  # one step further: nearest first
        assert noise.id not in found["memories"] and found["links"][0]["note"] == "disk full"

    def test_through_the_api(self, mgr):
        api = MemoryAPI(mgr, mode="embedded")
        err = _add(mgr, "docker exec fails", category=MemoryCategory.ERROR)
        assert route(api, "GET", f"/api/memories/{err.id}/chain", {}, {}) == (
            200,
            {"links": [], "memories": {err.id: chain(mgr.graph_store, err.id)["memories"][err.id]}},
        )
        assert route(api, "GET", "/api/memories/nope/chain", {}, {})[0] == 404


class TestDuplicates:
    @pytest.fixture
    def pair(self, mgr):
        a = _add(mgr, "The Docker VM disk for Colima was enlarged to 77GB after the disk-full incident")
        # past dedup, the way the real 77GB/76GB pair slipped in
        mgr.deduplicator.find_duplicate_detailed = lambda *a, **k: None
        b = _add(mgr, "The Docker VM disk for Colima was enlarged to 76GB after the disk-full incident")
        other = _add(mgr, "nightly backups run at two in the morning and keep fourteen days")
        return MemoryAPI(mgr, mode="embedded"), a.id, b.id, other.id

    def test_flags_the_near_duplicate_only(self, pair):
        api, a, b, other = pair
        status, payload = route(api, "GET", f"/api/memories/{a}/duplicates", {}, {})
        assert status == 200
        assert [d["id"] for d in payload["duplicates"]] == [b]
        assert payload["duplicates"][0]["likely"] is True and payload["duplicates"][0]["similarity"] >= 0.92
        assert route(api, "GET", f"/api/memories/{other}/duplicates", {}, {}) == (200, {"duplicates": []})
        assert route(api, "GET", "/api/memories/nope/duplicates", {}, {})[0] == 404

    def test_supersede_retires_one_into_the_version_chain(self, pair):
        api, a, b, _ = pair
        assert route(api, "POST", f"/api/memories/{a}/supersede", {}, {"superseded_by": b}) == (
            200,
            {"superseded": a, "superseded_by": b},
        )
        old = api.manager.get_memory(a)
        assert old.superseded_by == b and api.manager.get_memory(b).superseded_by is None
        assert route(api, "GET", f"/api/memories/{b}/duplicates", {}, {}) == (200, {"duplicates": []})
        # nothing to merge twice
        assert route(api, "POST", f"/api/memories/{b}/supersede", {}, {"superseded_by": a})[0] == 409

    @pytest.mark.parametrize(
        "body, status", [({}, 400), ({"superseded_by": ""}, 400), ({"superseded_by": "nope"}, 404), (None, 400)]
    )
    def test_supersede_validation(self, pair, body, status):
        api, a, _, _ = pair
        assert (
            route(api, "POST", f"/api/memories/{a}/supersede", {}, body if body is not None else {"superseded_by": a})[
                0
            ]
            == status
        )


class TestLocalRoute:
    def test_route(self, hubbed):
        mgr, ids = hubbed
        api = MemoryAPI(mgr, mode="embedded")
        status, g = route(
            api, "GET", "/api/local", {"id": ids["center"], "depth": "2", "expand": "docker", "limit": "8"}, {}
        )
        assert status == 200 and len(g["nodes"]) == 8 and g["center"] == ids["center"]

    @pytest.mark.parametrize(
        "params",
        [
            {},
            {"id": ""},
            {"id": "x", "depth": "4"},
            {"id": "x", "limit": "0"},
            {"id": "x", "limit": "501"},
            {"id": "x", "types": "NOPE"},
            {"id": "x", "project": "x" * 201},
        ],
    )
    def test_bad_params(self, hubbed, params):
        assert route(MemoryAPI(hubbed[0]), "GET", "/api/local", params, {})[0] == 400


def test_the_cap_never_keeps_a_node_without_its_path_to_the_centre(mgr):
    """Review finding: strongly linked hop-3 nodes used to outrank the hop-2 nodes that reach them."""
    center = _add(mgr, "the centre memory", ["topic"])
    for i in range(20):
        mid = _add(mgr, f"mid memory {i} on topic {i * 13}", ["topic"])
        far = _add(mgr, f"far fix {i} for mid {i * 17}")
        mgr.link(far.id, mid.id, RelType.RESOLVES)
    g = neighbourhood(mgr.graph_store, center.id, depth=3, limit=25)
    kept = set(_ids(g))
    linked = {e["source"] for e in g["edges"]} | {e["target"] for e in g["edges"]}
    assert kept - {center.id} <= linked  # every kept node is drawn connected
    assert [n["hop"] for n in g["nodes"]] == sorted(n["hop"] for n in g["nodes"])
