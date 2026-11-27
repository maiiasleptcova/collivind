"""The memory wiki (web/wiki.py), topic descriptions and the OKF export, against a real SQLite store.

The wiki exists because the graph showed that memories were linked but not
what a topic is, what each memory is for, or why two things connect. These
tests pin what a page must say: a description, memories grouped by purpose
with a gist, an error next to its fix, relations with their reason in words,
hubs read in pages. The export tests check the output against the OKF v0.2
frontmatter rules and OpenWiki's own frontmatter parser, and that it is
byte-for-byte deterministic.
"""

import hashlib
import json
import math
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from collivind.cli.commands.backup import export_cmd
from collivind.config import CollivindConfig, ConfigurationError, HooksConfig, QdrantConfig
from collivind.engine import summaries
from collivind.engine.descriptions import describe_entity, description_due
from collivind.engine.memory_manager import MemoryManager
from collivind.mcp.tools import CollivindTools
from collivind.models import EntityCreate, MemoryCategory, MemoryCreate, RelationshipCreate, RelType
from collivind.models.entity import EntityNode, EntityType
from collivind.storage.graph_sqlite import SqliteGraphStore
from collivind.storage.vector_sqlite import SqliteVectorStore
from collivind.web.api import MemoryAPI, route
from collivind.web.graph import HUB_DEGREE
from collivind.web.okf import MARKER, esc, render_bundle, write_bundle
from collivind.web.wiki import PAGE_SIZE, Wiki, gist

DIM = 64


class WordEmbedder:
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
    manager.deduplicator.find_duplicate_detailed = lambda *a, **k: None
    return manager


def _add(mgr, summary, content, cat=MemoryCategory.FACT, entities=(), project="p", session=None):
    return mgr.add_memory(
        MemoryCreate(content=content, summary=summary, category=cat, project_id=project, session_id=session),
        entities=[EntityCreate(name=n, type=t) for n, t in entities],
    )


DOCKER, COLIMA, NEO = ("Docker", EntityType.TOOL), ("Colima", EntityType.TOOL), ("Neo4j", EntityType.SERVICE)


@pytest.fixture
def incident(mgr):
    """An error about Docker and Colima, its cause, a fix filed only under Colima, and a decision about Docker."""
    text = "docker exec fails with no space left on device. Every exec into the cytm containers fails since Monday."
    err = _add(mgr, "docker exec fails with no space left", text, MemoryCategory.ERROR, [DOCKER, COLIMA])
    cause = _add(mgr, "Colima VM disk full", "The Colima VM disk filled up with old images.", entities=[COLIMA])
    fix = _add(mgr, "Resized the Colima disk to 80 GB", "Resized the Colima disk to 80 GB. Exec works again.",
               MemoryCategory.DECISION, [COLIMA])  # fmt: skip
    mgr.link(err.id, cause.id, RelType.CAUSED_BY, note="the disk was full")
    mgr.link(fix.id, err.id, RelType.RESOLVES, note="more disk")
    choice = _add(mgr, "Use compose profiles", "Use compose profiles for optional services.", MemoryCategory.DECISION,
                  [DOCKER])  # fmt: skip
    mgr.graph_store.create_relationship(
        RelationshipCreate(source_id="docker", target_id="colima", type=RelType.USES, properties={"note": "on the Mac"})
    )
    return {"err": err, "cause": cause, "fix": fix, "choice": choice}


# --- pages -------------------------------------------------------------------


def test_entity_page_says_what_it_is_from_its_links_until_an_agent_describes_it(mgr, incident):
    page = Wiki(mgr.graph_store).page("docker")
    assert page["title"] == "Docker" and page["type"] == "tool"
    assert (
        page["description"]
        == "Docker is a tool that uses Colima. 2 memories are about it, mostly decisions and problems."
        and page["described_by"] == "collivind"
    )
    assert page["stats"].startswith("2 memories: 1 problem (1 with a recorded fix), 1 decision")

    describe_entity(mgr, "docker", "Docker runs the stack locally.")
    page = Wiki(mgr.graph_store).page("docker")
    assert page["description"] == "Docker runs the stack locally." and page["described_by"] == "agent"


def test_error_is_shown_with_its_cause_and_fix_even_when_they_are_filed_elsewhere(mgr, incident):
    sections = {s["key"]: s for s in Wiki(mgr.graph_store).page("docker")["sections"]}
    (problem,) = sections["problems"]["items"]
    assert problem["id"] == incident["err"].id
    assert [(f["id"], f["note"]) for f in problem["fixes"]] == [(incident["fix"].id, "more disk")]
    assert [(c["id"], c["note"]) for c in problem["causes"]] == [(incident["cause"].id, "the disk was full")]
    assert [i["id"] for i in sections["decisions"]["items"]] == [incident["choice"].id]

    # on Colima's page the fix is part of the pair, not listed again under Decisions
    colima = {s["key"]: s for s in Wiki(mgr.graph_store).page("colima")["sections"]}
    assert colima["decisions"]["items"] == [] and colima["decisions"]["with_problems"] == 1
    assert colima["facts"]["items"] == [] and colima["facts"]["with_problems"] == 1


def test_a_memory_row_says_what_kind_what_it_says_and_what_else_it_is_under(mgr, incident):
    row = Wiki(mgr.graph_store).memory_row(incident["err"].id, here="docker")
    assert row["kind"] == "Problem" and row["label"] == "docker exec fails with no space left"
    assert row["gist"] == "Every exec into the cytm containers fails since Monday."
    assert [t["title"] for t in row["about"]] == ["Colima"]
    assert {(link["phrase"], link["memory"]["id"]) for link in row["links"]} == {
        ("is resolved by", incident["fix"].id),
        ("was caused by", incident["cause"].id),
    }


def test_gist_never_repeats_the_label():
    class M:
        summary = "Colima disk was enlarged to 77GB on the dev Mac after"
        content = "Colima disk was enlarged to 77GB on the dev Mac after builds failed. Images were pruned too."

    assert gist(M()) == "Images were pruned too."
    M.content = M.summary
    assert gist(M()) == ""


def test_related_topics_carry_their_reason_in_words(mgr, incident):
    rel = {r["title"]: r["reason"] for r in Wiki(mgr.graph_store).page("docker")["related"]}
    assert rel["Colima"] == "Docker uses Colima: on the Mac (1 shared memory)"
    assert rel["p"] == "2 of these are in this project"
    rel = {r["title"]: r["reason"] for r in Wiki(mgr.graph_store).page("colima")["related"]}
    assert rel["Docker"] == "Colima is used by Docker: on the Mac (1 shared memory)"


def test_summary_page_and_the_topics_its_memories_are_about(mgr, incident):
    node = summaries.save_summary(
        mgr,
        "p",
        summaries.Group("topic", "disk"),
        "- disk filled\n- disk resized",
        [incident["err"].id, incident["fix"].id],
    )
    wiki = Wiki(mgr.graph_store)
    page = wiki.page(node.id)
    assert page["kind"] == "summary" and page["facts"] == ["disk filled", "disk resized"]
    assert page["description"] == "disk filled"
    rel = {r["title"]: r["reason"] for r in page["related"]}
    assert (
        rel["Colima"] == "2 of its memories are about Colima" and rel["Docker"] == "1 of its memories is about Docker"
    )
    docker_rel = {r["id"]: r["reason"] for r in wiki.page("docker")["related"]}
    assert docker_rel[node.id] == "its summary covers 1 of these"
    assert {b["id"] for b in wiki.page("docker")["backlinks"]} >= {node.id, "colima"}


def test_a_hub_is_read_in_pages_and_can_be_narrowed_to_one_related_topic(mgr):
    for i in range(HUB_DEGREE + 5):
        _add(
            mgr,
            f"docker fact {i}",
            f"Docker fact number {i} about ports.",
            entities=[DOCKER] + ([NEO] if i < 4 else []),
        )
    wiki = Wiki(mgr.graph_store)
    page = wiki.page("docker")
    facts = page["sections"][0]
    assert page["hub"] and facts["total"] == HUB_DEGREE + 5
    assert len(facts["items"]) == PAGE_SIZE and facts["more"] == HUB_DEGREE + 5 - PAGE_SIZE
    rest = wiki.page("docker", only="facts", offset=PAGE_SIZE, limit=200)
    assert len(rest["sections"][0]["items"]) == HUB_DEGREE + 5 - PAGE_SIZE and "related" not in rest
    narrowed = wiki.page("docker", within="neo4j")
    assert narrowed["sections"][0]["total"] == 4 and narrowed["within"]["title"] == "Neo4j"
    found = wiki.page("docker", query="NUMBER 17 ports")["sections"][0]
    assert [i["label"] for i in found["items"]] == ["docker fact 17"]
    neo = {r["title"]: r for r in wiki.page("neo4j")["related"]}
    assert neo["Docker"]["reason"] == "4 shared memories"


def test_project_page_lists_topics_every_problem_and_the_memories_no_topic_files(mgr, incident):
    loose = _add(mgr, "Release on Fridays", "We release on Fridays.", MemoryCategory.PREFERENCE)
    page = Wiki(mgr.graph_store).page("project:p")
    assert page["unfiled"] == 1
    assert [i["id"] for s in page["sections"] for i in s["items"]] == [incident["err"].id, loose.id]
    assert {r["title"] for r in page["related"] if r["kind"] == "entity"} == {"Docker", "Colima"}
    assert Wiki(mgr.graph_store, project="other").topics == {}


def test_superseded_memories_are_not_listed_but_the_new_version_says_it_replaces_one(mgr, incident):
    old = incident["choice"]
    new = _add(
        mgr,
        "Use compose profiles everywhere",
        "Use compose profiles for every service.",
        MemoryCategory.DECISION,
        [DOCKER],
    )
    mgr.invalidate(old.id, new.id, "broader")
    wiki = Wiki(mgr.graph_store)
    ids = [i["id"] for s in wiki.page("docker")["sections"] for i in s["items"]]
    assert old.id not in ids and new.id in ids
    links = wiki.memory_row(new.id)["links"]
    assert [(link["phrase"], link["memory"]["live"]) for link in links] == [("supersedes", False)]


# --- API ---------------------------------------------------------------------


def test_wiki_routes(mgr, incident):
    api = MemoryAPI(mgr)
    status, index = route(api, "GET", "/api/wiki", {}, {})
    assert status == 200 and {t["id"] for t in index["topics"]} == {"docker", "colima", "project:p"}
    assert {(link["source"], link["target"]) for link in index["links"]} >= {("colima", "docker")} or {
        ("docker", "colima")
    } <= {(link["source"], link["target"]) for link in index["links"]}
    status, page = route(api, "GET", "/api/wiki/page", {"id": "docker"}, {})
    assert status == 200 and page["title"] == "Docker"
    assert route(api, "GET", "/api/wiki/page", {"id": "nope"}, {})[0] == 404
    assert route(api, "GET", "/api/wiki/page", {"id": "docker", "section": "bogus"}, {})[0] == 400
    assert route(api, "GET", "/api/wiki/page", {"id": "docker", "limit": "0"}, {})[0] == 400
    assert route(api, "GET", "/api/wiki", {"project": "*"}, {})[0] == 400

    status, body = route(api, "POST", "/api/wiki/describe", {}, {"name": "Colima", "description": "  The VM.  "})
    assert status == 200 and body["description"] == "The VM."
    assert route(api, "POST", "/api/wiki/describe", {}, {"name": "nope", "description": "x"})[0] == 404
    assert route(api, "POST", "/api/wiki/describe", {}, {"name": "Colima", "description": ""})[0] == 400
    assert route(api, "POST", "/api/wiki/describe", {}, {"name": "Colima", "description": "x" * 401})[0] == 400


# --- descriptions ------------------------------------------------------------


def test_describe_keeps_aliases_and_other_properties(mgr, incident):
    e = mgr.graph_store.get_entity("colima")
    mgr.graph_store.restore_entity(
        EntityNode(id=e.id, name=e.name, type=e.type, properties={"aliases": ["colima vm"]}, created_at=e.created_at)
    )
    mgr._aliases_read_at = float("-inf")  # the alias table is re-read at most once a minute
    describe_entity(mgr, "colima vm", "The Linux VM Docker runs in.")
    props = mgr.graph_store.get_entity("colima").properties
    assert props == {"aliases": ["colima vm"], "description": "The Linux VM Docker runs in."}
    with pytest.raises(LookupError):
        describe_entity(mgr, "nothing", "x")


def test_description_due_asks_once_a_topic_has_enough_memories_and_no_description(mgr, incident):
    ids = [incident["fix"].id]
    assert description_due(mgr, ids, 4) is None  # Colima has 3
    due = description_due(mgr, ids, 3)
    assert due["topics"] == [{"entity": "Colima"}] and "collivind_describe_topic" in due["instruction"]
    assert description_due(mgr, ids, 0) is None
    describe_entity(mgr, "Colima", "The VM.")
    assert description_due(mgr, ids, 3) is None


def test_mcp_batch_add_reports_description_due_and_the_tool_stores_it(mgr, incident, monkeypatch):
    monkeypatch.setattr("collivind.capture.record", lambda *a, **k: None)
    mgr.config.hooks.describe_after = 4
    tools = CollivindTools(mgr, session_id="s", project_id="p")
    memory = {"content": "Colima runs with 4 CPUs.", "summary": "Colima has 4 CPUs", "category": "fact",
              "entities": [{"name": "Colima", "type": "tool"}]}  # fmt: skip
    out = json.loads(tools.handle_call("collivind_batch_add", {"memories": [memory]}))
    assert out["description_due"]["topics"] == [{"entity": "Colima"}]
    out = json.loads(tools.handle_call("collivind_describe_topic", {"entity_name": "Colima", "description": "The VM."}))
    assert out["status"] == "success" and out["description"] == "The VM."
    assert "error" in json.loads(
        tools.handle_call("collivind_describe_topic", {"entity_name": "x", "description": "y"})
    )


def test_describe_after_must_be_a_non_negative_integer():
    with pytest.raises(ConfigurationError):
        HooksConfig(describe_after=-1)
    with pytest.raises(ConfigurationError):
        HooksConfig(describe_after=True)
    assert HooksConfig().describe_after == 5


# --- OKF export --------------------------------------------------------------

LINK = re.compile(r"\]\(([^)\s#]*)(#[^)\s]*)?\)")
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$")


def split_frontmatter(raw: str):
    """A port of OpenWiki's splitFrontmatter (src/visualize/graph.ts): the reader the export must satisfy."""
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    meta = {}
    pending = None
    for line in raw[3:end].strip().splitlines():
        item = re.match(r"^\s*-\s+(.*)$", line)
        if item and pending:
            meta[pending].append(item.group(1).strip().strip("'\""))
            continue
        kv = re.match(r"^([A-Za-z0-9_]+):\s*(.*)$", line)
        if not kv:
            continue
        key, value = kv.group(1), kv.group(2).strip()
        if value == "":
            pending, meta[key] = key, []
        else:
            pending, meta[key] = None, re.sub(r"^['\"]|['\"]$", "", value)
    return meta, raw[raw.find("\n", end + 1) + 1 :]


def validate_bundle(root: Path) -> int:
    """OKF v0.2 frontmatter rules (SPEC §4, §5, §8, §11) plus: every relative link resolves to a file and anchor."""
    pages = sorted(root.rglob("*.md"))
    for path in pages:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        if path.name == "index.md":
            if rel == "index.md":
                assert yaml.safe_load(text.split("---")[1]) == {"okf_version": "0.2"}
            else:
                assert not text.startswith("---"), f"{rel}: an index has no frontmatter (§8)"
        else:
            assert text.startswith("---\n"), rel
            meta = yaml.safe_load(text.split("\n---\n")[0][4:])
            assert isinstance(meta.get("type"), str) and meta["type"].strip(), f"{rel}: type is required"
            for key in ("title", "description"):
                assert isinstance(meta.get(key), str) and meta[key].strip(), f"{rel}: {key}"
            assert isinstance(meta["tags"], list) and all(isinstance(t, str) and t for t in meta["tags"]), rel
            assert meta["generated"]["by"].startswith("collivind/")
            assert "at" not in meta["generated"] or ISO.match(meta["generated"]["at"]), rel
            for source in meta.get("sources", []):
                assert isinstance(source["resource"], str) and source["resource"], rel
                assert f'<a id="{source["id"]}"></a>' in text, f"{rel}: source {source['id']} has no anchor"
            ow, _ = split_frontmatter(text)  # OpenWiki reads the same values
            assert (ow["type"], ow["title"], ow["description"], ow["tags"]) == (
                meta["type"], meta["title"], meta["description"], meta["tags"]
            ), rel  # fmt: skip
        body = text.split("\n---\n", 1)[1] if text.startswith("---") else text  # links are read from the body
        for target, anchor in LINK.findall(body):
            if target.startswith(("http://", "https://")):
                continue
            assert not target.startswith("/"), f"{rel}: OpenWiki drops bundle-absolute links"
            dest = (path.parent / target).resolve() if target else path
            assert dest.is_file(), f"{rel}: broken link {target}"
            if anchor:
                assert f'<a id="{anchor[1:]}"></a>' in dest.read_text(encoding="utf-8"), (
                    f"{rel}: no {anchor} in {target}"
                )
    return len(pages)


def test_export_is_valid_okf_and_deterministic(mgr, incident, tmp_path):
    summaries.save_summary(mgr, "p", summaries.Group("topic", "disk"), "- disk filled", [incident["err"].id])
    describe_entity(mgr, "Docker", 'Docker: the "runtime"; it runs # everything.')
    first, second = tmp_path / "a", tmp_path / "b"
    counts = write_bundle(Wiki(mgr.graph_store), first)
    write_bundle(Wiki(mgr.graph_store), second)
    assert counts == {"entity": 2, "summary": 1, "project": 1, "files": 8}
    assert validate_bundle(first) == 8
    files = sorted(p.relative_to(first).as_posix() for p in first.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(second).as_posix() for p in second.rglob("*") if p.is_file())
    assert all((first / f).read_bytes() == (second / f).read_bytes() for f in files)
    assert MARKER in files and "entities/docker.md" in files and "projects/p.md" in files

    docker = (first / "entities/docker.md").read_text()
    # the full text lives on its most specific page; the hub links there
    assert "[Resized the Colima disk to 80 GB](colima.md#m-" in docker and "**Fixed by**" in docker
    assert "[Colima](colima.md) — Docker uses Colima: on the Mac" in docker


def test_memory_text_cannot_break_the_page_or_forge_links(mgr, tmp_path):
    evil = "# Not a heading\n---\nsee [x](../../etc/passwd.md) and ](evil.md) `code` <script>"
    _add(mgr, "Evil [label](x.md)", evil, entities=[DOCKER])
    write_bundle(Wiki(mgr.graph_store), tmp_path / "out")
    text = (
        (tmp_path / "out/entities/docker.md").read_text().split("\n---\n", 1)[1]
    )  # the body: what links are read from
    assert "](evil.md)" not in text and "](x.md)" not in text and "<script>" not in text
    assert "\n# Not" not in text and "\n---\nsee" not in text
    validate_bundle(tmp_path / "out")
    assert esc("a](b.md)") == "a\\]\\(b.md\\)"


def test_reexport_replaces_its_own_pages_and_refuses_a_foreign_directory(mgr, incident, tmp_path):
    out = tmp_path / "wiki"
    write_bundle(Wiki(mgr.graph_store), out)
    mgr.forget(incident["choice"].id)
    for m in (incident["err"], incident["cause"], incident["fix"]):
        mgr.forget(m.id)
    write_bundle(Wiki(mgr.graph_store), out)
    assert not (out / "entities/docker.md").exists()

    foreign = tmp_path / "notes"
    foreign.mkdir()
    (foreign / "mine.md").write_text("keep me")
    with pytest.raises(FileExistsError):
        write_bundle(Wiki(mgr.graph_store), foreign)
    assert (foreign / "mine.md").read_text() == "keep me"


def test_project_filter_and_slug_collisions(mgr, tmp_path):
    _add(mgr, "a", "About C plus plus.", entities=[("C++", EntityType.LIBRARY)], project="x")
    _add(mgr, "b", "About C sharp.", entities=[("C#", EntityType.LIBRARY)], project="x")
    _add(mgr, "c", "About index.", entities=[("index", EntityType.CONCEPT)], project="y")
    files = render_bundle(Wiki(mgr.graph_store, project="x"))
    names = sorted(f for f in files if f.startswith("entities/") and f != "entities/index.md")
    assert (
        len(names) == 2
        and "entities/c.md" in names
        and any(re.fullmatch(r"entities/c-[0-9a-f]{6}\.md", n) for n in names)
    )
    assert "projects/y.md" not in files
    files = render_bundle(Wiki(mgr.graph_store, project="y"))
    assert "entities/index-topic.md" in files


def test_export_cli_writes_the_bundle(mgr, incident, tmp_path, monkeypatch):
    monkeypatch.setattr(mgr.graph_store, "close", lambda: None)
    with patch("collivind.cli.commands.backup._open_graph", return_value=mgr.graph_store):
        result = CliRunner().invoke(export_cmd, ["--okf", str(tmp_path / "w")])
        assert result.exit_code == 0, result.output
        assert "Wrote 7 files" in result.output
        (tmp_path / "x").mkdir()
        (tmp_path / "x" / "a.txt").write_text("x")
        result = CliRunner().invoke(export_cmd, ["--okf", str(tmp_path / "x")])
        assert result.exit_code == 1 and "not a collivind export" in result.output
