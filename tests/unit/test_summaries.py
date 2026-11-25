"""Summary memories: groups, writes, rewrites, routed recall, MCP/CLI surfaces and round-trips."""

import io
import json
import re
from dataclasses import replace

import pytest
from click.testing import CliRunner

from collivind.config import CollivindConfig, QdrantConfig, SearchConfig
from collivind.engine import summaries
from collivind.engine.bundle import export_bundle, import_bundle, read_bundle
from collivind.engine.memory_manager import MemoryManager
from collivind.engine.migrate import Stores, migrate
from collivind.mcp.tools import CollivindTools
from collivind.models import MemoryCategory, MemoryCreate, RelType, SearchQuery
from collivind.storage.graph_sqlite import SqliteGraphStore
from collivind.storage.vector_sqlite import SqliteVectorStore

VOCAB = (
    "alice bob paris rome trip museum dog cat adopt painting guitar concert job bank interview "
    "deploy helm postgres redis cache migration summary"
).split()


def _words(text):
    return re.findall(r"[a-z]+", text.lower())


class WordEmbedder:
    """Bag of words over a fixed vocabulary; the reranker counts query words in the text."""

    can_rerank = True

    def __init__(self):
        self.rerank_calls = 0

    def embed(self, text):
        words = _words(text)
        vec = [float(words.count(v)) for v in VOCAB]
        return vec + [0.0 if any(vec) else 1.0]

    embed_query = embed

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def rerank(self, query, texts):
        self.rerank_calls += 1
        q = set(_words(query)) & set(VOCAB)
        return [4.0 * len(q & set(_words(t))) - 2.0 - len(t) / 1000 for t in texts]

    def health_check(self):
        return {"status": "ok"}

    @property
    def dimension(self):
        return len(VOCAB) + 1


def _manager(path, **search):
    path.mkdir(parents=True, exist_ok=True)
    config = CollivindConfig(data_dir=str(path), search=replace(SearchConfig(), **search))
    graph = SqliteGraphStore(str(path))
    vector = SqliteVectorStore(str(path), QdrantConfig(), WordEmbedder().dimension)
    return MemoryManager(vector, graph, WordEmbedder(), config)


@pytest.fixture
def mgr(tmp_path):
    return _manager(tmp_path / "store")


def _add(mgr, text, session="s1", project="p"):
    return mgr.add_memory(
        MemoryCreate(content=text, summary=text, category=MemoryCategory.FACT, project_id=project, session_id=session)
    )


def _trip_session(mgr):
    ids = [
        _add(mgr, "Alice booked a trip to Paris for May").id,
        _add(mgr, "Alice wants to see the museum with the painting").id,
        _add(mgr, "Alice will bring her dog on the trip").id,
    ]
    _add(mgr, "Bob plays guitar at a concert", session="s2")
    return ids


SUMMARY = "- 2026-05-01: Alice booked a trip to Paris for May.\n- Alice plans the museum and brings her dog."


# --- groups and writes ------------------------------------------------------


def test_session_summary_links_every_member_both_ways(mgr):
    ids = _trip_session(mgr)
    node = summaries.save_summary(mgr, "p", summaries.Group("session", "s1"), SUMMARY)
    assert node.category == MemoryCategory.SUMMARY and node.session_id == "s1"
    assert {m.id for m in summaries.members_of(mgr, [node.id])[node.id]} == set(ids)
    assert [s.id for s in summaries.summaries_of(mgr, ids[0])] == [node.id]
    assert summaries.unsummarized(mgr, "p", summaries.Group("session", "s1")) == []
    # searchable as a memory: it has a vector
    assert node.id in {vid for vid, _, _ in mgr.vector_store.iter_all()}


def test_rewrite_supersedes_and_keeps_the_version_chain(mgr):
    ids = _trip_session(mgr)
    group = summaries.Group("session", "s1")
    first = summaries.save_summary(mgr, "p", group, SUMMARY)
    late = _add(mgr, "Alice will adopt a cat after the trip").id
    assert [m.id for m in summaries.unsummarized(mgr, "p", group)] == [late]
    view = summaries.group_view(mgr, "p", group)
    assert view["current_summary"]["id"] == first.id
    assert [m["id"] for m in view["members"] if m["new"]] == [late]

    second = summaries.save_summary(mgr, "p", group, SUMMARY + "\n- Alice will adopt a cat.", view["member_ids"])
    assert second.version == 2 and second.previous_version_id == first.id
    assert [m.id for m in mgr.get_version_chain(second.id)] == [first.id, second.id]
    assert summaries.live_summary(mgr, "p", group).id == second.id
    assert {m.id for m in summaries.members_of(mgr, [second.id])[second.id]} == set(ids) | {late}
    live = {vid for vid, _, _ in mgr.vector_store.iter_all()}
    assert first.id not in live and second.id in live


def test_topic_group_needs_members_and_keeps_them_on_rewrite(mgr):
    ids = _trip_session(mgr)
    group = summaries.group_for(topic="Paris trip")
    assert group == summaries.Group("topic", "paris_trip")
    with pytest.raises(ValueError, match="at least one member"):
        summaries.save_summary(mgr, "p", group, SUMMARY)
    first = summaries.save_summary(mgr, "p", group, SUMMARY, ids[:1])
    second = summaries.save_summary(mgr, "p", group, SUMMARY, ids[1:2])
    assert first.session_id is None
    assert {m.id for m in summaries.members_of(mgr, [second.id])[second.id]} == set(ids[:2])


@pytest.mark.parametrize(
    "text, error",
    [("", "required"), ("Alice went to Paris.", "list of facts"), ("- x" * 8000, "under")],
)
def test_summary_text_is_validated(mgr, text, error):
    _trip_session(mgr)
    with pytest.raises(ValueError, match=error):
        summaries.save_summary(mgr, "p", summaries.Group("session", "s1"), text)


def test_bad_members_are_refused_before_anything_is_written(mgr):
    ids = _trip_session(mgr)
    other = _add(mgr, "Alice job interview at the bank", project="q").id
    s = summaries.save_summary(mgr, "p", summaries.Group("session", "s1"), SUMMARY)
    group = summaries.group_for(topic="t")
    for bad, error in (([other], "belongs to project"), ([s.id], "is a summary"), (["nope"], "no memory")):
        with pytest.raises((ValueError, LookupError), match=error):
            summaries.save_summary(mgr, "p", group, SUMMARY, ids + bad)
    assert summaries.live_summary(mgr, "p", group) is None


def test_group_for_needs_exactly_one():
    for kwargs in ({}, {"session_id": "s", "topic": "t"}, {"topic": "   "}):
        with pytest.raises(ValueError):
            summaries.group_for(**kwargs)


def test_summaries_never_go_through_add_or_reclassify(mgr):
    m = _add(mgr, "Alice booked a trip")
    with pytest.raises(ValueError, match="collivind_summarize"):
        mgr.add_memory(MemoryCreate(content="- x", summary="x", category=MemoryCategory.SUMMARY, project_id="p"))
    with pytest.raises(ValueError):
        mgr.update_memory(m.id, category=MemoryCategory.SUMMARY)
    s = summaries.save_summary(mgr, "p", summaries.Group("session", "s1"), SUMMARY)
    with pytest.raises(ValueError, match="reclassified"):
        mgr.update_memory(s.id, category=MemoryCategory.FACT)


def test_dedup_never_merges_a_summary_and_a_detail(mgr):
    _trip_session(mgr)
    s = summaries.save_summary(mgr, "p", summaries.Group("session", "s1"), SUMMARY)
    twin = mgr.add_memory(MemoryCreate(content=SUMMARY, summary="x", category=MemoryCategory.FACT, project_id="p"))
    assert twin.id != s.id
    assert summaries.is_live(mgr.get_memory(s.id)) and summaries.is_live(twin)


def test_summary_due_counts_only_unsummarized_session_memories(mgr):
    _trip_session(mgr)
    assert summaries.summary_due(mgr, "p", "s1", 4) is None
    due = summaries.summary_due(mgr, "p", "s1", 3)
    assert due["new_memories"] == 3 and "collivind_summarize" in due["instruction"]
    summaries.save_summary(mgr, "p", summaries.Group("session", "s1"), SUMMARY)
    assert summaries.summary_due(mgr, "p", "s1", 1) is None
    assert summaries.summary_due(mgr, "p", "s1", 0) is None
    assert summaries.summary_due(mgr, "p", None, 1) is None


# --- routed recall ------------------------------------------------------------


def _filler(mgr, n=12):
    for i in range(n):
        _add(mgr, f"Bob deploy helm postgres redis cache note {i}", session=f"f{i}")


def _ids(results):
    return [r.memory.id for r in results]


def test_no_summaries_means_identical_results(tmp_path):
    a, b = _manager(tmp_path / "a"), _manager(tmp_path / "b", summary_routes=0)
    for m in (a, b):
        _trip_session(m)
        _filler(m)
    for q in ("Where is Alice's trip?", "Bob deploy helm", "museum painting dog"):
        query = SearchQuery(query=q, project_id="p", limit=6)
        ra, rb = a.search(query), b.search(query)
        assert [r.memory.content for r in ra] == [r.memory.content for r in rb]
        assert [r.score for r in ra] == pytest.approx([r.score for r in rb], rel=1e-6)  # decay ticks


def test_a_summary_routes_to_its_most_relevant_members(mgr):
    ids = _trip_session(mgr)
    _filler(mgr)
    s = summaries.save_summary(mgr, "p", summaries.Group("session", "s1"), SUMMARY)
    results = mgr.search(SearchQuery(query="Will Alice bring her dog to Paris?", project_id="p", limit=4))
    assert results[0].memory.id == s.id and results[0].rerank_score is not None
    # the member that answers the question follows its summary
    assert ids[2] in _ids(results[1:3])
    assert len(results) <= 4 and len(set(_ids(results))) == len(results)


def test_routing_respects_filters_cutoff_and_switch(tmp_path):
    mgr = _manager(tmp_path / "s")
    _trip_session(mgr)
    _filler(mgr)
    s = summaries.save_summary(mgr, "p", summaries.Group("session", "s1"), SUMMARY)
    q = "Will Alice bring her dog to Paris?"
    assert s.id not in _ids(mgr.search(SearchQuery(query=q, project_id="p", limit=4, category="fact")))
    assert s.id not in _ids(mgr.search(SearchQuery(query=q, project_id="other", limit=4)))
    # a cutoff above every score routes nothing (and recalls nothing)
    assert mgr.search(SearchQuery(query=q, project_id="p", limit=4, min_rerank_score=0.9999999)) == []
    off = _manager(tmp_path / "s", summary_routes=0)
    plain = off.search(SearchQuery(query=q, project_id="p", limit=4))
    assert off.embedding_provider.rerank_calls == 1  # one rerank, no routing pass
    assert s.id in _ids(plain)  # still an ordinary, searchable result


def test_get_context_puts_the_summary_first_inside_the_budget(mgr):
    _trip_session(mgr)
    _filler(mgr)
    summaries.save_summary(mgr, "p", summaries.Group("session", "s1"), SUMMARY)
    text = mgr.get_context("Will Alice bring her dog to Paris?", project_id="p", limit=6, max_tokens=60)
    parts = text.split("\n\n")
    assert parts[0].splitlines()[1].startswith("[SUMMARY]")
    assert len(text) // 4 <= 60 + 10


# --- MCP and CLI -----------------------------------------------------------------


def test_mcp_read_then_write_and_batch_add_asks_for_it(mgr, monkeypatch):
    monkeypatch.setattr("collivind.capture.record", lambda *a, **k: None)
    mgr.config.hooks.summary_after = 3
    tools = CollivindTools(mgr, session_id="mcp-1", project_id="p")
    out = json.loads(
        tools.handle_call(
            "collivind_batch_add",
            {
                "memories": [
                    {"content": t, "summary": t, "category": "fact"} for t in ("Alice in Paris", "Alice and the dog")
                ]
            },
        )
    )
    assert "summary_due" not in out
    out = json.loads(
        tools.handle_call(
            "collivind_batch_add",
            {"memories": [{"content": "Alice at the museum", "summary": "c", "category": "fact"}]},
        )
    )
    assert out["summary_due"]["session_id"] == "mcp-1" and out["summary_due"]["new_memories"] == 3

    view = json.loads(tools.handle_call("collivind_summarize", {}))
    assert len(view["member_ids"]) == 3 and view["current_summary"] is None and "- " in view["instruction"]
    saved = json.loads(
        tools.handle_call("collivind_summarize", {"summary": "- Alice a, b and c.", "member_ids": view["member_ids"]})
    )
    assert saved["status"] == "success" and saved["members"] == 3 and saved["version"] == 1
    assert "error" in json.loads(tools.handle_call("collivind_summarize", {"summary": "no bullets"}))
    assert "error" in json.loads(tools.handle_call("collivind_summarize", {"topic": "x", "member_ids": ["nope"]}))
    schema = {t["name"]: t for t in tools.get_tool_list()}
    cats = schema["collivind_batch_add"]["inputSchema"]["properties"]["memories"]["items"]["properties"]["category"]
    assert "summary" not in cats["enum"] and "collivind_summarize" in schema


def test_cli_shows_members_and_saves(mgr, monkeypatch):
    from collivind.cli.commands import memory
    from collivind.cli.main import cli

    _trip_session(mgr)
    monkeypatch.setattr(memory, "_manager", lambda: mgr)
    runner = CliRunner()
    shown = runner.invoke(cli, ["summarize", "--session", "s1", "-p", "p"])
    assert shown.exit_code == 0 and "3 members" in shown.output and "Paris" in shown.output
    saved = runner.invoke(cli, ["summarize", "--session", "s1", "-p", "p", "--text", "-", "--json"], input=SUMMARY)
    assert saved.exit_code == 0 and json.loads(saved.output)["members"] == 3
    bad = runner.invoke(cli, ["summarize", "--session", "s1", "-p", "p", "--text", "prose"])
    assert bad.exit_code == 1 and "list of facts" in bad.output


def test_checkpoint_prompt_mentions_summary_due():
    from collivind import capture

    assert "summary_due" in capture.extraction_prompt("abc", True)


# --- round-trips -------------------------------------------------------------------


def _summary_state(mgr):
    s = summaries.live_summaries(mgr, "p")
    return [(x.content, x.version, sorted(m.content for m in summaries.members_of(mgr, [x.id])[x.id])) for x in s]


def test_bundle_and_migrate_round_trip_summaries(tmp_path):
    src = _manager(tmp_path / "src")
    _trip_session(src)
    group = summaries.Group("session", "s1")
    summaries.save_summary(src, "p", group, SUMMARY)
    _add(src, "Alice will adopt a cat after the trip")
    summaries.save_summary(src, "p", group, SUMMARY + "\n- cat")
    before = _summary_state(src)
    assert before and before[0][1] == 2

    stores = lambda m: Stores(m.vector_store, m.graph_store, m.embedding_provider)  # noqa: E731
    dst = _manager(tmp_path / "dst")
    assert migrate(stores(src), stores(dst), reembed=False, dry_run=False).ok
    assert _summary_state(dst) == before
    assert len(dst.get_version_chain(summaries.live_summary(dst, "p", group).id)) == 2

    buf = io.StringIO()
    space = ("test", None, WordEmbedder().dimension)
    export_bundle(stores(src), buf, space=space, project="p")
    imported = _manager(tmp_path / "imp")
    assert import_bundle(read_bundle(buf.getvalue().splitlines()), stores(imported), space=space, dry_run=False).ok
    assert _summary_state(imported) == before
    hits = imported.search(SearchQuery(query="Will Alice bring her dog to Paris?", project_id="p", limit=4))
    assert hits[0].memory.category == MemoryCategory.SUMMARY
    assert any(e.type == RelType.SUMMARIZES for e in imported.graph_store.iter_relationships())


class TruncatingEmbedder(WordEmbedder):
    """A reranker that reads only the start of each text, as the 256-token cross-encoder does."""

    def rerank(self, query, texts):
        return super().rerank(query, [t[:60] for t in texts])


def test_a_summary_is_scored_by_its_best_line_not_its_opening(tmp_path):
    mgr = _manager(tmp_path / "t")
    mgr.embedding_provider = mgr.search_engine.embedding_provider = TruncatingEmbedder()
    _trip_session(mgr)
    _filler(mgr)
    late_fact = "\n".join([f"- Bob note {i} about the bank job interview and the concert." for i in range(6)])
    text = late_fact + "\n- Alice brings her dog on the Paris trip."  # the answer is past the cut
    s = summaries.save_summary(mgr, "p", summaries.Group("session", "s1"), text)
    # a decoy whose opening line matches better than the true summary's opening
    _add(mgr, "Alice visited Paris once", session="s9")
    decoy = summaries.save_summary(mgr, "p", summaries.Group("session", "s9"), "- Alice visited Paris once, years ago.")
    hits = mgr.search(SearchQuery(query="Will Alice bring her dog to Paris?", project_id="p", limit=8))
    ranked = [h.memory.id for h in hits if h.memory.category == MemoryCategory.SUMMARY]
    assert ranked[:2] == [s.id, decoy.id]
