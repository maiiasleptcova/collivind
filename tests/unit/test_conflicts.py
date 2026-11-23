"""Conflict candidates on write, and `collivind contradictions review` on an embedded temp store."""

import json

import pytest
from click.testing import CliRunner

from collivind.cli.commands.contradictions import contradictions
from collivind.config import CollivindConfig, QdrantConfig
from collivind.engine.conflicts import (
    DUPLICATE,
    RESOLVED,
    REVIEW,
    UNRELATED,
    classify_edge,
    review_backlog,
    same_subject,
)
from collivind.engine.memory_manager import MemoryManager
from collivind.mcp.tools import CollivindTools
from collivind.models import MemoryCreate, MemoryNode, RelationshipCreate, RelType
from collivind.models.memory import MemoryCategory
from collivind.storage.graph_sqlite import SqliteGraphStore
from collivind.storage.vector_sqlite import SqliteVectorStore

# Cosine to BASE: NEAR 0.90 (conflict band), DUP 0.95 (dedup band), FAR 0.0.
BASE = [1.0, 0.0, 0.0, 0.0]
NEAR = [1.0, 0.484, 0.0, 0.0]
NEAR2 = [1.0, 0.0, 0.484, 0.0]
DUP = [1.0, 0.0, 0.0, 0.329]
FAR = [0.0, 0.0, 0.0, 1.0]


class KeyedEmbedder:
    """Vector chosen by a marker word in the text, so similarity is exact and readable."""

    def __init__(self, table):
        self.table = table

    def embed(self, text):
        for key, vec in self.table.items():
            if key in text:
                return vec
        return FAR

    def embed_query(self, text):
        return self.embed(text)

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def health_check(self):
        return {"status": "ok"}

    @property
    def dimension(self):
        return 4


@pytest.fixture
def manager(tmp_path):
    table = {"alpha": BASE, "bravo": NEAR, "charlie": NEAR2, "delta": DUP}
    path = str(tmp_path / "store")
    return MemoryManager(
        SqliteVectorStore(path, QdrantConfig(), 4), SqliteGraphStore(path), KeyedEmbedder(table), CollivindConfig()
    )


def _add(mgr, content, category=MemoryCategory.FACT):
    return mgr.add_memory(MemoryCreate(content=content, summary=content[:40], category=category, project_id="p"))


def _edges(mgr, rel_type):
    return [(e.source_id, e.target_id, e.source) for e in mgr.graph_store.iter_relationships() if e.type == rel_type]


# --- detector precision ---------------------------------------------------


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("PR 776 still has no reviews at head 5887f622", "PR 773 still has no reviews at head 87d113c1", False),
        ("PR 774 merged main at 399f581c", "PR 774 was rebased onto main c1f53a82", True),
        (
            "the review loop posts nothing while the head is unchanged",
            "review loop posts nothing on unchanged head",
            True,
        ),
        ("use git archive to export a clean tree", "contrast must be measured on rendered pixels", False),
        # dates are not identifiers: sharing one does not make two memories about the same thing
        ("2026-11-13 PR 825 approved", "2026-11-13 PR 824 approved", False),
    ],
)
def test_same_subject(a, b, expected):
    assert same_subject(a, b) is expected


def _node(content, **kw):
    return MemoryNode(content=content, summary="s", category=MemoryCategory.FACT, **kw)


@pytest.mark.parametrize(
    "a, b, score, bucket",
    [
        (_node("PR 1 x"), _node("PR 1 y", superseded_by="z"), 0.9, RESOLVED),
        (_node("PR 1 x"), None, 0.9, RESOLVED),
        (_node("PR 10 open"), _node("PR 10 merged"), 0.80, UNRELATED),
        (_node("PR 10 open"), _node("PR 11 open"), 0.90, UNRELATED),
        (_node("PR 10 open"), _node("PR 10 merged"), 0.95, DUPLICATE),
        (_node("PR 10 open"), _node("PR 10 merged"), 0.88, REVIEW),
        (_node("PR 10 open"), _node("PR 10 merged"), None, REVIEW),
    ],
)
def test_classify_edge(a, b, score, bucket):
    assert classify_edge(a, b, score, dedup_threshold=0.92)[0] == bucket


# --- write time -----------------------------------------------------------


def test_similar_memory_is_returned_not_linked(manager):
    old = _add(manager, "alpha: PR 42 uses Postgres 15")
    new, found = manager.add_memory_checked(
        MemoryCreate(
            content="bravo: PR 42 uses Postgres 16", summary="db", category=MemoryCategory.FACT, project_id="p"
        )
    )
    assert [c.memory.id for c in found] == [old.id]
    assert _edges(manager, RelType.CONTRADICTS) == []  # similarity alone never writes an edge
    assert manager.graph_store.get_memory(old.id).valid_to is None  # nor retires anything


def test_compatible_neighbours_are_not_candidates(manager):
    _add(manager, "alpha: PR 776 has no reviews yet")
    _add(manager, "charlie: something else entirely", category=MemoryCategory.DECISION)
    _, found = manager.add_memory_checked(
        MemoryCreate(
            content="bravo: PR 773 has no reviews yet", summary="s", category=MemoryCategory.FACT, project_id="p"
        )
    )
    assert found == []


def test_exact_duplicate_reports_no_conflicts(manager):
    _add(manager, "alpha: PR 42 uses Postgres 15")
    _, found = manager.add_memory_checked(
        MemoryCreate(
            content="alpha: PR 42 uses Postgres 15", summary="db", category=MemoryCategory.FACT, project_id="p"
        )
    )
    assert found == []


def test_add_memory_tool_returns_conflicts_and_instruction(manager):
    old = _add(manager, "alpha: PR 42 uses Postgres 15")
    tools = CollivindTools(manager)
    res = json.loads(
        tools.handle_call(
            "collivind_add_memory",
            {"content": "bravo: PR 42 uses Postgres 16", "summary": "db", "category": "fact", "project_id": "p"},
        )
    )
    assert res["status"] == "success"
    assert [c["id"] for c in res["possible_conflicts"][res["memory_id"]]] == [old.id]
    assert "collivind_invalidate" in res["conflict_instruction"]

    # the agent resolves it the way the instruction says; history is kept
    tools.handle_call("collivind_invalidate", {"memory_id": old.id, "superseded_by": res["memory_id"], "reason": "x"})
    assert manager.graph_store.get_memory(old.id).superseded_by == res["memory_id"]


def test_batch_add_tool_returns_conflicts_per_memory(manager):
    old = _add(manager, "alpha: PR 42 uses Postgres 15")
    res = json.loads(
        CollivindTools(manager).handle_call(
            "collivind_batch_add",
            {
                "memories": [
                    {"content": "bravo: PR 42 uses Postgres 16", "summary": "s", "category": "fact", "project_id": "p"},
                    {"content": "unrelated note", "summary": "s", "category": "fact", "project_id": "p"},
                ]
            },
        )
    )
    first, second = res["memory_ids"]
    assert [c["id"] for c in res["possible_conflicts"][first]] == [old.id]
    assert second not in res["possible_conflicts"]


def test_no_conflict_keys_when_nothing_is_close(manager):
    res = json.loads(
        CollivindTools(manager).handle_call(
            "collivind_add_memory", {"content": "unrelated", "summary": "s", "category": "fact", "project_id": "p"}
        )
    )
    assert "possible_conflicts" not in res and "conflict_instruction" not in res


# --- backlog review -------------------------------------------------------


@pytest.fixture
def backlog(manager):
    """Four auto-detected edges, one per bucket, plus one a person drew."""
    base = _add(manager, "alpha: PR 42 runs on Postgres 15")
    review = _add(manager, "bravo: PR 42 runs on Postgres 16")
    unrelated = _add(manager, "charlie: PR 99 has no reviews")
    retired = _add(manager, "PR 42 note that was replaced")
    manager.invalidate(retired.id, review.id, "replaced")
    # a near-duplicate can sit in the store if dedup was off or the threshold changed
    dup = manager.graph_store.create_memory(
        MemoryCreate(
            content="delta: PR 42 runs on Postgres 15 again", summary="d", category=MemoryCategory.FACT, project_id="p"
        )
    )
    manager.vector_store.upsert(dup.id, DUP, dup.to_dict())
    pairs = {
        REVIEW: (review.id, base.id),
        UNRELATED: (unrelated.id, base.id),
        RESOLVED: (review.id, retired.id),
        DUPLICATE: (dup.id, base.id),
    }
    for src, tgt in pairs.values():
        manager.graph_store.create_relationship(
            RelationshipCreate(src, tgt, RelType.CONTRADICTS, confidence=0.8, source="auto_detection")
        )
    manager.graph_store.create_relationship(RelationshipCreate(unrelated.id, review.id, RelType.CONTRADICTS))
    return manager, pairs, base, dup


def test_review_dry_run_changes_nothing(backlog):
    manager, pairs, _, _ = backlog
    before = sorted(_edges(manager, RelType.CONTRADICTS))
    verdicts = review_backlog(manager)
    assert {(v.source_id, v.target_id): v.bucket for v in verdicts} == {p: b for b, p in pairs.items()}
    assert sorted(_edges(manager, RelType.CONTRADICTS)) == before


def test_review_apply_relabels_and_invalidates(backlog):
    manager, pairs, base, dup = backlog
    review_backlog(manager, apply=True)

    remaining = _edges(manager, RelType.CONTRADICTS)
    assert sorted((s, t) for s, t, _ in remaining) == sorted([pairs[REVIEW], (pairs[UNRELATED][0], pairs[REVIEW][0])])
    relabelled = {(s, t) for s, t, src in _edges(manager, RelType.RELATES_TO) if src == "contradictions_review"}
    assert relabelled == {pairs[UNRELATED], pairs[RESOLVED]}
    # the older duplicate is retired, not deleted, and leaves the vector index
    assert manager.graph_store.get_memory(base.id).superseded_by == dup.id
    assert base.id not in {vid for vid, _, _ in manager.vector_store.iter_all()}


def test_review_cli_dry_run_and_apply(backlog, monkeypatch):
    manager, pairs, _, _ = backlog
    monkeypatch.setattr("collivind.cli.commands.contradictions._manager", lambda: manager)
    runner = CliRunner()

    out = runner.invoke(contradictions, ["review"])
    assert out.exit_code == 0, out.output
    assert "4 auto-detected" in out.output and "Dry run" in out.output
    assert len(_edges(manager, RelType.CONTRADICTS)) == 5

    out = runner.invoke(contradictions, ["review", "--json"])
    assert {v["bucket"] for v in json.loads(out.output)} == {RESOLVED, UNRELATED, DUPLICATE, REVIEW}

    out = runner.invoke(contradictions, ["review", "--apply"])
    assert out.exit_code == 0 and "Applied" in out.output
    assert len(_edges(manager, RelType.CONTRADICTS)) == 2


def test_changed_value_without_identifiers_is_a_candidate(manager):
    """The textbook contradiction: same statement, different number, no PR to anchor it."""
    old = _add(manager, "alpha: the request timeout is 30 seconds")
    _, found = manager.add_memory_checked(
        MemoryCreate(
            content="bravo: the request timeout is 60 seconds",
            summary="t",
            category=MemoryCategory.FACT,
            project_id="p",
        )
    )
    assert [c.memory.id for c in found] == [old.id]


def test_mark_conflict_tool_flags_the_pair(manager):
    a = _add(manager, "alpha: the request timeout is 30 seconds")
    b = _add(manager, "unrelated: the request timeout is 60 seconds")
    tools = CollivindTools(manager)
    res = json.loads(tools.handle_call("collivind_mark_conflict", {"memory_id": b.id, "other_id": a.id}))
    assert res["status"] == "success"
    assert (b.id, a.id, "manual") in _edges(manager, RelType.CONTRADICTS)
    missing = json.loads(tools.handle_call("collivind_mark_conflict", {"memory_id": b.id, "other_id": "nope"}))
    assert "error" in missing


def test_review_apply_keeps_version_chain_when_newer_side_repeats(manager):
    """One newer memory duplicating two older ones: only the first becomes its previous version."""
    newer = _add(manager, "delta: PR 7 status, newest")
    olders = []
    for i in range(2):
        m = manager.graph_store.create_memory(
            MemoryCreate(content=f"delta: PR 7 status {i}", summary="o", category=MemoryCategory.FACT, project_id="p")
        )
        manager.vector_store.upsert(m.id, BASE, m.to_dict())
        olders.append(m)
    # make the stored "newer" memory actually newer than both
    manager.graph_store.update_memory(newer.id, created_at="2999-01-01T00:00:00+00:00")
    for m in olders:
        manager.graph_store.create_relationship(
            RelationshipCreate(newer.id, m.id, RelType.CONTRADICTS, confidence=0.95, source="auto_detection")
        )
    assert {v.bucket for v in review_backlog(manager)} == {DUPLICATE}

    review_backlog(manager, apply=True)
    head = manager.graph_store.get_memory(newer.id)
    retired = [m.id for m in olders if manager.graph_store.get_memory(m.id).superseded_by == newer.id]
    assert retired == [head.previous_version_id]  # one link, not one overwritten by the next
