"""Write-time dedup on a real SQLite store: only a resubmission is rejected, nothing is retired on similarity."""

import json
from pathlib import Path

import pytest

from collivind.config import CollivindConfig, QdrantConfig
from collivind.engine.memory_manager import MemoryManager
from collivind.models import MemoryCategory, MemoryCreate, RelationshipCreate, RelType
from collivind.storage.graph_sqlite import SqliteGraphStore
from collivind.storage.vector_sqlite import SqliteVectorStore

PAIRS = json.loads((Path(__file__).parent.parent / "fixtures" / "dedup_pairs.json").read_text())

# cosine to BASE: DUP 0.95 (above the 0.92 dedup threshold), MID 0.80, FAR 0.0
BASE = [1.0, 0.0, 0.0, 0.0]
DUP = [1.0, 0.329, 0.0, 0.0]
MID = [1.0, 0.0, 0.75, 0.0]
FAR = [0.0, 0.0, 0.0, 1.0]


class KeyedEmbedder:
    """Vector chosen by the first marker word found in the text; FAR when none."""

    def __init__(self, table):
        self.table = table

    def embed(self, text):
        return next((vec for key, vec in self.table.items() if key in text), FAR)

    def embed_query(self, text):
        return self.embed(text)

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def health_check(self):
        return {"status": "ok"}

    @property
    def dimension(self):
        return 4


def _manager(tmp_path, embedder, dim=4):
    path = str(tmp_path / "store")
    return MemoryManager(
        SqliteVectorStore(path, QdrantConfig(), dim), SqliteGraphStore(path), embedder, CollivindConfig()
    )


@pytest.fixture
def manager(tmp_path):
    return _manager(tmp_path, KeyedEmbedder({"alpha": BASE, "bravo": DUP, "mike": MID, "sumsame": BASE}))


def _add(mgr, content, summary=None, category=MemoryCategory.FACT, relationships=None):
    return mgr.add_memory_checked(
        MemoryCreate(content=content, summary=summary or content, category=category, project_id="p"),
        relationships=relationships,
    )


def _live_ids(mgr):
    return {i for i, _, _ in mgr.vector_store.iter_all()}


def test_resubmission_is_rejected_and_keeps_its_links(manager):
    old, _ = _add(manager, "alpha: PR 42 uses Postgres 15")
    other, _ = _add(manager, "unrelated note")
    again, found = _add(
        manager,
        "  alpha: PR 42 uses Postgres 15\n",
        relationships=[RelationshipCreate(source_id="", target_id=other.id, type=RelType.RELATES_TO)],
    )
    assert again.id == old.id and found == []
    assert _live_ids(manager) == {old.id, other.id}
    assert any(e.target_id == other.id for e in manager.graph_store.iter_relationships() if e.source_id == old.id)


@pytest.mark.parametrize(
    "content, tags",
    [
        ("alpha: PR 42 uses postgres 15", []),  # case is meaningful in code
        ("alpha: PR 42  uses Postgres 15", []),  # so is inner whitespace
        ("alpha: PR 42 uses Postgres 15", ["pr-43"]),  # same line, another subject
    ],
)
def test_almost_identical_is_kept_and_surfaced(manager, content, tags):
    old, _ = _add(manager, "alpha: PR 42 uses Postgres 15")
    new, found = manager.add_memory_checked(
        MemoryCreate(content=content, summary=content, category=MemoryCategory.FACT, project_id="p", tags=tags)
    )
    assert new.id != old.id
    assert old.id in [c.memory.id for c in found]


def test_stale_payload_does_not_reject_a_new_fact(manager):
    """The vector payload says A but the memory was edited to B: saving A again is a new fact."""
    old, _ = _add(manager, "alpha: timeout is 30s")
    manager.graph_store.update_memory(old.id, content="alpha: timeout is 60s", summary="alpha: timeout is 60s")
    new, _ = _add(manager, "alpha: timeout is 30s")
    assert new.id != old.id


def test_resaving_a_retired_memory_stores_it_again(manager):
    old, _ = _add(manager, "alpha: PR 42 uses Postgres 15")
    newer, _ = _add(manager, "bravo: PR 42 uses Postgres 16")
    manager.graph_store.invalidate_memory(old.id, newer.id, "graph only: the vector delete never ran")
    again, _ = _add(manager, "alpha: PR 42 uses Postgres 15")
    assert again.id not in (old.id, newer.id)


def test_near_duplicate_is_kept_and_surfaced_not_retired(manager):
    """Changed: this used to retire the older memory (cosine 0.95 >= 0.92)."""
    old, _ = _add(manager, "alpha: PR 42 uses Postgres 15")
    new, found = _add(manager, "bravo: PR 42 uses Postgres 16")
    assert new.id != old.id
    assert [c.memory.id for c in found] == [old.id]
    assert manager.graph_store.get_memory(old.id).superseded_by is None
    assert _live_ids(manager) == {old.id, new.id}


def test_likely_duplicate_in_another_category_is_surfaced(manager):
    old, _ = _add(manager, "alpha: snapshot before the qdrant 1.19 upgrade", category=MemoryCategory.FACT)
    _, found = _add(manager, "bravo: snapshot before the qdrant 1.19 upgrade, always", category=MemoryCategory.PATTERN)
    assert [c.memory.id for c in found] == [old.id]


def test_memory_the_old_rule_would_have_merged_is_always_surfaced(manager):
    """Within dedup_threshold the agent is told even when the PR numbers differ, and nothing is retired."""
    old, _ = _add(manager, "alpha: PR 776 has no reviews yet")
    new, found = _add(manager, "bravo: PR 773 has no reviews yet")
    assert [c.memory.id for c in found] == [old.id]
    assert _live_ids(manager) == {old.id, new.id}


def test_close_memory_about_another_pr_is_not_surfaced_below_the_threshold(manager):
    _add(manager, "alpha: PR 776 has no reviews yet")
    _, found = _add(manager, "mike: PR 773 has no reviews yet")  # cosine 0.80
    assert found == []


def test_same_fact_with_different_detail_is_surfaced_by_summary(manager):
    # content vectors 0.80 apart (below the dedup threshold), summaries identical in meaning
    old, _ = _add(manager, "alpha: the Colima VM disk is 80 GiB", summary="sumsame Colima disk is 80 GiB")
    _, found = _add(
        manager,
        "mike: while debugging docker exec, the Colima VM disk turned out to be 80 GiB",
        summary="sumsame Colima VM disk 80 GiB",
        category=MemoryCategory.PATTERN,  # not a same-category conflict candidate
    )
    assert [c.memory.id for c in found] == [old.id]


def test_agent_confirmed_update_retires_stale_memory_and_keeps_the_chain(manager):
    old, _ = _add(manager, "alpha: PR 42 uses Postgres 15")
    new, found = _add(manager, "bravo: PR 42 uses Postgres 16")
    assert [c.memory.id for c in found] == [old.id]
    manager.invalidate(old.id, new.id, "Postgres upgraded")
    assert [m.id for m in manager.get_version_chain(new.id)] == [old.id, new.id]  # oldest first
    assert _live_ids(manager) == {new.id}


class SameVector:
    """Every text gets the same vector: the worst case, where similarity says everything is a duplicate."""

    def embed(self, text):
        return [1.0, 0.0]

    embed_query = embed

    def embed_batch(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def health_check(self):
        return {"status": "ok"}

    @property
    def dimension(self):
        return 2


def test_similarity_alone_never_loses_a_pair_memory(tmp_path):
    """Every fixture pair at cosine 1.0: only identical text may be merged, so nothing is lost."""
    manager = _manager(tmp_path, SameVector(), dim=2)
    for pair in PAIRS:
        a, _ = manager.add_memory_checked(_create(pair["a"], pair["id"]))
        b, _ = manager.add_memory_checked(_create(pair["b"], pair["id"]))
        assert b.id != a.id, pair["id"]
    assert len(_live_ids(manager)) == 2 * len(PAIRS)


def _create(mem, project):
    return MemoryCreate(
        content=mem["content"],
        summary=mem["summary"],
        category=MemoryCategory(mem["category"]),
        project_id=project,
        tags=mem["tags"],
    )


def test_fixture_pairs_with_the_real_model(tmp_path):
    """The synthetic coding-agent pairs through the real embedder (needs the embedded extra)."""
    pytest.importorskip("sentence_transformers")
    from collivind.config import EmbeddingsConfig
    from collivind.storage.embedding_local import LocalEmbeddingProvider

    embedder = LocalEmbeddingProvider(
        EmbeddingsConfig(provider="local", model="BAAI/bge-small-en-v1.5", daemon=False), data_dir=str(tmp_path)
    )
    manager = _manager(tmp_path, embedder, dim=embedder.dimension)
    missed, lost = [], []
    for pair in PAIRS:
        a, _ = manager.add_memory_checked(_create(pair["a"], pair["id"]))
        b, found = manager.add_memory_checked(_create(pair["b"], pair["id"]))
        if pair["kind"] == "different" and (b.id == a.id or a.id not in _live_ids(manager)):
            lost.append(pair["id"])
        if pair["kind"] == "same" and b.id != a.id and a.id not in {c.memory.id for c in found}:
            missed.append(pair["id"])
    same = sum(p["kind"] == "same" for p in PAIRS)
    assert lost == []
    assert len(missed) <= 0.2 * same, missed


def test_superseded_report_lists_same_call_retirements_only(manager, monkeypatch):
    from click.testing import CliRunner

    from collivind.cli.commands.contradictions import contradictions
    from collivind.engine.conflicts import auto_superseded

    # what the pre-0.22 add path did: create the successor, retire the old one in the same call
    auto_old, _ = _add(manager, "alpha: PR 1 note")
    auto_new, _ = _add(manager, "unrelated: PR 1 note, newer")
    manager.invalidate(auto_old.id, auto_new.id, "superseded_by_update")
    # an agent retiring a memory well after the successor was stored
    agent_old, _ = _add(manager, "unrelated: PR 2 note")
    agent_new, _ = _add(manager, "unrelated: PR 2 note, newer")
    manager.graph_store.update_memory(agent_new.id, created_at="2020-01-01T00:00:00+00:00")
    manager.invalidate(agent_old.id, agent_new.id, "agent")

    assert [(o.id, n.id) for o, n in auto_superseded(manager.graph_store)] == [(auto_old.id, auto_new.id)]
    monkeypatch.setattr("collivind.cli.commands.contradictions._manager", lambda: manager)
    before = {m.id: m.superseded_by for m in manager.graph_store.iter_memories()}
    out = CliRunner().invoke(contradictions, ["superseded", "--json"])
    assert out.exit_code == 0, out.output
    assert [r["retired"] for r in json.loads(out.output)] == [auto_old.id]
    assert {m.id: m.superseded_by for m in manager.graph_store.iter_memories()} == before  # read only
