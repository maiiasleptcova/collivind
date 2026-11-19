"""`collivind migrate`: embedded -> embedded (second data dir) round trips.

Real SQLite stores on both sides, so ids, edges, version chains and vectors
are checked against what is actually persisted.
"""

import hashlib
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from collivind.cli.commands.migrate import migrate_cmd, mode_config
from collivind.config import CollivindConfig, QdrantConfig
from collivind.engine.memory_manager import MemoryManager
from collivind.engine.migrate import Stores, embedding_space, migrate
from collivind.models import EntityCreate, MemoryCreate, RelationshipCreate, RelType
from collivind.models.entity import EntityType
from collivind.models.memory import MemoryCategory
from collivind.storage.graph_sqlite import SqliteGraphStore
from collivind.storage.vector_sqlite import SqliteVectorStore

DIM = 8


class HashEmbedder:
    """Deterministic, distinct-per-text vectors; records what it embedded."""

    def __init__(self, salt: str = ""):
        self.salt = salt
        self.texts = []

    def embed(self, text):
        self.texts.append(text)
        digest = hashlib.sha256((self.salt + text).encode()).digest()
        return [b / 255.0 + 0.01 for b in digest[:DIM]]

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
    """The CLI reads ~/.collivind/config.toml; never the developer's real one."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("COLLIVIND_MODE", raising=False)


def _stores(path, salt=""):
    return Stores(
        vector=SqliteVectorStore(str(path), QdrantConfig(), DIM),
        graph=SqliteGraphStore(str(path)),
        embedder=HashEmbedder(salt),
    )


def _graph_snapshot(stores):
    return (
        sorted((m.to_dict() for m in stores.graph.iter_memories()), key=lambda d: d["id"]),
        sorted(((e.id, e.name, e.type, e.properties) for e in stores.graph.iter_entities())),
        sorted(
            (r.id, r.source_id, r.target_id, r.type.value, r.confidence, r.source, str(r.properties))
            for r in stores.graph.iter_relationships()
        ),
    )


def _vectors(stores):
    return {vid: (np.array(vec), payload) for vid, vec, payload in stores.vector.iter_all()}


@pytest.fixture
def populated(tmp_path):
    """Two projects, entity links, a contradiction, a dangling edge and a
    superseded version whose vector was retired as add_memory does."""
    src = _stores(tmp_path / "src")
    mgr = MemoryManager(src.vector, src.graph, src.embedder, CollivindConfig())
    mgr.search_engine.find_contradictions = lambda memory: []  # hash vectors make auto-detection arbitrary

    def add(content, project="alpha", entities=()):
        return mgr.add_memory(
            MemoryCreate(content=content, summary=content[:20], category=MemoryCategory.DECISION, project_id=project),
            entities=[EntityCreate(name=n, type=EntityType.SERVICE, properties={"k": "v"}) for n in entities],
        )

    old = add("we use postgres for the event log", entities=["Postgres"])
    new = add("we moved the event log to kafka", entities=["Kafka", "data-integrity"])
    mgr.invalidate(old.id, new.id, "replaced")
    src.vector.delete(old.id)  # history stays in the graph only
    other = add("billing runs on stripe", project="beta", entities=["Stripe"])
    src.graph.create_relationship(
        RelationshipCreate(source_id=other.id, target_id=new.id, type=RelType.CONTRADICTS, confidence=0.4)
    )
    src.graph.create_relationship(RelationshipCreate(source_id=other.id, target_id="gone", type=RelType.RELATES_TO))
    return src, {"old": old.id, "new": new.id, "other": other.id}


def test_round_trip_preserves_ids_edges_chains_and_vectors(populated, tmp_path):
    src, ids = populated
    before = (_graph_snapshot(src), _vectors(src))
    dst = _stores(tmp_path / "dst")

    report = migrate(src, dst, reembed=False, dry_run=False)

    assert report.ok
    counts = {c.kind: (c.source, c.target, c.missing) for c in report.counts}
    assert counts == {
        "memories": (3, 3, 0),
        "entities": (4, 4, 0),
        "relationships": (5, 5, 0),  # 4 ABOUT + 1 CONTRADICTS; the dangling one is reported, not copied
        "vectors": (2, 2, 0),
    }
    assert [(e.source_id, e.target_id) for e in report.dangling] == [(ids["other"], "gone")]

    mem_s, ent_s, rel_s = _graph_snapshot(src)
    mem_d, ent_d, rel_d = _graph_snapshot(dst)
    assert mem_d == mem_s  # every field verbatim, timestamps included
    assert ent_d == ent_s
    assert set(rel_d) == {r for r in rel_s if r[2] != "gone"}

    chain = [m.id for m in dst.graph.get_version_chain(ids["new"])]
    assert chain == [ids["old"], ids["new"]]
    assert dst.graph.get_memory(ids["old"]).superseded_by == ids["new"]
    assert {m.id for m in dst.graph.find_related_memories("Kafka")} == {ids["new"]}

    src_vecs, dst_vecs = _vectors(src), _vectors(dst)
    assert set(dst_vecs) == {ids["new"], ids["other"]}  # superseded stays out of search
    for vid, (vec, payload) in src_vecs.items():
        np.testing.assert_allclose(dst_vecs[vid][0], vec, atol=1e-6)
        assert dst_vecs[vid][1] == payload

    assert (_graph_snapshot(src), _vectors(src).keys()) == (before[0], before[1].keys())  # source untouched


def test_rerun_is_idempotent(populated, tmp_path):
    src, _ = populated
    dst = _stores(tmp_path / "dst")
    migrate(src, dst, reembed=False, dry_run=False)
    first = (_graph_snapshot(dst), set(_vectors(dst)))

    report = migrate(src, dst, reembed=False, dry_run=False)

    assert report.ok
    assert (_graph_snapshot(dst), set(_vectors(dst))) == first


def test_dry_run_writes_nothing(populated, tmp_path):
    src, _ = populated
    dst = _stores(tmp_path / "dst")

    report = migrate(src, dst, reembed=False, dry_run=True)

    assert _graph_snapshot(dst) == ([], [], [])
    assert _vectors(dst) == {}
    assert {c.kind: (c.target, c.missing) for c in report.counts}["memories"] == (0, 3)
    assert not report.ok  # a dry run reports what is still missing


def test_reembed_uses_the_target_provider_with_the_enriched_text(populated, tmp_path):
    src, ids = populated
    dst = _stores(tmp_path / "dst", salt="other-model")

    migrate(src, dst, reembed=True, dry_run=False)

    dst_vecs = _vectors(dst)
    assert set(dst_vecs) == {ids["new"], ids["other"]}
    assert len(dst.embedder.texts) == 2
    kafka_text = next(t for t in dst.embedder.texts if "kafka" in t)
    assert "Kafka" in kafka_text and "data-integrity" in kafka_text  # entity names kept
    expected = np.array(dst.embedder.embed(kafka_text))
    np.testing.assert_allclose(dst_vecs[ids["new"]][0], expected / np.linalg.norm(expected), atol=1e-6)


def test_missing_target_record_fails_verification(populated, tmp_path):
    src, _ = populated
    dst = _stores(tmp_path / "dst")
    with patch.object(dst.graph, "restore_entity", lambda node: None):
        report = migrate(src, dst, reembed=False, dry_run=False)
    assert not report.ok
    assert {c.kind: c.missing for c in report.counts}["entities"] == 4


def test_embedding_space_treats_local_and_docker_http_as_one_model():
    assert embedding_space("local", "BAAI/bge-small-en-v1.5", 384) == embedding_space(
        "http", "BAAI/bge-small-en-v1.5", 384
    )
    assert embedding_space("local", "m", 384) != embedding_space("openai", "m", 384)
    assert embedding_space("local", "m", 384) != embedding_space("local", "m", 768)


def test_mode_config_clears_provider_overrides():
    base = CollivindConfig()
    base.neo4j.provider = "sqlite"
    cfg = mode_config(base, "docker")
    assert (cfg.mode, cfg.neo4j.provider, base.neo4j.provider) == ("docker", "", "sqlite")


def test_cli_refuses_same_mode():
    result = CliRunner().invoke(migrate_cmd, ["--from", "embedded", "--to", "embedded"])
    assert result.exit_code == 1
    assert "nothing to migrate" in result.output


def test_cli_rejects_remote_which_shares_dockers_stores():
    result = CliRunner().invoke(migrate_cmd, ["--from", "docker", "--to", "remote"])
    assert result.exit_code == 2
    assert "remote" in result.output


def test_cli_unreachable_target_writes_nothing(populated):
    src, _ = populated

    def fake_backends(config):
        if config.mode == "docker":
            raise RuntimeError("Connection refused")
        return src.vector, src.graph, src.embedder

    with patch("collivind.cli.commands.migrate.create_all_backends", fake_backends):
        result = CliRunner().invoke(migrate_cmd, ["--from", "embedded", "--to", "docker"])
    assert result.exit_code == 1
    assert "not reachable; nothing was written" in result.output


def test_cli_prints_counts_and_exits_zero(populated, tmp_path):
    src, _ = populated
    dst = _stores(tmp_path / "dst")
    by_mode = {"embedded": src, "docker": dst}

    with patch(
        "collivind.cli.commands.migrate.create_all_backends",
        lambda config: (by_mode[config.mode].vector, by_mode[config.mode].graph, by_mode[config.mode].embedder),
    ):
        result = CliRunner().invoke(migrate_cmd, ["--from", "embedded", "--to", "docker"])
    assert result.exit_code == 0, result.output
    assert "Vectors copied as stored" in result.output
    assert "skipped dangling edge" in result.output
    assert "memories             3       3        0" in result.output


def test_rerun_does_not_revert_a_target_edit(populated, tmp_path):
    """Review r1 #4: a re-run used to overwrite target records, undoing edits made there."""
    src, ids = populated
    dst = _stores(tmp_path / "dst")
    migrate(src, dst, reembed=False, dry_run=False)
    edited = next(m for m in dst.graph.iter_memories() if m.id == ids["other"])
    dst.graph.restore_memory(type(edited).from_dict({**edited.to_dict(), "content": "edited on the target"}))

    report = migrate(src, dst, reembed=False, dry_run=False)
    assert report.ok
    after = next(m for m in dst.graph.iter_memories() if m.id == ids["other"])
    assert after.content == "edited on the target"


def test_docker_service_model_is_not_taken_from_config():
    """Review r1 #5: MiniLM (384d) on embedded is not the space the bge docker service writes."""
    assert embedding_space("local", "all-MiniLM-L6-v2", 384) != embedding_space("http", "all-MiniLM-L6-v2", 384)
    assert embedding_space("local", "BAAI/bge-small-en-v1.5", 384) == embedding_space("http", "anything", 384)
