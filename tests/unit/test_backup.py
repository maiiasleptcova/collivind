"""`collivind export` / `import` bundle (v2): embedded store -> file -> fresh embedded store.

Real SQLite stores on both sides; only the config/backend lookup is patched.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner
from test_migrate import _stores, _vectors, populated  # noqa: F401 - fixture

from collivind.cli.commands.backup import export_cmd, import_cmd
from collivind.engine.bundle import _edge_dict, _entity_dict
from collivind.models import MemoryCategory, MemoryNode

SPACE = ("sentence-transformers", "model-a", 8)


@pytest.fixture(autouse=True)
def _tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def _cli(cmd, args, stores, space=SPACE):
    with (
        patch("collivind.cli.commands.backup._open", return_value=stores),
        patch("collivind.cli.commands.backup._space", return_value=space),
    ):
        return CliRunner().invoke(cmd, [str(a) for a in args])


def _full(stores):
    """Every field of every graph record, timestamps included."""
    return (
        sorted((m.to_dict() for m in stores.graph.iter_memories()), key=lambda d: d["id"]),
        sorted((_entity_dict(e) for e in stores.graph.iter_entities()), key=lambda d: d["id"]),
        sorted((_edge_dict(r) for r in stores.graph.iter_relationships()), key=lambda d: d["id"]),
    )


@pytest.fixture
def exported(populated, tmp_path):  # noqa: F811
    """populated source exported with vectors; returns (src snapshot, src vectors, ids, bundle path)."""
    src, ids = populated
    snapshot, vectors = _full(src), _vectors(src)
    path = tmp_path / "backup.jsonl"
    result = _cli(export_cmd, ["-o", path, "--vectors"], src)
    assert result.exit_code == 0, result.output
    return snapshot, vectors, ids, path


def test_round_trip_preserves_ids_links_edges_chains_and_vectors(exported, tmp_path):
    snapshot, src_vecs, ids, path = exported
    dst = _stores(tmp_path / "dst")

    result = _cli(import_cmd, [path], dst)

    assert result.exit_code == 0, result.output
    dst = _stores(tmp_path / "dst")  # the CLI closed its handles
    mem_d, ent_d, rel_d = _full(dst)
    mem_s, ent_s, rel_s = snapshot
    assert mem_d == mem_s and ent_d == ent_s
    assert rel_d == [r for r in rel_s if r["target_id"] != "gone"]  # the dangling edge never left
    assert [m.id for m in dst.graph.get_version_chain(ids["new"])] == [ids["old"], ids["new"]]
    assert {m.id for m in dst.graph.find_related_memories("Kafka")} == {ids["new"]}

    dst_vecs = _vectors(dst)
    assert set(dst_vecs) == set(src_vecs) == {ids["new"], ids["other"]}
    for vid, (vec, payload) in src_vecs.items():
        np.testing.assert_allclose(dst_vecs[vid][0], vec, atol=1e-6)
        assert dst_vecs[vid][1] == payload
    assert "memories             3       3        0" in result.output


def test_bundle_header_is_an_allow_list(exported):
    *_, path = exported
    header = json.loads(path.read_text().splitlines()[0])
    assert set(header) == {
        "format",
        "format_version",
        "collivind_version",
        "exported_at",
        "project",
        "embedding",
        "vectors_included",
        "counts",
    }
    assert header["counts"] == {"memory": 3, "entity": 4, "relationship": 5, "vector": 2}
    assert header["embedding"] == {"family": "sentence-transformers", "model": "model-a", "dimension": 8}


def test_without_vectors_reembeds_only_what_was_searchable(populated, tmp_path):  # noqa: F811
    src, ids = populated
    path = tmp_path / "backup.jsonl"
    assert _cli(export_cmd, ["-o", path], src).exit_code == 0
    assert all(json.loads(line).get("vector") is None for line in path.read_text().splitlines()[1:])
    dst = _stores(tmp_path / "dst")

    result = _cli(import_cmd, [path], dst)

    assert result.exit_code == 0, result.output
    assert "Re-embedding" in result.output
    assert len(dst.embedder.texts) == 2  # the superseded version stays out of search
    assert set(_vectors(_stores(tmp_path / "dst"))) == {ids["new"], ids["other"]}


def test_other_embedding_space_reembeds(exported, tmp_path):
    *_, path = exported
    dst = _stores(tmp_path / "dst")
    result = _cli(import_cmd, [path], dst, space=("openai", "text-embedding-3-small", 8))
    assert result.exit_code == 0, result.output
    assert len(dst.embedder.texts) == 2


def test_reimport_is_idempotent_and_never_reverts_edits(exported, tmp_path):
    *_, ids, path = exported
    assert _cli(import_cmd, [path], _stores(tmp_path / "dst")).exit_code == 0
    dst = _stores(tmp_path / "dst")
    edited = dst.graph.get_memory(ids["other"])
    dst.graph.restore_memory(
        MemoryNode(**{**edited.__dict__, "content": "billing moved to adyen", "category": MemoryCategory.FACT})
    )
    before = _full(dst)

    result = _cli(import_cmd, [path], dst)

    assert result.exit_code == 0, result.output
    after = _stores(tmp_path / "dst")
    assert _full(after) == before
    assert after.graph.get_memory(ids["other"]).content == "billing moved to adyen"


def test_project_filter_keeps_linked_entities_and_drops_cross_project_edges(populated, tmp_path):  # noqa: F811
    src, ids = populated
    path = tmp_path / "beta.jsonl"

    result = _cli(export_cmd, ["-o", path, "-p", "beta", "--vectors"], src)

    assert result.exit_code == 0, result.output
    assert "Left out 2 edge(s)" in result.output  # CONTRADICTS into alpha + the dangling one
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["project"] == "beta"
    kinds = {}
    for rec in lines[1:]:
        kinds.setdefault(rec["kind"], []).append(rec)
    assert [m["id"] for m in kinds["memory"]] == [ids["other"]]
    assert [e["id"] for e in kinds["entity"]] == ["stripe"]
    assert [(r["source_id"], r["target_id"]) for r in kinds["relationship"]] == [(ids["other"], "stripe")]
    assert [v["id"] for v in kinds["vector"]] == [ids["other"]]


def test_dry_run_writes_nothing(exported, tmp_path):
    *_, path = exported
    dst = _stores(tmp_path / "dst")

    result = _cli(import_cmd, [path, "--dry-run"], dst)

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    dst = _stores(tmp_path / "dst")
    assert _full(dst) == ([], [], []) and _vectors(dst) == {}


@pytest.mark.parametrize(
    "damage",
    [
        lambda lines: lines[:-1],  # cut at a line boundary: still valid JSONL
        lambda lines: lines[:-1] + [lines[-1][: len(lines[-1]) // 2]],  # cut mid-line
        lambda lines: [lines[0].replace('"format_version": 2', '"format_version": 3')] + lines[1:],
        lambda lines: lines[:3] + ['{"kind": "memory", "content": "no id"}'] + lines[4:],
    ],
    ids=["truncated-at-line", "truncated-mid-line", "newer-format", "bad-record"],
)
def test_damaged_bundle_fails_before_any_write(exported, tmp_path, damage):
    *_, path = exported
    path.write_text("\n".join(damage(path.read_text().splitlines())) + "\n")
    dst = _stores(tmp_path / "dst")

    result = _cli(import_cmd, [path], dst)

    assert result.exit_code == 1
    assert "nothing written" in result.output
    dst = _stores(tmp_path / "dst")
    assert _full(dst) == ([], [], []) and _vectors(dst) == {}


def test_missing_record_after_import_exits_nonzero(exported, tmp_path):
    *_, path = exported
    dst = _stores(tmp_path / "dst")
    with patch.object(dst.graph, "restore_entity", lambda node: None):
        result = _cli(import_cmd, [path], dst)
    assert result.exit_code == 1
    assert "Verification failed" in result.output
