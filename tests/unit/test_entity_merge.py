"""Entity merge on a real SQLite graph store: the whole edge set is checked, not samples."""

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from entity_merge_scenario import GONE, KEEP, OTHERS, build, edge_set, expected

from collivind.cli.main import cli
from collivind.config import CollivindConfig
from collivind.engine.bundle import _edge, _entity
from collivind.engine.entity_merge import apply_merge, plan_merge
from collivind.engine.memory_manager import MemoryManager
from collivind.engine.projects import set_project_meta
from collivind.models import EntityCreate, EntityType
from collivind.storage.graph_sqlite import SqliteGraphStore


@pytest.fixture
def graph(tmp_path):
    return SqliteGraphStore(str(tmp_path / "graph"))


@pytest.fixture
def mem(graph):
    return build(graph)


def _entity_ids(graph):
    return {e.id for e in graph.iter_entities()}


def test_merge_repoints_every_edge_and_drops_loops_and_duplicates(graph, mem, tmp_path):
    done = apply_merge(graph, plan_merge(graph, KEEP, OTHERS), tmp_path / "b.jsonl")
    assert edge_set(graph) == sorted(expected(mem))
    assert not GONE & _entity_ids(graph)
    assert graph.get_entity(KEEP).properties["aliases"] == list(OTHERS)
    assert done["entities_removed"] == 2 and done["edges_removed"] == 9
    assert done["self_loops_dropped"] == 2 and done["duplicates_dropped"] == 3


def test_a_moved_edge_keeps_its_note_confidence_and_source(graph, mem, tmp_path):
    apply_merge(graph, plan_merge(graph, KEEP, OTHERS), tmp_path / "b.jsonl")
    (uses,) = [e for e in graph.edges_of("flask")]
    assert (uses.source_id, uses.properties, uses.confidence, uses.source) == (
        "cybergine",
        {"note": "web framework"},
        0.7,
        "extraction",
    )


def test_dry_run_writes_nothing(graph, mem):
    before = (edge_set(graph), _entity_ids(graph))
    plan = plan_merge(graph, KEEP, OTHERS)
    assert plan.counts()["entity_edges_moved"] == 2
    assert (edge_set(graph), _entity_ids(graph)) == before


def test_backup_holds_the_pre_merge_state_and_undoes_the_merge(graph, mem, tmp_path):
    before = (edge_set(graph), _entity_ids(graph))
    backup = tmp_path / "b.jsonl"
    apply_merge(graph, plan_merge(graph, KEEP, OTHERS), backup)
    lines = [json.loads(line) for line in backup.read_text().splitlines()]
    assert lines[0]["kind"] == "entity_merge"
    for rec in lines[1:]:  # undo: drop what the merge created, put back what it removed
        if rec["kind"] == "created":
            graph.delete_relationship(rec["source_id"], rec["target_id"], rec["type"])
    for rec in lines[1:]:
        if rec.get("kind") == "entity":
            graph.restore_entity(_entity({k: v for k, v in rec.items() if k != "kind"}))
    for rec in lines[1:]:
        if rec.get("kind") == "relationship":
            graph.restore_relationship(_edge({k: v for k, v in rec.items() if k != "kind"}))
    assert (edge_set(graph), _entity_ids(graph)) == before
    assert "aliases" not in graph.get_entity(KEEP).properties


def test_a_second_backup_never_overwrites_the_first(graph, mem, tmp_path):
    backup = tmp_path / "b.jsonl"
    backup.write_text("earlier\n")
    with pytest.raises(FileExistsError):
        apply_merge(graph, plan_merge(graph, KEEP, OTHERS), backup)
    assert backup.read_text() == "earlier\n"
    assert "v5_flask" in _entity_ids(graph)


def test_an_interrupted_merge_finishes_when_run_again(graph, mem, tmp_path):
    real_delete, calls = graph.delete_relationship, []

    def flaky(*args):
        calls.append(args)
        if len(calls) == 3:
            raise RuntimeError("connection lost")
        return real_delete(*args)

    with patch.object(graph, "delete_relationship", flaky), pytest.raises(RuntimeError):
        apply_merge(graph, plan_merge(graph, KEEP, OTHERS), tmp_path / "1.jsonl")
    apply_merge(graph, plan_merge(graph, KEEP, OTHERS), tmp_path / "2.jsonl")
    assert edge_set(graph) == sorted(expected(mem))
    assert not GONE & _entity_ids(graph)
    with pytest.raises(ValueError, match="nothing to merge"):  # a third run: all done
        plan_merge(graph, KEEP, OTHERS)


def test_merge_refuses_bad_targets(graph, mem):
    with pytest.raises(LookupError):
        plan_merge(graph, KEEP, ["nope"])
    with pytest.raises(ValueError, match="being kept"):
        plan_merge(graph, KEEP, ["cybergine"])
    set_project_meta(graph, "v5_flask", {"description": "x"})
    with pytest.raises(ValueError, match="project metadata"):
        plan_merge(graph, KEEP, ["project:v5_flask"])


def test_delete_entity_refuses_while_edges_remain(graph, mem):
    with pytest.raises(ValueError, match="still has"):
        graph.delete_entity("v5_flask")
    assert "v5_flask" in _entity_ids(graph)


def test_after_a_merge_a_save_naming_the_old_name_lands_on_the_kept_entity(graph, mem, tmp_path):
    vectors, embedder = MagicMock(), MagicMock()
    vectors.search.return_value = []
    embedder.embed.return_value = [0.1]
    manager = MemoryManager(vectors, graph, embedder, CollivindConfig())
    manager.canonical_entity_name("warm the alias cache before the merge")
    manager.merge_entities(manager.plan_entity_merge(KEEP, OTHERS), tmp_path / "b.jsonl")
    manager.batch_add_memories(
        [
            {
                "content": "v5_flask deploys to Cloud Run",
                "summary": "deploy",
                "category": "fact",
                "entities": [{"name": "v5_flask", "type": "project"}],
            }
        ]
    )
    assert not GONE & _entity_ids(graph)
    assert len(graph.find_related_memories("cybergine", limit=50)) == 4


def test_cli_merge_is_a_dry_run_until_apply(graph, mem, tmp_path):
    manager = MemoryManager(MagicMock(), graph, MagicMock(), CollivindConfig(data_dir=str(tmp_path)))
    before = edge_set(graph)
    with patch("collivind.cli.commands.memory._manager", return_value=manager):
        dry = CliRunner().invoke(cli, ["entities", "merge", KEEP, *OTHERS])
        assert dry.exit_code == 0, dry.output
        assert "Dry run" in dry.output and edge_set(graph) == before
        done = CliRunner().invoke(cli, ["entities", "merge", KEEP, *OTHERS, "--apply"])
    assert done.exit_code == 0, done.output
    assert edge_set(graph) == sorted(expected(mem))
    assert len(list((tmp_path / "backups").glob("entity-merge-cybergine-*.jsonl"))) == 1


# ---- detection ----------------------------------------------------------------


def _seed(graph, names_and_memories):
    from collivind.models import MemoryCategory, MemoryCreate, RelationshipCreate, RelType

    mems = {}
    for name, labels in names_and_memories.items():
        e = graph.create_entity(EntityCreate(name=name, type=EntityType.CONCEPT))
        for label in labels:
            if label not in mems:
                mems[label] = graph.create_memory(
                    MemoryCreate(content=label, summary=label, category=MemoryCategory.FACT, project_id="p")
                ).id
            graph.create_relationship(RelationshipCreate(source_id=mems[label], target_id=e.id, type=RelType.ABOUT))


def _pairs(groups):
    return {frozenset((p["a"], p["b"])): {e["reason"] for e in p["evidence"]} for g in groups for p in g["pairs"]}


def test_duplicates_report_evidence_and_stay_conservative(graph):
    _seed(
        graph,
        {
            "PR 774": ["a"],
            "PR #774": ["b"],
            "PR 775": ["c"],
            "Cybergine": ["d", "e"],
            "Cybergine v5_flask": ["e"],
            "Cybergine Ltd": ["f"],  # contains, but no shared memory
            "src/app/deploy_clean.sh": ["g"],
            "deploy_clean.sh": ["h"],  # the only path ending in it: pairs without a shared memory
            "pkg/__init__.py": ["i"],
            "other/__init__.py": ["j"],
            "__init__.py": ["k"],  # two paths end in it and none shares a memory: no pair
            "channels/email_fetch.py": ["l"],
            "channels/email_mime.py": ["m"],
        },
    )
    embedder = MagicMock()
    embedder.embed_batch.side_effect = lambda texts: [[1.0, 0.0]] * len(texts)  # every name identical
    pairs = _pairs(MemoryManager(MagicMock(), graph, embedder, CollivindConfig()).find_duplicate_entities())
    assert pairs[frozenset(("PR 774", "PR #774"))] == {"same-name", "embedding"}
    assert pairs[frozenset(("Cybergine", "Cybergine v5_flask"))] == {"contains"}
    assert pairs[frozenset(("src/app/deploy_clean.sh", "deploy_clean.sh"))] == {"basename"}
    assert frozenset(("PR 774", "PR 775")) not in pairs  # digits differ
    assert pairs.get(frozenset(("Cybergine", "Cybergine Ltd"))) == {"embedding"}  # weak only, no contains
    assert not any("__init__.py" in p for pair in pairs for p in pair)
    assert frozenset(("channels/email_fetch.py", "channels/email_mime.py")) not in pairs


def test_explorer_route_lists_only_this_entitys_groups_and_writes_nothing(graph):
    from collivind.web.api import MemoryAPI, route

    _seed(graph, {"PR 774": ["a"], "PR #774": ["b"], "Docker": ["c"], "docker-compose": ["d"]})
    before = edge_set(graph)
    api = MemoryAPI(MemoryManager(MagicMock(), graph, MagicMock(), CollivindConfig()), mode="embedded")
    status, body = route(api, "GET", "/api/entities/pr_774/duplicates", {}, {})
    assert status == 200 and len(body["groups"]) == 1
    assert body["groups"][0]["command"].startswith("collivind entities merge ")
    assert route(api, "GET", "/api/entities/docker/duplicates", {}, {}) == (200, {"groups": []})
    assert edge_set(graph) == before
