"""Project resolution from the working directory, and `collivind projects rename`."""

import subprocess
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner
from test_migrate import _graph_snapshot, _vectors, populated  # noqa: F401 - fixture

from collivind.cli.commands.projects import projects
from collivind.config import CollivindConfig
from collivind.engine.memory_manager import MemoryManager
from collivind.engine.project_labels import repo_name_from_url, resolve_project
from collivind.models import SearchQuery


def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repo `ws/folder-name` under a fake home, with no env override."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("COLLIVIND_PROJECT")
    root = tmp_path / "ws" / "folder-name"
    (root / "src" / "pkg").mkdir(parents=True)
    _git(root, "init", "-q")
    return root


# --- resolution order -------------------------------------------------------


def test_folder_name_without_remote(repo):
    assert resolve_project(str(repo / "src" / "pkg")) == "folder-name"


def test_remote_repo_name_beats_folder_name(repo):
    _git(repo, "remote", "add", "origin", "git@github.com:someone/v5_flask.git")
    assert resolve_project(str(repo)) == "v5_flask"


def test_local_file_beats_remote(repo):
    _git(repo, "remote", "add", "origin", "https://github.com/someone/v5_flask")
    (repo / ".collivind.toml").write_text('project = "cybergine"\n')
    assert resolve_project(str(repo / "src" / "pkg")) == "cybergine"


def test_env_beats_local_file(repo, monkeypatch):
    (repo / ".collivind.toml").write_text('project = "cybergine"\n')
    monkeypatch.setenv("COLLIVIND_PROJECT", "pinned")
    assert resolve_project(str(repo)) == "pinned"


def test_local_file_above_the_repo_is_ignored(repo):
    (repo.parent / ".collivind.toml").write_text('project = "outer"\n')
    assert resolve_project(str(repo)) == "folder-name"


def test_invalid_local_file_falls_through(repo):
    (repo / ".collivind.toml").write_text("project = [not toml")
    assert resolve_project(str(repo)) == "folder-name"


def test_worktree_resolves_to_the_main_checkout(repo):
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "init")
    wt = repo / ".claude" / "worktrees" / "feature-x"
    _git(repo, "worktree", "add", "-q", str(wt))
    assert resolve_project(str(wt)) == "folder-name"


def test_plain_directory_uses_its_name(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("COLLIVIND_PROJECT")
    d = tmp_path / "scratch"
    d.mkdir()
    assert resolve_project(str(d)) == "scratch"


def test_home_and_missing_dirs_are_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("COLLIVIND_PROJECT")
    assert resolve_project(str(tmp_path)) == "default"


@pytest.mark.parametrize(
    "url,name",
    [
        ("git@github.com:o/v5_flask.git", "v5_flask"),
        ("https://github.com/o/v5_flask", "v5_flask"),
        ("https://u@github.com/o/collivind.git/", "collivind"),
        ("/srv/git/repo.git", "repo"),
    ],
)
def test_repo_name_from_url(url, name):
    assert repo_name_from_url(url) == name


# --- rename -----------------------------------------------------------------


def _run(stores, *args):
    mgr = MemoryManager(stores.vector, stores.graph, stores.embedder, CollivindConfig())
    with patch("collivind.cli.commands.memory._manager", return_value=mgr):
        return CliRunner().invoke(projects, ["rename", *args])


def test_rename_is_a_dry_run_by_default(populated):  # noqa: F811
    src, _ = populated
    before = (_graph_snapshot(src), _vectors(src))
    result = _run(src, "alpha", "gamma")
    assert result.exit_code == 0, result.output
    assert "2 memories (1 live) in 'alpha' -> 'gamma'" in result.output
    assert "Dry run" in result.output
    after = (_graph_snapshot(src), _vectors(src))
    assert repr(after) == repr(before)


def test_rename_keeps_ids_links_chains_and_vectors(populated):  # noqa: F811
    src, ids = populated
    mems_before, ents_before, rels_before = _graph_snapshot(src)
    vecs_before = _vectors(src)

    result = _run(src, "alpha", "gamma", "--apply")

    assert result.exit_code == 0, result.output
    assert "Relabelled 2 memories and 1 vectors" in result.output
    mems, ents, rels = _graph_snapshot(src)
    assert (ents, rels) == (ents_before, rels_before)
    by_id = {m["id"]: m for m in mems}
    assert {by_id[ids["old"]]["project_id"], by_id[ids["new"]]["project_id"]} == {"gamma"}
    assert by_id[ids["other"]]["project_id"] == "beta"
    assert by_id[ids["old"]]["superseded_by"] == ids["new"]  # chain intact
    for m in mems_before:  # nothing but the label (and updated_at) changed
        changed = {k for k, v in m.items() if by_id[m["id"]][k] != v}
        assert changed <= {"project_id", "updated_at"}
    vecs = _vectors(src)
    assert set(vecs) == set(vecs_before)
    assert np.allclose(vecs[ids["new"]][0], vecs_before[ids["new"]][0])  # not re-embedded
    assert vecs[ids["new"]][1]["project_id"] == "gamma"
    assert vecs[ids["other"]][1]["project_id"] == "beta"

    mgr = MemoryManager(src.vector, src.graph, src.embedder, CollivindConfig())
    hits = mgr.search(SearchQuery(query="we moved the event log to kafka", project_id="gamma"))
    assert ids["new"] in [h.memory.id for h in hits]
    assert not mgr.get_timeline("alpha")


def test_rename_by_match_moves_whole_version_chains_only(populated):  # noqa: F811
    src, ids = populated
    # only the superseding version mentions kafka; its predecessor must follow it
    result = _run(src, "alpha", "gamma", "--match", "KAFKA", "--apply")
    assert result.exit_code == 0, result.output
    labels = {m.id: m.project_id for m in src.graph.iter_memories()}
    assert labels[ids["new"]] == labels[ids["old"]] == "gamma"
    assert labels[ids["other"]] == "beta"


def test_rename_by_match_leaves_unmatched_memories(populated):  # noqa: F811
    src, ids = populated
    result = _run(src, "beta", "gamma", "--match", "postgres", "--apply")
    assert result.exit_code == 0, result.output
    assert "0 memories" in result.output
    assert {m.project_id for m in src.graph.iter_memories() if m.id == ids["other"]} == {"beta"}


def test_rename_is_rerunnable_after_an_interrupted_run(populated):  # noqa: F811
    src, ids = populated
    real = src.graph.update_memory
    calls = []

    def flaky(mid, **kw):
        calls.append(mid)
        if len(calls) == 2:
            raise RuntimeError("connection lost")
        return real(mid, **kw)

    with patch.object(src.graph, "update_memory", side_effect=flaky):
        first = _run(src, "alpha", "gamma", "--apply")
    assert first.exit_code == 1 and "run the same command again" in first.output

    second = _run(src, "alpha", "gamma", "--apply")
    assert second.exit_code == 0, second.output
    assert {m.project_id for m in src.graph.iter_memories() if m.id != ids["other"]} == {"gamma"}


@pytest.mark.parametrize("args", [["alpha", "alpha"], ["alpha", "*"], ["alpha", "x", "--match", "("]])
def test_rename_rejects_bad_arguments(populated, args):  # noqa: F811
    src, _ = populated
    assert _run(src, *args).exit_code == 1


# --- agents omitting project_id -------------------------------------------


def _tools(manager):
    from collivind.mcp.tools import CollivindTools

    return CollivindTools(manager, session_id="s1", project_id="v5_flask")


def test_mcp_schemas_carry_no_project_default():
    from collivind.mcp.tools import CollivindTools

    text = repr(CollivindTools.get_tool_list())
    assert "'default': 'default'" not in text
    assert "'required': ['project_id']" not in text


def test_batch_add_labels_unlabelled_memories_with_the_current_project():
    from unittest.mock import MagicMock

    manager = MagicMock()
    manager.batch_add_memories.return_value = ["a", "b", "c"]
    mems = [
        {"content": "c", "summary": "s", "category": "fact"},
        {"content": "c", "summary": "s", "category": "fact", "project_id": "other"},
        {"content": "c", "summary": "s", "category": "fact", "project_id": "default"},
    ]
    with patch("collivind.mcp.tools.capture.record"):
        _tools(manager).handle_call("collivind_batch_add", {"memories": mems})
    sent = manager.batch_add_memories.call_args.args[0]
    assert [m["project_id"] for m in sent] == ["v5_flask", "other", "v5_flask"]


def test_get_context_falls_back_to_default_only_when_current_project_has_nothing():
    from unittest.mock import MagicMock

    manager = MagicMock()
    manager.get_context.side_effect = ["No relevant context found in Collivind.", "legacy hit"]
    assert _tools(manager).handle_call("collivind_get_context", {"query": "q"}) == "legacy hit"
    assert [c.kwargs["project_id"] for c in manager.get_context.call_args_list] == ["v5_flask", "default"]

    manager.get_context.reset_mock(side_effect=True)
    manager.get_context.return_value = "explicit"
    _tools(manager).handle_call("collivind_get_context", {"query": "q", "project_id": "x"})
    assert [c.kwargs["project_id"] for c in manager.get_context.call_args_list] == ["x"]


def test_user_prompt_hook_recalls_the_payload_cwd_project_first(repo, monkeypatch):
    import json
    from unittest.mock import MagicMock

    from collivind.cli.commands.hook import hook

    _git(repo, "remote", "add", "origin", "git@github.com:someone/v5_flask.git")
    manager = MagicMock()
    manager.get_context.return_value = "--- Collivind Context ---\n[FACT] scrape order"
    monkeypatch.setattr("collivind.cli.commands.hook._load_manager", lambda: manager)
    monkeypatch.setattr("collivind.capture.state_path", lambda: repo / "state.json")
    payload = json.dumps({"prompt": "what is the scrape programme order", "cwd": str(repo)})

    result = CliRunner().invoke(hook, ["user-prompt"], input=payload)

    assert result.exit_code == 0, result.output
    assert "scrape order" in result.output
    assert manager.get_context.call_args.kwargs["project_id"] == "v5_flask"


def test_session_start_names_the_current_project_when_falling_back(repo, monkeypatch):
    import json
    from unittest.mock import MagicMock

    from collivind.cli.commands.hook import hook
    from collivind.models import MemoryCategory, MemoryNode

    manager = MagicMock()
    old = MemoryNode(content="legacy fact", summary="legacy fact", category=MemoryCategory.FACT)
    manager.get_timeline.side_effect = lambda project, limit: [old] if project == "default" else []
    monkeypatch.setattr("collivind.cli.commands.hook._load_manager", lambda: manager)
    monkeypatch.setattr("collivind.capture.state_path", lambda: repo / "state.json")

    result = CliRunner().invoke(hook, ["session-start"], input=json.dumps({"cwd": str(repo)}))

    text = json.loads(result.output)["hookSpecificOutput"]["additionalContext"]
    assert "Project 'folder-name' has no memories yet" in text and "legacy fact" in text
    assert "project 'default'" not in text
