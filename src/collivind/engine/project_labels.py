"""Which project a memory belongs to, and relabelling memories after the fact.

Resolution order for the agent's working directory (first hit wins):
1. COLLIVIND_PROJECT environment variable
2. `project = "..."` in a `.collivind.toml` at or above the directory (up to the repo root)
3. the repo name of the `origin` remote (stable across clones and worktrees)
4. the main checkout's folder name (a worktree resolves to the repo it belongs to)
5. outside git: the directory's own name, unless it is home or the filesystem root
6. "default"
"""

import os
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from collivind.engine.projects import validate_project_id
from collivind.models.memory import MemoryNode
from collivind.storage.interfaces import GraphStore, VectorStore

DEFAULT_PROJECT = "default"
ENV_VAR = "COLLIVIND_PROJECT"
LOCAL_FILE = ".collivind.toml"
# what MemoryManager.get_context answers when nothing matched
NO_CONTEXT = "No relevant context found"


def _clean(value: object) -> Optional[str]:
    try:
        return validate_project_id(value).strip()
    except ValueError:
        return None


def _git(cwd: Path, *args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def _repo_roots(cwd: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """(this checkout's top level, the main checkout's top level); Nones outside git."""
    out = _git(cwd, "rev-parse", "--path-format=absolute", "--show-toplevel", "--git-common-dir")
    if not out or len(out.splitlines()) != 2:
        return None, None
    top, common = (Path(line) for line in out.splitlines())
    main = common.parent if common.name == ".git" else top
    return top, main


def _local_override(start: Path, stop: Optional[Path]) -> Optional[str]:
    for d in (start, *start.parents):
        f = d / LOCAL_FILE
        if f.is_file():
            try:
                return _clean(tomllib.loads(f.read_text()).get("project"))
            except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError):
                return None
        if stop is None or d == stop:
            return None
    return None


def repo_name_from_url(url: str) -> Optional[str]:
    """`git@github.com:o/v5_flask.git` or `https://host/o/v5_flask` -> `v5_flask`."""
    tail = re.split(r"[/:]", url.strip().rstrip("/"))[-1]
    return _clean(tail[:-4] if tail.endswith(".git") else tail)


def resolve_project(cwd: Optional[str] = None) -> str:
    """The project for work happening in `cwd` (default: this process's cwd). Never raises."""
    env = _clean(os.environ.get(ENV_VAR, ""))
    if env:
        return env
    try:
        start = Path(cwd or os.getcwd()).expanduser().resolve()
    except OSError:
        return DEFAULT_PROJECT
    home = Path.home().resolve()
    top, main = _repo_roots(start)
    if top is not None and top.resolve() == home:
        top = main = None  # a dotfiles repo at ~ says nothing about the project
    override = _local_override(start, top)
    if override:
        return override
    if top is None or main is None:
        is_named_dir = start not in (home, Path(start.anchor))
        return (_clean(start.name) if is_named_dir else None) or DEFAULT_PROJECT
    url = _git(main, "config", "--get", "remote.origin.url")
    return (repo_name_from_url(url) if url else None) or _clean(main.name) or DEFAULT_PROJECT


def selected_memories(graph: GraphStore, old: str, match: Optional[str] = None) -> List[MemoryNode]:
    """Memories of `old` whose content, summary or tags match `match` (a regex), with every
    version of each: a match on any version selects its whole chain."""
    rx = re.compile(match, re.IGNORECASE) if match else None

    def hit(m: MemoryNode) -> bool:
        return rx is None or bool(rx.search("\n".join([m.content or "", m.summary or "", *(m.tags or [])])))

    pool = {m.id: m for m in graph.iter_memories() if m.project_id == old}
    links: Dict[str, List[str]] = {i: [] for i in pool}
    for m in pool.values():  # version links, both ways, within `old`
        for other in (m.superseded_by, m.previous_version_id):
            if other in pool:
                links[m.id].append(other)
                links[other].append(m.id)
    # Close the matches over their version chains so a chain never straddles two projects.
    todo = [i for i, m in pool.items() if hit(m)]
    chosen = set(todo)
    while todo:
        for other in links[todo.pop()]:
            if other not in chosen:
                chosen.add(other)
                todo.append(other)
    return [m for i, m in pool.items() if i in chosen]


def rename_project(
    graph: GraphStore,
    vectors: VectorStore,
    memories: List[MemoryNode],
    new: str,
    progress: Callable[[int], None] = lambda n: None,
) -> Dict[str, int]:
    """Relabel `memories` as project `new`. Ids, links and vectors are kept.

    Vectors are patched first and the graph second, so an interrupted run is
    finished by running it again: the graph still lists what is left.
    """
    new = validate_project_id(new).strip()
    ids = {m.id for m in memories}
    # ponytail: holds every vector in memory (~1.5 KB each); stream in pages past ~1M memories
    stale = [(i, v, p) for i, v, p in vectors.iter_all() if i in ids and p.get("project_id") != new]
    for i, vector, payload in stale:
        vectors.upsert(i, vector, {**payload, "project_id": new})
    for n, m in enumerate(memories, 1):
        graph.update_memory(m.id, project_id=new)
        progress(n)
    return {"memories": len(memories), "vectors": len(stale)}
