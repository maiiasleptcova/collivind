"""Cross-process semantics of the SQLite graph store.

WAL mode has tolerated concurrent processes for a while; these tests assert
it — plus the busy_timeout that turns concurrent-writer collisions from an
immediate `database is locked` into a short wait.
"""

import sqlite3
import subprocess
import sys
import threading

from collivind.models.memory import MemoryCategory, MemoryCreate
from collivind.storage.graph_sqlite import SqliteGraphStore


def _memory(content: str) -> MemoryCreate:
    return MemoryCreate(content=content, summary=content[:40], category=MemoryCategory.FACT, project_id="proj")


WRITER_SCRIPT = """
import sys
sys.path.insert(0, {src!r})
from collivind.storage.graph_sqlite import SqliteGraphStore
from collivind.models.memory import MemoryCategory, MemoryCreate

store = SqliteGraphStore(data_dir=sys.argv[1])
store.initialize()
node = store.create_memory(
    MemoryCreate(content="written by subprocess", summary="sub", category=MemoryCategory.FACT, project_id="proj")
)
print(node.id)
store.close()
"""


def test_subprocess_write_visible_to_open_parent_store(tmp_path):
    parent = SqliteGraphStore(data_dir=str(tmp_path))
    parent.initialize()
    parent.get_timeline("proj")  # open a read before the other process writes

    src = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-c", WRITER_SCRIPT.format(src=src), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    memory_id = result.stdout.strip()

    found = parent.get_memory(memory_id)
    assert found is not None
    assert found.content == "written by subprocess"
    parent.close()


def test_write_waits_for_concurrent_writer_instead_of_failing(tmp_path):
    """busy_timeout: a held write lock means waiting, not `database is locked`."""
    store = SqliteGraphStore(data_dir=str(tmp_path))
    store.initialize()

    blocker = sqlite3.connect(str(tmp_path / "collivind_graph.db"), check_same_thread=False)
    blocker.execute("BEGIN IMMEDIATE")
    release = threading.Timer(1.0, blocker.commit)
    release.start()
    try:
        node = store.create_memory(_memory("written while another writer held the lock"))
        assert store.get_memory(node.id) is not None
    finally:
        release.join()
        blocker.close()
        store.close()
