"""Multi-process acceptance test for the embedded vector store.

Two real Python processes hold the same store open simultaneously — the
exact situation the old qdrant-based engine refused ("already accessed by
another instance") — and each must see the other's acknowledged writes
without reopening anything (strict cross-process read-your-writes).
"""

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

DIM = 8

SRC = str(Path(__file__).resolve().parents[2] / "src")

DRIVER = f"""
import json, sys
sys.path.insert(0, {SRC!r})
from collivind.config import QdrantConfig
from collivind.storage.qdrant_embedded import EmbeddedQdrantStore

store = EmbeddedQdrantStore(data_dir=sys.argv[1], config=QdrantConfig(), dimension={DIM})
store.initialize()
print(json.dumps({{"ready": True}}), flush=True)
for line in sys.stdin:
    cmd = json.loads(line)
    if cmd["op"] == "upsert":
        store.upsert(cmd["id"], cmd["vector"], cmd.get("payload", {{}}))
        print(json.dumps({{"ok": True}}), flush=True)
    elif cmd["op"] == "search":
        hits = store.search(cmd["vector"], limit=10, threshold=0.5)
        print(json.dumps({{"ok": True, "ids": [h["id"] for h in hits]}}), flush=True)
"""


def unit(i: int) -> list[float]:
    vec = [0.0] * DIM
    vec[i] = 1.0
    return vec


def spawn(data_dir: Path) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-c", DRIVER, str(data_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    ready = proc.stdout.readline()
    assert json.loads(ready) == {"ready": True}, proc.stderr.read()
    return proc


def send(proc: subprocess.Popen, command: dict) -> dict:
    proc.stdin.write(json.dumps(command) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def reap(proc: subprocess.Popen) -> None:
    proc.kill()
    proc.communicate(timeout=10)  # reap and close the pipes


@pytest.fixture
def two_processes(tmp_path):
    a, b = spawn(tmp_path), spawn(tmp_path)
    yield a, b
    for proc in (a, b):
        reap(proc)


def test_two_processes_open_the_store_concurrently(two_processes):
    """Both processes are fully operational — nobody is refused or degraded."""
    a, b = two_processes
    assert send(a, {"op": "search", "vector": unit(0)})["ok"]
    assert send(b, {"op": "search", "vector": unit(0)})["ok"]


def test_cross_process_read_your_writes_both_directions(two_processes):
    a, b = two_processes
    # warm both caches while the store is empty
    send(a, {"op": "search", "vector": unit(0)})
    send(b, {"op": "search", "vector": unit(0)})

    assert send(a, {"op": "upsert", "id": "from-a", "vector": unit(0)})["ok"]
    assert send(b, {"op": "search", "vector": unit(0)})["ids"] == ["from-a"]

    assert send(b, {"op": "upsert", "id": "from-b", "vector": unit(1)})["ok"]
    assert send(a, {"op": "search", "vector": unit(1)})["ids"] == ["from-b"]


# Cold-start worker: imports (the slow, jittery part) happen BEFORE the start
# signal; construction happens immediately after it. This compresses process
# startup jitter (~1 s) below the rollback->WAL conversion window (~1 ms) that
# plain Popen timing never hits.
COLD_START_WORKER = f"""
import json, sys, time
sys.path.insert(0, {SRC!r})
from pathlib import Path
from collivind.config import QdrantConfig
from collivind.storage.vector_sqlite import SqliteVectorStore

data_dir, start_marker = sys.argv[1], sys.argv[2]
print("ready", flush=True)
deadline = time.monotonic() + 30
while not Path(start_marker).exists():  # spin, no sleep: keep the collision tight
    if time.monotonic() > deadline:
        sys.exit(2)
try:
    store = SqliteVectorStore(data_dir=data_dir, config=QdrantConfig(), dimension={DIM})
    store.close()
    print(json.dumps({{"ok": True}}), flush=True)
except Exception as e:
    print(json.dumps({{"ok": False, "error": f"{{type(e).__name__}}: {{e}}"}}), flush=True)
"""

N_COLD_START_PROCS = 4
N_COLD_START_TRIALS = 4


def test_concurrent_cold_start_on_fresh_dir(tmp_path):
    """First-ever open by N simultaneous processes must not crash.

    The rollback->WAL journal-mode conversion needs an exclusive lock and
    SQLite does not consult the busy handler for it, so unsynchronized
    constructors racing on a fresh dir raised raw 'database is locked'
    (~60% of opens, review finding R1). Barrier-style start makes the
    ~1 ms collision deterministic enough to reproduce.
    """
    for trial in range(N_COLD_START_TRIALS):
        data_dir = tmp_path / f"trial-{trial}"
        data_dir.mkdir()
        marker = tmp_path / f"start-{trial}"
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", COLD_START_WORKER, str(data_dir), str(marker)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(N_COLD_START_PROCS)
        ]
        try:
            for proc in procs:
                assert proc.stdout.readline().strip() == "ready"
            marker.touch()  # release the barrier: all constructors fire at once
            outcomes = [json.loads(proc.stdout.readline()) for proc in procs]
        finally:
            for proc in procs:
                reap(proc)
        errors = [o["error"] for o in outcomes if not o["ok"]]
        assert not errors, f"trial {trial}: {len(errors)}/{N_COLD_START_PROCS} cold-start opens failed: {errors}"


def test_sigkill_leaves_store_usable_and_writes_durable(tmp_path):
    """Crash recovery: no lock/state cleanup needed, acknowledged writes survive."""
    victim = spawn(tmp_path)
    assert send(victim, {"op": "upsert", "id": "acked", "vector": unit(0)})["ok"]
    victim.send_signal(signal.SIGKILL)
    victim.communicate(timeout=10)

    survivor = spawn(tmp_path)
    try:
        assert send(survivor, {"op": "search", "vector": unit(0)})["ids"] == ["acked"]
    finally:
        reap(survivor)
