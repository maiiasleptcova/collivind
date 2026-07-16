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
