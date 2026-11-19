"""Resident embedding daemon (#16). A fake model stands in for
sentence-transformers; no test spawns a real process.

Sockets live in a short /tmp dir: pytest's tmp_path can exceed the 104-byte
AF_UNIX path limit on macOS."""

import os
import shutil
import socket
import stat
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from collivind.config import EmbeddingsConfig
from collivind.storage import embed_daemon
from collivind.storage.embedding_local import LocalEmbeddingProvider
from collivind.version import __version__

MODEL = "fake/model"


class FakeModel:
    def encode(self, texts, batch_size=32, **kwargs):
        if isinstance(texts, str):
            return np.array([float(len(texts)), 1.0])
        return np.array([[float(len(t)), 1.0] for t in texts])


@pytest.fixture
def data_dir():
    d = tempfile.mkdtemp(prefix="cv", dir="/tmp")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def daemon_on(monkeypatch):
    monkeypatch.setenv(embed_daemon.ENV_DISABLE, "1")


def start_daemon(data_dir, idle=30.0, model_name=MODEL):
    path = embed_daemon.socket_path_for(data_dir)
    result = {}
    t = threading.Thread(
        target=lambda: result.setdefault(
            "rc", embed_daemon.serve(path, model_name, idle, load_model=lambda name: FakeModel())
        ),
        daemon=True,
    )
    t.start()
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists(), "daemon did not bind"
    return path, t, result


def stop(path, thread):
    try:
        embed_daemon.request(path, {"op": "shutdown"})
    except OSError:
        pass
    thread.join(5)


def provider_for(data_dir):
    return LocalEmbeddingProvider(EmbeddingsConfig(model=MODEL), data_dir=str(data_dir))


def test_protocol_round_trip(data_dir, daemon_on):
    path, t, _ = start_daemon(data_dir)
    try:
        info = embed_daemon.request(path, {"op": "info"})
        assert info["version"] == __version__ and info["model"] == MODEL
        provider = provider_for(data_dir)
        with patch.object(provider, "_load_model", side_effect=AssertionError("must not load in-process")):
            assert provider.embed("abc") == [3.0, 1.0]
            assert provider.embed_batch(["a", "bb"]) == [[1.0, 1.0], [2.0, 1.0]]
    finally:
        stop(path, t)
    assert not path.exists()


def test_socket_is_owner_only(data_dir, daemon_on):
    path, t, _ = start_daemon(data_dir)
    try:
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    finally:
        stop(path, t)


def test_absent_daemon_spawns_and_embeds_in_process(data_dir, daemon_on):
    provider = provider_for(data_dir)
    provider._model = None
    with (
        patch.object(embed_daemon, "spawn") as spawn,
        patch.object(provider, "_load_model", side_effect=lambda: setattr(provider, "_model", FakeModel())),
    ):
        assert provider.embed("abcd") == [4.0, 1.0]
    spawn.assert_called_once()


def test_no_spawn_while_another_daemon_holds_the_lock(data_dir, daemon_on):
    """A daemon still loading its model holds the lock but has no socket yet."""
    lock = embed_daemon.acquire_lock(data_dir / embed_daemon.LOCK_NAME)
    try:
        with patch.object(embed_daemon, "spawn") as spawn:
            assert embed_daemon.embed_via_daemon(embed_daemon.socket_path_for(data_dir), MODEL, ["x"], 60) is None
        spawn.assert_not_called()
    finally:
        os.close(lock)


def test_broken_daemon_falls_back_in_process(data_dir, daemon_on):
    path = embed_daemon.socket_path_for(data_dir)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(path))
    srv.listen(1)

    def garbage():
        conn, _ = srv.accept()
        conn.recv(65536)
        conn.sendall(b"not json\n")
        conn.close()

    threading.Thread(target=garbage, daemon=True).start()
    provider = provider_for(data_dir)
    try:
        with patch.object(provider, "_load_model", side_effect=lambda: setattr(provider, "_model", FakeModel())):
            assert provider.embed("ab") == [2.0, 1.0]
    finally:
        srv.close()


def test_hung_daemon_times_out_and_falls_back(data_dir, daemon_on, monkeypatch):
    """A daemon that accepts and never replies must not eat the hook's budget."""
    monkeypatch.setattr(embed_daemon, "REPLY_TIMEOUT_BASE", 0.2)
    path = embed_daemon.socket_path_for(data_dir)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(path))
    srv.listen(1)
    provider = provider_for(data_dir)
    try:
        start = time.monotonic()
        with patch.object(provider, "_load_model", side_effect=lambda: setattr(provider, "_model", FakeModel())):
            assert provider.embed("abc") == [3.0, 1.0]
        assert time.monotonic() - start < 2
    finally:
        srv.close()


def test_no_daemon_without_flock(data_dir, daemon_on, monkeypatch):
    """Windows has no fcntl: the daemon is skipped, not an import error."""
    monkeypatch.setattr(embed_daemon, "fcntl", None)
    with patch.object(embed_daemon, "spawn") as spawn:
        assert embed_daemon.embed_via_daemon(embed_daemon.socket_path_for(data_dir), MODEL, ["x"], 60) is None
    spawn.assert_not_called()


def test_daemon_disabled_by_env_and_config(data_dir, monkeypatch):
    for env, flag in (("0", True), ("1", False)):
        monkeypatch.setenv(embed_daemon.ENV_DISABLE, env)
        provider = LocalEmbeddingProvider(EmbeddingsConfig(model=MODEL, daemon=flag), data_dir=str(data_dir))
        with (
            patch.object(embed_daemon, "embed_via_daemon") as via,
            patch.object(provider, "_load_model", side_effect=lambda p=provider: setattr(p, "_model", FakeModel())),
        ):
            provider.embed("x")
        via.assert_not_called()


def test_version_mismatch_stops_old_daemon_and_starts_new(data_dir, daemon_on):
    with patch.object(embed_daemon, "__version__", "0.0.0-old"):
        path, t, _ = start_daemon(data_dir)  # a daemon left over from an older install
    with patch.object(embed_daemon, "spawn") as spawn:
        assert embed_daemon.embed_via_daemon(path, MODEL, ["x"], 60) is None
    t.join(5)
    assert not t.is_alive(), "mismatched daemon must exit"
    spawn.assert_called_once()


def test_model_mismatch_also_restarts(data_dir, daemon_on):
    path, t, _ = start_daemon(data_dir, model_name="other/model")
    with patch.object(embed_daemon, "spawn") as spawn:
        assert embed_daemon.embed_via_daemon(path, MODEL, ["x"], 60) is None
    t.join(5)
    assert not t.is_alive()
    spawn.assert_called_once()


def test_idle_exit_removes_socket(data_dir, daemon_on):
    path, t, result = start_daemon(data_dir, idle=0.2)
    t.join(5)
    assert not t.is_alive()
    assert result["rc"] == 0
    assert not path.exists()


def test_second_daemon_exits_while_first_holds_lock(data_dir, daemon_on, monkeypatch):
    monkeypatch.setattr(embed_daemon, "LOCK_WAIT", 0.1)
    path, t, _ = start_daemon(data_dir)
    try:
        loaded = []
        rc = embed_daemon.serve(path, MODEL, 30, load_model=lambda n: loaded.append(n))
        assert rc == 0 and loaded == []  # never loaded a model, never touched the socket
        assert embed_daemon.request(path, {"op": "info"})["ok"]
    finally:
        stop(path, t)


def test_stale_socket_is_replaced(data_dir, daemon_on):
    path = embed_daemon.socket_path_for(data_dir)
    dead = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    dead.bind(str(path))
    dead.close()  # file left behind, nobody listening — a crashed daemon
    path2, t, _ = start_daemon_after_stale(data_dir)
    try:
        assert embed_daemon.request(path2, {"op": "info"})["ok"]
    finally:
        stop(path2, t)


def start_daemon_after_stale(data_dir):
    path = embed_daemon.socket_path_for(data_dir)
    t = threading.Thread(
        target=embed_daemon.serve, args=(path, MODEL, 30), kwargs={"load_model": lambda n: FakeModel()}, daemon=True
    )
    t.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            embed_daemon.request(path, {"op": "info"})
            return path, t, None
        except OSError:
            time.sleep(0.02)
    raise AssertionError("daemon never replaced the stale socket")


def test_overlong_socket_path_skips_daemon_without_spawning():
    path = Path("/tmp/" + "x" * 120) / embed_daemon.SOCKET_NAME
    with patch.object(embed_daemon, "spawn") as spawn:
        assert embed_daemon.embed_via_daemon(path, MODEL, ["x"], 60) is None
    spawn.assert_not_called()


def test_spawn_detaches_stdio(data_dir):
    with patch("subprocess.Popen") as popen:
        embed_daemon.spawn(embed_daemon.socket_path_for(data_dir), MODEL, 60)
    kwargs = popen.call_args.kwargs
    assert kwargs["stdin"] == kwargs["stdout"] == embed_daemon.subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
