"""Resident embedding process for embedded mode (#16).

Every hook run is a fresh process, so without this each prompt pays the
sentence-transformers import and model load (~9s warm, worse cold). The daemon
loads the model once and answers JSON-line requests over a unix socket in the
data dir (mode 0600), or in a private per-user dir under /tmp
when that path is over the AF_UNIX limit. It exits after an idle timeout.

Protocol, one request per connection:
  -> {"op": "embed", "texts": [...], "version": "...", "model": "..."}
  <- {"ok": true, "vectors": [[...], ...]}
  -> {"op": "info"}       <- {"ok": true, "version": "...", "model": "...", "pid": N}
  -> {"op": "shutdown"}   <- {"ok": true}
A request whose version/model differ from the daemon's gets
{"ok": false, "error": "mismatch"} and the daemon shuts down, so the client
can start a matching one.

Single instance: the daemon holds an exclusive flock on `embed.lock` for its
whole life, taken before it touches the socket. Whoever holds the lock owns
the socket path, so it can unlink a stale socket left by a crashed daemon.

This module must stay cheap to import: hooks import it on every prompt.
"""

import argparse
import hashlib
import json
import logging
import os
import socket
import socketserver
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

from collivind.version import __version__

fcntl: Optional[ModuleType]
try:
    import fcntl as _fcntl

    fcntl = _fcntl
except ImportError:  # Windows: no flock, no unix sockets; embed in-process
    fcntl = None

logger = logging.getLogger(__name__)

SOCKET_NAME = "embed.sock"
LOCK_NAME = "embed.lock"
LOG_NAME = "embed-daemon.log"
ENV_DISABLE = "COLLIVIND_EMBED_DAEMON"  # "0" disables the daemon
MAX_SOCKET_PATH = 100  # AF_UNIX limit is 104 on macOS, 108 on Linux
# Fixed, not tempfile.gettempdir(): $TMPDIR differs between a sandboxed agent
# and the user's shell, and both must find the same daemon. The 0700 + owner
# check below is what makes a shared sticky /tmp safe.
SHORT_SOCKET_BASE = "/tmp"
CONNECT_TIMEOUT = 1.0
# A busy or hung daemon must leave the hook time to load in-process (6-9s)
# within Claude Code's 30s hook timeout, so small requests give up fast.
REPLY_TIMEOUT_BASE = 3.0
REPLY_TIMEOUT_PER_TEXT = 0.05
HANDLER_TIMEOUT = 10.0  # a stuck client must not pin the daemon
LOCK_WAIT = 3.0  # lets a replacement wait out a daemon that is shutting down


def daemon_enabled(config_flag: bool) -> bool:
    return config_flag and os.environ.get(ENV_DISABLE, "1") != "0"


def socket_path_for(data_dir: Path) -> Path:
    """The socket for a data dir: inside it, or, when that path is over the
    AF_UNIX limit, in a private per-user dir under /tmp keyed by a hash of the
    data dir. Lock and log live next to the socket either way. Callers resolve
    it per use, so the directory check is redone before every connect/spawn.

    If the short dir cannot be made safe (someone else owns it, or it is open
    to others) the long path is returned and the daemon stays off for it.
    """
    data_dir = Path(data_dir).expanduser()
    path = data_dir / SOCKET_NAME
    if len(str(path)) <= MAX_SOCKET_PATH or not hasattr(os, "getuid"):
        return path
    key = hashlib.sha256(str(data_dir.resolve()).encode()).hexdigest()[:16]
    short = Path(SHORT_SOCKET_BASE) / f"collivind-{os.getuid()}-{key}"
    try:
        short.mkdir(mode=0o700, exist_ok=True)
        st = short.lstat()  # lstat: a planted symlink is not a directory we own
    except OSError:
        return path
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        logger.warning(f"refusing embed daemon dir {short}: not a private directory owned by this user")
        return path
    fallback = short / SOCKET_NAME
    return fallback if len(str(fallback)) <= MAX_SOCKET_PATH else path


# --- server ---------------------------------------------------------------


class _Handler(socketserver.StreamRequestHandler):
    server: "EmbedServer"

    def handle(self):
        self.request.settimeout(HANDLER_TIMEOUT)
        try:
            request = json.loads(self.rfile.readline())
            reply = self.server.dispatch(request)
        except Exception as e:  # bad JSON, encode failure, ...
            logger.exception("embed daemon request failed")
            reply = {"ok": False, "error": str(e)}
        try:
            self.wfile.write(json.dumps(reply).encode() + b"\n")
        except OSError:
            pass  # client gave up; it falls back in-process


if sys.platform == "win32":  # no unix sockets: the class only has to import (fcntl is None there)
    _ServerBase = socketserver.TCPServer
else:
    _ServerBase = socketserver.UnixStreamServer


class EmbedServer(_ServerBase):
    """Single-threaded on purpose: requests are serialized, so the model is
    never used from two threads."""

    def __init__(self, path: str, model: Any, model_name: str, idle_timeout: float):
        self.model = model
        self.model_name = model_name
        self.version = __version__
        self.stopping = False
        self.timeout = idle_timeout  # handle_request() waits at most this long
        old_umask = os.umask(0o177)  # socket is never briefly world-accessible
        try:
            super().__init__(path, _Handler)
        finally:
            os.umask(old_umask)
        os.chmod(path, 0o600)

    def handle_timeout(self):
        logger.info("embed daemon idle, exiting")
        self.stopping = True

    def dispatch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        op = request.get("op")
        if op == "info":
            return {"ok": True, "version": self.version, "model": self.model_name, "pid": os.getpid()}
        if op == "shutdown":
            self.stopping = True
            return {"ok": True}
        if op == "embed":
            if request.get("version") != self.version or request.get("model") != self.model_name:
                self.stopping = True
                return {"ok": False, "error": "mismatch", "version": self.version, "model": self.model_name}
            texts = request["texts"]
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                return {"ok": False, "error": "texts must be a list of strings"}
            vectors = self.model.encode(texts, batch_size=128, show_progress_bar=False)
            return {"ok": True, "vectors": vectors.tolist()}
        return {"ok": False, "error": f"unknown op: {op!r}"}

    def run(self):
        while not self.stopping:
            self.handle_request()


def acquire_lock(lock_path: Path, wait: float = 0.0):
    """Exclusive flock on lock_path, or None if someone else holds it."""
    if fcntl is None:  # callers gate on fcntl first; this only makes it explicit
        raise OSError("flock is not available on this platform")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.monotonic() + wait
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                return None
            time.sleep(0.05)


def lock_is_held(lock_path: Path) -> bool:
    if not lock_path.exists():
        return False
    fd = acquire_lock(lock_path)
    if fd is None:
        return True
    os.close(fd)
    return False


def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def serve(socket_path: Path, model_name: str, idle_timeout: float, load_model=_load_sentence_transformer) -> int:
    """Run the daemon until idle/shutdown. Returns a process exit code."""
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = acquire_lock(socket_path.with_name(LOCK_NAME), wait=LOCK_WAIT)
    if lock_fd is None:
        logger.info("another embed daemon holds the lock, exiting")
        return 0
    try:
        model = load_model(model_name)  # before binding: clients never wait on a load
        model.encode(["warm up"], batch_size=1, show_progress_bar=False)  # first encode pays device init (~0.6s on mps)
        socket_path.unlink(missing_ok=True)  # we hold the lock, so any socket here is stale
        server = EmbedServer(str(socket_path), model, model_name, idle_timeout)
        logger.info(f"embed daemon pid={os.getpid()} model={model_name} on {socket_path}")
        try:
            server.run()
        finally:
            server.server_close()
            socket_path.unlink(missing_ok=True)
        return 0
    finally:
        os.close(lock_fd)


# --- client ---------------------------------------------------------------


def request(socket_path: Path, payload: Dict[str, Any], timeout: float = REPLY_TIMEOUT_BASE) -> Dict[str, Any]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(CONNECT_TIMEOUT)
        sock.connect(str(socket_path))
        sock.settimeout(timeout)
        sock.sendall(json.dumps(payload).encode() + b"\n")
        with sock.makefile("rb") as f:
            line = f.readline()
    reply = json.loads(line)
    if not isinstance(reply, dict):
        raise ValueError("malformed daemon reply")
    return reply


def spawn(socket_path: Path, model_name: str, idle_timeout: float) -> None:
    """Start a detached daemon. All stdio is detached from the caller: a hook
    whose stdout pipe is inherited would hang until the daemon exits."""
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    # -I and a neutral cwd: hooks run inside the user's project, and `-m` would
    # otherwise put that directory first on sys.path, so a repo's own json.py
    # or argparse.py would be imported (and executed) by the daemon.
    cmd = [sys.executable, "-I", "-m", "collivind.storage.embed_daemon", "--socket", str(socket_path)]
    cmd += ["--model", model_name, "--idle-timeout", str(idle_timeout)]
    with open(socket_path.with_name(LOG_NAME), "ab") as log:
        subprocess.Popen(
            cmd,
            cwd=str(socket_path.parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log,
            start_new_session=True,
        )


def embed_via_daemon(socket_path: Path, model_name: str, texts: List[str], idle_timeout: float):
    """Vectors from the daemon, or None (caller embeds in-process).

    None never means failure for the caller. When no daemon is listening this
    starts one so the next call is fast; a version/model mismatch makes the old
    daemon exit and starts a matching one."""
    if fcntl is None or len(str(socket_path)) > MAX_SOCKET_PATH:
        return None  # can't bind there; spawning would just fail every time
    payload = {"op": "embed", "texts": texts, "version": __version__, "model": model_name}
    try:
        reply = request(socket_path, payload, REPLY_TIMEOUT_BASE + REPLY_TIMEOUT_PER_TEXT * len(texts))
    except (FileNotFoundError, ConnectionRefusedError):
        if not lock_is_held(socket_path.with_name(LOCK_NAME)):  # held = alive or still loading
            _spawn_quietly(socket_path, model_name, idle_timeout)
        return None
    except Exception as e:
        logger.debug(f"embed daemon unusable, embedding in-process: {e}")
        return None
    if reply.get("ok"):
        vectors = reply.get("vectors")
        if isinstance(vectors, list) and len(vectors) == len(texts):
            return vectors
        return None
    if reply.get("error") == "mismatch":  # the old daemon is already exiting
        _spawn_quietly(socket_path, model_name, idle_timeout)
    return None


def _spawn_quietly(socket_path: Path, model_name: str, idle_timeout: float) -> None:
    try:
        spawn(socket_path, model_name, idle_timeout)
    except Exception as e:
        logger.debug(f"could not start embed daemon: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Collivind resident embedding daemon")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--idle-timeout", type=float, default=1800)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(process)d %(message)s")
    return serve(Path(args.socket), args.model, args.idle_timeout)


if __name__ == "__main__":
    sys.exit(main())
