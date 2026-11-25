"""Resident embedding process for embedded mode (#16).

Every hook run is a fresh process, so without this each prompt pays the
sentence-transformers import and model load (~9s warm, worse cold). The daemon
loads the model once and answers JSON-line requests over a unix socket in the
data dir (mode 0600), or in a private per-user dir under /tmp
when that path is over the AF_UNIX limit. It exits after an idle timeout.

Protocol, one request per connection:
  -> {"op": "embed", "texts": [...], "version": "...", "model": "...", "rerank_model": "..."}
  <- {"ok": true, "vectors": [[...], ...]}
  -> {"op": "rerank", "query": "...", "texts": [...], "version": "...", "model": "...", "rerank_model": "..."}
  <- {"ok": true, "scores": [...]}
  -> {"op": "info"}       <- {"ok": true, "version": "...", "model": "...", "rerank_model": "...", "pid": N}
  -> {"op": "shutdown"}   <- {"ok": true}
A request whose version/model differ from the daemon's gets
{"ok": false, "error": "mismatch"} and the daemon shuts down, so the client
can start a matching one. A rerank request for a cross-encoder the daemon did
not load gets {"ok": false, "error": "no reranker loaded"} and the daemon
keeps running: processes with different [search] rerank settings must not
keep killing each other's daemon. The daemon loads the rerank_model of the
client that started it ("" = none) until it exits.

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

    def __init__(
        self,
        path: str,
        model: Any,
        model_name: str,
        idle_timeout: float,
        reranker: Any = None,
        rerank_model: str = "",
    ):
        self.model = model
        self.model_name = model_name
        self.reranker = reranker  # None: not configured, or it failed to load
        self.rerank_model = rerank_model
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
            return {
                "ok": True,
                "version": self.version,
                "model": self.model_name,
                "rerank_model": self.rerank_model,
                "pid": os.getpid(),
            }
        if op == "shutdown":
            self.stopping = True
            return {"ok": True}
        if op in ("embed", "rerank") and (
            request.get("version") != self.version
            or request.get("model") != self.model_name
            # started by a rerank-off process and now asked to rerank: replace it,
            # or an active user (embeds reset the idle timer) never gets rerank.
            # Rerank-off clients never send this op, so two clients cannot loop;
            # a rerank-off process can still win a spawn race now and then, which
            # costs one more model load on the next rerank request, not a loop.
            or (op == "rerank" and not self.rerank_model and bool(request.get("rerank_model")))
        ):
            self.stopping = True
            return {"ok": False, "error": "mismatch", "version": self.version, "model": self.model_name}
        if op == "rerank":
            query, texts = request.get("query"), request.get("texts")
            if not isinstance(query, str) or not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                return {"ok": False, "error": "query must be a string and texts a list of strings"}
            if self.reranker is None or request.get("rerank_model") != self.rerank_model:
                return {"ok": False, "error": "no reranker loaded", "rerank_model": self.rerank_model}
            scores = self.reranker.predict([(query, t) for t in texts], batch_size=64, show_progress_bar=False)
            return {"ok": True, "scores": [float(x) for x in scores]}
        if op == "embed":
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


def load_cross_encoder(model_name: str):
    """The reranker. max_length 256 tokens covers query + one memory; longer is truncated."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=256)


def _load_reranker_quietly(rerank_model: str, load_reranker) -> Any:
    """A reranker that fails to load must not take embeddings down with it."""
    if not rerank_model:
        return None
    try:
        reranker = load_reranker(rerank_model)
        reranker.predict([("warm up", "warm up")], show_progress_bar=False)
        return reranker
    except Exception as e:
        logger.warning(f"reranker {rerank_model} failed to load, serving embeddings only: {e}")
        return None


def serve(
    socket_path: Path,
    model_name: str,
    idle_timeout: float,
    load_model=_load_sentence_transformer,
    rerank_model: str = "",
    load_reranker=load_cross_encoder,
) -> int:
    """Run the daemon until idle/shutdown. Returns a process exit code."""
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = acquire_lock(socket_path.with_name(LOCK_NAME), wait=LOCK_WAIT)
    if lock_fd is None:
        logger.info("another embed daemon holds the lock, exiting")
        return 0
    try:
        model = load_model(model_name)  # before binding: clients never wait on a load
        model.encode(["warm up"], batch_size=1, show_progress_bar=False)  # first encode pays device init (~0.6s on mps)
        reranker = _load_reranker_quietly(rerank_model, load_reranker)  # also before binding
        socket_path.unlink(missing_ok=True)  # we hold the lock, so any socket here is stale
        server = EmbedServer(str(socket_path), model, model_name, idle_timeout, reranker, rerank_model)
        logger.info(f"embed daemon pid={os.getpid()} model={model_name} rerank={rerank_model or '-'} on {socket_path}")
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


def spawn(socket_path: Path, model_name: str, idle_timeout: float, rerank_model: str = "") -> None:
    """Start a detached daemon. All stdio is detached from the caller: a hook
    whose stdout pipe is inherited would hang until the daemon exits."""
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    # -I and a neutral cwd: hooks run inside the user's project, and `-m` would
    # otherwise put that directory first on sys.path, so a repo's own json.py
    # or argparse.py would be imported (and executed) by the daemon.
    cmd = [sys.executable, "-I", "-m", "collivind.storage.embed_daemon", "--socket", str(socket_path)]
    cmd += ["--model", model_name, "--idle-timeout", str(idle_timeout)]
    if rerank_model:
        cmd += ["--rerank-model", rerank_model]
    with open(socket_path.with_name(LOG_NAME), "ab") as log:
        subprocess.Popen(
            cmd,
            cwd=str(socket_path.parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log,
            start_new_session=True,
        )


def daemon_usable(socket_path: Path) -> bool:
    """False where no daemon can ever bind (no flock, or the path is over the AF_UNIX limit)."""
    return fcntl is not None and len(str(socket_path)) <= MAX_SOCKET_PATH


def _call(
    socket_path: Path,
    payload: Dict[str, Any],
    timeout: float,
    model_name: str,
    idle_timeout: float,
    rerank_model: str,
) -> Optional[Dict[str, Any]]:
    """The daemon's ok reply, or None. Starts a daemon when none is listening
    and replaces one whose version/model/rerank_model mismatch."""
    if not daemon_usable(socket_path):
        return None  # can't bind there; spawning would just fail every time
    payload = {**payload, "version": __version__, "model": model_name, "rerank_model": rerank_model}
    try:
        reply = request(socket_path, payload, timeout)
    except (FileNotFoundError, ConnectionRefusedError):
        if not lock_is_held(socket_path.with_name(LOCK_NAME)):  # held = alive or still loading
            _spawn_quietly(socket_path, model_name, idle_timeout, rerank_model)
        return None
    except Exception as e:
        logger.debug(f"embed daemon unusable: {e}")
        return None
    if reply.get("ok"):
        return reply
    if reply.get("error") == "mismatch":  # the old daemon is already exiting
        _spawn_quietly(socket_path, model_name, idle_timeout, rerank_model)
    elif reply.get("error") == "no reranker loaded":
        raise RerankerUnavailable(reply.get("rerank_model", ""))
    return None


class RerankerUnavailable(Exception):
    """The daemon is up but serves no (or another) cross-encoder."""


def embed_via_daemon(socket_path: Path, model_name: str, texts: List[str], idle_timeout: float, rerank_model: str = ""):
    """Vectors from the daemon, or None (caller embeds in-process).

    None never means failure for the caller. When no daemon is listening this
    starts one so the next call is fast; a version/model mismatch makes the old
    daemon exit and starts a matching one."""
    timeout = REPLY_TIMEOUT_BASE + REPLY_TIMEOUT_PER_TEXT * len(texts)
    reply = _call(socket_path, {"op": "embed", "texts": texts}, timeout, model_name, idle_timeout, rerank_model)
    vectors = reply.get("vectors") if reply else None
    if isinstance(vectors, list) and len(vectors) == len(texts):
        return vectors
    return None


def rerank_via_daemon(
    socket_path: Path, model_name: str, rerank_model: str, query: str, texts: List[str], idle_timeout: float
) -> Optional[List[float]]:
    """Cross-encoder scores from the daemon, or None (caller keeps its first-stage ranking).

    Raises RerankerUnavailable when the daemon answers but has no matching
    reranker, so the caller can stop asking for a while."""
    payload = {"op": "rerank", "query": query, "texts": texts}
    timeout = REPLY_TIMEOUT_BASE + REPLY_TIMEOUT_PER_TEXT * len(texts)
    reply = _call(socket_path, payload, timeout, model_name, idle_timeout, rerank_model)
    scores = reply.get("scores") if reply else None
    if isinstance(scores, list) and len(scores) == len(texts) and all(isinstance(x, (int, float)) for x in scores):
        return [float(x) for x in scores]
    return None


def _spawn_quietly(socket_path: Path, model_name: str, idle_timeout: float, rerank_model: str = "") -> None:
    try:
        spawn(socket_path, model_name, idle_timeout, rerank_model)
    except Exception as e:
        logger.debug(f"could not start embed daemon: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Collivind resident embedding daemon")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--idle-timeout", type=float, default=1800)
    parser.add_argument("--rerank-model", default="")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(process)d %(message)s")
    return serve(Path(args.socket), args.model, args.idle_timeout, rerank_model=args.rerank_model)


if __name__ == "__main__":
    sys.exit(main())
