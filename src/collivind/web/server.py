"""HTTP plumbing for the local memory UI.

Deliberately stdlib: this is a single-user tool bound to loopback, so adding
an ASGI framework to the runtime dependencies would cost every collivind
install for one optional command.
"""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from collivind.web.api import MemoryAPI, route

logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
    ".svg": "image/svg+xml",
}
# Everything the page loads is its own: no inline script, no third party.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)
LOCAL_HOSTNAMES = {"127.0.0.1", "localhost", "[::1]", "::1"}
WRITE_METHODS = {"POST", "PATCH", "PUT", "DELETE"}
MAX_BODY = 1_000_000


def _hostname(netloc: str) -> str:
    netloc = netloc.strip().lower()
    if netloc.startswith("["):
        return netloc[: netloc.find("]") + 1]
    return netloc.rsplit(":", 1)[0] if ":" in netloc else netloc


def check_request(method: str, headers: Mapping[str, str]) -> Optional[Tuple[int, str]]:
    """Refuse requests a page on another origin could make. None means allowed.

    The server has no auth, so the browser is the threat: a page elsewhere
    must not read the store or write to it.
    - Host must name loopback. A DNS-rebinding page reaches 127.0.0.1 under
      its own hostname, and this is what tells the two apart. Applies to reads
      too, since reads expose every memory.
    - Writes must carry an Origin equal to the Host they were sent to (browsers
      always send one on POST/PATCH/DELETE), so another local port, say a dev
      server on :3000, cannot write either.
    - Writes must be JSON: a cross-site HTML form can only send text/plain or
      form encodings without a preflight.
    """
    host = headers.get("Host") or ""
    if _hostname(host) not in LOCAL_HOSTNAMES:
        return 403, "this UI only answers on localhost"
    if method not in WRITE_METHODS:
        return None
    origin = headers.get("Origin") or ""
    parsed = urlparse(origin)
    if parsed.scheme not in ("http", "https") or parsed.netloc.lower() != host.strip().lower():
        return 403, "cross-origin write refused (Origin must match this server)"
    content_type = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if content_type != "application/json":
        return 415, "writes must be application/json"
    return None


def _make_handler(api: MemoryAPI):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter than the stdlib default
            logger.debug(fmt, *args)

        def _send(self, status: int, body: bytes, content_type: str):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # A local tool has no business being embedded elsewhere.
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload):
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _static(self, path: str):
            name = "index.html" if path in ("/", "") else path.lstrip("/")
            target = (STATIC / name).resolve()
            # Never serve outside the static dir, whatever the URL claims.
            if not target.is_relative_to(STATIC.resolve()) or not target.is_file():
                return self._json(404, {"error": "not found"})
            self._send(200, target.read_bytes(), CONTENT_TYPES.get(target.suffix, "application/octet-stream"))

        def _handle(self, method: str):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            refused = check_request(method, self.headers)
            if refused:
                return self._json(refused[0], {"error": refused[1]})
            if not path.startswith("/api"):
                if method != "GET":
                    return self._json(405, {"error": "method not allowed"})
                return self._static(path)

            body = {}
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0:
                self.close_connection = True
                return self._json(400, {"error": "bad Content-Length"})
            if length > MAX_BODY:
                self.close_connection = True
                return self._json(413, {"error": "request body too large"})
            if length:
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return self._json(400, {"error": "body is not valid JSON"})

            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            try:
                status, payload = route(api, method, path, params, body)
            except Exception as e:  # a UI action must not take the server down
                logger.exception("request failed")
                status, payload = 500, {"error": f"{type(e).__name__}: {e}"}
            self._json(status, payload)

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

        def do_PATCH(self):
            self._handle("PATCH")

        def do_DELETE(self):
            self._handle("DELETE")

    return Handler


def build_server(api: MemoryAPI, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Bind the UI server. Loopback by default: this exposes the whole store.

    The container binds 0.0.0.0 inside its own network namespace and is
    published on the host's 127.0.0.1 only; check_request still refuses any
    Host header that is not loopback.
    """
    return ThreadingHTTPServer((host, port), _make_handler(api))
