"""HTTP plumbing for the local memory UI.

Deliberately stdlib: this is a single-user tool bound to loopback, so adding
an ASGI framework to the runtime dependencies would cost every collivind
install for one optional command.
"""

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from collivind.web.api import MemoryAPI, route

logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript"}


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
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload):
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _static(self, path: str):
            name = "index.html" if path in ("/", "") else path.lstrip("/")
            target = (STATIC / name).resolve()
            # Never serve outside the static dir, whatever the URL claims.
            if not str(target).startswith(str(STATIC.resolve())) or not target.is_file():
                return self._json(404, {"error": "not found"})
            self._send(200, target.read_bytes(), CONTENT_TYPES.get(target.suffix, "application/octet-stream"))

        def _handle(self, method: str):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if not path.startswith("/api"):
                if method != "GET":
                    return self._json(405, {"error": "method not allowed"})
                return self._static(path)

            body = {}
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    return self._json(400, {"error": "body is not valid JSON"})

            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            try:
                status, payload = route(api, method, path, params, body)
            except Exception as e:  # a UI action must not take the server down
                logger.exception("request failed")
                status, payload = 500, {"error": str(e)}
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
    """Bind the UI server. Loopback only — this exposes the whole store."""
    return ThreadingHTTPServer((host, port), _make_handler(api))
