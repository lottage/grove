"""Drive StoneSage's real HTTP handler in-process (no sockets): call(method, path, body) -> (status, content_type, body).

Used by the route golden tests: the layer split moves route code between files, and a route must answer exactly as before.
Runs under tests/run_tests.py's network guard, so anything a route tries to reach (Home Assistant, Frigate, the engine host) fails fast.
"""

import http.client
import io
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))

from stonesage import server  # noqa: E402


TEST_KEY = "test-server-key-0123456789abcdef"      # a made-up key; the harness makes sure the server under test has one


def server_key() -> str:
    """The key the server would check against (env, else config); a test key is put in the config when there is none."""
    from stonesage import auth
    key = auth.configured_key(server.config)
    if not key:
        server.config.setdefault("api", {})["key"] = key = TEST_KEY
    return key


class _Handler(server.StoneSageHandler):
    def __init__(self, method, path, body, headers, client=("127.0.0.1", 50000), tls=False):
        raw = b"" if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode("utf-8"))
        msg = http.client.HTTPMessage()
        msg["Host"] = "localhost:8888"
        msg["Content-Length"] = str(len(raw))
        if body is not None and not isinstance(body, bytes):
            msg["Content-Type"] = "application/json"
        for k, v in (headers or {}).items():
            del msg[k]                    # HTTPMessage appends: a header given here replaces the default one
            msg[k] = v
        self.command, self.path, self.headers = method, path, msg
        self.request_version, self.requestline = "HTTP/1.1", f"{method} {path} HTTP/1.1"
        self.client_address = client
        self.directory = server.FRONTEND_DIR
        if tls:
            import ssl
            self.connection = ssl.SSLSocket.__new__(ssl.SSLSocket)      # only its type is looked at
        self.server = types.SimpleNamespace(server_name="localhost", server_port=8888)
        self.rfile, self.wfile = io.BytesIO(raw), io.BytesIO()
        self.close_connection = True

    def log_message(self, *args):          # quiet
        pass


def call_full(method, path, body=None, headers=None, auth=True, client=("127.0.0.1", 50000), tls=False):
    """-> (status, response headers as a list of (name, value), payload). auth=True (the default) sends the server's key as a
    bearer, like a script would; auth=False sends no credentials. Explicit headers (an Authorization or Cookie of your own)
    are never overridden. `client` is the connecting address; tls=True makes the request look like it came over HTTPS."""
    headers = dict(headers or {})
    if auth and not any(k.lower() in ("authorization", "cookie") for k in headers):
        headers["Authorization"] = "Bearer " + server_key()
    h = _Handler(method, path, body, headers, client, tls)
    getattr(h, "do_" + method)()
    out = h.wfile.getvalue()
    head, _, payload = out.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split()[1]) if lines and lines[0].startswith("HTTP/") else 0
    hdrs = [tuple(l.split(":", 1)[0:1] + [l.split(":", 1)[1].strip()]) for l in lines[1:] if ":" in l]
    return status, hdrs, payload


def header(hdrs, name):
    """The first value of a response header (case-insensitive), or None."""
    return next((v for k, v in hdrs if k.lower() == name.lower()), None)


def call(method, path, body=None, headers=None, auth=True, client=("127.0.0.1", 50000), tls=False):
    """-> (status, content type, payload); see call_full."""
    status, hdrs, payload = call_full(method, path, body, headers, auth, client, tls)
    return status, header(hdrs, "content-type") or "", payload


def shape(method, path, body=None, **kw):
    """What a route's answer looks like, stable across runs: status, content type, and top-level JSON keys if any."""
    status, ctype, payload = call(method, path, body, **kw)
    out = {"status": status, "type": ctype.split(";")[0]}
    if "json" in ctype:
        try:
            data = json.loads(payload.decode("utf-8") or "null")
            out["keys"] = sorted(data) if isinstance(data, dict) else type(data).__name__
        except ValueError:
            out["keys"] = "not json"
    return out
