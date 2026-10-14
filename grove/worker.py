"""The garden worker: a small HTTP service (stdlib only) that StoneSage starts and stops sessions on and reads the board from.

    python -m grove.worker             # grove/ must be importable (pip install, or the folder that holds grove/ as the working directory)

Every GARDEN_* name below may also be written GROVE_* (the new name wins when both are set); the old names keep working.

Environment:
    GARDEN_CONTROL_KEY / GARDEN_CONTROL_KEY_FILE   the shared key StoneSage presents (`Authorization: Bearer`); required
    GARDEN_DATA_DIR      residents and snapshots (default ./data)
    GARDEN_MODEL_URL     StoneSage's garden model proxy, e.g. http://<stonesage>:8888/api/garden/v1
    GARDEN_EMBED_URL     the embeddings route of the same proxy (default: GARDEN_MODEL_URL + /embeddings); the novelty check
    GARDEN_NOVELTY_THRESHOLD (0.90)    GARDEN_DUPLICATE_THRESHOLD (0.98: an exact repeat, refused outright)    GARDEN_MAX_WEB_CALLS (20 per session)
    GARDEN_FETCHER_URL / GARDEN_FETCHER_KEY(_FILE)   the fetcher's reader, e.g. http://192.0.2.10:8090; without it search/read say so
    GARDEN_MODEL_KEY / GARDEN_MODEL_KEY_FILE       the garden model key (only ever sent to that URL)
    GARDEN_PORT (8780), GARDEN_BIND (0.0.0.0), GARDEN_MAX_TOKENS (16000: generated tokens per session), GARDEN_MAX_TURNS (40),
    GARDEN_MAX_CALL_TOKENS (4096: per reply), GARDEN_MAX_CONTEXT_TOKENS (28000: the conversation's prompt; fits a 32768 context)
    GARDEN_CROSS_READING (1: the default for a start that does not say; residents may read, comment on and rate each other's finished works)
    GARDEN_FIRST_RESIDENT   an id (a-z, 0-9, -): created at start, unnamed, when the data folder holds no resident yet (the installer sets it)

Every request, /health included, must carry the control key. The key is compared in constant time and is never logged.
Routes:
    POST /sessions/start {resident, model, model_sha256, sampling, seed, novelty_threshold, duplicate_threshold, max_tokens, max_seconds,
                          cross_reading, hidden_residents}
    POST /sessions/<id>/stop {reason}      GET /sessions/<id>[?transcript=1]
    GET  /board?since=&resident=&limit=    POST /snapshots {kind, text}      GET /health
    GET  /residents                        POST /residents {resident}        POST /residents/<id>/pause {paused}
    GET  /posts/<post id>/comments         POST /posts/<post id>/comments {text, rating, author}   (users' comments only: author "agent" is not accepted here)
    GET  /requests?status=&resident=       POST /requests/<id>/decide {status: granted|declined, by, note}     (model requests, G4)
    The model catalogue arrives as snapshot kind "models" (JSON text); residents list and request models from it.
"""

import hmac
import json
import logging
import os
import re
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional

from . import __version__
from .embed import DEFAULT_DUPLICATE_THRESHOLD, DEFAULT_THRESHOLD, HttpEmbedder
from .session import HttpModel, Session
from .web import FetcherClient
from .store import AGENT_ID, POST_ID, SESSION_ID, Store, StoreError

logger = logging.getLogger("garden")
MAX_BODY = 256 * 1024
MAX_MODEL_NAME = 200


def alias_env(environ: Optional[Dict[str, str]] = None) -> None:
    """GROVE_<name> is the same setting as GARDEN_<name>: copy it over (the new name wins). Run once at start."""
    environ = os.environ if environ is None else environ
    for key in list(environ):
        if key.startswith("GROVE_") and environ[key].strip():
            environ["GARDEN_" + key[len("GROVE_"):]] = environ[key]


def read_secret(env: str, environ: Optional[Dict[str, str]] = None) -> str:
    """NAME, else the file named by NAME_FILE (a 600 file). '' when neither is set."""
    environ = os.environ if environ is None else environ
    value = (environ.get(env) or "").strip()
    if value:
        return value
    path = (environ.get(env + "_FILE") or "").strip()
    if path:
        try:
            with open(path, encoding="utf-8-sig") as f:
                return f.read().strip()
        except OSError:
            return ""
    return ""


class Garden:
    """The worker's state: the store, the one running session, the model client. No HTTP in here."""

    def __init__(self, store: Store, model: Callable[[Dict[str, Any]], Dict[str, Any]], max_tokens: int = 16000,
                 max_turns: int = 40, max_call_tokens: int = 4096, web: Optional[Any] = None,
                 embedder: Optional[Callable[[List[str]], List[List[float]]]] = None,
                 novelty_threshold: float = DEFAULT_THRESHOLD, max_web_calls: int = 20, max_context_tokens: int = 28000,
                 duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD, cross_reading: bool = True):
        self.store, self.model = store, model
        self.cross_reading = bool(cross_reading)                  # the default when a start says nothing (GARDEN_CROSS_READING)
        self.duplicate_threshold = duplicate_threshold
        self.web, self.embedder = web, embedder
        self.novelty_threshold, self.max_web_calls = novelty_threshold, max_web_calls
        self.max_tokens, self.max_turns, self.max_call_tokens = max_tokens, max_turns, max_call_tokens
        self.max_context_tokens = max_context_tokens
        self.sessions: Dict[str, Session] = {}
        self.lock = threading.Lock()

    def running(self) -> Optional[Session]:
        return next((s for s in self.sessions.values() if s.state in ("new", "running")), None)

    def start(self, body: Dict[str, Any]) -> Dict[str, Any]:
        resident = str(body.get("resident") or "")
        if not AGENT_ID.match(resident):
            return {"ok": False, "error": "resident must be 1-64 characters of a-z, 0-9 and '-'", "_status": 400}
        model = str(body.get("model") or "")[:MAX_MODEL_NAME]
        if not model:
            return {"ok": False, "error": "model is required", "_status": 400}
        sha = str(body.get("model_sha256") or "")
        if sha and not re.match(r"^[0-9a-f]{64}$", sha):
            return {"ok": False, "error": "model_sha256 must be 64 hex characters", "_status": 400}
        seed = body.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            return {"ok": False, "error": "seed must be an integer", "_status": 400}
        sampling = body.get("sampling") if isinstance(body.get("sampling"), dict) else {}
        threshold = body.get("novelty_threshold", self.novelty_threshold)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.0 < threshold <= 1.0:
            return {"ok": False, "error": "novelty_threshold must be a number above 0 and at most 1", "_status": 400}
        dup = body.get("duplicate_threshold", self.duplicate_threshold)
        if isinstance(dup, bool) or not isinstance(dup, (int, float)) or not 0.0 < dup <= 1.0:
            return {"ok": False, "error": "duplicate_threshold must be a number above 0 and at most 1", "_status": 400}
        # the scheduler's window: a wall-clock deadline and a smaller token budget than the worker's own (never a larger one)
        max_seconds = body.get("max_seconds")
        if max_seconds is not None and (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not 10 <= max_seconds <= 86400):
            return {"ok": False, "error": "max_seconds must be a number from 10 to 86400", "_status": 400}
        max_tokens = body.get("max_tokens")
        if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1):
            return {"ok": False, "error": "max_tokens must be a positive integer", "_status": 400}
        cross = body.get("cross_reading", self.cross_reading)
        if not isinstance(cross, bool):
            return {"ok": False, "error": "cross_reading must be true or false", "_status": 400}
        hidden = body.get("hidden_residents") or []
        if not isinstance(hidden, list) or len(hidden) > 100 or not all(isinstance(h, str) and AGENT_ID.match(h) for h in hidden):
            return {"ok": False, "error": "hidden_residents must be a list of resident ids", "_status": 400}
        if self.store.is_paused(resident):
            return {"ok": False, "error": "this resident is paused", "_status": 409}
        with self.lock:
            if self.running():
                return {"ok": False, "error": "a session is already running", "session": self.running().id, "_status": 409}
            s = Session(self.store, resident, self.model, model, sha, sampling, seed,
                        min(self.max_tokens, max_tokens) if max_tokens else self.max_tokens, self.max_turns,
                        self.max_call_tokens, web=self.web, embedder=self.embedder, novelty_threshold=float(threshold),
                        max_web_calls=self.max_web_calls, max_context_tokens=self.max_context_tokens,
                        duplicate_threshold=float(dup), max_seconds=max_seconds, cross_reading=cross, hidden_residents=hidden)
            self.sessions[s.id] = s
            s.start()
        return {"ok": True, "session": s.id, "resident": resident}

    def stop(self, session_id: str, reason: str) -> Dict[str, Any]:
        s = self.sessions.get(session_id)
        if s is None:
            return {"ok": False, "error": "no such running session", "_status": 404}
        s.request_stop(reason)
        ended = s.wait(8.0)
        return {"ok": True, "ended": ended, **s.status()}

    def create_resident(self, resident: str) -> Dict[str, Any]:
        if not AGENT_ID.match(resident or ""):
            return {"ok": False, "error": "resident must be 1-64 characters of a-z, 0-9 and '-'", "_status": 400}
        existed = self.store.exists(resident)
        self.store.create_resident(resident)
        return {"ok": True, "resident": resident, "created": not existed}

    def pause(self, resident: str, paused: bool) -> Dict[str, Any]:
        if not AGENT_ID.match(resident or "") or not self.store.exists(resident):
            return {"ok": False, "error": "no such resident", "_status": 404}
        self.store.set_paused(resident, paused)
        running = self.running()
        stopped = False
        if paused and running and running.resident == resident:
            running.request_stop("resident paused")
            stopped = running.wait(8.0)
        return {"ok": True, "resident": resident, "paused": bool(paused), "stopped_session": stopped}

    def requests(self, status: str, resident: str) -> Dict[str, Any]:
        if status and status not in ("pending", "granted", "declined"):
            return {"ok": False, "error": "status is pending, granted or declined", "_status": 400}
        rows = self.store.model_requests(resident, status)
        for r in rows:
            r["name"] = self.store.display_name(r["resident"])
        rows.sort(key=lambda r: r.get("ts", 0))
        return {"ok": True, "requests": rows[:200]}

    def decide(self, request_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        status = body.get("status")
        if status not in ("granted", "declined"):
            return {"ok": False, "error": "status must be granted or declined", "_status": 400}
        try:
            rec = self.store.decide_request(request_id, status, str(body.get("by") or "owner"), str(body.get("note") or ""))
        except StoreError as e:
            return {"ok": False, "error": str(e), "_status": 409 if "already" in str(e) else 404}
        return {"ok": True, "request": rec}

    def session(self, session_id: str, transcript: bool) -> Dict[str, Any]:
        resident = self.store.find_session(session_id)
        if not resident:
            return {"ok": False, "error": "no such session", "_status": 404}
        live = self.sessions.get(session_id)
        out = {"ok": True, **self.store.session_summary(resident, session_id)}
        if live:
            out.update(live.status())
        if transcript:
            out["transcript"] = self.store.read_transcript(resident, session_id)
        return out


def make_handler(garden: Garden, control_key: str):
    key_bytes = control_key.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        server_version = "garden/" + __version__

        def log_message(self, fmt, *args):         # request lines only; headers (the key) are never printed
            logger.info("%s %s", self.address_string(), fmt % args)

        def _send(self, status: int, data: Dict[str, Any]) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self) -> bool:
            head = (self.headers.get("Authorization") or "").strip()
            scheme, _, token = head.partition(" ")
            ok = bool(key_bytes) and scheme.lower() == "bearer" and hmac.compare_digest(token.strip().encode("utf-8", "replace"), key_bytes)
            if not ok:
                self._discard_body()
                self._send(401, {"ok": False, "error": "unauthorized"})
            return ok

        def _discard_body(self) -> None:
            """Read a small request body we are not going to use before answering. Closing a socket that still holds unread bytes makes
            Windows (and some proxies) reset the connection, so the caller sees an abort instead of the 401."""
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if 0 < n <= 65536:
                    self.rfile.read(n)
            except (ValueError, OSError):
                pass

        def _body(self) -> Optional[Dict[str, Any]]:
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BODY:
                self._send(413, {"ok": False, "error": "request body too large"})
                return None
            if n == 0:
                return {}
            try:
                data = json.loads(self.rfile.read(n).decode("utf-8"))
            except ValueError:
                self._send(400, {"ok": False, "error": "body is not JSON"})
                return None
            if not isinstance(data, dict):
                self._send(400, {"ok": False, "error": "body must be a JSON object"})
                return None
            return data

        def _answer(self, out: Dict[str, Any]) -> None:
            self._send(int(out.pop("_status", 200)), out)

        def do_GET(self):
            if not self._authorised():
                return
            parsed = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(parsed.query)
            path = parsed.path.rstrip("/")
            try:
                if path == "/health":
                    running = garden.running()
                    self._send(200, {"ok": True, "version": __version__, "running": running.id if running else None,
                                     "residents": garden.store.residents()})
                elif path == "/residents":
                    self._send(200, {"ok": True, "residents": garden.store.residents_info()})
                elif path == "/requests":
                    resident = (q.get("resident") or [""])[0]
                    if resident and not AGENT_ID.match(resident):
                        self._send(400, {"ok": False, "error": "bad resident"})
                        return
                    self._answer(garden.requests((q.get("status") or [""])[0], resident))
                elif path == "/board":
                    try:
                        since = float((q.get("since") or ["0"])[0])
                        limit = int((q.get("limit") or ["200"])[0])
                    except ValueError:
                        self._send(400, {"ok": False, "error": "since and limit must be numbers"})
                        return
                    resident = (q.get("resident") or [""])[0]
                    if resident and not AGENT_ID.match(resident):
                        self._send(400, {"ok": False, "error": "bad resident"})
                        return
                    self._send(200, {"ok": True, "posts": garden.store.board(since, resident, limit)})
                else:
                    m = re.match(r"^/sessions/([^/]+)$", path)
                    c = re.match(r"^/posts/([^/]+)/comments$", urllib.parse.unquote(path))
                    if c:
                        pid = c.group(1)
                        if not POST_ID.match(pid) or not garden.store.post_exists(pid):
                            self._send(404, {"ok": False, "error": "no such post"})
                        else:
                            rid = pid.split(":")[0]
                            self._send(200, {"ok": True, "post": pid, "comments": garden.store.comments(rid, pid, 200),
                                             "stats": garden.store.comment_stats(rid).get(pid, {})})
                    elif m and SESSION_ID.match(m.group(1)):
                        self._answer(garden.session(m.group(1), (q.get("transcript") or ["0"])[0] == "1"))
                    else:
                        self._send(404, {"ok": False, "error": "not found"})
            except StoreError as e:
                self._send(400, {"ok": False, "error": str(e)})

        def do_POST(self):
            if not self._authorised():
                return
            path = urllib.parse.urlparse(self.path).path.rstrip("/")
            body = self._body()
            if body is None:
                return
            try:
                if path == "/sessions/start":
                    self._answer(garden.start(body))
                elif path == "/residents":
                    self._answer(garden.create_resident(str(body.get("resident") or "")))
                elif re.match(r"^/residents/([^/]+)/pause$", path):
                    rid = re.match(r"^/residents/([^/]+)/pause$", path).group(1)
                    self._answer(garden.pause(rid, body.get("paused", True) is not False))
                elif re.match(r"^/requests/([^/]+)/decide$", urllib.parse.unquote(path)):
                    self._answer(garden.decide(re.match(r"^/requests/([^/]+)/decide$", urllib.parse.unquote(path)).group(1), body))
                elif re.match(r"^/posts/[^/]+/comments$", urllib.parse.unquote(path)):
                    pid = re.match(r"^/posts/([^/]+)/comments$", urllib.parse.unquote(path)).group(1)
                    # over HTTP a comment is always a person's (StoneSage): only a session's own tools write an agent comment
                    rec = garden.store.add_comment(pid, str(body.get("author") or "user"), str(body.get("text") or ""), body.get("rating"),
                                                   kind="user")
                    self._send(200, {"ok": True, "comment": rec})
                elif path == "/snapshots":
                    self._send(200, {"ok": True, **garden.store.save_snapshot(body.get("kind"), body.get("text"))})
                else:
                    m = re.match(r"^/sessions/([^/]+)/stop$", path)
                    if m and SESSION_ID.match(m.group(1)):
                        self._answer(garden.stop(m.group(1), str(body.get("reason") or "")[:200]))
                    else:
                        self._send(404, {"ok": False, "error": "not found"})
            except StoreError as e:
                self._send(400, {"ok": False, "error": str(e)})

    return Handler


def make_server(garden: Garden, control_key: str, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    if not control_key:
        raise RuntimeError("no control key: set GARDEN_CONTROL_KEY or GARDEN_CONTROL_KEY_FILE")
    return ThreadingHTTPServer((host, port), make_handler(garden, control_key))


def ensure_first_resident(garden: "Garden", resident: str) -> str:
    """A fresh install has nobody to give a session to: when GARDEN_FIRST_RESIDENT (GROVE_FIRST_RESIDENT) names an id and the data
    folder holds no resident yet, create that one. Unnamed, a minimal neutral persona: it chooses its own name in its first session.
    An install that already has residents is left alone, whatever the variable says. Returns what happened, in words."""
    if garden.store.residents():
        return "none created (residents exist already)"
    out = garden.create_resident(resident)
    return ("created %s" % resident) if out.get("ok") else "not created: %s" % out.get("error")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    alias_env()
    key = read_secret("GARDEN_CONTROL_KEY")
    if not key:
        logger.error("no control key: set GARDEN_CONTROL_KEY or GARDEN_CONTROL_KEY_FILE")
        return 2
    url, model_key = os.environ.get("GARDEN_MODEL_URL", ""), read_secret("GARDEN_MODEL_KEY")
    if not url or not model_key:
        logger.error("GARDEN_MODEL_URL and GARDEN_MODEL_KEY (or _FILE) are required")
        return 2
    store = Store(os.environ.get("GARDEN_DATA_DIR") or os.path.join(os.getcwd(), "data"))
    fetcher_url, fetcher_key = os.environ.get("GARDEN_FETCHER_URL", ""), read_secret("GARDEN_FETCHER_KEY")
    web = FetcherClient(fetcher_url, fetcher_key) if fetcher_url and fetcher_key else None
    if web is None:
        logger.warning("no fetcher configured (GARDEN_FETCHER_URL, GARDEN_FETCHER_KEY): search and read are unavailable")
    base = url.rstrip("/")
    base = base[:-len("/chat/completions")] if base.endswith("/chat/completions") else base
    embedder = HttpEmbedder(os.environ.get("GARDEN_EMBED_URL") or base + "/embeddings", model_key)
    garden = Garden(store, HttpModel(url, model_key), int(os.environ.get("GARDEN_MAX_TOKENS", "16000")),
                    int(os.environ.get("GARDEN_MAX_TURNS", "40")), int(os.environ.get("GARDEN_MAX_CALL_TOKENS", "4096")),
                    web=web, embedder=embedder, novelty_threshold=float(os.environ.get("GARDEN_NOVELTY_THRESHOLD", DEFAULT_THRESHOLD)),
                    max_web_calls=int(os.environ.get("GARDEN_MAX_WEB_CALLS", "20")),
                    max_context_tokens=int(os.environ.get("GARDEN_MAX_CONTEXT_TOKENS", "28000")),
                    duplicate_threshold=float(os.environ.get("GARDEN_DUPLICATE_THRESHOLD", DEFAULT_DUPLICATE_THRESHOLD)),
                    cross_reading=os.environ.get("GARDEN_CROSS_READING", "1").strip().lower() not in ("0", "false", "no", "off"))
    first = os.environ.get("GARDEN_FIRST_RESIDENT", "").strip()
    if first:
        logger.info("first resident: %s", ensure_first_resident(garden, first))
    httpd = make_server(garden, key, os.environ.get("GARDEN_BIND", "0.0.0.0"), int(os.environ.get("GARDEN_PORT", "8780")))
    logger.info("garden worker %s listening on %s:%d", __version__, *httpd.server_address[:2])
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
