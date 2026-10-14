"""StoneSage's side of Grove (the design notes, step G1): a scoped model proxy, and a client for the worker.

Settings live in config.json section `grove` (or `garden`, its older name; stonesage_app/section.py says which is used). The routes
keep the path prefix /api/garden/ so a worker configured before the add-on had its name keeps working.

The garden worker runs on its own machine and holds the residents. It touches StoneSage in exactly one way, the model proxy:

    POST /api/garden/v1/chat/completions    accepts the model key (declared as a scoped key: valid on these two routes only) or the owner's key
    POST /api/garden/v1/embeddings          the novelty check: forwards {input: [text, ...]} (each under 950 characters) to the embedder

The proxy is deliberately dumb. It forwards a chat body to the one engine in config `garden.engine_url`, keeps only a short list of
fields, caps the size and `max_tokens`, and returns what the engine said. The tool schemas the model was given are the garden's own;
the proxy never runs a tool, never adds memory, retrieval or grounding, and never routes to Computer's loop.

Going the other way (login required, the worker's control key from config `garden.control_key` or env STONESAGE_GARDEN_CONTROL_KEY):

    GET  /api/garden/board?since=&resident=&limit=
    POST /api/garden/sessions/start               {resident, model, model_sha256, sampling, seed}  (config garden.* fills the gaps)
    POST /api/garden/sessions/<id>/stop           {reason}
    GET  /api/garden/worker/status                       the worker's health (the running session, the residents) and the window feeder's state
    GET  /api/garden/residents                    POST {resident} creates one         POST /api/garden/residents/<id>/pause {paused}
    GET  /api/garden/posts/<post id>/comments     POST {text, rating}  (comments and 1-5 ratings; the resident reads them with read_comments)
    POST /api/garden/window/refresh               build and push the window snapshots now (stonesage/garden_window.py)

    GET  /api/garden/schedule                     the scheduler's status (garden_scheduler.py)       POST {enabled}  switch it on / off (persisted to config)
    POST /api/garden/residents/<id>/model         {model}  assign a model from garden.models ("" clears)

    GET  /api/garden/requests                     the residents' pending model requests (G4), pulled from the worker first
    POST /api/garden/requests/<id>/decide         {status: granted|declined, note, layout?, engine_url?, unit?}  the owner's answer (garden_models.py)

Requests (G4, garden_models.py): `garden.models` + `garden.residents.<id>.allowed_models` (auto-grant), `garden.grants.push` (phone buttons, default
true). Cross-reading (G5): `garden.cross_reading` (default true) and `garden.residents.<id>.cross_reading` (false = that resident neither reads nor
is read); both are sent with every session start.

Models (G3): config `garden.models` = {name: {layout, engine_url, model_sha256, build?, unit?, tok_s?}}; `layout` null = the CPU slot (`unit`
is started on demand by the scheduler). A resident's model is `garden.residents.<id>.model` (a name in that table). The proxy forwards to the
engine of the session that is running (kept in memory when StoneSage starts it), else `garden.engine_url`: switching models needs no config
edit and no restart.

Config (`garden`): worker_url, control_key, model_key, engine_url, embed_url (default: the embedder endpoint), novelty_threshold (0.90),
duplicate_threshold (0.98), resident, model, model_sha256, sampling, max_body_bytes, max_tokens_cap, engine_timeout_s, models, residents, schedule;
window feeds: see garden_window.py. Session starts and stops are traced as kind "garden" (metadata only).
"""

import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Optional

from lodestone import endpoints

from stonesage import auth

from . import section as sec
from .section import section as _cfg

OWNER = "grove"                    # the route owner in the platform's table
SCOPED = "grove"                   # the principal of a request that carries the model key (auth.register_scoped_key)
MODEL_ROUTE = ("POST", "/api/garden/v1/chat/completions")
EMBED_ROUTE = ("POST", "/api/garden/v1/embeddings")        # the novelty check's embedder, same key
SCOPED_ROUTES = (MODEL_ROUTE, EMBED_ROUTE)                  # the only two routes the model key opens
SESSION_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[a-f0-9]{4}$")
DEFAULT_MAX_BODY = 256 * 1024
DEFAULT_MAX_TOKENS = 4096
ALLOWED_ROLES = ("system", "user", "assistant", "tool")
MAX_MESSAGES = 200
# the only request fields that reach the engine: the conversation, the model's own tool schemas, and sampling
FORWARD_KEYS = ("messages", "tools", "tool_choice", "temperature", "top_p", "top_k", "min_p", "seed", "stop",
                "presence_penalty", "frequency_penalty", "repeat_penalty", "parse_tool_calls")
PARSE_FAILURE = "Failed to parse tool call"       # the engine's 500 text when its grammar rejects the model's own tool-call JSON
ANSWER_KEYS = ("id", "object", "created", "model", "choices", "usage")
MAX_EMBED_INPUTS = 64
MAX_EMBED_CHARS = 950             # the BGE embedder's limit (CLAUDE.md: inputs < 950 chars)
SCHEDULER: Optional[Any] = None       # the running scheduler, for /metrics (set by install)
POST_ID = re.compile(r"^[a-z0-9-]{1,64}:(?:work|journal|wish|draft|session):[A-Za-z0-9._-]{1,80}$")
REQUEST_ID = re.compile(r"^[a-z0-9-]{1,64}\.r[0-9]{5,}$")




AGENT_ID = re.compile(r"^[a-z0-9-]{1,64}$")
ACTIVE_SLACK_S = 600
ACTIVE_DEFAULT_S = 4 * 3600


def models_table(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """garden.models: {name: {layout, engine_url, model_sha256, build?, unit?, tok_s?}}; entries that are not objects are ignored."""
    m = _cfg(config).get("models")
    return {str(k): v for k, v in m.items() if isinstance(v, dict)} if isinstance(m, dict) else {}


def cross_settings(config: Dict[str, Any], resident: str):
    """(may this resident read, comment on and rate the others' finished works, the residents who opted out and are hidden from the rest)."""
    g = _cfg(config)
    res = g.get("residents") if isinstance(g.get("residents"), dict) else {}
    out = sorted(r for r, e in res.items() if isinstance(e, dict) and e.get("cross_reading") is False and AGENT_ID.match(str(r)))
    return (g.get("cross_reading", True) is not False and resident not in out), out


def resident_model(config: Dict[str, Any], resident: str) -> str:
    """The model assigned to a resident (garden.residents.<id>.model), or ''."""
    r = (_cfg(config).get("residents") or {}).get(resident)
    return str(r.get("model") or "") if isinstance(r, dict) else ""


class ActiveEngine:
    """The engine of the garden session that is running, kept in memory: the proxy forwards there. Set by StoneSage when it starts a
    session (by hand or by the scheduler), cleared when the session is stopped or seen to have ended; it also expires on its own."""

    def __init__(self, clock: Callable[[], float] = time.time):
        self.clock, self._lock = clock, threading.Lock()
        self.session: Optional[str] = None
        self.engine_url = ""
        self.model = ""
        self.by = ""
        self.until = 0.0

    def engine(self) -> Optional[str]:
        with self._lock:
            return self.engine_url if self.engine_url and self.clock() < self.until else None

    def _state(self) -> Dict[str, Any]:
        return {"session": self.session, "engine_url": self.engine_url, "model": self.model, "by": self.by, "until": self.until}

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._state()

    def set(self, engine_url: str, model: str, by: str, ttl_s: float = ACTIVE_DEFAULT_S) -> Dict[str, Any]:
        """Take the engine for a session about to start; returns the previous state for restore() if the start fails."""
        with self._lock:
            prev = self._state()
            self.session, self.engine_url, self.model, self.by = None, engine_url, model, by
            self.until = self.clock() + float(ttl_s) + ACTIVE_SLACK_S
            return prev

    def bind(self, session: str) -> None:
        with self._lock:
            self.session = session

    def restore(self, prev: Dict[str, Any]) -> None:
        with self._lock:
            self.session, self.engine_url, self.model = prev["session"], prev["engine_url"], prev["model"]
            self.by, self.until = prev["by"], prev["until"]

    def clear(self, session: Optional[str] = None) -> None:
        """Forget the engine. With a session id, only when that is the session it was set for."""
        with self._lock:
            if session is not None and self.session not in (None, session):
                return
            self.session, self.engine_url, self.model, self.by, self.until = None, "", "", "", 0.0


def control_key(config: Dict[str, Any]) -> str:
    return sec.env("CONTROL_KEY") or str(_cfg(config).get("control_key") or "").strip()


def model_key(config: Optional[Dict[str, Any]]) -> str:
    """The worker's model key: env STONESAGE_GROVE_MODEL_KEY (or the older STONESAGE_GARDEN_MODEL_KEY), else config
    grove.model_key / garden.model_key; '' when none. It authenticates SCOPED_ROUTES (chat completions, embeddings) and nothing else,
    and the platform refuses it when it equals the server key (auth.scoped_key_value)."""
    return sec.env("MODEL_KEY") or str(_cfg(config or {}).get("model_key") or "").strip()


# ---- the model proxy ----------------------------------------------------------------------------------
def sanitize_body(body: Dict[str, Any], max_tokens_cap: int = DEFAULT_MAX_TOKENS) -> Dict[str, Any]:
    """The chat body the engine gets: whitelisted fields only, one non-streamed answer, `max_tokens` capped. Raises ValueError."""
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or len(msgs) > MAX_MESSAGES:
        raise ValueError("messages must be a list of 1-%d items" % MAX_MESSAGES)
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ALLOWED_ROLES:
            raise ValueError("every message needs a role of system, user, assistant or tool")
    out = {k: body[k] for k in FORWARD_KEYS if k in body}
    if "tools" in out and not (isinstance(out["tools"], list) and all(isinstance(t, dict) for t in out["tools"])):
        raise ValueError("tools must be a list of objects")
    if "parse_tool_calls" in out and not isinstance(out["parse_tool_calls"], bool):
        del out["parse_tool_calls"]
    asked = body.get("max_tokens")
    asked = asked if isinstance(asked, int) and not isinstance(asked, bool) and asked > 0 else max_tokens_cap
    out["max_tokens"] = min(asked, max_tokens_cap)
    out["stream"] = False
    out["n"] = 1
    return out


def _chat_url(engine_url: str) -> str:
    u = engine_url.rstrip("/")
    return u if u.endswith("/chat/completions") else u + "/chat/completions"


def model_proxy(config: Dict[str, Any], opener: Optional[Callable[..., Any]] = None,
                engine_of: Optional[Callable[[], Optional[str]]] = None, cfg_fn: Optional[Callable[[], Dict[str, Any]]] = None):
    """The route function for POST /api/garden/v1/chat/completions. `opener` is urllib's urlopen unless a test gives its own.
    `engine_of()` is the engine of the session that is running (ActiveEngine.engine); without one, or when it says None, garden.engine_url."""
    opener = opener or urllib.request.urlopen

    def route(h, parsed, body: Dict[str, Any]) -> None:
        if getattr(h, "principal", None) not in (auth.OWNER, SCOPED):
            auth.reply(h, 401, {"ok": False, "error": "unauthorized"}, {"WWW-Authenticate": "Bearer"})
            return
        g = _cfg(cfg_fn() if cfg_fn else config)
        engine = (engine_of() if engine_of else None) or str(g.get("engine_url") or "").strip()
        if not engine:
            auth.reply(h, 503, {"ok": False, "error": "garden.engine_url is not set and no session is running"})
            return
        limit = int(g.get("max_body_bytes") or DEFAULT_MAX_BODY)
        if len(getattr(h, "_raw_body", b"") or b"") > limit:
            auth.reply(h, 413, {"ok": False, "error": "request body too large"})
            return
        try:
            send = sanitize_body(body if isinstance(body, dict) else {}, int(g.get("max_tokens_cap") or DEFAULT_MAX_TOKENS))
        except ValueError as e:
            auth.reply(h, 400, {"ok": False, "error": str(e)})
            return
        req = urllib.request.Request(_chat_url(engine), data=json.dumps(send).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with opener(req, timeout=float(g.get("engine_timeout_s") or 600)) as r:
                answer = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = e.read(4096).decode("utf-8", "replace")
            except OSError:
                detail = ""
            if PARSE_FAILURE in detail:              # the caller can retry without engine-side parsing; nothing else of the text leaves
                auth.reply(h, 502, {"ok": False, "error": PARSE_FAILURE + " in the engine's answer"})
                return
            auth.reply(h, 502, {"ok": False, "error": "the engine answered HTTP %d" % e.code})
            return
        except (urllib.error.URLError, OSError, ValueError):
            auth.reply(h, 502, {"ok": False, "error": "the engine did not answer"})
            return
        auth.reply(h, 200, {k: answer[k] for k in ANSWER_KEYS if k in answer} if isinstance(answer, dict) else {})

    return route


def embed_inputs(body: Dict[str, Any]) -> list:
    """The list of texts to embed, else ValueError. Every text must be non-empty and shorter than the embedder's limit: the garden
    chunks its own text, this route does not guess."""
    raw = body.get("input")
    items = [raw] if isinstance(raw, str) else raw
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_EMBED_INPUTS:
        raise ValueError("input must be a text or a list of 1-%d texts" % MAX_EMBED_INPUTS)
    out = []
    for t in items:
        if not isinstance(t, str) or not t.strip():
            raise ValueError("every input must be a non-empty text")
        if len(t) >= MAX_EMBED_CHARS:
            raise ValueError("every input must be shorter than %d characters: chunk it first" % MAX_EMBED_CHARS)
        out.append(t)
    return out


def _embed_url(config: Dict[str, Any]) -> str:
    u = str(_cfg(config).get("embed_url") or "").strip() or endpoints.url("embedder", config)
    u = u.rstrip("/")
    return u if u.endswith("/embeddings") else u + "/embeddings"


def embed_proxy(config: Dict[str, Any], opener: Optional[Callable[..., Any]] = None):
    """The route function for POST /api/garden/v1/embeddings: the garden key (or the owner's) may ask for vectors, nothing else.
    It forwards only `input` to the embedder and returns only the vectors."""
    opener = opener or urllib.request.urlopen

    def route(h, parsed, body: Dict[str, Any]) -> None:
        if getattr(h, "principal", None) not in (auth.OWNER, SCOPED):
            auth.reply(h, 401, {"ok": False, "error": "unauthorized"}, {"WWW-Authenticate": "Bearer"})
            return
        g = _cfg(config)
        limit = int(g.get("max_body_bytes") or DEFAULT_MAX_BODY)
        if len(getattr(h, "_raw_body", b"") or b"") > limit:
            auth.reply(h, 413, {"ok": False, "error": "request body too large"})
            return
        try:
            inputs = embed_inputs(body if isinstance(body, dict) else {})
        except ValueError as e:
            auth.reply(h, 400, {"ok": False, "error": str(e)})
            return
        req = urllib.request.Request(_embed_url(config), data=json.dumps({"input": inputs}).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with opener(req, timeout=float(g.get("embed_timeout_s") or 60)) as r:
                answer = json.loads(r.read().decode("utf-8"))
            rows = answer["data"]
            data = [{"object": "embedding", "index": int(d.get("index", i)), "embedding": [float(x) for x in d["embedding"]]}
                    for i, d in enumerate(rows)]
            if len(data) != len(inputs):
                raise ValueError("wrong number of vectors")
        except urllib.error.HTTPError as e:
            auth.reply(h, 502, {"ok": False, "error": "the embedder answered HTTP %d" % e.code})
            return
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError):
            auth.reply(h, 502, {"ok": False, "error": "the embedder did not answer"})
            return
        auth.reply(h, 200, {"object": "list", "data": data, "model": "embedder"})

    return route


# ---- the worker's client --------------------------------------------------------------------------------
class GardenError(RuntimeError):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


class GardenClient:
    def __init__(self, config: Dict[str, Any], opener: Optional[Callable[..., Any]] = None,
                 cfg_fn: Optional[Callable[[], Dict[str, Any]]] = None, active: Optional[ActiveEngine] = None):
        self.config = config
        self.cfg_fn = cfg_fn or (lambda: self.config)       # the live config (the file, re-read) when StoneSage gives one
        self.active = active if active is not None else ActiveEngine()
        self.opener = opener or urllib.request.urlopen

    def _call(self, method: str, path: str, body: Optional[Dict[str, Any]] = None, timeout: float = 15.0) -> Dict[str, Any]:
        conf = self.cfg_fn()
        base = str(_cfg(conf).get("worker_url") or "").rstrip("/")
        key = control_key(conf)
        if not base or not key:
            raise GardenError("garden.worker_url and garden.control_key are not set", 503)
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(base + path, data=data, method=method,
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with self.opener(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode("utf-8")).get("error", "")
            except (ValueError, OSError, AttributeError):
                detail = ""
            raise GardenError("the garden worker answered HTTP %d%s" % (e.code, ": " + str(detail)[:200] if detail else ""), e.code) from None
        except (urllib.error.URLError, OSError, ValueError):
            raise GardenError("the garden worker did not answer") from None

    def health(self) -> Dict[str, Any]:
        return self._call("GET", "/health")

    def board(self, since: float = 0.0, resident: str = "", limit: int = 200) -> Dict[str, Any]:
        q = {"since": since, "limit": limit}
        if resident:
            q["resident"] = resident
        return self._call("GET", "/board?" + urllib.parse.urlencode(q))

    def resolve_model(self, resident: str, requested: str = "") -> Dict[str, Any]:
        """Which model a session for `resident` runs: the one asked for, else the one assigned to the resident, else garden.model.
        A name in garden.models carries its own engine_url and sha256; any other name is the old single-engine setup."""
        conf = self.cfg_fn()
        g = _cfg(conf)
        name = str(requested or resident_model(conf, resident) or g.get("model") or "")
        entry = models_table(conf).get(name)
        if entry:
            return {"name": name, "sha256": str(entry.get("model_sha256") or ""), "engine_url": str(entry.get("engine_url") or "").strip(),
                    "entry": entry}
        return {"name": name, "sha256": str(g.get("model_sha256") or ""), "engine_url": "", "entry": None}

    def start_session(self, body: Dict[str, Any], by: str = "manual") -> Dict[str, Any]:
        g = _cfg(self.cfg_fn())
        resident = body.get("resident") or g.get("resident") or "resident-1"
        model = self.resolve_model(resident, str(body.get("model") or ""))
        send = {"resident": resident, "model": model["name"], "model_sha256": body.get("model_sha256") or model["sha256"],
                "sampling": body.get("sampling") if isinstance(body.get("sampling"), dict) else (g.get("sampling") or {})}
        if body.get("seed") is not None:
            send["seed"] = body["seed"]
        threshold = body.get("novelty_threshold", g.get("novelty_threshold"))
        if threshold is not None:
            send["novelty_threshold"] = threshold
        dup = body.get("duplicate_threshold", g.get("duplicate_threshold"))
        if dup is not None:
            send["duplicate_threshold"] = dup
        send["cross_reading"], send["hidden_residents"] = cross_settings(self.cfg_fn(), resident)
        for k in ("max_seconds", "max_tokens"):          # the scheduler's window; the worker clamps them to its own budgets
            if body.get(k) is not None:
                send[k] = body[k]
        # The proxy forwards to this session's engine from its first call, which can come before the worker has even answered us,
        # so the engine is set first and given back if the worker refuses (a session already running keeps its own).
        prev = self.active.set(model["engine_url"], model["name"], by, float(send.get("max_seconds") or ACTIVE_DEFAULT_S))
        try:
            out = self._call("POST", "/sessions/start", send)
        except Exception:
            self.active.restore(prev)
            raise
        if not out.get("session"):
            self.active.restore(prev)
            return out
        self.active.bind(out["session"])
        return out

    def stop_session(self, session_id: str, reason: str = "") -> Dict[str, Any]:
        if not SESSION_ID.match(session_id or ""):
            raise GardenError("bad session id", 400)
        try:
            return self._call("POST", "/sessions/%s/stop" % session_id, {"reason": reason}, timeout=20.0)
        finally:
            self.active.clear(session_id)

    def session(self, session_id: str) -> Dict[str, Any]:
        if not SESSION_ID.match(session_id or ""):
            raise GardenError("bad session id", 400)
        return self._call("GET", "/sessions/%s" % session_id)

    def push_snapshot(self, kind: str, text: str) -> Dict[str, Any]:
        return self._call("POST", "/snapshots", {"kind": kind, "text": text})

    def residents(self) -> Dict[str, Any]:
        return self._call("GET", "/residents")

    def create_resident(self, resident: str) -> Dict[str, Any]:
        return self._call("POST", "/residents", {"resident": resident})

    def pause_resident(self, resident: str, paused: bool) -> Dict[str, Any]:
        if not re.match(r"^[a-z0-9-]{1,64}$", resident or ""):
            raise GardenError("bad resident id", 400)
        return self._call("POST", "/residents/%s/pause" % resident, {"paused": bool(paused)}, timeout=20.0)

    def requests(self, status: str = "pending") -> Dict[str, Any]:
        return self._call("GET", "/requests?" + urllib.parse.urlencode({"status": status} if status else {}))

    def decide_request(self, request_id: str, status: str, by: str = "owner", note: str = "") -> Dict[str, Any]:
        if not REQUEST_ID.match(request_id or ""):
            raise GardenError("bad request id", 400)
        return self._call("POST", "/requests/%s/decide" % urllib.parse.quote(request_id, safe=""),
                          {"status": status, "by": by, "note": note}, timeout=20.0)

    def comments(self, post_id: str) -> Dict[str, Any]:
        if not POST_ID.match(post_id or ""):
            raise GardenError("bad post id", 400)
        return self._call("GET", "/posts/%s/comments" % urllib.parse.quote(post_id, safe=""))

    def add_comment(self, post_id: str, text: str, rating: Any, author: str = "user") -> Dict[str, Any]:
        if not POST_ID.match(post_id or ""):
            raise GardenError("bad post id", 400)
        return self._call("POST", "/posts/%s/comments" % urllib.parse.quote(post_id, safe=""),
                          {"text": text, "rating": rating, "author": author})


# ---- routes behind the normal login ------------------------------------------------------------------------
class _LazyLease:
    """The engine lease service is built on first use; the scheduler reaches it through this so install() never builds it."""

    def __init__(self, getter: Callable[[], Any]):
        self._getter = getter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._getter(), name)


def install(ctx, client: Optional[GardenClient] = None, proxy_opener: Optional[Callable[..., Any]] = None,
            feeder: Optional[Any] = None, scheduler: Optional[Any] = None) -> None:
    config = ctx.config
    load = ctx.services.get("load_config")
    save = ctx.services.get("save_config")
    # The platform's live config object (changed in place by persist() below, and written to config.json when it can be): the
    # assignments and the schedule switch made from the board apply at once, with no restart.
    cfg_fn: Callable[[], Dict[str, Any]] = lambda: config
    client = client or GardenClient(config, cfg_fn=cfg_fn)
    active = getattr(client, "active", None)
    if active is None:
        active = client.active = ActiveEngine()
    if scheduler is None:
        get_lease, get_gate = ctx.services.get("get_engine_lease"), ctx.services.get("get_idle_gate")
        if callable(get_lease) and callable(get_gate) and control_key(config) and str(_cfg(config).get("worker_url") or "").strip():
            from . import scheduler as garden_scheduler

            def write_trace(rec: Dict[str, Any]) -> None:
                getter = ctx.services.get("get_trace_log")
                if getter:
                    getter().write(rec)

            scheduler = garden_scheduler.GardenScheduler(
                cfg_fn, lambda fresh: get_gate().check(fresh=fresh), _LazyLease(get_lease), client, active,
                ctx.services.get("last_courage_turn") or (lambda: None), ctx.services.get("pending_approvals") or (lambda: 0),
                trace=write_trace, data_dir=ctx.data_dir(), systemctl=ctx.services.get("lease_systemctl"),
                http_ok=ctx.services.get("http_ok"))
            ctx.on_start(scheduler.start)     # always runs once the garden is configured: it records gate history; it acts only if enabled
    if feeder is None:
        from . import window as garden_window

        def recent(limit: int, kind: str):
            getter = ctx.services.get("get_trace_log")
            return getter().recent(limit, kind) if getter else []

        def terms():
            provide = ctx.hook("privacy_terms")
            return list(provide()) if provide else []

        def catalogue() -> str:
            from . import models as garden_models
            getter = ctx.services.get("get_provenance")
            try:
                summary = getter().summary() if getter else None
            except Exception:
                summary = None
            sess = getattr(scheduler, "sessions", None)
            return garden_models.catalogue_text(cfg_fn(), summary, sess.tok_s if sess is not None and hasattr(sess, "tok_s") else None)

        feeder = garden_window.WindowFeeder(config, client.push_snapshot, trace_recent=recent, extra_terms=terms, catalogue=catalogue)
        if control_key(config) and str(_cfg(config).get("worker_url") or "").strip():
            ctx.on_start(feeder.start)

    def trace(event: str, outcome: str, **fields: Any) -> None:
        try:
            getter = ctx.services.get("get_trace_log")
            if getter:
                getter().write({"kind": "garden", "at": round(time.time(), 3), "event": event, "outcome": outcome, **fields})
        except Exception:
            pass

    def owner_only(h) -> bool:
        if getattr(h, "principal", None) != auth.OWNER:
            auth.reply(h, 401, {"ok": False, "error": "unauthorized"}, {"WWW-Authenticate": "Bearer"})
            return False
        return True

    def failure(h, e: GardenError) -> None:
        auth.reply(h, e.status if 400 <= e.status < 600 else 502, {"ok": False, "error": str(e)})

    grants: Optional[Any] = None          # the model-request queue (garden_models.ModelGrants), built below

    def board(h, parsed, body):
        if not owner_only(h):
            return
        if grants is not None:
            try:
                grants.maybe_tick()           # requests are pulled and decided on a board read too (throttled)
            except Exception:
                pass
        q = urllib.parse.parse_qs(parsed.query)
        try:
            since = float((q.get("since") or ["0"])[0])
            limit = int((q.get("limit") or ["200"])[0])
            auth.reply(h, 200, client.board(since, (q.get("resident") or [""])[0], limit))
        except ValueError:
            auth.reply(h, 400, {"ok": False, "error": "since and limit must be numbers"})
        except GardenError as e:
            failure(h, e)

    global SCHEDULER
    SCHEDULER = scheduler

    def start(h, parsed, body):
        if not owner_only(h):
            return
        body = body if isinstance(body, dict) else {}
        held = scheduler.holding() if scheduler else None
        if held:                           # a manual start while the scheduler holds a lease would fight it for the engine
            trace("start", "refused", error=held[:200])
            auth.reply(h, 409, {"ok": False, "error": held + ": a session cannot be started by hand until its window ends "
                                                      "(switch the schedule off on the board to end it now)"})
            return
        conf = cfg_fn()
        model = str(body.get("model") or resident_model(conf, str(body.get("resident") or _cfg(conf).get("resident") or ""))
                    or _cfg(conf).get("model") or "")
        try:
            out = client.start_session(body)
        except GardenError as e:
            trace("start", "refused" if e.status in (400, 409) else "error", error=str(e)[:200])
            failure(h, e)
            return
        if scheduler:
            scheduler.note_manual_start(out, model)
        trace("start", "started", session=out.get("session"), resident=out.get("resident"), model=model[:120])
        auth.reply(h, 200, out)

    def stop(h, parsed, body):
        if not owner_only(h):
            return
        m = re.match(r"^/api/garden/sessions/([^/]+)/stop$", parsed.path)
        if not m:
            auth.reply(h, 404, {"ok": False, "error": "not found"})
            return
        reason = str((body or {}).get("reason") or "stopped from StoneSage")[:200]
        try:
            out = client.stop_session(m.group(1), reason)
        except GardenError as e:
            trace("stop", "error", session=m.group(1), error=str(e)[:200])
            failure(h, e)
            return
        trace("stop", "stopped", session=m.group(1), reason=reason, ended=bool(out.get("ended")))
        auth.reply(h, 200, out)

    def status(h, parsed, body):
        if not owner_only(h):
            return
        try:
            out = client.health()
        except GardenError as e:
            out = {"ok": False, "error": str(e)}
        out["window"] = dict(feeder.status)
        bound = active.snapshot().get("session")
        if bound and out.get("ok") is not False and out.get("running") != bound:
            active.clear(bound)            # that session has ended: the proxy goes back to garden.engine_url
        auth.reply(h, 200, out)

    def residents(h, parsed, body):
        if not owner_only(h):
            return
        try:
            auth.reply(h, 200, client.residents())
        except GardenError as e:
            failure(h, e)

    def create_resident(h, parsed, body):
        if not owner_only(h):
            return
        try:
            out = client.create_resident(str((body or {}).get("resident") or ""))
        except GardenError as e:
            failure(h, e)
            return
        trace("resident", "created" if out.get("created") else "exists", resident=out.get("resident"))
        auth.reply(h, 200, out)

    def persist(mutate: Callable[[Dict[str, Any]], None]) -> bool:
        """Apply a config change to the live config object, and to config.json when StoneSage can save it. -> saved?"""
        mutate(config)
        if callable(load) and callable(save):
            cfg = load()
            mutate(cfg)
            save(cfg)
            return True
        return False

    def assign(resident: str, name: str) -> bool:
        """garden.residents.<id>.model = name ('' clears): what the board's model select and a granted request both do."""
        def mutate(cfg: Dict[str, Any]) -> None:
            res = sec.writable(cfg).setdefault("residents", {})
            entry = res.get(resident) if isinstance(res.get(resident), dict) else {}
            if name:
                entry["model"] = name
            else:
                entry.pop("model", None)
            res[resident] = entry

        return persist(mutate)

    def add_model(name: str, entry: Dict[str, Any]) -> bool:
        def mutate(cfg: Dict[str, Any]) -> None:
            sec.writable(cfg).setdefault("models", {})[name] = dict(entry)

        return persist(mutate)

    def assign_model(h, resident: str, body):
        name = str((body or {}).get("model") or "").strip()
        if name and name not in models_table(cfg_fn()):
            auth.reply(h, 400, {"ok": False, "error": "unknown model %r: garden.models has %s" % (name[:60], ", ".join(sorted(models_table(cfg_fn()))) or "none")})
            return
        saved = assign(resident, name)
        trace("assign_model", "assigned" if name else "cleared", resident=resident, model=name[:120])
        auth.reply(h, 200, {"ok": True, "resident": resident, "model": name, "saved": saved})

    def pause(h, parsed, body):
        if not owner_only(h):
            return
        m = re.match(r"^/api/garden/residents/([a-z0-9-]{1,64})/(pause|model)$", parsed.path)
        if not m:
            auth.reply(h, 404, {"ok": False, "error": "not found"})
            return
        if m.group(2) == "model":
            assign_model(h, m.group(1), body)
            return
        paused = (body or {}).get("paused", True) is not False
        try:
            out = client.pause_resident(m.group(1), paused)
        except GardenError as e:
            failure(h, e)
            return
        trace("pause", "paused" if paused else "resumed", resident=m.group(1))
        auth.reply(h, 200, out)

    def comments(h, parsed, body):
        if not owner_only(h):
            return
        m = re.match(r"^/api/garden/posts/([^/]+)/comments$", parsed.path)
        if not m:
            auth.reply(h, 404, {"ok": False, "error": "not found"})
            return
        post = urllib.parse.unquote(m.group(1))
        try:
            if h.command == "GET":
                auth.reply(h, 200, client.comments(post))
                return
            body = body if isinstance(body, dict) else {}
            rating = body.get("rating")
            out = client.add_comment(post, str(body.get("text") or "")[:2000], rating if isinstance(rating, int) else None)
        except GardenError as e:
            failure(h, e)
            return
        trace("comment", "added", post=post, rated=isinstance(body.get("rating"), int))
        auth.reply(h, 200, out)

    def refresh(h, parsed, body):
        if not owner_only(h):
            return
        if not control_key(config) or not str(_cfg(config).get("worker_url") or "").strip():
            auth.reply(h, 503, {"ok": False, "error": "garden.worker_url and garden.control_key are not set"})
            return
        auth.reply(h, 200, {"ok": True, "window": feeder.run_once()})

    def schedule(h, parsed, body):
        if not owner_only(h):
            return
        if scheduler is None:
            auth.reply(h, 503, {"ok": False, "error": "the garden scheduler is not available (garden.worker_url / control_key are not set, "
                                                      "or this install has no engine lease service)"})
            return
        if h.command == "POST":
            enabled = (body or {}).get("enabled") if isinstance(body, dict) else None
            if not isinstance(enabled, bool):
                auth.reply(h, 400, {"ok": False, "error": "enabled must be true or false"})
                return

            def mutate(cfg: Dict[str, Any]) -> None:
                sec.writable(cfg).setdefault("schedule", {})["enabled"] = enabled

            saved = persist(mutate)
            trace("schedule", "enabled" if enabled else "disabled", saved=saved)
            out = scheduler.status()
            out["enabled"], out["saved"] = enabled, saved
            if not saved:
                out["note"] = "in memory only: StoneSage has no way to save its config here, so a restart returns it to the file's value"
            auth.reply(h, 200, out)
            return
        auth.reply(h, 200, scheduler.status())

    from . import models as garden_models

    def phone_push(title: str, message: str, buttons: Optional[list], tag: str) -> Any:
        notify = ctx.hook("notify_phone")      # the app that knows the household's phones; no app = board-only
        if notify is None:
            return {"ok": False, "error": "no app sends phone notifications"}
        data: Dict[str, Any] = {"tag": tag, "group": "courage"}
        if buttons:
            data["actions"] = buttons
        return notify(title, message, data)

    def provenance_summary():
        getter = ctx.services.get("get_provenance")
        return getter().summary() if getter else None

    grants = garden_models.ModelGrants(cfg_fn, client, assign, add_model=add_model, notify=phone_push, trace=trace,
                                       summary_fn=provenance_summary, data_dir=ctx.data_dir())
    register_action = ctx.services.get("add_phone_action_handler")
    if callable(register_action):
        register_action(grants.handle_action)         # the phone's Grant / Decline taps arrive through the approvals listener
    if scheduler is not None and hasattr(scheduler, "tick_hooks"):
        scheduler.tick_hooks.append(grants.maybe_tick)

    def requests_route(h, parsed, body):
        if not owner_only(h):
            return
        m = re.match(r"^/api/garden/requests/([^/]+)/decide$", parsed.path)
        try:
            if m:
                if h.command != "POST":
                    auth.reply(h, 405, {"ok": False, "error": "POST only"})
                    return
                body = body if isinstance(body, dict) else {}
                out = grants.owner_decides(urllib.parse.unquote(m.group(1)), str(body.get("status") or ""), str(body.get("note") or ""),
                                  body.get("layout") if "layout" in body else None, str(body.get("engine_url") or ""),
                                  str(body.get("unit") or ""))
                auth.reply(h, 200, out)
                return
            grants.tick()
            conf = cfg_fn()
            rows = []
            for r in grants.pending:
                model = str(r.get("model") or "")
                rows.append(dict(r, configured=model in models_table(conf),
                                 allowed=model in garden_models.allowed_models(conf, str(r.get("resident") or "")),
                                 current_model=resident_model(conf, str(r.get("resident") or ""))))
            auth.reply(h, 200, {"ok": True, "requests": rows, "layouts": sorted((conf.get("engine_layouts") or {}).keys()),
                                "error": grants.last_error or None})
        except garden_models.GrantError as e:
            auth.reply(h, e.status, {"ok": False, "error": str(e)})
        except GardenError as e:
            failure(h, e)

    # the worker's model key opens the two proxy routes and nothing else (platform: auth.SCOPED_KEYS); /metrics gets the scheduler's gauges
    ctx.add_scoped_key(SCOPED, SCOPED_ROUTES, model_key)
    ctx.add_metrics(lambda: SCHEDULER.metrics() if SCHEDULER is not None else [])
    ctx.routes.add("GET", "/api/garden/requests", requests_route, owner=OWNER)
    ctx.routes.add("POST", "/api/garden/requests/", requests_route, prefix=True, owner=OWNER)
    ctx.routes.add("POST", MODEL_ROUTE[1], model_proxy(config, proxy_opener, engine_of=active.engine, cfg_fn=cfg_fn), owner=OWNER)
    ctx.routes.add("GET", "/api/garden/schedule", schedule, owner=OWNER)
    ctx.routes.add("POST", "/api/garden/schedule", schedule, owner=OWNER)
    ctx.routes.add("POST", EMBED_ROUTE[1], embed_proxy(config, proxy_opener), owner=OWNER)
    ctx.routes.add("GET", "/api/garden/worker/status", status, owner=OWNER)
    ctx.routes.add("GET", "/api/garden/residents", residents, owner=OWNER)
    ctx.routes.add("POST", "/api/garden/residents", create_resident, owner=OWNER)
    ctx.routes.add("POST", "/api/garden/residents/", pause, prefix=True, owner=OWNER)
    ctx.routes.add("GET", "/api/garden/posts/", comments, prefix=True, owner=OWNER)
    ctx.routes.add("POST", "/api/garden/posts/", comments, prefix=True, owner=OWNER)
    ctx.routes.add("POST", "/api/garden/window/refresh", refresh, owner=OWNER)
    ctx.routes.add("GET", "/api/garden/board", board, owner=OWNER)
    ctx.routes.add("POST", "/api/garden/sessions/start", start, owner=OWNER)
    ctx.routes.add("POST", "/api/garden/sessions/", stop, prefix=True, owner=OWNER)
