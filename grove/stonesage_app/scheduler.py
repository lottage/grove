"""The garden's scheduler (the design notes, "Scheduling without a clock", step G3).

Decides when a resident may have a session, from the idle gate (lodestone/idle_gate.py) and recorded history, never from the time of
day. One thread (`start()`), a tick every ~20 s (every ~5 s while a session runs). It never needs a restart: the thread always runs
once the garden is configured, and `garden.schedule.enabled` (default false) only decides whether it ACTS. Gate transitions are
recorded either way (data/garden_gate_history.jsonl), so the opening-length history accumulates before the schedule is switched on.

What a start needs (all of it, checked in this order):
  - no engine lease is held (by anyone), no session is running (a manual one counts), and the last window ended >= cooldown_s ago
  - nobody is waiting on Computer: no approval pending, no Computer turn in the last idle_gate.computer_turn_min (default 2; the gate covers both; this
    checks them again, directly, because a lease would take the vision engine away from Computer's camera looks)
  - a resident that is not paused, has a model assigned (garden.residents.<id>.model), and is inside its budgets
    (max_sessions_24h rolling, max_tokens_hour rolling)
  - the model's layout: its engines' roles and GPUs known, and the idle gate's DEMAND view open for every GPU the layout evicts from
    (hardware demand only since 2026-10-14: GPU busy, engines idle, no approval, no Computer turn; whether anyone is awake, at home or
    watching a film is never a reason. The cockpit signal (someone using the StoneSage page) closes it for every garden lease. The garden's own load on a lent card never does: while its lease is held, that card's busy/engine reasons are ignored). A model may list
    several layouts (`layouts`, in order of preference; `layout` alone is a list of one): the first that is ready is used; with `allow_coordinator_layouts` on, a ready coordinator-evicting layout (the RX 6750 XT) comes first, since the
    6600 is vision's and lending it blinds the sentry. A layout that evicts
    the coordinator (Computer then answers from the Courage-lite stand-in) additionally needs ALL of: `allow_coordinator_layouts` true
    (default false), the layout's stand-in configured AND its eval passed (`engine_stand_ins.<name>.eval_passed`), and no takeback
    cooldown (below)
  - an opening-length estimate >= min_window_min (below)

Opening-length estimate: every gate opening (open -> close) is recorded per GPU. For the opening that is open now, with age A: take the
past openings that lasted at least A, and use the median of (length - A). Fewer than 5 such openings = not enough history, so the
estimate is default_window_min. Openings the scheduler itself ended (its window ran out, or the schedule was switched off) and openings
cut off by a StoneSage restart are left out: they say nothing about how long the hardware stays free. History records carry the gate
version (`v`); openings recorded under the old household rule have no version and are ignored, so estimates come from demand-gate openings only.

Window and lease: ttl = min(estimate, max_window_min) + margin_min. The session (and the window) end `stop_before_end_min` (12, never
less) before the lease's own end, so the engine lease's phone check-in (10 min before the end) never fires for a garden lease. A lease
is never renewed (renewing is the phone's job by design): the window ends, the lease is released, a later opening gets a new one.
Inside the window, sessions run one after another, round robin over the residents assigned to that model, each given a deadline
(`max_seconds`) and, when the model's speed is known (models.<m>.tok_s, else measured from earlier sessions), a generated-token cap
that fits the time left.

Takeback (2026-10-13): every garden lease is acquired with yield_to_computer. The platform's takeback (stonesage/courage/takeback.py) can
take a coordinator-evicting lease back at any moment: Computer on the stand-in fails or needs the full brain, or a person starts using
the cockpit (stonesage/activity.py; also closes the gate for a coordinator-evicting layout, so the garden neither starts nor keeps running meanwhile).
It calls `yield_to_computer(why, cooldown_s)` (registered through the `register_yielder` service; this module imports nothing of the
platform's courage code): the running session is stopped at once (the worker saves the draft, reason "takeback: <why>") and the next
tick ends the window; the takeback releases the lease itself and waits for the coordinator. After a takeback no coordinator-evicting layout
is taken for `courage.takeback.cooldown_min` (default 60; the platform's `takeback_cooldown`). Layouts that leave the coordinator alone are
not affected by the cooldown.
Per card (2026-10-14): a camera request while the garden holds the vision card (the RX 6600 lease) is a hot-load: the takeback calls
`yield_to_computer(why, cooldown_s, resource="vision")`, which stops the session only if the held lease really has that card out, and a
layout that evicts the vision server waits `courage.takeback.vision_cooldown_min` (10) afterwards. Each card has its own cooldown; a
garden session on the other card is never disturbed. `takeback_cooldown_fn(role)` is asked per role (an older 0-argument function counts
for the coordinator only).

Preemption (polled every running_poll_s): an approval appears, Computer gets a turn after the window began, the gate closes (its
demand reasons: approvals, a Computer turn; the per-GPU reasons are ignored while the lease is held, because our own
session is what makes the card busy and the lent-out engine is not scraped), the lease is lost, or the schedule is switched off:
the session is stopped at once (the worker saves the draft) and the lease released. Computer always wins.

CPU slot: a model with layout null and a `unit` (e.g. the Courage-lite stand-in's unit) is started on demand when `schedule.cpu_enabled`
is true (default FALSE) and no lease is held and the house gate is open; a lease that appears wins at once (the session stops; the unit
is left alone while a lease is held, because the lease may be using it as the stand-in). The CPU slot and the stand-in are the same engine (:8006
in the usual setup), so a CPU session never starts while Computer is on the stand-in (a lease is held or `lease.brain()` says lite), and one
that is running ends when a lease appears: the lease wins.

Every start, preemption and release is traced as kind "garden"; skips are traced at most once per reason per skip_trace_s.
Status: GET /api/garden/schedule; Prometheus gauges from metrics().
"""

import inspect
import json
import logging
import os
import re
import statistics
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .section import section

logger = logging.getLogger("stonesage.garden_scheduler")

DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "max_sessions_24h": 12,          # per resident, rolling
    "max_tokens_hour": 48000,        # per resident, generated tokens started in the last hour (3 full 16k sessions)
    "default_window_min": 30,        # the estimate when the history has fewer than 5 comparable openings
    "min_window_min": 15,            # start only when the estimate is at least this
    "max_window_min": 55,
    "margin_min": 10,                # lease ttl = min(estimate, max_window_min) + margin_min
    "stop_before_end_min": 12,       # floor 12: the lease checks in with the phone at 10
    "min_session_s": 180,            # no session is started with less time than this left in the window
    "recent_turn_min": None,         # None = idle_gate.computer_turn_min (default 2); a number here overrides it for the garden only
    "cooldown_s": 300,               # after any window ends, nothing starts for this long
    "retry_s": 60,                   # after a refused acquire / failed start
    "poll_s": 20,
    "running_poll_s": 5,
    "gate_fresh_s": 10,              # while running, the idle gate is re-read at least this often
    "skip_trace_s": 900,
    "allow_coordinator_layouts": False,
    "cpu_enabled": False,
    "cpu_wait_s": 120,
    "min_samples": 5,
}
OWN_END_WINDOW_S = 900            # a gate close within 15 min of our own window ending is probably our own doing
HOUSE = "house"                   # the pseudo-GPU: the gate's shared conditions alone (for the CPU slot)
STATES = ("disabled", "idle", "waiting", "running", "preempted")
COORDINATOR_ROLE = "coordinator"
VISION_ROLE = "vision"
COOLDOWN_ROLES = (COORDINATOR_ROLE, VISION_ROLE)       # the cards a takeback can cool down
GATE_VERSION = 2                  # history written under the old rule (household signals closed the gate) is not comparable and is ignored
MAX_HISTORY_LINES = 4000
KEEP_HISTORY_LINES = 2000
ACTION_PREEMPT = "preempted"
RESIDENT_ID = re.compile(r"^[a-z0-9-]{1,64}$")     # the same ids the worker and the routes accept


def _hhmm(ts: float) -> str:
    return time.strftime("%H:%M", time.localtime(ts))


def layout_choices(spec: Dict[str, Any]) -> List[str]:
    """The engine layouts a model can run in, in order of preference: `layouts` (a list), else `layout` alone. [] = the CPU slot."""
    many = spec.get("layouts")
    if isinstance(many, list):
        out = [str(x) for x in many if x]
        if out:
            return out
    return [str(spec["layout"])] if spec.get("layout") else []


# ---- the gate's history ---------------------------------------------------------------------------------------------------
class GateHistory:
    """Per GPU: when the gate opened, when it closed and why. data/garden_gate_history.jsonl, one JSON object per transition:
      {"gpu", "event": "open", "at"}  and  {"gpu", "event": "close", "at", "opened_at", "why": [...], "own": bool, "censored": bool}
    """

    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time):
        self.path, self.clock = path, clock
        self.open_since: Dict[str, float] = {}
        self.closed: Dict[str, List[Dict[str, Any]]] = {}
        self.own_end: Dict[str, float] = {}
        self._seen: set = set()
        self._lines = 0
        self._load()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    self._lines += 1
                    gpu = r.get("gpu")
                    if not gpu or r.get("v") != GATE_VERSION:     # older openings were measured under another rule
                        continue
                    if r.get("event") == "open":
                        self.open_since[gpu] = float(r["at"])
                    elif r.get("event") == "close":
                        self.open_since.pop(gpu, None)
                        self.closed.setdefault(gpu, []).append(r)
        except OSError:
            pass

    def _write(self, rec: Dict[str, Any]) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._lines += 1
            if self._lines > MAX_HISTORY_LINES:
                self._trim()
        except OSError as e:
            logger.warning("garden gate history: %s", e)

    def _trim(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            lines = f.readlines()[-KEEP_HISTORY_LINES:]
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.replace(tmp, self.path)
        self._lines = len(lines)

    def observe(self, states: Dict[str, Tuple[bool, List[str]]], now: Optional[float] = None) -> None:
        """states: {gpu: (open, reasons it is closed)}. Writes only transitions."""
        now = self.clock() if now is None else now
        for gpu, (is_open, reasons) in states.items():
            first = gpu not in self._seen
            self._seen.add(gpu)
            since = self.open_since.get(gpu)
            if is_open and since is None:
                self.open_since[gpu] = now
                self._write({"gpu": gpu, "event": "open", "at": round(now, 3), "v": GATE_VERSION})
            elif not is_open and since is not None:
                own_at = self.own_end.get(gpu)
                rec = {"gpu": gpu, "event": "close", "at": round(now, 3), "v": GATE_VERSION, "opened_at": since, "why": list(reasons)[:6],
                       "own": bool(own_at is not None and 0 <= now - own_at <= OWN_END_WINDOW_S),
                       # first look since StoneSage started: the opening's real end is unknown (the process was not watching)
                       "censored": first}
                self.open_since.pop(gpu, None)
                self.closed.setdefault(gpu, []).append(rec)
                self._write(rec)

    def mark_own_end(self, gpus: List[str], now: Optional[float] = None) -> None:
        now = self.clock() if now is None else now
        for g in gpus:
            self.own_end[g] = now

    def lengths(self, gpu: str) -> List[float]:
        """Lengths (s) of the past openings that tell us how long the house stays quiet: not ours, not censored."""
        out = []
        for r in self.closed.get(gpu, []):
            if r.get("own") or r.get("censored"):
                continue
            n = float(r["at"]) - float(r["opened_at"])
            if n > 0:
                out.append(n)
        return out

    def age(self, gpu: str, now: Optional[float] = None) -> Optional[float]:
        since = self.open_since.get(gpu)
        return None if since is None else (self.clock() if now is None else now) - since


def estimate_remaining(lengths: List[float], age: float, default_s: float, min_samples: int = 5) -> Dict[str, Any]:
    """How much longer this opening is likely to last: the median of (length - age) over past openings that lasted at least `age`.
    Fewer than `min_samples` of those = no basis, so `default_s`."""
    longer = sorted(n - age for n in lengths if n >= age)
    if len(longer) < min_samples:
        return {"seconds": float(default_s), "basis": "default", "samples": len(longer)}
    return {"seconds": float(statistics.median(longer)), "basis": "history", "samples": len(longer)}


# ---- sessions started (budgets, measured speed) --------------------------------------------------------------------------------
class SessionLog:
    """data/garden_sessions.jsonl: {"start": ...} when a session is started here, {"end": ...} when it is seen to have ended."""

    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time):
        self.path, self.clock = path, clock
        self.rows: Dict[str, Dict[str, Any]] = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        if "start" in r:
                            self.rows[r["session"]] = dict(r["start"], session=r["session"])
                        elif "end" in r and r.get("session") in self.rows:
                            self.rows[r["session"]].update(r["end"])
            except OSError:
                pass
        cutoff = self.clock() - 48 * 3600
        self.rows = {k: v for k, v in self.rows.items() if v.get("at", 0) >= cutoff}

    def _append(self, rec: Dict[str, Any]) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:
            logger.warning("garden session log: %s", e)

    def started(self, session: str, resident: str, model: str, by: str) -> None:
        row = {"at": round(self.clock(), 3), "resident": resident, "model": model, "by": by}
        self.rows[session] = dict(row, session=session)
        self._append({"session": session, "start": row})

    def ended(self, session: str, tokens: int, reason: str) -> None:
        row = self.rows.get(session)
        if not row or "ended_at" in row:
            return
        end = {"ended_at": round(self.clock(), 3), "tokens": int(tokens or 0), "reason": str(reason or "")[:120]}
        row.update(end)
        self._append({"session": session, "end": end})

    def progress(self, session: str, tokens: int) -> None:
        if session in self.rows:
            self.rows[session]["tokens_live"] = int(tokens or 0)

    def count_24h(self, resident: str, now: Optional[float] = None) -> int:
        now = self.clock() if now is None else now
        return sum(1 for r in self.rows.values() if r.get("resident") == resident and now - r["at"] < 86400)

    def tokens_hour(self, resident: str, now: Optional[float] = None) -> int:
        now = self.clock() if now is None else now
        return sum(int(r.get("tokens", r.get("tokens_live", 0)) or 0) for r in self.rows.values()
                   if r.get("resident") == resident and now - r["at"] < 3600)

    def tok_s(self, model: str) -> Optional[float]:
        """Generated tokens per second wall-clock, median of the last 8 finished sessions of this model that made 200+ tokens
        (it includes tool time and prompt processing, so it is a floor: a cap built from it is on the safe side)."""
        rates = []
        for r in sorted(self.rows.values(), key=lambda x: x["at"], reverse=True):
            if r.get("model") == model and r.get("tokens", 0) >= 200 and r.get("ended_at", 0) > r["at"]:
                rates.append(r["tokens"] / (r["ended_at"] - r["at"]))
            if len(rates) >= 8:
                break
        return statistics.median(rates) if rates else None

    def has_session(self, resident: str) -> bool:
        """Whether this resident has a session in the ledger (48 h: the ledger's own horizon)."""
        return any(r.get("resident") == resident for r in self.rows.values())

    def per_resident_24h(self, now: Optional[float] = None) -> Dict[str, int]:
        now = self.clock() if now is None else now
        out: Dict[str, int] = {}
        for r in self.rows.values():
            if now - r["at"] < 86400:
                out[r["resident"]] = out.get(r["resident"], 0) + 1
        return out


# ---- the scheduler -----------------------------------------------------------------------------------------------------------
class GardenScheduler:
    def __init__(self, cfg_fn: Callable[[], Dict[str, Any]], gate_fn: Callable[[bool], Dict[str, Any]], lease: Any, worker: Any,
                 active: Any, last_turn_fn: Callable[[], Optional[float]], pending_fn: Callable[[], int],
                 trace: Optional[Callable[[Dict[str, Any]], None]] = None, data_dir: Optional[str] = None,
                 systemctl: Optional[Callable[[str, str], Dict[str, Any]]] = None, http_ok: Optional[Callable[[str], bool]] = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 takeback_cooldown_fn: Optional[Callable[[], float]] = None,
                 takeback_status_fn: Optional[Callable[[], Dict[str, Any]]] = None):
        """gate_fn(fresh) -> idle_gate.check(); lease: engine_lease.EngineLease (.lease, .layout_info, .acquire, .release);
        worker: garden_client.GardenClient (.health, .residents, .start_session(body, by=), .stop_session, .session);
        active: garden_client.ActiveEngine; last_turn_fn() -> time of Computer's last turn; pending_fn() -> approvals waiting;
        takeback_cooldown_fn() -> seconds until a coordinator-evicting layout may be borrowed again; takeback_status_fn() -> the last takeback."""
        self.takeback_cooldown_fn, self.takeback_status_fn = takeback_cooldown_fn, takeback_status_fn
        self.yield_request: Optional[Dict[str, Any]] = None
        self.takeback_until: Dict[str, float] = {}           # role -> when a layout that evicts it may be borrowed again
        self.cfg_fn, self.gate_fn, self.lease, self.worker, self.active = cfg_fn, gate_fn, lease, worker, active
        self.last_turn_fn, self.pending_fn, self.trace = last_turn_fn, pending_fn, trace
        self.systemctl, self.http_ok, self.clock, self.sleep = systemctl, http_ok, clock, sleep
        gate_path = os.path.join(data_dir, "garden_gate_history.jsonl") if data_dir else None
        sess_path = os.path.join(data_dir, "garden_sessions.jsonl") if data_dir else None
        self.history = GateHistory(gate_path, clock)
        self.sessions = SessionLog(sess_path, clock)
        self.window: Optional[Dict[str, Any]] = None
        self.starting = False
        self.state = "disabled"
        self.reason = "not started"
        self.last_event: Dict[str, Any] = {}
        self.cooldown_until = 0.0
        self.retry_until = 0.0
        self.rr: Dict[str, int] = {}
        self._skip_traced: Dict[str, float] = {}
        self._gate_read = 0.0
        self._gate: Dict[str, Any] = {}
        self._thread: Optional[threading.Thread] = None
        self.tick_hooks: List[Callable[[], Any]] = []        # run after each tick (the garden's model-request queue pulls and decides here)

    # ------------------------------------------------------------------ config ----
    def cfg(self) -> Dict[str, Any]:
        g = section(self.cfg_fn() or {})
        s = g.get("schedule") if isinstance(g.get("schedule"), dict) else {}
        out = dict(DEFAULTS)
        out.update({k: v for k, v in s.items() if k in DEFAULTS})
        out["stop_before_end_min"] = max(12.0, float(out["stop_before_end_min"]))      # the check-in is at 10: never closer than 12
        return out

    def models(self) -> Dict[str, Dict[str, Any]]:
        m = section(self.cfg_fn() or {}).get("models")
        return {k: v for k, v in m.items() if isinstance(v, dict)} if isinstance(m, dict) else {}

    def assignment(self, resident: str) -> str:
        r = (section(self.cfg_fn() or {}).get("residents") or {}).get(resident)
        return str(r.get("model") or "") if isinstance(r, dict) else ""

    def _residents(self, listed: List[Any]) -> Tuple[List[str], List[str]]:
        """(open, paused), each sorted: the residents a session may go to. Candidates are the worker's residents plus the
        configured ones (garden.residents), so a resident that was assigned a model but has never had a session still gets
        one: the worker creates it on its first start. Paused = the worker's flag or `paused` in the configured entry.
        Ids the worker would refuse are left out."""
        flags: Dict[str, bool] = {}
        for r in listed or []:
            if isinstance(r, dict) and RESIDENT_ID.match(str(r.get("id") or "")):
                flags[r["id"]] = bool(r.get("paused"))
        conf = section(self.cfg_fn() or {}).get("residents")
        for rid, r in (conf.items() if isinstance(conf, dict) else []):
            if RESIDENT_ID.match(str(rid)):
                paused = isinstance(r, dict) and bool(r.get("paused"))
                flags[rid] = flags.get(rid, False) or paused
        return sorted(k for k, v in flags.items() if not v), sorted(k for k, v in flags.items() if v)

    def _fresh_first(self, ids: List[str]) -> List[str]:
        """Residents with no session yet come first (keeping their order), then the rest."""
        return [r for r in ids if not self.sessions.has_session(r)] + [r for r in ids if self.sessions.has_session(r)]

    # ------------------------------------------------------------------ small things ----
    def holding(self) -> Optional[str]:
        """Why a manual start must wait: the scheduler holds a window (or is just taking one), else None."""
        w = self.window
        if w:
            return "the garden scheduler holds %s until %s" % ("the " + w["layout"] + " lease" if w.get("layout") else "the CPU slot", _hhmm(w["deadline"]))
        if self.starting:
            return "the garden scheduler is starting a scheduled session"
        return None

    def _trace(self, event: str, outcome: str, **fields: Any) -> None:
        self.last_event = {"event": event, "outcome": outcome, "at": round(self.clock(), 3), **{k: v for k, v in fields.items() if k == "why"}}
        if self.trace:
            try:
                self.trace({"kind": "garden", "at": round(self.clock(), 3), "event": event, "outcome": outcome, "scheduler": True, **fields})
            except Exception:
                pass

    def _skip(self, why: str, state: str = "waiting") -> None:
        self.state, self.reason = state, why
        key = re.sub(r"\d+", "#", why)[:120]
        now = self.clock()
        if now - self._skip_traced.get(key, -1e9) >= float(self.cfg()["skip_trace_s"]):
            self._skip_traced[key] = now
            self._trace("skipped", "skipped", why=why[:200])

    def _read_gate(self, fresh: bool) -> Dict[str, Any]:
        try:
            gate = self.gate_fn(fresh)
            self._gate = gate if isinstance(gate, dict) else {}
        except Exception as e:
            self._gate = {"gpus": {}, "shared_reasons": ["the idle gate could not be read (%s)" % type(e).__name__]}
        if fresh:
            self._gate_read = self.clock()
        return self._gate

    def _shared(self, gate: Dict[str, Any]) -> List[str]:
        """Why the gate is closed for background work: hardware demand only (approvals, a Computer turn, nothing about who is
        awake). Falls back to the full list for a gate that does not report the demand view."""
        return list(gate["demand_shared_reasons"] if "demand_shared_reasons" in gate else gate.get("shared_reasons") or [])

    def _turn_min(self, cfg: Dict[str, Any]) -> float:
        if cfg.get("recent_turn_min") is not None:
            return float(cfg["recent_turn_min"])
        ig = (self.cfg_fn() or {}).get("idle_gate")
        try:
            return float((ig or {}).get("computer_turn_min", 2))
        except (TypeError, ValueError):
            return 2.0

    def _observe(self, gate: Dict[str, Any], now: float) -> None:
        shared = self._shared(gate)
        held = set((self.window or {}).get("gpus") or [])
        states: Dict[str, Tuple[bool, List[str]]] = {}
        for gpu, g in (gate.get("gpus") or {}).items():
            if gpu in held:            # our own lease: the card is busy because of us and the lent-out engine is not scraped
                states[gpu] = (not shared, shared)
            else:
                states[gpu] = (bool(g.get("demand_open", g.get("open"))), list(g.get("reasons") or []) + shared)
        states[HOUSE] = (not shared, shared)
        self.history.observe(states, now)

    def estimate(self, gpu: str, now: float) -> Optional[Dict[str, Any]]:
        age = self.history.age(gpu, now)
        if age is None:
            return None
        c = self.cfg()
        e = estimate_remaining(self.history.lengths(gpu), age, float(c["default_window_min"]) * 60, int(c["min_samples"]))
        e["age_s"] = round(age, 1)
        e["minutes"] = round(e["seconds"] / 60, 1)
        return e

    # ------------------------------------------------------------------ the tick ----
    def tick(self) -> None:
        reconcile = getattr(self.lease, "reconcile", None)      # engine lease self-healing: a layout left running with no lease held
        if callable(reconcile):
            try:
                reconcile()
            except Exception as e:
                logger.warning("garden scheduler: lease reconcile failed: %s", e)
        now = self.clock()
        cfg = self.cfg()
        w = self.window
        fresh = bool(w) and now - self._gate_read >= float(cfg["gate_fresh_s"])
        gate = self._read_gate(fresh or not w)
        self._observe(gate, now)
        if w:
            self._supervise(cfg, gate, now)
        elif not cfg["enabled"]:
            self.state, self.reason = "disabled", "garden.schedule.enabled is false (gate history is still recorded)"
        else:
            self._maybe_start(cfg, gate, now)

    # ---- starting ----
    def _model_ready(self, name: str, cfg: Dict[str, Any], gate: Dict[str, Any], now: float) -> Tuple[bool, str, Dict[str, Any]]:
        spec = self.models().get(name)
        if not spec:
            return False, "model %r is not in garden.models" % name, {}
        shared = self._shared(gate)
        if not layout_choices(spec):                                 # the CPU slot
            if not cfg["cpu_enabled"]:
                return False, "the CPU slot is switched off (schedule.cpu_enabled)", {}
            if not spec.get("unit") or not spec.get("engine_url"):
                return False, "model %r needs unit and engine_url for the CPU slot" % name, {}
            on_stand_in = self._on_stand_in()
            if on_stand_in:                                          # same engine as Computer's spare brain: the lease wins
                return False, on_stand_in, {}
            if shared:
                return False, "the house gate is closed: " + "; ".join(shared[:3]), {}
            est = self.estimate(HOUSE, now)
            return self._with_estimate(est, cfg, {"kind": "cpu", "unit": spec["unit"], "gpus": [HOUSE]}, "the house gate")
        why: List[str] = []
        ready: List[Dict[str, Any]] = []
        for layout in layout_choices(spec):                           # in the order of preference
            ok, reason, plan = self._layout_ready(layout, cfg, gate, now)
            if ok:
                ready.append(plan)
            else:
                why.append(reason)
        if ready:
            # With coordinator layouts allowed, the RX 6750 XT is the garden's card: among the layouts whose gate is open it comes
            # first (the 6600 is vision's, and lending it blinds the sentry). Otherwise the configured order decides.
            if cfg["allow_coordinator_layouts"]:
                ready.sort(key=lambda p: 0 if p.get("coord") else 1)
            return True, "", ready[0]
        return False, "; ".join(why)[:300], {}

    def takeback_left(self, role: str = COORDINATOR_ROLE) -> float:
        """Seconds until a layout that evicts `role` (coordinator | vision) may be borrowed again after a takeback."""
        left = self.takeback_until.get(role, 0.0) - self.clock()
        if self.takeback_cooldown_fn:
            try:
                try:
                    takes_role = len(inspect.signature(self.takeback_cooldown_fn).parameters) > 0
                except (TypeError, ValueError):
                    takes_role = False
                if takes_role:
                    left = max(left, float(self.takeback_cooldown_fn(role) or 0.0))
                elif role == COORDINATOR_ROLE:             # an older function knows the coordinator only
                    left = max(left, float(self.takeback_cooldown_fn() or 0.0))
            except Exception:
                pass
        return max(0.0, left)

    def _layout_ready(self, layout: str, cfg: Dict[str, Any], gate: Dict[str, Any], now: float) -> Tuple[bool, str, Dict[str, Any]]:
        shared = self._shared(gate)
        info = self.lease.layout_info(layout) if hasattr(self.lease, "layout_info") else None
        if not info:
            return False, "layout %r is not in engine_layouts" % layout, {}
        if info.get("disabled"):
            return False, "layout %s is disabled: %s" % (layout, info["disabled"]), {}
        if not info.get("resolved") or not info.get("gpus"):
            return False, "layout %s: its engines' roles or GPUs are not known right now" % layout, {}
        if COORDINATOR_ROLE in (info.get("roles") or []):
            if not cfg["allow_coordinator_layouts"]:
                return False, "layout %s would take Computer's coordinator (schedule.allow_coordinator_layouts is false)" % layout, {}
            if not info.get("stand_in_ready"):
                return False, ("layout %s would take Computer's coordinator and its stand-in is not ready (engine_stand_ins "
                               "eval_passed is not set, or the layout names none)" % layout), {}
            left = self.takeback_left(COORDINATOR_ROLE)
            if left > 0:
                return False, "layout %s: Computer took the coordinator back; not again for %d min" % (layout, round(left / 60)), {}
        if VISION_ROLE in (info.get("roles") or []):
            left = self.takeback_left(VISION_ROLE)
            if left > 0:
                return False, "layout %s: Computer took the camera engine back; not again for %d min" % (layout, max(1, round(left / 60))), {}
        closed = []
        coord = COORDINATOR_ROLE in (info.get("roles") or [])
        if gate.get("cockpit"):                            # every garden lease yields to Computer: someone on the page closes it
            return False, "the idle gate is closed: " + str(gate["cockpit"]), {}
        for gpu in info["gpus"]:
            g = (gate.get("gpus") or {}).get(gpu)
            if not g or not g.get("demand_open", g.get("open") if g else False):
                closed.append("%s: %s" % (gpu, ", ".join((g or {}).get("reasons") or shared or ["not reported"])))
        if closed:
            return False, "the idle gate is closed: " + "; ".join(closed)[:200], {}
        ests = [self.estimate(g, now) for g in info["gpus"]]
        if not all(ests):
            return False, "no opening recorded yet for %s" % ", ".join(info["gpus"]), {}
        est = min(ests, key=lambda e: e["seconds"])
        return self._with_estimate(est, cfg, {"kind": "lease", "layout": layout, "gpus": list(info["gpus"]), "coord": coord}, "the gate")

    def _on_stand_in(self) -> Optional[str]:
        """Why the CPU slot is not ours: Computer is on the stand-in (it shares the CPU engine), or a lease is held. None = free."""
        held = getattr(self.lease, "lease", None)
        if held:
            return "an engine lease is held (%s): Computer may be on the stand-in, which is the CPU slot" % held.get("layout")
        brain = getattr(self.lease, "brain", None)
        try:
            b = brain(COORDINATOR_ROLE) if callable(brain) else None
        except Exception:
            b = None
        if b and b.get("mode") == "lite":
            return "Computer is on the spare brain, which is the CPU slot"
        return None

    def _with_estimate(self, est: Optional[Dict[str, Any]], cfg: Dict[str, Any], base: Dict[str, Any], what: str):
        if not est:
            return False, "no opening recorded yet for " + what, {}
        if est["seconds"] < float(cfg["min_window_min"]) * 60:
            return False, "the opening is expected to last %.0f more min (needs %s)" % (est["minutes"], cfg["min_window_min"]), {}
        return True, "", dict(base, estimate=est)

    def _budget_ok(self, resident: str, cfg: Dict[str, Any], now: float) -> Optional[str]:
        n = self.sessions.count_24h(resident, now)
        if n >= int(cfg["max_sessions_24h"]):
            return "%s: %d sessions in 24 h (limit %s)" % (resident, n, cfg["max_sessions_24h"])
        t = self.sessions.tokens_hour(resident, now)
        if t >= int(cfg["max_tokens_hour"]):
            return "%s: %d tokens in the last hour (limit %s)" % (resident, t, cfg["max_tokens_hour"])
        return None

    def _maybe_start(self, cfg: Dict[str, Any], gate: Dict[str, Any], now: float) -> None:
        self.yield_request = None                  # a takeback that arrived with no window open has nothing left to end
        if self.state == "preempted" and now >= self.cooldown_until:
            self.state = "waiting"
        if now < self.cooldown_until:
            self.reason = "cooling down after the last window until %s" % _hhmm(self.cooldown_until)
            if self.state != "preempted":
                self.state = "waiting"
            return
        if now < self.retry_until:
            return
        held = getattr(self.lease, "lease", None)
        if held:
            return self._skip("an engine lease is held (%s until %s)" % (held.get("layout"), _hhmm(held["expires"])))
        pending = int(self.pending_fn() or 0)
        if pending:
            return self._skip("%d approval(s) waiting for Computer" % pending)
        last = self.last_turn_fn()
        if last and now - last < self._turn_min(cfg) * 60:
            return self._skip("Computer answered %d min ago" % round((now - last) / 60))
        try:
            health = self.worker.health()
            residents = self.worker.residents().get("residents") or []
        except Exception as e:
            return self._skip("the garden worker did not answer (%s)" % str(e)[:80])
        if health.get("running"):
            return self._skip("a session is already running (started by hand)")
        open_ids, paused_ids = self._residents(residents)
        if not open_ids and not paused_ids:
            return self._skip("the garden has no residents")
        if not open_ids:
            return self._skip("every resident is paused")
        start_at = self.rr.get("_first", 0) % len(open_ids)
        order = self._fresh_first(open_ids[start_at:] + open_ids[:start_at])
        ready: Dict[str, Tuple[bool, str, Dict[str, Any]]] = {}
        why_not: List[str] = []
        for rid in order:
            model = self.assignment(rid)
            if not model:
                why_not.append("%s has no model assigned" % rid)
                continue
            over = self._budget_ok(rid, cfg, now)
            if over:
                why_not.append(over)
                continue
            if model not in ready:
                ready[model] = self._model_ready(model, cfg, gate, now)
            ok, why, plan = ready[model]
            if not ok:
                why_not.append("%s (%s): %s" % (rid, model, why))
                continue
            self.rr["_first"] = (open_ids.index(rid) + 1) % len(open_ids)
            return self._open_window(cfg, model, rid, plan, now)
        self._skip("; ".join(why_not)[:300] or "nothing to start", "idle" if all("assigned" in w for w in why_not) else "waiting")

    def _open_window(self, cfg: Dict[str, Any], model: str, resident: str, plan: Dict[str, Any], now: float) -> None:
        est = plan["estimate"]
        usable_s = min(est["seconds"], float(cfg["max_window_min"]) * 60)
        self.starting = True
        try:
            if plan["kind"] == "lease":
                ttl = usable_s + float(cfg["margin_min"]) * 60
                res = self.lease.acquire(plan["layout"], ttl, reason="garden: scheduled session (%s)" % model, approver="garden-scheduler",
                                         yield_to_computer=True)
                if not res.get("ok"):
                    self.retry_until = now + float(cfg["retry_s"])
                    self._skip("the lease was refused: %s" % str(res.get("error"))[:160])
                    return
                expires = float(res["expires"])
                window = {"kind": "lease", "layout": plan["layout"], "lease_id": res["lease_id"], "expires": expires,
                          "coord": bool(plan.get("coord")), "deadline": expires - float(cfg["stop_before_end_min"]) * 60}
            else:
                err = self._start_cpu(plan["unit"], self.models()[model]["engine_url"], cfg)
                if err:
                    self.retry_until = now + float(cfg["retry_s"])
                    self._skip("the CPU slot did not start: %s" % err[:160])
                    return
                window = {"kind": "cpu", "layout": None, "unit": plan["unit"], "deadline": self.clock() + usable_s}
            window.update(model=model, gpus=list(plan["gpus"]), t0=now, started=self.clock(), session=None, sessions=0,
                          estimate_minutes=est["minutes"], basis=est["basis"])
            self.window = window
        finally:
            self.starting = False
        self._trace("scheduled_start", "started", model=model, layout=window.get("layout"), estimate_min=est["minutes"], basis=est["basis"],
                    samples=est["samples"], window_until=_hhmm(window["deadline"]), resident=resident)
        self.state, self.reason = "running", "window until %s" % _hhmm(window["deadline"])
        if not self._start_session(cfg, resident):
            if not window["sessions"]:
                self.retry_until = self.clock() + float(cfg["retry_s"])
                self._end_window("no session could be started", preempted=False, own=False)

    def _start_cpu(self, unit: str, engine_url: str, cfg: Dict[str, Any]) -> Optional[str]:
        if not self.systemctl or not self.http_ok:
            return "no way to start units here"
        res = self.systemctl("start", unit)
        if not res.get("ok"):
            return str(res.get("error") or "systemctl start failed")
        base = re.sub(r"/v1/?$", "", engine_url.rstrip("/"))
        deadline = self.clock() + float(cfg["cpu_wait_s"])
        while self.clock() < deadline:
            if self.http_ok(base + "/health"):
                return None
            self.sleep(3)
        self.systemctl("stop", unit)
        return "no /health within %ss" % cfg["cpu_wait_s"]

    def _start_session(self, cfg: Dict[str, Any], preferred: Optional[str] = None) -> bool:
        """Start the next session in the window: round robin over the residents assigned to the window's model."""
        w = self.window
        now = self.clock()
        remaining = w["deadline"] - now
        if remaining < float(cfg["min_session_s"]):
            return False
        try:
            residents = self.worker.residents().get("residents") or []
        except Exception:
            return False
        open_ids, _paused = self._residents(residents)
        mine = [r for r in open_ids if self.assignment(r) == w["model"]]
        mine = [r for r in mine if not self._budget_ok(r, cfg, now)]
        if not mine:
            return False
        fresh = [r for r in mine if not self.sessions.has_session(r)]
        if preferred in mine:
            pick = preferred
        elif fresh:
            pick = fresh[0]                              # a configured resident's first turn comes before the rotation moves on
        else:
            last = w.get("last_resident")
            pick = next((r for r in mine if last is None or r > last), mine[0])
        spec = self.models().get(w["model"]) or {}
        body: Dict[str, Any] = {"resident": pick, "model": w["model"], "max_seconds": int(remaining)}
        rate = spec.get("tok_s") or self.sessions.tok_s(w["model"])
        if isinstance(rate, (int, float)) and rate > 0:
            body["max_tokens"] = max(256, int(remaining * float(rate)))
        try:
            out = self.worker.start_session(body, by="scheduler")
        except Exception as e:
            self._skip("the session could not start: %s" % str(e)[:160], "running")
            return False
        w.update(session=out.get("session"), last_resident=pick, session_started=now, sessions=w["sessions"] + 1, resident=pick)
        self.sessions.started(out.get("session") or "?", pick, w["model"], "scheduler")
        self.state, self.reason = "running", "%s is in a session (window until %s)" % (pick, _hhmm(w["deadline"]))
        return True

    # ---- supervising a window ----
    def _supervise(self, cfg: Dict[str, Any], gate: Dict[str, Any], now: float) -> None:
        w = self.window
        held = getattr(self.lease, "lease", None)
        if self.yield_request:                                  # Computer took the GPU back (yield_to_computer): the session is stopped
            why = self.yield_request["why"]
            self.yield_request = None
            return self._end_window(why, preempted=True, own=False, release=False)
        if w["kind"] == "lease" and (not held or held.get("id") != w["lease_id"]):
            return self._end_window("the lease is gone", preempted=True, own=False, release=False)
        if w["kind"] == "cpu" and held:
            return self._end_window("an engine lease took over (the lease wins)", preempted=True, own=False)
        if not cfg["enabled"]:
            return self._end_window("the schedule was switched off", preempted=False, own=True)
        if int(self.pending_fn() or 0):
            return self._end_window("an approval is waiting for Computer", preempted=True, own=False)
        last = self.last_turn_fn()
        if last and last > w["t0"]:
            return self._end_window("Computer got a turn", preempted=True, own=False)
        shared = self._shared(gate)
        if gate.get("cockpit"):
            shared = shared + [str(gate["cockpit"])]
        if shared:
            return self._end_window("the gate closed: " + "; ".join(shared[:3]), preempted=True, own=False)
        if now >= w["deadline"]:
            return self._end_window("the window ran out", preempted=False, own=True)
        if w.get("session"):
            try:
                running = self.worker.health().get("running")
            except Exception as e:
                # cannot see the worker: the session is not supervised, so it does not run
                return self._end_window("the garden worker stopped answering (%s)" % str(e)[:60], preempted=True, own=False)
            if running != w["session"]:                       # it ended by itself (rest, budget, a loop, an error)
                try:
                    st = self.worker.session(w["session"])
                except Exception:
                    st = {}
                self.sessions.ended(w["session"], st.get("tokens") or 0, st.get("reason") or "ended")
                self.active.clear(w["session"])
                w["session"] = None
                if not self._start_session(cfg):
                    return self._end_window("no time or no resident left in the window", preempted=False, own=True)
        else:
            if not self._start_session(cfg):
                return self._end_window("no time or no resident left in the window", preempted=False, own=True)

    def yield_to_computer(self, why: str, cooldown_s: float = 0.0, resource: str = COORDINATOR_ROLE) -> Dict[str, Any]:
        """Computer takes the GPU back (the platform's takeback calls this from any thread; it registered through `register_yielder`).
        Stops the running session at once, so the worker saves the draft (reason = `why`, e.g. "takeback: tool"), and asks the next tick to
        end the window. The takeback releases the lease itself. Starts the cooldown for layouts that evict `resource`; a window whose lease does
        not have that card out (the garden holds the other GPU) is left alone."""
        held = getattr(self.lease, "lease", None)
        roles = (held.get("evicted_roles") or []) if isinstance(held, dict) else []
        if roles and resource not in roles:
            return {"ok": True, "stopped": False, "note": "the garden's lease does not hold the %s" % resource}
        if cooldown_s:
            self.takeback_until[resource] = max(self.takeback_until.get(resource, 0.0), self.clock() + float(cooldown_s))
        w = self.window
        if not w and not self.starting:
            return {"ok": True, "stopped": False, "note": "no garden window"}
        self.yield_request = {"why": str(why)[:200], "at": self.clock()}
        sid = (w or {}).get("session")
        stopped = False
        if sid:
            try:
                out = self.worker.stop_session(sid, str(why)[:200])         # the draft is kept with this reason: "takeback: <why>"
                self.sessions.ended(sid, (out or {}).get("tokens") or 0, "preempted: " + why)
                w["stopped_by_takeback"] = sid
                stopped = True
            except Exception as e:
                logger.warning("garden scheduler: takeback stop failed: %s", e)
            self.active.clear(sid)
        return {"ok": True, "stopped": stopped, "session": sid}

    def _end_window(self, reason: str, preempted: bool, own: bool, release: bool = True) -> None:
        w = self.window
        if not w:
            return
        sid = w.get("session")
        if sid and w.get("stopped_by_takeback") == sid:
            sid = None                                          # already stopped (and logged) by yield_to_computer
        if sid:
            try:
                out = self.worker.stop_session(sid, ("garden scheduler: " + reason)[:200])
                self.sessions.ended(sid, (out or {}).get("tokens") or 0, "preempted: " + reason)
            except Exception as e:
                logger.warning("garden scheduler: stop failed: %s", e)
            self.active.clear(sid)
        if preempted:
            self._trace("preempted", "preempted", why=reason[:200], session=sid, layout=w.get("layout"),
                        held_min=round((self.clock() - w["started"]) / 60, 1))
        released: Any = None
        if w["kind"] == "lease" and release:
            try:
                released = self.lease.release(w["lease_id"], why="garden: " + reason[:80], approver="garden-scheduler")
            except Exception as e:
                released = {"ok": False, "error": str(e)}
        elif w["kind"] == "cpu" and not getattr(self.lease, "lease", None) and self.systemctl:
            released = self.systemctl("stop", w["unit"])
            released = {"ok": bool(released.get("ok")), "error": released.get("error")}
        if own:
            self.history.mark_own_end(w["gpus"], self.clock())
        self._trace("released", "released" if (released or {}).get("ok", True) else "failed", why=reason[:200], layout=w.get("layout"),
                    sessions=w["sessions"], error=(released or {}).get("error") if released and not released.get("ok", True) else None)
        self.window = None
        self.active.clear()
        self.cooldown_until = self.clock() + float(self.cfg()["cooldown_s"])
        self.state = "preempted" if preempted else "waiting"
        self.reason = reason

    # ------------------------------------------------------------------ report ----
    def status(self) -> Dict[str, Any]:
        now = self.clock()
        cfg = self.cfg()
        gate = self._gate or {}
        gates = {gpu: {"open": bool(g.get("demand_open", g.get("open"))), "reasons": g.get("reasons") or []} for gpu, g in (gate.get("gpus") or {}).items()}
        shared = self._shared(gate)
        est = {}
        for gpu in list(gates) + [HOUSE]:
            e = self.estimate(gpu, now)
            if e:
                est[gpu] = {k: e[k] for k in ("minutes", "basis", "samples", "age_s")}
        w = self.window
        sessions_24h = self.sessions.per_resident_24h(now)
        return {"ok": True, "enabled": bool(cfg["enabled"]), "state": self.state, "reason": self.reason,
                "last_event": self.last_event, "gates": gates, "house_quiet": not shared, "shared_reasons": shared,
                "estimates": est, "min_window_min": cfg["min_window_min"],
                "window": ({"model": w["model"], "layout": w.get("layout"), "kind": w["kind"], "until": _hhmm(w["deadline"]),
                            "lease_until": _hhmm(w["expires"]) if w.get("expires") else None, "resident": w.get("resident"),
                            "session": w.get("session"), "sessions": w["sessions"]} if w else None),
                "cooldown_until": self.cooldown_until if self.cooldown_until > now else None,
                "takeback": self._takeback_status(),
                "sessions_24h": sessions_24h,
                "budgets": {"max_sessions_24h": cfg["max_sessions_24h"], "max_tokens_hour": cfg["max_tokens_hour"]},
                "cpu_enabled": bool(cfg["cpu_enabled"]), "allow_coordinator_layouts": bool(cfg["allow_coordinator_layouts"]),
                "models": {k: {"layout": v.get("layout"), "unit": v.get("unit")} for k, v in self.models().items()},
                "assignments": {rid: str(r.get("model") or "") for rid, r in (section(self.cfg_fn() or {}).get("residents") or {}).items()
                                if isinstance(r, dict)},
                "history": {gpu: len(self.history.lengths(gpu)) for gpu in set(self.history.closed) | set(gates)}}

    def _takeback_status(self) -> Dict[str, Any]:
        """The last takeback (what the platform recorded) and how long coordinator-evicting layouts stay off the table."""
        last = None
        if self.takeback_status_fn:
            try:
                last = (self.takeback_status_fn() or {}).get("last")
            except Exception:
                last = None
        left, vleft = self.takeback_left(COORDINATOR_ROLE), self.takeback_left(VISION_ROLE)
        return {"last": last, "coordinator_layouts_blocked_for_s": round(left) if left > 0 else 0,
                "vision_layouts_blocked_for_s": round(vleft) if vleft > 0 else 0}

    def metrics(self) -> List[str]:
        enabled = 1 if self.cfg()["enabled"] else 0
        lines = ["# HELP stonesage_garden_scheduler_state 1 for the garden scheduler's current state.",
                 "# TYPE stonesage_garden_scheduler_state gauge"]
        for s in STATES:
            lines.append('stonesage_garden_scheduler_state{state="%s"} %d' % (s, 1 if self.state == s else 0))
        lines += ["# HELP stonesage_garden_scheduler_enabled 1 when garden.schedule.enabled is true.",
                  "# TYPE stonesage_garden_scheduler_enabled gauge", "stonesage_garden_scheduler_enabled %d" % enabled,
                  "# HELP stonesage_garden_sessions_24h Garden sessions started by the scheduler or by hand in the last 24 h, per resident.",
                  "# TYPE stonesage_garden_sessions_24h gauge"]
        for rid, n in sorted(self.sessions.per_resident_24h().items()):
            lines.append('stonesage_garden_sessions_24h{resident="%s"} %d' % (re.sub(r'[^a-z0-9-]', "", rid), n))
        return lines

    # ------------------------------------------------------------------ the thread ----
    def note_manual_start(self, out: Dict[str, Any], model: str) -> None:
        if out.get("session"):
            self.sessions.started(out["session"], out.get("resident") or "?", model, "manual")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        def loop():
            while True:
                try:
                    self.tick()
                except Exception:
                    logger.exception("garden scheduler tick failed")
                for hook in list(self.tick_hooks):
                    try:
                        hook()
                    except Exception:
                        logger.exception("garden scheduler tick hook failed")
                c = self.cfg()
                self.sleep(float(c["running_poll_s"] if self.window else c["poll_s"]))

        self._thread = threading.Thread(target=loop, name="garden-scheduler", daemon=True)
        self._thread.start()
