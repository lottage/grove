"""The garden's models (the design notes, "Models: assigned or requested", step G4): the catalogue StoneSage pushes to the worker, and
the queue of model requests a resident makes (`request_model`).

Catalogue (pushed with the window snapshots as kind "models", JSON text, capped at MAX_CATALOGUE_CHARS):
    configured   every entry of config `garden.models`: name, a one-line description from the model's provenance record (family, size,
                 quant, build), tokens/s when known, and the slot it runs in (an engine layout, or the CPU)
    other        GGUFs on the machine (provenance) that some build could load and that are not in `garden.models`, marked "needs <owner label>" (config `garden.owner_label`, default "the owner")

Requests (the worker keeps them in the resident's wishes.jsonl; StoneSage pulls and decides):
    - the model is in `garden.models` AND in `garden.residents.<id>.allowed_models` (default: none)  ->  granted at once, by "auto"
    - anything else waits for the owner: the board's Grant / Decline, and (config `garden.grants.push`, default true) a phone notification with
      Grant / Decline buttons through the same push machinery the lease check-in uses. A model that is not in `garden.models` has no slot:
      only the owner can grant it, on the board, by naming the layout (and engine url) it runs in, which adds the `garden.models` entry;
      otherwise the answer is a decline with the reason "no slot configured".
    - a grant sets `garden.residents.<id>.model` (the same save path as the board's model select): it applies from the resident's NEXT
      session, never to one that is running. The resident learns the outcome in that session's opening.

Nothing here calls the garden from the garden's side: StoneSage pulls `GET /requests?status=pending` and answers with
`POST /requests/<id>/decide`.
"""

import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .section import section as _cfg

logger = logging.getLogger("StoneSage.garden_models")

MODELS_KIND = "models"
MAX_CATALOGUE_CHARS = 28000          # the worker keeps up to 64 KB per snapshot; the window feeder allows 30000
MAX_OTHER = 40                       # other models listed
MAX_DESCRIPTION = 200
DEFAULT_MAX_OTHER_GB = 16.0          # an "other" model larger than this fits no slot (the largest GPU slot is 12 GB, CPU slots are slow)
GRANT, DECLINE = "GARDEN_GRANT_", "GARDEN_DECLINE_"       # phone buttons (push_approvals hands unknown actions to handle_action below)
NO_SLOT = "no slot configured"
REQUEST_ID = re.compile(r"^[a-z0-9-]{1,64}\.r[0-9]{5,}$")
NOT_LM_ARCH = ("clip", "bert", "nomic-bert", "t5encoder")     # embedders and vision projectors are not residents' minds
QUANT = re.compile(r"(?<![A-Za-z0-9])(IQ\d[A-Z0-9_]*|Q\d[A-Z0-9_]*|BF16|F16|F32|PTQ\d_\d|PQ\d_\d|MXFP4)(?![A-Za-z0-9])", re.I)




def models_table(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    m = _cfg(config).get("models")
    return {str(k): v for k, v in m.items() if isinstance(v, dict)} if isinstance(m, dict) else {}


def allowed_models(config: Dict[str, Any], resident: str) -> List[str]:
    r = (_cfg(config).get("residents") or {}).get(resident)
    a = r.get("allowed_models") if isinstance(r, dict) else None
    return [str(x) for x in a] if isinstance(a, list) else []


def owner_label(config: Dict[str, Any]) -> str:
    """How the catalogue names who must add an "other" model: config garden.owner_label, else "the owner"."""
    return " ".join(str(_cfg(config).get("owner_label") or "the owner").split())[:40]


def push_enabled(config: Dict[str, Any]) -> bool:
    g = _cfg(config).get("grants")
    return not (isinstance(g, dict) and g.get("push") is False)


def stem(file: str) -> str:
    base = os.path.basename(str(file or "").replace("\\", "/"))
    return base[:-5] if base.lower().endswith(".gguf") else base


def one_line(parts: List[str], limit: int = MAX_DESCRIPTION) -> str:
    text = " · ".join(" ".join(str(p).split()) for p in parts if p)
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def describe(m: Optional[Dict[str, Any]], build: str = "") -> str:
    """One line from a provenance record: family / name, size label, quant, the build that loads it."""
    if not m:
        return "no provenance record for this model"
    lin = m.get("lineage") or {}
    name = lin.get("name") or lin.get("basename") or stem(m.get("file", ""))
    size = lin.get("size_label")
    quant = QUANT.search(stem(m.get("file", "")))
    builds = build or ", ".join(m.get("loadable_by") or [])
    return one_line([name, size if size and str(size).lower() not in str(name).lower() else "", quant.group(1).upper() if quant else "",
                     ("build " + builds) if builds else ""])


def build_catalogue(config: Dict[str, Any], summary: Optional[Dict[str, Any]], tok_s: Optional[Callable[[str], Optional[float]]] = None,
                    now: Optional[float] = None) -> Dict[str, Any]:
    """The catalogue as data. `summary` = Provenance.summary() (models with sha256, lineage, loadable_by, size_gb), or None when
    provenance is not available: the configured models are still listed, without descriptions from it."""
    g = _cfg(config)
    prov = [m for m in (summary or {}).get("models") or [] if isinstance(m, dict)]
    by_sha = {m["sha256"]: m for m in prov if m.get("sha256")}
    configured: List[Dict[str, Any]] = []
    used = set()
    for name, spec in sorted(models_table(config).items()):
        sha = str(spec.get("model_sha256") or "")
        if sha:
            used.add(sha)
        entry: Dict[str, Any] = {"name": name, "description": describe(by_sha.get(sha), str(spec.get("build") or "")),
                                 "slot": ("engine layout " + str(spec["layout"])) if spec.get("layout") else "CPU"}
        rate = spec.get("tok_s")
        if not isinstance(rate, (int, float)) and tok_s:
            try:
                rate = tok_s(name)
            except Exception:
                rate = None
        if isinstance(rate, (int, float)) and not isinstance(rate, bool):
            entry["tok_s"] = round(float(rate), 1)
        configured.append(entry)
    names = {e["name"] for e in configured}
    limit_gb = float((g.get("catalogue") or {}).get("max_other_gb") or DEFAULT_MAX_OTHER_GB) if isinstance(g.get("catalogue"), dict) \
        else DEFAULT_MAX_OTHER_GB
    other: List[Dict[str, Any]] = []
    for m in sorted(prov, key=lambda x: str(x.get("file", "")).lower()):
        sha, file = m.get("sha256"), str(m.get("file", ""))
        if (not sha or sha in used or m.get("error") or not m.get("loadable_by") or "mmproj" in file.lower()
                or str(m.get("architecture") or "").lower() in NOT_LM_ARCH or (m.get("size_gb") or 0) > limit_gb):
            continue
        name = stem(file)
        if not name or name in names:
            continue
        names.add(name)
        used.add(sha)                                  # one content, several paths: listed once
        other.append({"name": name, "description": describe(m), "needs": owner_label(config)})
        if len(other) >= MAX_OTHER:
            break
    cat = {"configured": configured, "other": other, "at": round(time.time() if now is None else now, 1)}
    while len(json.dumps(cat, ensure_ascii=False)) > MAX_CATALOGUE_CHARS and cat["other"]:
        cat["other"].pop()
    return cat


def catalogue_text(config: Dict[str, Any], summary: Optional[Dict[str, Any]], tok_s: Optional[Callable[[str], Optional[float]]] = None,
                   now: Optional[float] = None) -> str:
    return json.dumps(build_catalogue(config, summary, tok_s, now), ensure_ascii=False)


class GrantError(RuntimeError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class ModelGrants:
    """Pulls pending model requests from the worker and decides them (see the module docstring).

    worker: GardenClient (.requests(status), .decide_request(id, status, by, note)); assign(resident, model) -> saved: the board's model
    select; add_model(name, entry) -> saved: add a garden.models entry; notify(title, message, buttons, tag) -> result: the phone push, or None."""

    def __init__(self, cfg_fn: Callable[[], Dict[str, Any]], worker: Any, assign: Callable[[str, str], Any],
                 add_model: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
                 notify: Optional[Callable[[str, str, Optional[List[Dict[str, str]]], str], Any]] = None,
                 trace: Optional[Callable[..., None]] = None, summary_fn: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
                 data_dir: Optional[str] = None, clock: Callable[[], float] = time.time):
        self.cfg_fn, self.worker, self.assign, self.add_model = cfg_fn, worker, assign, add_model
        self.notify, self.trace, self.summary_fn, self.clock = notify, trace, summary_fn, clock
        self._lock = threading.RLock()
        self._last = 0.0
        self.last_error = ""
        self.pending: List[Dict[str, Any]] = []
        self._path = os.path.join(data_dir, "garden_requests_notified.json") if data_dir else None
        self._notified = self._load()

    # ---- remembering which requests the phone was told about (so a restart does not push them again) ----
    def _load(self) -> set:
        if not self._path:
            return set()
        try:
            with open(self._path, encoding="utf-8-sig") as f:
                return {str(x) for x in json.load(f)}
        except (OSError, ValueError, TypeError):
            return set()

    def _save(self) -> None:
        if not self._path:
            return
        try:
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                json.dump(sorted(self._notified)[-500:], f)
            os.replace(tmp, self._path)
        except OSError:
            pass

    def _trace(self, event: str, outcome: str, **fields: Any) -> None:
        if self.trace:
            try:
                self.trace(event, outcome, **fields)
            except Exception:
                pass

    # ---- the pull ----
    def maybe_tick(self, min_interval_s: float = 20.0) -> None:
        """tick(), at most once per `min_interval_s`: called on every scheduler tick and on every board read."""
        with self._lock:
            if self.clock() - self._last < min_interval_s:
                return
            self.tick()

    def tick(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._last = self.clock()
            try:
                rows = list((self.worker.requests("pending") or {}).get("requests") or [])
                self.last_error = ""
            except Exception as e:
                self.last_error = str(e)[:200]
                return []
            self.pending = []
            for r in rows:
                try:
                    if not self._consider(r):
                        self.pending.append(r)
                except Exception as e:                       # one bad request must not stop the rest
                    logger.warning("garden request %s: %s", r.get("id"), type(e).__name__)
                    self.pending.append(r)
            return list(self.pending)

    def _consider(self, r: Dict[str, Any]) -> bool:
        """Decide it if the rules say so (True), else make sure the owner knows (False)."""
        rid, model, who = str(r.get("resident") or ""), str(r.get("model") or ""), str(r.get("id") or "")
        conf = self.cfg_fn()
        if not REQUEST_ID.match(who) or not rid or not model:
            return True                                      # malformed: leave it, do not push it
        if model in models_table(conf) and model in allowed_models(conf, rid):
            self.decide(who, "granted", "auto", "on the resident's allowed list", req=r)
            return True
        if who not in self._notified:
            self._notified.add(who)
            self._save()
            self._trace("request", "waiting", resident=rid, request=who, model=model[:120], configured=model in models_table(conf))
            self._push(r, model in models_table(conf))
        return False

    def _push(self, r: Dict[str, Any], configured: bool) -> None:
        if not self.notify or not push_enabled(self.cfg_fn()):
            return
        who = r["id"]
        buttons = ([{"action": GRANT + who, "title": "Grant"}] if configured else []) + [{"action": DECLINE + who, "title": "Decline"}]
        message = "%s asks to run on %s. %s%s" % (r.get("name") or r.get("resident"), r.get("model"), " ".join(str(r.get("why") or "").split())[:200],
                                                    "" if configured else " (No slot is configured for it: grant it on the board.)")
        try:
            self.notify("Garden: model request", message, buttons, "garden-" + who)
        except Exception as e:
            logger.warning("garden request push failed: %s", type(e).__name__)

    # ---- deciding ----
    def decide(self, request_id: str, status: str, by: str, note: str = "", req: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Record the decision on the worker; for a grant, then assign the model (it applies from the resident's next session).
        The worker refuses a request that was already decided, and then nothing is assigned."""
        req = req or self._find(request_id)
        if status == "granted":
            out = self.worker.decide_request(request_id, "granted", by, note)
            self.assign(req["resident"], req["model"])
            self._trace("request", "auto_granted" if by == "auto" else "granted", resident=req["resident"], request=request_id,
                        model=req["model"][:120], by=by)
            return out
        out = self.worker.decide_request(request_id, "declined", by, note)
        self._trace("request", "declined", resident=req["resident"], request=request_id, model=req["model"][:120], by=by)
        return out

    def _find(self, request_id: str) -> Dict[str, Any]:
        if not REQUEST_ID.match(request_id or ""):
            raise GrantError("bad request id", 400)
        for r in (self.worker.requests("pending") or {}).get("requests") or []:
            if r.get("id") == request_id:
                return r
        raise GrantError("no such pending request", 404)

    def owner_decides(self, request_id: str, status: str, note: str = "", layout: Any = None, engine_url: str = "", unit: str = "",
             by: str = "owner") -> Dict[str, Any]:
        """The owner's answer from the board. Granting a model that is not in garden.models needs the slot he confirms (`layout` an engine
        layout name, or '' with `unit` + `engine_url` for the CPU slot); without it the grant is refused and nothing changes."""
        if status not in ("granted", "declined"):
            raise GrantError("status must be granted or declined", 400)
        note = " ".join(str(note or "").split())[:500]
        with self._lock:
            req = self._find(request_id)
            model = str(req.get("model") or "")
            conf = self.cfg_fn()
            if status == "declined":
                return self.decide(request_id, "declined", by, note or ("" if model in models_table(conf) else NO_SLOT), req=req)
            if model not in models_table(conf):
                entry = self._new_entry(conf, model, layout, engine_url, unit)
                if self.add_model is None:
                    raise GrantError("this install cannot add a garden.models entry", 503)
                self.add_model(model, entry)
            return self.decide(request_id, "granted", by, note, req=req)

    def _new_entry(self, conf: Dict[str, Any], model: str, layout: Any, engine_url: str, unit: str) -> Dict[str, Any]:
        engine_url = str(engine_url or "").strip()
        if layout is None and not unit:
            raise GrantError("%s has no slot yet: name the layout it runs in (and the engine url), or decline" % model, 400)
        if not engine_url:
            raise GrantError("the engine url the model is served at is needed", 400)
        layouts = conf.get("engine_layouts") or {}
        if layout:
            if str(layout) not in layouts:
                raise GrantError("unknown layout %r: engine_layouts has %s" % (str(layout)[:60], ", ".join(sorted(layouts)) or "none"), 400)
        elif not unit:
            raise GrantError("the CPU slot needs the systemd unit that serves the model", 400)
        sha = ""
        summary = (self.summary_fn() if self.summary_fn else None) or {}
        for m in summary.get("models") or []:
            if stem(m.get("file", "")) == model and m.get("sha256"):
                sha = m["sha256"]
                break
        entry: Dict[str, Any] = {"layout": str(layout) if layout else None, "engine_url": engine_url, "model_sha256": sha}
        if unit and not layout:
            entry["unit"] = str(unit)
        return entry

    # ---- the phone's buttons ----
    def handle_action(self, action: str) -> Optional[str]:
        """A tapped Grant / Decline on the phone (hooked into push_approvals). None when it is not ours."""
        if not isinstance(action, str) or not action.startswith((GRANT, DECLINE)):
            return None
        grant = action.startswith(GRANT)
        who = action[len(GRANT if grant else DECLINE):]
        try:
            req = self._find(who)
            out = (self.owner_decides(who, "granted" if grant else "declined", by="phone")
                   if not grant or req.get("model") in models_table(self.cfg_fn()) else None)
            if out is None:
                outcome = "That model has no slot yet: grant it on the board."
            else:
                outcome = ("Granted %s to %s. It applies from the next session." % (req.get("model"), req.get("name") or req.get("resident"))
                           if grant else "Declined %s for %s." % (req.get("model"), req.get("name") or req.get("resident")))
        except GrantError as e:
            outcome = str(e)
        except Exception as e:
            outcome = "failed: %s" % type(e).__name__
        if self.notify:
            try:
                self.notify("Garden: model request", outcome, None, "garden-" + who)
            except Exception:
                pass
        return outcome
