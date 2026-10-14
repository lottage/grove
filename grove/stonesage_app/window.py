"""The garden's window (the design notes): plain-text snapshots of the world, pushed to the garden worker.

StoneSage writes, the garden never asks. Every 30 minutes (config `garden.window.interval_s`) and on demand
(`POST /api/garden/window/refresh`) this builds, from read-only sources, a few short pieces of text and posts them to the worker's
`/snapshots`:

    weather     Home Assistant's weather entity (state, temperature, humidity, wind)
    cameras     Frigate's events of the last 24 h: label, camera, time. NO pictures; names of people only if garden.window_names is true
    book        a passage from a Kavita book (config garden.kavita: url + key; skipped when unset)
    headlines   RSS headlines (config garden.rss: a list of feed urls, and/or garden.freshrss: user + password; skipped when unset)
    computer    Computer's own trace: counts, and short text of recent questions and failures
    household-terms   NOT a window item: the household / pet / camera / family names the worker's egress scan refuses in a search
    models      NOT a window item: the model catalogue (garden_models.py) behind a resident's list_models / request_model; switch "models"

Each kind is switched in config `garden.window` ({"weather": true, "cameras": true, "book": true, "headlines": true, "computer": true,
"terms": true, "interval_s": 1800}). Text only, every snapshot capped at MAX_CHARS. A source that fails leaves the previous snapshot in
place (the window shows its age). Nothing here imports the garden: the worker is reached over HTTP with its control key.
"""

import html
import json
import logging
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from lodestone import endpoints

from .section import section as _cfg

logger = logging.getLogger("StoneSage.garden_window")

KINDS = ("weather", "cameras", "book", "headlines", "computer")
TERMS_KIND = "household-terms"
MODELS_KIND = "models"          # the model catalogue (JSON text; garden_models.py): not a window item, the residents' list_models reads it
MAX_CHARS = 6000
MAX_TERMS_CHARS = 30000
DEFAULT_INTERVAL_S = 1800
MIN_INTERVAL_S = 60
MAX_FEED_BYTES = 1024 * 1024

Http = Callable[..., Any]          # http(method, url, headers=None, body=None, timeout=15.0, raw=False) -> parsed JSON, or text if raw




def window_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    w = _cfg(config).get("window")
    return w if isinstance(w, dict) else {}


def kind_enabled(config: Dict[str, Any], kind: str) -> bool:
    return window_cfg(config).get(kind, True) is not False


def _clip(text: str, limit: int = MAX_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _ago(seconds: float) -> str:
    m = int(max(0, seconds) // 60)
    if m < 1:
        return "just now"
    if m < 90:
        return "%d min ago" % m
    h = m // 60
    return "%d h ago" % h if h < 48 else "%d days ago" % (h // 24)


# ---- the plain HTTP the sources use ---------------------------------------------------------------------------------
def default_http(opener: Optional[Callable[..., Any]] = None) -> Http:
    opener = opener or urllib.request.urlopen

    def http(method: str, url: str, headers: Optional[Dict[str, str]] = None, body: Any = None, timeout: float = 15.0,
             raw: bool = False) -> Any:
        data = None
        hdrs = dict(headers or {})
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
        with opener(req, timeout=timeout) as r:
            payload = r.read(MAX_FEED_BYTES + 1)
        if len(payload) > MAX_FEED_BYTES:
            raise ValueError("answer too large")
        text = payload.decode("utf-8", "replace")
        return text if raw else json.loads(text)

    return http


# ---- builders: pure functions of what the sources returned ---------------------------------------------------------
def build_weather(states: List[Dict[str, Any]], entity: str = "") -> str:
    """One line from Home Assistant's weather entity (the named one, else the first)."""
    weather = [s for s in states if str(s.get("entity_id", "")).startswith("weather.")]
    if entity:
        weather = [s for s in weather if s.get("entity_id") == entity] or weather
    if not weather:
        return ""
    s = weather[0]
    a = s.get("attributes") or {}
    parts = [str(s.get("state", "unknown")).replace("-", " ").replace("_", " ")]
    if a.get("temperature") is not None:
        parts.append("%s %s" % (a["temperature"], a.get("temperature_unit", "")))
    if a.get("humidity") is not None:
        parts.append("humidity %s%%" % a["humidity"])
    if a.get("wind_speed") is not None:
        parts.append("wind %s %s" % (a["wind_speed"], a.get("wind_speed_unit", "")))
    if a.get("pressure") is not None:
        parts.append("pressure %s %s" % (a["pressure"], a.get("pressure_unit", "")))
    return _clip("Weather at home right now: " + ", ".join(p.strip() for p in parts) + ".")


def _camera_label(name: str) -> str:
    return name.replace("_", " ").strip()


def build_cameras(events: List[Dict[str, Any]], now: float, allow_names: bool = False, hours: float = 24.0) -> str:
    """What the cameras saw: counts per label and camera, and the newest few events. Labels and times only, never pictures."""
    recent = [e for e in events if isinstance(e, dict) and (e.get("start_time") or 0) >= now - hours * 3600]
    if not recent:
        return "The cameras recorded nothing in the last %d hours." % int(hours)
    recent.sort(key=lambda e: e.get("start_time") or 0, reverse=True)
    by_label: Dict[str, Dict[str, int]] = {}
    for e in recent:
        by_label.setdefault(str(e.get("label") or "something"), {})
        cam = _camera_label(str(e.get("camera") or "a camera"))
        by_label[str(e.get("label") or "something")][cam] = by_label[str(e.get("label") or "something")].get(cam, 0) + 1
    lines = ["On the home cameras in the last %d hours (labels and times only):" % int(hours)]
    for label, cams in sorted(by_label.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])):
        where = ", ".join("%s %d" % (c, n) for c, n in sorted(cams.items(), key=lambda kv: (-kv[1], kv[0])))
        lines.append("- %s x%d (%s)" % (label, sum(cams.values()), where))
    lines.append("Most recent:")
    for e in recent[:6]:
        who = ""
        sub = e.get("sub_label")
        if allow_names and sub:
            who = " (%s)" % (sub[0] if isinstance(sub, (list, tuple)) and sub else sub)
        lines.append("- %s: %s at %s%s" % (_ago(now - (e.get("start_time") or now)), e.get("label") or "something",
                                           _camera_label(str(e.get("camera") or "a camera")), who))
    return _clip("\n".join(lines))


def build_computer(records: List[Dict[str, Any]], now: float, hours: float = 24.0) -> str:
    """Computer's trace in short: how many turns, how they ended, the last questions and the turns that went wrong."""
    turns = [r for r in records if r.get("kind", "courage") == "courage" and (r.get("at") or 0) >= now - hours * 3600]
    if not turns:
        return "Computer, the house assistant, was not asked anything in the last %d hours." % int(hours)
    outcomes: Dict[str, int] = {}
    for r in turns:
        outcomes[str(r.get("outcome") or "unknown")] = outcomes.get(str(r.get("outcome") or "unknown"), 0) + 1
    lines = ["Computer, the house assistant, in the last %d hours: %d turns (%s)." % (
        int(hours), len(turns), ", ".join("%d %s" % (n, k) for k, n in sorted(outcomes.items(), key=lambda kv: -kv[1])))]
    asked = sorted(turns, key=lambda r: r.get("at") or 0, reverse=True)
    questions = [" ".join(str(r.get("user") or "").split())[:100] for r in asked if r.get("user")][:5]
    if questions:
        lines.append("Recent things people asked it:")
        lines += ['- "%s"' % q for q in questions]
    bad = [r for r in asked if r.get("outcome") not in ("answered", "asked_approval") or r.get("triggers")][:5]
    if bad:
        lines.append("Where it struggled:")
        for r in bad:
            trig = ", ".join(str(t) for t in (r.get("triggers") or [])[:3])
            lines.append('- "%s" -> %s%s' % (" ".join(str(r.get("user") or "").split())[:80], r.get("outcome") or "unknown",
                                             " (" + trig + ")" if trig else ""))
    return _clip("\n".join(lines))


def build_headlines(items: List[Tuple[str, str]], limit: int = 25) -> str:
    seen, lines = set(), []
    for source, title in items:
        t = " ".join(str(title).split())
        if t and t.lower() not in seen:
            seen.add(t.lower())
            lines.append("- [%s] %s" % (source[:40], t[:200]))
        if len(lines) >= limit:
            break
    return _clip("Headlines:\n" + "\n".join(lines)) if lines else ""


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("p", "div", "br", "h1", "h2", "h3", "li") and not self._skip:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_text(source: str) -> str:
    p = _Text()
    try:
        p.feed(source)
        p.close()
    except Exception:
        pass
    paras = [" ".join(x.split()) for x in re.split(r"\n\s*\n", "".join(p.parts))]
    return "\n\n".join(x for x in paras if x)


def pick_passage(text: str, rng: random.Random, target: int = 1200) -> str:
    """A run of whole paragraphs of about `target` characters from a random place in `text`."""
    paras = [p for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 40]
    if not paras:
        return ""
    start = rng.randrange(len(paras))
    out, size = [], 0
    for p in paras[start:]:
        out.append(p.strip())
        size += len(p)
        if size >= target:
            break
    return "\n\n".join(out)[:target * 2]


def household_terms(config: Dict[str, Any], states: Iterable[Dict[str, Any]] = (), extra: Iterable[str] = ()) -> List[str]:
    """Names the egress scan refuses in a search: the app's privacy terms (passed as `extra`), config garden.household_terms,
    camera names and recognition identities from config, and the friendly names of Home Assistant's people."""
    terms: List[str] = []

    def add(t: Any) -> None:
        t = " ".join(str(t or "").split())
        if len(t) >= 3 and t.lower() not in {x.lower() for x in terms}:
            terms.append(t)

    for t in extra:
        add(t)
    for t in _cfg(config).get("household_terms") or []:
        add(t)
    fr = (config or {}).get("frigate") or {}
    for cam in (fr.get("cameras") or {}).values():
        add(str(cam).replace("_", " "))
    for ident in (fr.get("identities") or {}).values():
        add(ident)
    for name in ((config or {}).get("commentary") or {}).get("residents") or []:
        add(name)
    for s in states:
        if str(s.get("entity_id", "")).startswith("person."):
            full = str((s.get("attributes") or {}).get("friendly_name") or "")
            add(full)
            for w in full.split():
                add(w)
    return terms


# ---- the sources ----------------------------------------------------------------------------------------------------
def fetch_ha_states(config: Dict[str, Any], http: Http) -> List[Dict[str, Any]]:
    ha = (config or {}).get("homeassistant") or (config or {}).get("hass") or {}
    token = str(ha.get("token") or "").strip()
    base = endpoints.url("homeassistant", config)
    if not token or not base:
        raise RuntimeError("Home Assistant is not configured")
    data = http("GET", base.rstrip("/") + "/api/states", {"Authorization": "Bearer " + token})
    return data if isinstance(data, list) else []


def fetch_frigate_events(config: Dict[str, Any], http: Http, now: float) -> List[Dict[str, Any]]:
    base = str(((config or {}).get("frigate") or {}).get("url") or "").rstrip("/")
    if not base:
        raise RuntimeError("Frigate is not configured")
    q = urllib.parse.urlencode({"limit": 300, "after": int(now - 24 * 3600)})
    data = http("GET", base + "/api/events?" + q)
    return data if isinstance(data, list) else []


def kavita_passage(cfg: Dict[str, Any], http: Http, rng: random.Random) -> str:
    """A passage from a random page of a random book on Kavita (api key -> plugin token -> series -> chapter -> book page).
    The endpoint shapes follow Kavita's API; not yet exercised against the live server, so every step is defensive."""
    base, key = str(cfg.get("url") or "").rstrip("/"), str(cfg.get("key") or "").strip()
    if not base or not key:
        raise RuntimeError("Kavita is not configured")
    auth = http("POST", base + "/api/Plugin/authenticate?" + urllib.parse.urlencode({"apiKey": key, "pluginName": "StoneSageGarden"}),
                {}, b"")
    headers = {"Authorization": "Bearer " + str((auth or {}).get("token") or "")}
    series_ids = [int(x) for x in (cfg.get("series_ids") or []) if str(x).isdigit()]
    names: Dict[int, str] = {}
    if not series_ids:
        page = http("POST", base + "/api/Series/all-v2?PageNumber=1&PageSize=100", headers, {})
        for s in page if isinstance(page, list) else []:
            if s.get("format") in (3, None):                       # 3 = epub
                series_ids.append(int(s["id"]))
                names[int(s["id"])] = str(s.get("name") or "")
    if not series_ids:
        raise RuntimeError("no books found on Kavita")
    sid = rng.choice(series_ids)
    volumes = http("GET", base + "/api/Series/volumes?" + urllib.parse.urlencode({"seriesId": sid}), headers)
    chapters = [c for v in (volumes if isinstance(volumes, list) else []) for c in (v.get("chapters") or []) if c.get("pages")]
    if not chapters:
        raise RuntimeError("that book has no readable chapters")
    ch = rng.choice(chapters)
    pageno = rng.randrange(int(ch["pages"]))
    body = http("GET", base + "/api/Book/%d/book-page?page=%d" % (int(ch["id"]), pageno), headers, raw=True)
    try:
        body = json.loads(body)                                     # Kavita answers a JSON string of HTML
    except ValueError:
        pass
    passage = pick_passage(html_text(str(body)), rng)
    if not passage:
        raise RuntimeError("that page had no text")
    title = names.get(sid) or ("book %d" % sid)
    return _clip("From a book (%s), a passage:\n\n%s" % (title, passage))


def _feed_titles(raw: str, source: str) -> List[Tuple[str, str]]:
    if "<!ENTITY" in raw or "<!DOCTYPE" in raw.upper() and "ENTITY" in raw.upper():
        return []                                                   # no entity tricks in a feed
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    out = []
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag in ("item", "entry"):
            for ch in el:
                if ch.tag.rsplit("}", 1)[-1] == "title" and (ch.text or "").strip():
                    out.append((source, html.unescape(ch.text.strip())))
                    break
    return out[:8]


def fetch_headlines(config: Dict[str, Any], http: Http) -> List[Tuple[str, str]]:
    g = _cfg(config)
    feeds = [u for u in (g.get("rss") or []) if isinstance(u, str) and u.startswith(("http://", "https://"))]
    fresh = g.get("freshrss") if isinstance(g.get("freshrss"), dict) else {}
    fresh_url = endpoints.url("freshrss", config)
    if not feeds and not (fresh and fresh_url):
        raise RuntimeError("no RSS feeds configured")
    items: List[Tuple[str, str]] = []
    errors = 0
    for u in feeds[:10]:
        try:
            items += _feed_titles(http("GET", u, {"User-Agent": "StoneSageGarden/1"}, raw=True), urllib.parse.urlsplit(u).hostname or "feed")
        except Exception:
            errors += 1
    if fresh and fresh_url:
        try:
            login = http("POST", fresh_url.rstrip("/") + "/api/greader.php/accounts/ClientLogin",
                         {"Content-Type": "application/x-www-form-urlencoded"},
                         urllib.parse.urlencode({"Email": fresh.get("user", ""), "Passwd": fresh.get("password", "")}).encode("utf-8"),
                         raw=True)
            m = re.search(r"Auth=(\S+)", login)
            if m:
                data = http("GET", fresh_url.rstrip("/") + "/api/greader.php/reader/api/0/stream/contents/reading-list?n=20&output=json",
                            {"Authorization": "GoogleLogin auth=" + m.group(1)})
                for it in (data or {}).get("items") or []:
                    items.append((str((it.get("origin") or {}).get("title") or "FreshRSS"), str(it.get("title") or "")))
        except Exception:
            errors += 1
    if not items and errors:
        raise RuntimeError("the feeds did not answer")
    return items


# ---- the feeder -----------------------------------------------------------------------------------------------------
class WindowFeeder:
    """Builds the snapshots and pushes them with `push(kind, text)`; runs every interval on a thread, and on demand."""

    def __init__(self, config: Dict[str, Any], push: Callable[[str, str], Any], http: Optional[Http] = None,
                 trace_recent: Optional[Callable[[int, str], List[Dict[str, Any]]]] = None,
                 extra_terms: Optional[Callable[[], Iterable[str]]] = None, catalogue: Optional[Callable[[], str]] = None, clock: Callable[[], float] = time.time,
                 rng: Optional[random.Random] = None):
        self.config, self.push, self.http = config, push, http or default_http()
        self.trace_recent, self.extra_terms, self.clock = trace_recent, extra_terms, clock
        self.catalogue = catalogue
        self.rng = rng or random.Random()
        self.status: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _note(self, kind: str, state: str, detail: str = "", chars: int = 0) -> None:
        self.status[kind] = {"state": state, "at": round(self.clock(), 1), "detail": detail[:200], "chars": chars}

    def _put(self, kind: str, text: str) -> None:
        text = _clip(text, MAX_TERMS_CHARS if kind in (TERMS_KIND, MODELS_KIND) else MAX_CHARS)
        if not text:
            self._note(kind, "empty", "nothing to show")
            return
        self.push(kind, text)
        self._note(kind, "pushed", chars=len(text))

    def _guard(self, kind: str, fn: Callable[[], str]) -> None:
        if not kind_enabled(self.config, kind if kind != TERMS_KIND else "terms"):
            self._note(kind, "off", "switched off in garden.window")
            return
        try:
            self._put(kind, fn())
        except RuntimeError as e:
            self._note(kind, "skipped", str(e))
        except Exception as e:
            logger.warning("garden window %s: %s", kind, type(e).__name__)
            self._note(kind, "error", "%s: %s" % (type(e).__name__, str(e)[:120]))

    def run_once(self) -> Dict[str, Dict[str, Any]]:
        """Build and push every enabled kind. A kind that fails is noted and the rest still run."""
        with self._lock:
            now = self.clock()
            g = _cfg(self.config)
            states_cache: Dict[str, List[Dict[str, Any]]] = {}

            def states() -> List[Dict[str, Any]]:
                if "s" not in states_cache:
                    states_cache["s"] = fetch_ha_states(self.config, self.http)
                return states_cache["s"]

            self._guard("weather", lambda: build_weather(states(), str(g.get("weather_entity") or "")))
            self._guard("cameras", lambda: build_cameras(fetch_frigate_events(self.config, self.http, now), now,
                                                         allow_names=bool(g.get("window_names"))))
            self._guard("book", lambda: kavita_passage(g["kavita"], self.http, self.rng) if isinstance(g.get("kavita"), dict)
                        else self._skip("Kavita is not configured"))
            self._guard("headlines", lambda: build_headlines(fetch_headlines(self.config, self.http)))
            self._guard("computer", lambda: build_computer(self.trace_recent(300, "courage") if self.trace_recent else [], now)
                        if self.trace_recent else self._skip("no trace log"))

            def terms() -> str:
                try:
                    st = states()
                except Exception:
                    st = []
                extra = list(self.extra_terms()) if self.extra_terms else []
                return "\n".join(household_terms(self.config, st, extra))

            self._guard(TERMS_KIND, terms)
            self._guard(MODELS_KIND, lambda: self.catalogue() if self.catalogue else self._skip("no catalogue source"))
            return dict(self.status)

    @staticmethod
    def _skip(why: str) -> str:
        raise RuntimeError(why)

    # ---- the thread -------------------------------------------------------------------------------
    def interval(self) -> float:
        try:
            return max(MIN_INTERVAL_S, float(window_cfg(self.config).get("interval_s") or DEFAULT_INTERVAL_S))
        except (TypeError, ValueError):
            return DEFAULT_INTERVAL_S

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="garden-window", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        self._stop.wait(20.0)                                       # let the server finish starting
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as e:
                logger.warning("garden window run: %s", type(e).__name__)
            self._stop.wait(self.interval())
