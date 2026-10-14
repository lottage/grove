"""What may leave the garden in a search query or a url (the design notes: "nothing private leaves in a query or URL").

The rules are the ones in StoneSage's `lodestone/boost/egress.py` (HA entity ids, LAN addresses, household names, tokens), written out
again here because the garden never imports StoneSage. The household term list is pushed by StoneSage with the window snapshots
(snapshot kind `household-terms`, one term per line). Deterministic: no model is asked.

    problem = check_query(text, terms)  -> None when it may go, else a sentence that tells the resident why not
    problem = check_url(url, terms)
"""

import ipaddress
import re
import urllib.parse
from typing import Iterable, List, Optional

_HA_DOMAINS = ("light|switch|camera|sensor|binary_sensor|climate|lock|cover|person|device_tracker|media_player|fan|"
               "alarm_control_panel|input_boolean|input_number|automation|script|scene|zone|vacuum|notify")
ENTITY = re.compile(r"\b(?:%s)\.[a-z0-9_]{3,}\b" % _HA_DOMAINS, re.I)
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
IPV6 = re.compile(r"(?<![0-9a-f:])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![0-9a-f:])", re.I)
PRIVATE_HOST = re.compile(r"(?i)\b(?:localhost|[a-z0-9-]+\.(?:local|lan|internal|home|corp|intranet|localdomain|ts\.net))\b")
SECRET_PATTERNS = [
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("pve_token", re.compile(r"PVEAPIToken\s*=?\s*\S+", re.I)),
    ("api_key", re.compile(r"\b(?:sk|gsk|sk-or|sk-ant|csk|AIza|ghp|gho|xox[bpa])[-_][A-Za-z0-9_-]{16,}")),
    ("assignment", re.compile(r"(?i)(?:api[_-]?key|token|password|passwd|secret)\s*[:=]\s*[\"']?[^\s\"']{8,}")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_+/=.-]{16,}")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]
_OPAQUE = re.compile(r"[A-Za-z0-9_+/=-]{32,}")


def _private_ip(text: str) -> bool:
    for m in IPV4.findall(text):
        try:
            ip = ipaddress.ip_address(m)
        except ValueError:
            continue
        if not ip.is_global:
            return True
        if ip in ipaddress.ip_network("100.64.0.0/10"):
            return True
    for m in IPV6.findall(text):
        if m.count(":") < 2:
            continue
        try:
            ip = ipaddress.ip_address(m)
        except ValueError:
            continue
        if not ip.is_global:
            return True
    return False


def _token_like(text: str) -> bool:
    """A long opaque run of letters and digits: a key, a hash, a session id. Long slugs with many hyphens are not."""
    for m in _OPAQUE.findall(text):
        if (re.search(r"\d", m) and re.search(r"[A-Za-z]", m) and len(set(m)) >= 12 and m.count("-") < 4
                and not re.fullmatch(r"[0-9]+", m)):
            return True
    return False


def parse_terms(text: str) -> List[str]:
    """The household terms from the pushed snapshot: one per line, 3 or more characters, no duplicates."""
    seen, out = set(), []
    for line in (text or "").splitlines():
        t = " ".join(line.split())
        if len(t) >= 3 and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def scan(text: str, terms: Iterable[str] = ()) -> List[str]:
    """Kinds of private content in `text`: 'secret:<kind>', 'home:ha_entity', 'home:lan_ip', 'home:private_host', 'home:term:<t>'."""
    hits: List[str] = []
    for kind, rx in SECRET_PATTERNS:
        if rx.search(text):
            hits.append("secret:" + kind)
    if _token_like(text):
        hits.append("secret:opaque_token")
    if ENTITY.search(text):
        hits.append("home:ha_entity")
    if _private_ip(text):
        hits.append("home:lan_ip")
    if PRIVATE_HOST.search(text):
        hits.append("home:private_host")
    low = text.lower()
    for term in terms:
        t = " ".join(str(term).lower().split())
        if len(t) >= 3 and re.search(r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(t), low):
            hits.append("home:term:" + t)
    return hits


def _why(hits: List[str], what: str) -> str:
    parts = []
    for h in hits:
        if h.startswith("secret:"):
            parts.append("something that looks like a key, token or password")
        elif h == "home:ha_entity":
            parts.append("something shaped like a smart-home device id (domain.name)")
        elif h == "home:lan_ip":
            parts.append("a private network address")
        elif h == "home:private_host":
            parts.append("a private network host name")
        elif h.startswith("home:term:"):
            parts.append("the household name '%s'" % h[len("home:term:"):])
    parts = list(dict.fromkeys(parts))
    return ("Not sent: this %s contains %s. Nothing about the household or its network may leave the garden. "
            "Nothing was searched or read; you can ask again without it." % (what, " and ".join(parts)))


def check_query(text: str, terms: Iterable[str] = ()) -> Optional[str]:
    hits = scan(str(text or ""), terms)
    return _why(hits, "query") if hits else None


def check_url(url: str, terms: Iterable[str] = ()) -> Optional[str]:
    """The url is scanned as written and percent-decoded (twice, so %2570 tricks do not hide anything)."""
    raw = str(url or "")
    seen = [raw]
    for _ in range(2):
        seen.append(urllib.parse.unquote_plus(seen[-1]))
    hits: List[str] = []
    for variant in dict.fromkeys(seen):
        hits += scan(variant, terms)
    hits = list(dict.fromkeys(hits))
    return _why(hits, "address") if hits else None
