"""The fetcher's reader: a small read-only web service for the garden (standard library only, Python 3.10+).

    python -m grove.reader        # grove/ must be importable (pip install, or the folder that holds grove/ as the working directory)

It runs on its own container or machine (the fetcher), whose firewall lets it reach the public internet and nothing on the LAN. This file is a
second wall behind that firewall: it refuses anything that is not an ordinary public web page, and it connects to the very address
it checked.

Routes (every one needs `Authorization: Bearer <FETCHER_KEY>`, compared in constant time; GET only):
    GET /read?url=       -> {ok, url, final_url, title, text, bytes, truncated}
    GET /search?q=&n=    -> {ok, query, results: [{title, url, snippet}]}      (SearXNG on 127.0.0.1:8080)
    GET /health

What /read will and will not do:
  - http and https only, GET only, ports 80 and 443 only, no user:password@ in the URL. No cookies, no auth headers, no referrer are
    ever sent; the User-Agent says what this is.
  - Redirects are followed by hand (at most 5) and every hop is validated again from scratch.
  - A host that is an IP address (any spelling) is refused. A name is resolved ONCE; if any answer is private, loopback, link-local,
    CGNAT (100.64/10), multicast, reserved, or a tunnel/NAT64 range, the request is refused. The connection then goes to that resolved
    address (Host header and TLS server name stay the real name), so the check and the connection cannot disagree (DNS rebinding).
  - Only text/html, text/plain and application/json come back; at most 2 MB is read and 40 KB of text is returned (`truncated`).
  - Rate limits per minute and per day, separately for reads and searches.

Environment:
    FETCHER_KEY / FETCHER_KEY_FILE   required. The worker presents it.
    READER_PORT (8090), READER_BIND (0.0.0.0), READER_SEARX_URL (http://127.0.0.1:8080)
    READER_READS_PER_MIN (10), READER_READS_PER_DAY (300), READER_SEARCHES_PER_MIN (10), READER_SEARCHES_PER_DAY (300)
"""

import codecs
import hmac
import html
import http.client
import ipaddress
import json
import logging
import os
import re
import socket
import ssl
import sys
import threading
import time
import urllib.parse
from collections import deque
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("garden.reader")

USER_AGENT = "StoneSageGardenReader/1 (read-only text fetcher; no cookies)"
ALLOWED_TYPES = ("text/html", "text/plain", "application/json")
ALLOWED_PORTS = (80, 443)
MAX_REDIRECTS = 5
MAX_RAW_BYTES = 2 * 1024 * 1024
MAX_TEXT_BYTES = 40 * 1024
CONNECT_TIMEOUT = 10.0
TOTAL_TIMEOUT = 25.0
MAX_URL_CHARS = 2000
MAX_RESULTS = 10
SNIPPET_CHARS = 300

_NETS = ["100.64.0.0/10", "192.0.0.0/24", "192.0.2.0/24", "198.18.0.0/15", "198.51.100.0/24", "203.0.113.0/24", "240.0.0.0/4",
         "64:ff9b::/96", "64:ff9b:1::/48", "2001::/32", "2001:db8::/32", "2002::/16", "fc00::/7", "fe80::/10", "ff00::/8", "::/128",
         "::1/128"]
BLOCKED_NETS = [ipaddress.ip_network(n) for n in _NETS]
_IPISH = re.compile(r"^(0x[0-9a-f]+|[0-9]+)(\.(0x[0-9a-f]+|[0-9]+))*$", re.I)   # 127.1, 0x7f.1, 2130706433, 0177.0.0.1: IPv4 spellings
_HOSTNAME = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
_BAD_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home", ".corp", ".intranet", ".ts.net", ".test", ".invalid", ".example")


class ReaderError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# ---- the address rules -------------------------------------------------------------------------------------
def blocked_address(addr: str) -> bool:
    """True when `addr` is anything but an ordinary public internet address."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified
            or not ip.is_global):
        return True
    return any(ip in net for net in BLOCKED_NETS if net.version == ip.version)


def validate_url(url: str) -> Tuple[str, str, int, str]:
    """(scheme, host, port, path_and_query) of a URL that may be fetched, else ReaderError. Does no DNS."""
    if not isinstance(url, str) or not url.strip() or len(url) > MAX_URL_CHARS:
        raise ReaderError("a url is required (at most %d characters)" % MAX_URL_CHARS)
    if re.search(r"[\x00-\x20\x7f]", url.strip()):
        raise ReaderError("the url contains spaces or control characters")
    try:
        p = urllib.parse.urlsplit(url.strip())
        port = p.port
    except ValueError:
        raise ReaderError("the url is not valid") from None
    scheme = p.scheme.lower()
    if scheme not in ("http", "https"):
        raise ReaderError("only http and https urls can be read")
    if p.username is not None or p.password is not None or "@" in p.netloc:
        raise ReaderError("urls with a user name or password are refused")
    host = (p.hostname or "").lower().rstrip(".")
    if not host:
        raise ReaderError("the url has no host")
    if _IPISH.match(host) or host.startswith("[") or ":" in host:
        raise ReaderError("addresses are refused: use a host name")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        raise ReaderError("the host name is not valid") from None
    if not _HOSTNAME.match(host) or host == "localhost" or host.endswith(_BAD_SUFFIXES):
        raise ReaderError("that host name is not a public web site")
    port = port or (443 if scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise ReaderError("only ports 80 and 443 are allowed")
    path = p.path or "/"
    if p.query:
        path += "?" + p.query
    return scheme, host, port, path


def _system_resolver(host: str, port: int) -> List[str]:
    seen: List[str] = []
    for _fam, _t, _p, _c, sa in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM):
        a = sa[0]
        if a not in seen:
            seen.append(a)
    return seen


def resolve_public(host: str, port: int, resolver: Optional[Callable[[str, int], List[str]]] = None) -> str:
    """Resolve once; every answer must be public. Returns the address to connect to."""
    try:
        addrs = (resolver or _system_resolver)(host, port)
    except (OSError, UnicodeError):
        raise ReaderError("the host name did not resolve", 502) from None
    if not addrs:
        raise ReaderError("the host name did not resolve", 502)
    for a in addrs:
        if blocked_address(a):
            raise ReaderError("that host resolves to a private or reserved address")
    return addrs[0]


# ---- the connection: to the address that was checked ----------------------------------------------------------
class _PinnedHTTP(http.client.HTTPConnection):
    pinned_ip = ""

    def connect(self):
        self.sock = socket.create_connection((self.pinned_ip, self.port), self.timeout)


class _PinnedHTTPS(http.client.HTTPSConnection):
    pinned_ip = ""

    def connect(self):
        sock = socket.create_connection((self.pinned_ip, self.port), self.timeout)
        ctx = ssl.create_default_context()          # certificate and host name verified against the real name, not the address
        self.sock = ctx.wrap_socket(sock, server_hostname=self.host)


def default_connector(scheme: str, host: str, ip: str, port: int, timeout: float):
    cls = _PinnedHTTPS if scheme == "https" else _PinnedHTTP
    conn = cls(host, port, timeout=timeout)         # .host stays the name: it becomes the Host header and the TLS server name
    conn.pinned_ip = ip
    return conn


# ---- HTML to text -------------------------------------------------------------------------------------------
_SKIP = {"script", "style", "noscript", "nav", "svg", "template", "iframe", "head", "object", "embed", "canvas"}
_BLOCK = {"p", "div", "section", "article", "main", "header", "footer", "aside", "ul", "ol", "li", "table", "tr", "blockquote",
          "pre", "figure", "figcaption", "form", "dl", "dt", "dd", "address", "details", "summary"}
_HEADING = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ", "h5": "##### ", "h6": "###### "}


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.parts: List[str] = []
        self._skip = 0
        self._in_title = False
        self._title_parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
            return
        if tag in _SKIP:
            self._skip += 1
            return
        if self._skip:
            return
        if tag in _HEADING:
            self.parts.append("\n\n" + _HEADING[tag])
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in ("br", "hr"):
            self.parts.append("\n")
        elif tag in _BLOCK:
            self.parts.append("\n\n")

    def handle_startendtag(self, tag, attrs):
        if tag in ("br", "hr") and not self._skip:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
            if not self.title:
                self.title = " ".join("".join(self._title_parts).split())
            return
        if tag in _SKIP:
            if self._skip:
                self._skip -= 1
            return
        if self._skip:
            return
        if tag == "li":
            return                                      # the next item (or the end of the list) supplies the break
        if tag in _HEADING or tag in _BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._in_title:
            self._title_parts.append(data)
        elif not self._skip:
            self.parts.append(data)


def html_to_text(source: str) -> Tuple[str, str]:
    """(title, plain text) of an HTML page: scripts, styles and navigation dropped; headings and paragraph breaks kept."""
    p = _TextParser()
    try:
        p.feed(source)
        p.close()
    except Exception:
        pass                                          # a page that breaks the parser still yields what was read so far
    raw = "".join(p.parts)
    out: List[str] = []
    blank = 0
    for line in raw.replace("\r", "").split("\n"):
        line = " ".join(line.split())
        if line:
            out.append(line)
            blank = 0
        else:
            blank += 1
            if blank == 1 and out:
                out.append("")
    return html.unescape(p.title)[:300], "\n".join(out).strip()


def _charset(content_type: str, head: bytes) -> str:
    m = re.search(r"charset\s*=\s*[\"']?([A-Za-z0-9_.:-]+)", content_type or "", re.I)
    name = m.group(1) if m else ""
    if not name:
        m2 = re.search(rb"<meta[^>]+charset\s*=\s*[\"']?([A-Za-z0-9_.:-]+)", head[:4096], re.I)
        name = m2.group(1).decode("ascii", "ignore") if m2 else ""
    try:
        return codecs.lookup(name or "utf-8").name
    except LookupError:
        return "utf-8"


def cut_text(text: str, limit: int = MAX_TEXT_BYTES) -> Tuple[str, bool]:
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text, False
    return data[:limit].decode("utf-8", "ignore").rstrip(), True


# ---- fetching -----------------------------------------------------------------------------------------------
class Fetcher:
    def __init__(self, resolver: Optional[Callable[[str, int], List[str]]] = None,
                 connector: Optional[Callable[..., Any]] = None, clock: Callable[[], float] = time.monotonic):
        self.resolver, self.connector, self.clock = resolver, connector or default_connector, clock

    def read(self, url: str) -> Dict[str, Any]:
        deadline = self.clock() + TOTAL_TIMEOUT
        current = url
        for _hop in range(MAX_REDIRECTS + 1):
            scheme, host, port, path = validate_url(current)                       # every hop is checked from scratch
            ip = resolve_public(host, port, self.resolver)
            conn = self.connector(scheme, host, ip, port, CONNECT_TIMEOUT)
            redirect = ""
            try:
                conn.request("GET", path, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "text/html,text/plain;q=0.9,application/json;q=0.8",
                                                   "Accept-Encoding": "identity", "Connection": "close"})
                resp = conn.getresponse()
                status = resp.status
                if status in (301, 302, 303, 307, 308):
                    redirect = resp.getheader("Location") or ""
                    if not redirect:
                        raise ReaderError("a redirect without a location", 502)
                elif status < 200 or status >= 300:
                    raise ReaderError("the site answered HTTP %d" % status, 502)
                else:
                    ctype = (resp.getheader("Content-Type") or "").split(";")[0].strip().lower()
                    if ctype not in ALLOWED_TYPES:
                        raise ReaderError("only text/html, text/plain and application/json can be read (this is %s)"
                                          % (ctype or "unknown"), 415)
                    raw = b""
                    while len(raw) <= MAX_RAW_BYTES:
                        if self.clock() > deadline:
                            raise ReaderError("the site was too slow", 504)
                        chunk = resp.read(min(65536, MAX_RAW_BYTES + 1 - len(raw)))
                        if not chunk:
                            break
                        raw += chunk
                    truncated = len(raw) > MAX_RAW_BYTES
                    if truncated:
                        raw = raw[:MAX_RAW_BYTES]
                    head_type = resp.getheader("Content-Type") or ""
            except ReaderError:
                raise
            except (OSError, http.client.HTTPException, ssl.SSLError):
                raise ReaderError("the site could not be reached", 502) from None
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            if redirect:
                current = urllib.parse.urljoin(current, redirect)
                continue
            body = raw.decode(_charset(head_type, raw), "replace")
            if ctype == "text/html":
                title, text = html_to_text(body)
            elif ctype == "application/json":
                title = ""
                try:
                    text = json.dumps(json.loads(body), ensure_ascii=False, indent=1)
                except ValueError:
                    text = body
            else:
                title, text = "", body.replace("\r", "").strip()
            text, cut = cut_text(text)
            return {"ok": True, "url": url, "final_url": current, "title": title, "text": text, "bytes": len(raw),
                    "truncated": bool(truncated or cut)}
        raise ReaderError("too many redirects", 508)


# ---- rate limits ---------------------------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, per_minute: int, per_day: int, clock: Callable[[], float] = time.time):
        self.per_minute, self.per_day, self.clock = per_minute, per_day, clock
        self._hits: Deque[float] = deque()
        self._lock = threading.Lock()

    def take(self) -> Optional[int]:
        """None when allowed (and counted), else the seconds to wait."""
        now = self.clock()
        with self._lock:
            while self._hits and self._hits[0] <= now - 86400:
                self._hits.popleft()
            if len(self._hits) >= self.per_day:
                return int(self._hits[0] + 86400 - now) + 1
            recent = [t for t in self._hits if t > now - 60]
            if len(recent) >= self.per_minute:
                return int(recent[0] + 60 - now) + 1
            self._hits.append(now)
            return None


# ---- search --------------------------------------------------------------------------------------------------
def searx_fetch(base: str, query: str) -> Dict[str, Any]:
    u = urllib.parse.urlsplit(base)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=15)
    try:
        conn.request("GET", "/search?" + urllib.parse.urlencode({"q": query, "format": "json"}),
                     headers={"Accept": "application/json", "User-Agent": USER_AGENT, "X-Forwarded-For": "127.0.0.1",
                              "X-Real-IP": "127.0.0.1"})
        resp = conn.getresponse()
        data = resp.read(2 * 1024 * 1024)
        if resp.status != 200:
            raise ReaderError("the search engine answered HTTP %d" % resp.status, 502)
        return json.loads(data.decode("utf-8", "replace"))
    except ReaderError:
        raise
    except (OSError, http.client.HTTPException, ValueError):
        raise ReaderError("the search engine did not answer", 502) from None
    finally:
        conn.close()


def search(query: str, n: int, fetch: Callable[[str], Dict[str, Any]]) -> Dict[str, Any]:
    query = " ".join(str(query or "").split())
    if not query or len(query) > 300:
        raise ReaderError("a query is required (at most 300 characters)")
    n = max(1, min(int(n), MAX_RESULTS))
    out = []
    for r in (fetch(query).get("results") or []):
        if not isinstance(r, dict):
            continue
        url = str(r.get("url") or "")
        if not url.lower().startswith(("http://", "https://")):
            continue
        out.append({"title": " ".join(str(r.get("title") or "").split())[:200], "url": url[:500],
                    "snippet": " ".join(str(r.get("content") or "").split())[:SNIPPET_CHARS]})
        if len(out) >= n:
            break
    return {"ok": True, "query": query, "results": out}


# ---- the service ----------------------------------------------------------------------------------------------
class Reader:
    def __init__(self, fetcher: Optional[Fetcher] = None, searx: Optional[Callable[[str], Dict[str, Any]]] = None,
                 reads: Optional[RateLimiter] = None, searches: Optional[RateLimiter] = None):
        self.fetcher = fetcher or Fetcher()
        self.searx = searx or (lambda q: searx_fetch("http://127.0.0.1:8080", q))
        self.reads = reads or RateLimiter(10, 300)
        self.searches = searches or RateLimiter(10, 300)

    def handle(self, path: str, query: Dict[str, List[str]]) -> Tuple[int, Dict[str, Any]]:
        try:
            if path == "/health":
                return 200, {"ok": True}
            if path == "/read":
                wait = self.reads.take()
                if wait:
                    return 429, {"ok": False, "error": "read limit reached; try again in %d seconds" % wait}
                return 200, self.fetcher.read((query.get("url") or [""])[0])
            if path == "/search":
                wait = self.searches.take()
                if wait:
                    return 429, {"ok": False, "error": "search limit reached; try again in %d seconds" % wait}
                try:
                    n = int((query.get("n") or ["8"])[0])
                except ValueError:
                    n = 8
                return 200, search((query.get("q") or [""])[0], n, self.searx)
            return 404, {"ok": False, "error": "not found"}
        except ReaderError as e:
            return e.status, {"ok": False, "error": str(e)}


def make_handler(reader: Reader, key: str):
    key_bytes = key.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        server_version = "garden-reader/1"

        def log_message(self, fmt, *args):             # the request line carries the url asked for; never the key
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
            return bool(key_bytes) and scheme.lower() == "bearer" and hmac.compare_digest(token.strip().encode("utf-8", "replace"), key_bytes)

        def do_GET(self):
            if not self._authorised():
                self._send(401, {"ok": False, "error": "unauthorized"})
                return
            parsed = urllib.parse.urlsplit(self.path)
            status, data = reader.handle(parsed.path.rstrip("/") or "/", urllib.parse.parse_qs(parsed.query))
            self._send(status, data)

        def _refuse(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if 0 < n <= 65536:
                    self.rfile.read(n)                  # read the body so the answer is not cut off by a reset
            except (ValueError, OSError):
                pass
            self.close_connection = True
            self._send(405, {"ok": False, "error": "GET only"})

        do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _refuse

    return Handler


def make_server(reader: Reader, key: str, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    if not key:
        raise RuntimeError("no key: set FETCHER_KEY or FETCHER_KEY_FILE")
    return ThreadingHTTPServer((host, port), make_handler(reader, key))


def _secret(env: str) -> str:
    value = (os.environ.get(env) or "").strip()
    if value:
        return value
    path = (os.environ.get(env + "_FILE") or "").strip()
    if path:
        try:
            with open(path, encoding="utf-8-sig") as f:
                return f.read().strip()
        except OSError:
            return ""
    return ""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    key = _secret("FETCHER_KEY")
    if not key:
        logger.error("no key: set FETCHER_KEY or FETCHER_KEY_FILE")
        return 2
    env = os.environ.get
    reader = Reader(searx=lambda q: searx_fetch(env("READER_SEARX_URL", "http://127.0.0.1:8080"), q),
                    reads=RateLimiter(int(env("READER_READS_PER_MIN", "10")), int(env("READER_READS_PER_DAY", "300"))),
                    searches=RateLimiter(int(env("READER_SEARCHES_PER_MIN", "10")), int(env("READER_SEARCHES_PER_DAY", "300"))))
    httpd = make_server(reader, key, env("READER_BIND", "0.0.0.0"), int(env("READER_PORT", "8090")))
    logger.info("garden reader listening on %s:%d", *httpd.server_address[:2])
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
