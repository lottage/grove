"""The reader (grove/reader.py): URL rules, DNS rules, redirects, content types, size caps, HTML to text, rate limits,
the HTTP service. No LAN: a fake resolver and a fake connection stand in for DNS and the web; one test runs the HTTP service on localhost.

Imports only the reader (grove), like tests/test_grove_worker.py.
"""

import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))

from grove import reader  # noqa: E402
from grove.reader import Fetcher, RateLimiter, Reader, ReaderError  # noqa: E402

PUBLIC = "93.184.216.34"


class FakeResp:
    def __init__(self, status=200, headers=None, body=b""):
        self.status, self._h, self._body, self._pos = status, {k.lower(): v for k, v in (headers or {}).items()}, body, 0

    def getheader(self, name, default=None):
        return self._h.get(name.lower(), default)

    def read(self, n=-1):
        n = len(self._body) - self._pos if n is None or n < 0 else n
        chunk = self._body[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk


class Web:
    """A fake internet: hosts -> {path: (status, headers, body)}. Records every connection and request."""

    def __init__(self, sites, dns=None):
        self.sites, self.connections, self.requests, self.dns_calls = sites, [], [], []
        self.dns = dns if dns is not None else {h: [PUBLIC] for h in sites}

    def resolver(self, host, port):
        self.dns_calls.append(host)
        answer = self.dns.get(host)
        if answer is None:
            raise OSError("no such host")
        return list(answer)

    def connector(self, scheme, host, ip, port, timeout):
        web = self
        web.connections.append({"scheme": scheme, "host": host, "ip": ip, "port": port})

        class Conn:
            def request(self, method, path, headers=None):
                self.path = path
                web.requests.append({"method": method, "host": host, "path": path, "headers": dict(headers or {})})

            def getresponse(self):
                status, headers, body = web.sites[host].get(self.path, (404, {}, b"not here"))
                return FakeResp(status, headers, body)

            def close(self):
                pass

        return Conn()

    def fetcher(self, **kw):
        return Fetcher(resolver=self.resolver, connector=self.connector, **kw)


def html(body, title="T"):
    return (200, {"Content-Type": "text/html; charset=utf-8"}, ("<html><head><title>%s</title></head><body>%s</body></html>" % (title, body)).encode())


class UrlRuleTests(unittest.TestCase):
    def test_good_urls(self):
        self.assertEqual(reader.validate_url("https://example.com/a?b=1"), ("https", "example.com", 443, "/a?b=1"))
        self.assertEqual(reader.validate_url("http://Example.COM")[1:], ("example.com", 80, "/"))
        self.assertEqual(reader.validate_url("https://en.wikipedia.org:443/wiki/X")[2], 443)

    def test_schemes_other_than_http_are_refused(self):
        for u in ("ftp://example.com/x", "file:///etc/passwd", "javascript:alert(1)", "gopher://example.com", "data:text/html,hi", "//example.com/x",
                  "example.com/x", ""):
            with self.assertRaises(ReaderError, msg=u):
                reader.validate_url(u)

    def test_addresses_in_any_spelling_are_refused(self):
        for u in ("http://127.0.0.1/", "http://10.0.0.1/", "http://10.0.0.82:8123/", "http://[::1]/", "http://2130706433/", "http://0x7f.0.0.1/",
                  "http://0177.0.0.1/", "http://127.1/", "http://169.254.169.254/latest/meta-data/", "http://8.8.8.8/", "https://[2606:4700::1111]/",
                  "http://3232235777/"):
            with self.assertRaises(ReaderError, msg=u):
                reader.validate_url(u)

    def test_private_names_and_odd_urls_are_refused(self):
        for u in ("http://localhost/", "http://homeassistant.local:8123/", "http://printer.lan/", "http://x.internal/", "http://box.ts.net/",
                  "http://nodots/", "http://user:pass@example.com/", "http://user@example.com/", "https://example.com:8443/", "http://example.com:8080/",
                  "https://exa mple.com/", "https://example.com/a b", "https://example.com/\x00", "https://" + "a" * 300 + ".com/",
                  "https://example.com/" + "x" * 2100):
            with self.assertRaises(ReaderError, msg=u[:60]):
                reader.validate_url(u)

    def test_blocked_addresses(self):
        for a in ("10.1.1.1", "172.16.0.1", "172.31.255.255", "10.0.0.1", "127.0.0.1", "169.254.1.1", "100.64.0.1", "100.127.255.254",
                  "224.0.0.1", "240.0.0.1", "0.0.0.0", "255.255.255.255", "::1", "fe80::1", "fc00::1", "fd12:3456::1", "::ffff:10.0.0.1",
                  "::ffff:127.0.0.1", "2002:c0a8:0101::1", "64:ff9b::c0a8:101", "ff02::1", "not an address", "192.0.2.1"):
            self.assertTrue(reader.blocked_address(a), a)
        for a in ("8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111", "100.63.255.255", "100.128.0.1"):
            self.assertFalse(reader.blocked_address(a), a)


class FetchTests(unittest.TestCase):
    def test_a_page_is_fetched_and_connects_to_the_checked_address(self):
        web = Web({"example.com": {"/": html("<h1>Hello</h1><p>World</p>", "Greeting")}})
        out = web.fetcher().read("https://example.com/")
        self.assertTrue(out["ok"])
        self.assertEqual(out["title"], "Greeting")
        self.assertIn("# Hello", out["text"])
        self.assertIn("World", out["text"])
        self.assertEqual(out["final_url"], "https://example.com/")
        self.assertFalse(out["truncated"])
        self.assertEqual(web.connections, [{"scheme": "https", "host": "example.com", "ip": PUBLIC, "port": 443}])

    def test_dns_to_a_private_address_is_refused_before_any_connection(self):
        for answer in (["10.0.0.82"], ["127.0.0.1"], ["10.0.0.5"], ["169.254.169.254"], ["100.64.1.1"], ["::1"], ["fe80::1"],
                       [PUBLIC, "10.0.0.5"], ["10.0.0.5", PUBLIC]):
            web = Web({"evil.example.com": {"/": html("secret")}}, dns={"evil.example.com": answer})
            with self.assertRaises(ReaderError, msg=str(answer)) as cm:
                web.fetcher().read("http://evil.example.com/")
            self.assertIn("private or reserved", str(cm.exception))
            self.assertEqual(web.connections, [], "connected despite " + str(answer))

    def test_rebinding_the_connection_uses_the_one_answer_that_was_checked(self):
        """A resolver that answers public the first time and private after: the fetcher asks once per hop and connects to what it checked."""
        calls = []

        def resolver(host, port):
            calls.append(host)
            return [PUBLIC] if len(calls) == 1 else ["10.0.0.82"]

        web = Web({"rebind.example.com": {"/": html("fine")}})
        out = Fetcher(resolver=resolver, connector=web.connector).read("http://rebind.example.com/")
        self.assertTrue(out["ok"])
        self.assertEqual(calls, ["rebind.example.com"])
        self.assertEqual([c["ip"] for c in web.connections], [PUBLIC])

    def test_unresolvable_host(self):
        web = Web({}, dns={})
        with self.assertRaises(ReaderError) as cm:
            web.fetcher().read("https://nothing.example.com/")
        self.assertEqual(cm.exception.status, 502)

    def test_redirects_are_followed_and_every_hop_is_checked(self):
        web = Web({"a.example.com": {"/": (302, {"Location": "https://b.example.com/x"}, b"")},
                   "b.example.com": {"/x": (301, {"Location": "/y"}, b""), "/y": html("<p>arrived</p>")}})
        out = web.fetcher().read("http://a.example.com/")
        self.assertEqual(out["final_url"], "https://b.example.com/y")
        self.assertIn("arrived", out["text"])
        self.assertEqual([c["host"] for c in web.connections], ["a.example.com", "b.example.com", "b.example.com"])

    def test_a_redirect_to_a_private_target_is_refused(self):
        for target in ("http://10.0.0.82:8123/", "http://127.0.0.1/", "http://169.254.169.254/", "file:///etc/passwd", "http://localhost/admin",
                       "ftp://example.org/", "http://[::1]/", "http://0x7f.1/", "https://router.local/"):
            web = Web({"a.example.com": {"/": (302, {"Location": target}, b"")}})
            with self.assertRaises(ReaderError, msg=target):
                web.fetcher().read("http://a.example.com/")
            self.assertEqual([c["host"] for c in web.connections], ["a.example.com"], target)

    def test_a_redirect_to_a_name_that_resolves_private_is_refused(self):
        web = Web({"a.example.com": {"/": (302, {"Location": "http://sneaky.example.net/"}, b"")}, "sneaky.example.net": {"/": html("x")}},
                  dns={"a.example.com": [PUBLIC], "sneaky.example.net": ["10.0.0.1"]})
        with self.assertRaises(ReaderError):
            web.fetcher().read("http://a.example.com/")
        self.assertEqual([c["host"] for c in web.connections], ["a.example.com"])

    def test_too_many_redirects(self):
        web = Web({"loop.example.com": {"/": (302, {"Location": "/"}, b"")}})
        with self.assertRaises(ReaderError) as cm:
            web.fetcher().read("http://loop.example.com/")
        self.assertIn("redirects", str(cm.exception))
        self.assertEqual(len(web.connections), reader.MAX_REDIRECTS + 1)

    def test_redirect_without_location(self):
        web = Web({"a.example.com": {"/": (302, {}, b"")}})
        with self.assertRaises(ReaderError):
            web.fetcher().read("http://a.example.com/")

    def test_content_types(self):
        for ctype in ("image/png", "application/pdf", "application/octet-stream", "video/mp4", "application/zip", "", "text/css",
                      "application/javascript", "image/svg+xml"):
            web = Web({"f.example.com": {"/": (200, {"Content-Type": ctype} if ctype else {}, b"data")}})
            with self.assertRaises(ReaderError, msg=ctype) as cm:
                web.fetcher().read("http://f.example.com/")
            self.assertEqual(cm.exception.status, 415)
        web = Web({"f.example.com": {"/p": (200, {"Content-Type": "text/plain"}, b"plain words\r\nline two"),
                                     "/j": (200, {"Content-Type": "application/json; charset=utf-8"}, b'{"a": [1, 2], "b": "x"}')}})
        self.assertEqual(web.fetcher().read("http://f.example.com/p")["text"], "plain words\nline two")
        self.assertIn('"b": "x"', web.fetcher().read("http://f.example.com/j")["text"])

    def test_errors_from_the_site(self):
        for status in (403, 404, 500, 204 + 100):
            web = Web({"f.example.com": {"/": (status, {"Content-Type": "text/html"}, b"no")}})
            with self.assertRaises(ReaderError, msg=str(status)):
                web.fetcher().read("http://f.example.com/")

    def test_size_caps(self):
        big = b"word " * (3 * 1024 * 1024 // 5)
        web = Web({"big.example.com": {"/": (200, {"Content-Type": "text/plain"}, big)}})
        out = web.fetcher().read("http://big.example.com/")
        self.assertEqual(out["bytes"], reader.MAX_RAW_BYTES)
        self.assertTrue(out["truncated"])
        self.assertLessEqual(len(out["text"].encode("utf-8")), reader.MAX_TEXT_BYTES)
        small = Web({"s.example.com": {"/": (200, {"Content-Type": "text/plain"}, b"x" * 100000)}}).fetcher().read("http://s.example.com/")
        self.assertEqual(small["bytes"], 100000)
        self.assertTrue(small["truncated"])                     # 100 KB of text is cut to 40 KB
        self.assertEqual(len(small["text"]), reader.MAX_TEXT_BYTES)
        ok = Web({"s.example.com": {"/": (200, {"Content-Type": "text/plain"}, b"short")}}).fetcher().read("http://s.example.com/")
        self.assertFalse(ok["truncated"])

    def test_a_slow_site_is_cut_off(self):
        ticks = iter([0.0, 0.0, 1.0, 100.0, 100.0, 100.0, 100.0])
        web = Web({"slow.example.com": {"/": (200, {"Content-Type": "text/plain"}, b"x" * 200000)}})
        with self.assertRaises(ReaderError) as cm:
            web.fetcher(clock=lambda: next(ticks)).read("http://slow.example.com/")
        self.assertEqual(cm.exception.status, 504)

    def test_nothing_identifying_is_sent(self):
        web = Web({"example.com": {"/": html("hi")}})
        web.fetcher().read("https://example.com/")
        req = web.requests[0]
        self.assertEqual(req["method"], "GET")
        names = {k.lower() for k in req["headers"]}
        self.assertFalse(names & {"cookie", "authorization", "referer", "proxy-authorization", "x-forwarded-for", "origin"})
        self.assertEqual(req["headers"]["User-Agent"], reader.USER_AGENT)
        self.assertEqual(req["headers"]["Accept-Encoding"], "identity")

    def test_charsets(self):
        web = Web({"c.example.com": {"/": (200, {"Content-Type": "text/plain; charset=iso-8859-1"}, "café".encode("latin-1")),
                                     "/x": (200, {"Content-Type": "text/plain; charset=nonsense-9"}, "café".encode("utf-8"))}})
        self.assertEqual(web.fetcher().read("http://c.example.com/")["text"], "café")
        self.assertEqual(web.fetcher().read("http://c.example.com/x")["text"], "café")


class HtmlToTextTests(unittest.TestCase):
    def test_scripts_styles_and_navigation_are_dropped(self):
        title, text = reader.html_to_text(
            "<html><head><title>Page &amp; Title</title><style>p{color:red}</style><script>var x='<p>nope</p>';</script></head><body>"
            "<nav><a href='/'>Home</a><ul><li>Menu</li></ul></nav><script>alert(1)</script><noscript>enable js</noscript>"
            "<h1>Main</h1><p>First   paragraph</p><p>Second<br>line</p><h2>Sub</h2><ul><li>one</li><li>two</li></ul>"
            "<svg><text>glyph</text></svg><iframe>frame</iframe><footer>the end</footer></body></html>")
        self.assertEqual(title, "Page & Title")
        for gone in ("alert", "Menu", "Home", "nope", "color", "enable js", "glyph", "frame"):
            self.assertNotIn(gone, text)
        self.assertIn("# Main", text)
        self.assertIn("## Sub", text)
        self.assertIn("First paragraph", text)
        self.assertIn("Second\nline", text)
        self.assertIn("- one\n- two", text)
        self.assertIn("the end", text)
        self.assertIn("\n\n", text)                               # paragraph breaks survive
        self.assertNotIn("\n\n\n", text)

    def test_broken_html_still_gives_text(self):
        title, text = reader.html_to_text("<p>unclosed <b>bold <script>stuff")
        self.assertIn("unclosed", text)
        self.assertEqual(title, "")

    def test_entities_and_unicode(self):
        _t, text = reader.html_to_text("<p>caf&eacute; &lt;tag&gt; &#8212; über</p>")
        self.assertEqual(text, "café <tag> — über")

    def test_cut_text_never_splits_a_character(self):
        cut, was = reader.cut_text("é" * 100, 51)
        self.assertTrue(was)
        cut.encode("utf-8")
        self.assertLessEqual(len(cut.encode("utf-8")), 51)


class RateLimitTests(unittest.TestCase):
    def test_per_minute_and_per_day(self):
        now = [1000.0]
        lim = RateLimiter(3, 5, clock=lambda: now[0])
        self.assertEqual([lim.take() for _ in range(3)], [None, None, None])
        wait = lim.take()
        self.assertTrue(0 < wait <= 61)
        now[0] += 61
        self.assertIsNone(lim.take())
        self.assertIsNone(lim.take())
        now[0] += 61
        wait = lim.take()                                          # the day's 5 are used
        self.assertGreater(wait, 3600)
        now[0] += 86400
        self.assertIsNone(lim.take())


class ServiceTests(unittest.TestCase):
    KEY = "fetcher-key-for-tests-0123456789"

    def setUp(self):
        self.web = Web({"example.com": {"/": html("<p>served</p>", "Served")}})
        searx = {"results": [{"title": "A", "url": "https://a.example.com/", "content": "snippet a " * 80},
                             {"title": "Bad", "url": "javascript:alert(1)", "content": "x"},
                             {"title": "File", "url": "file:///etc/passwd", "content": "x"}] +
                [{"title": "T%d" % i, "url": "https://t%d.example.com/" % i, "content": "c"} for i in range(20)]}
        self.searched = []

        def fake_searx(q):
            self.searched.append(q)
            return searx

        self.reader = Reader(self.web.fetcher(), fake_searx, RateLimiter(50, 500), RateLimiter(50, 500))
        self.httpd = reader.make_server(self.reader, self.KEY)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = "http://127.0.0.1:%d" % self.httpd.server_address[1]

    def get(self, path, key=KEY, method="GET"):
        req = urllib.request.Request(self.base + path, method=method, headers={"Authorization": "Bearer " + key} if key else {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_the_key_is_required_everywhere_and_compared_in_constant_time(self):
        for path in ("/health", "/read?url=https://example.com/", "/search?q=x"):
            self.assertEqual(self.get(path, key="")[0], 401, path)
            self.assertEqual(self.get(path, key="wrong")[0], 401, path)
            self.assertEqual(self.get(path, key=self.KEY[:-1])[0], 401, path)
        self.assertEqual(self.web.connections, [])
        self.assertEqual(self.searched, [])
        import inspect
        self.assertIn("hmac.compare_digest", inspect.getsource(reader.make_handler))

    def test_only_get(self):
        for m in ("POST", "PUT", "DELETE", "PATCH"):
            req = urllib.request.Request(self.base + "/read?url=https://example.com/", method=m, data=b"x" if m != "DELETE" else None,
                                         headers={"Authorization": "Bearer " + self.KEY})
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(req, timeout=10)
            self.assertEqual(cm.exception.code, 405, m)

    def test_read_and_errors(self):
        status, body = self.get("/read?url=" + urllib.request.quote("https://example.com/", safe=""))
        self.assertEqual(status, 200)
        self.assertEqual((body["title"], body["text"]), ("Served", "served"))
        status, body = self.get("/read?url=" + urllib.request.quote("http://10.0.0.82:8123/", safe=""))
        self.assertEqual((status, body["ok"]), (400, False))
        self.assertEqual(self.get("/read")[0], 400)
        self.assertEqual(self.get("/nothing")[0], 404)

    def test_read_rate_limit(self):
        self.reader.reads, self.reader.searches = RateLimiter(2, 10), RateLimiter(2, 10)
        url = "/read?url=" + urllib.request.quote("https://example.com/", safe="")
        self.assertEqual([self.get(url)[0] for _ in range(3)], [200, 200, 429])
        self.assertEqual(self.get("/search?q=x")[0], 200)           # searches have their own counter

    def test_search_returns_top_results_with_clean_urls(self):
        status, body = self.get("/search?q=" + urllib.request.quote("sourdough starter") + "&n=3")
        self.assertEqual(status, 200)
        self.assertEqual(self.searched, ["sourdough starter"])
        self.assertEqual([r["url"] for r in body["results"]], ["https://a.example.com/", "https://t0.example.com/", "https://t1.example.com/"])
        self.assertLessEqual(len(body["results"][0]["snippet"]), reader.SNIPPET_CHARS)
        status, body = self.get("/search?q=x&n=500")
        self.assertEqual(len(body["results"]), reader.MAX_RESULTS)
        self.assertEqual(self.get("/search?q=")[0], 400)
        self.assertEqual(self.get("/search")[0], 400)

    def test_no_key_means_no_server(self):
        with self.assertRaises(RuntimeError):
            reader.make_server(self.reader, "")


if __name__ == "__main__":
    unittest.main()
