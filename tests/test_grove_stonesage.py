"""StoneSage's side of garden G2 (stonesage/garden_client.py, garden_window.py, auth.py): the embeddings route and its key scope, the
board / comments / residents routes, the window snapshot builders and their feeder. No LAN: fake engines, a fake worker, fake sources.

Does not import `garden` (the worker): it talks to a fake worker over localhost, so the public split can carry this file.
"""

import io
import json
import os
import random
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from route_harness import call_full, server_key  # noqa: E402
import grove_fixture  # noqa: E402
from stonesage import apps, auth, server  # noqa: E402
from grove.stonesage_app import client as garden_client, window as garden_window  # noqa: E402

grove_fixture.install()

GARDEN_KEY = "garden-model-key-for-tests-0123456789"
CONTROL_KEY = "garden-control-key-for-tests-0123456789"
EMBED_ROUTE = "/api/garden/v1/embeddings"
CHAT_ROUTE = "/api/garden/v1/chat/completions"
NOW = 1_700_000_000.0


class Serve:
    """A localhost server whose handler is fn(method, path, body) -> (status, json)."""

    def __init__(self, fn):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _do(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                body = json.loads(raw) if raw else {}
                outer.requests.append({"method": self.command, "path": self.path, "headers": dict(self.headers), "body": body})
                status, data = fn(self.command, self.path, body)
                out = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            do_GET = do_POST = _do

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def vectors(inputs):
    return {"object": "list", "data": [{"object": "embedding", "index": i, "embedding": [float(len(t)), 0.5, -1.0]} for i, t in enumerate(inputs)],
            "model": "bge-secret-name", "usage": {"total_tokens": 9}, "system_fingerprint": "leak"}


def garden_call(method, path, body=None, key=GARDEN_KEY):
    return call_full(method, path, body, headers={"Authorization": "Bearer " + key})


# ---- the embeddings route ---------------------------------------------------------------------------------------------------
class EmbeddingsRouteTests(unittest.TestCase):
    def setUp(self):
        auth.THROTTLE._fails.clear()
        self.addCleanup(auth.THROTTLE._fails.clear)
        os.environ.pop("STONESAGE_GARDEN_MODEL_KEY", None)
        self.engine = Serve(lambda m, p, b: (200, vectors(b.get("input", []))))
        self.addCleanup(self.engine.close)
        patcher = mock.patch.dict(server.config, {"garden": {"model_key": GARDEN_KEY, "embed_url": self.engine.url + "/v1", "engine_url": self.engine.url + "/v1",
                                                            "max_body_bytes": 20000}})
        patcher.start()
        self.addCleanup(patcher.stop)
        server_key()

    def test_the_garden_key_gets_vectors_and_only_vectors(self):
        status, _h, payload = garden_call("POST", EMBED_ROUTE, {"input": ["first text", "second"], "model": "x", "encoding_format": "base64", "user": "u"})
        self.assertEqual(status, 200, payload)
        answer = json.loads(payload)
        self.assertEqual([d["index"] for d in answer["data"]], [0, 1])
        self.assertEqual(answer["data"][0]["embedding"], [10.0, 0.5, -1.0])
        self.assertEqual(set(answer), {"object", "data", "model"})                       # no usage, fingerprint or engine model name
        self.assertNotIn("bge-secret-name", payload.decode() if isinstance(payload, bytes) else payload)
        sent = self.engine.requests[0]
        self.assertEqual(sent["path"], "/v1/embeddings")
        self.assertEqual(sent["body"], {"input": ["first text", "second"]})               # only `input` is forwarded
        self.assertNotIn("Authorization", {k.title() for k in sent["headers"]})            # the garden key never reaches the engine

    def test_one_string_is_accepted(self):
        status, _h, payload = garden_call("POST", EMBED_ROUTE, {"input": "just one"})
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(payload)["data"]), 1)

    def test_the_garden_key_is_refused_on_every_other_route(self):
        routes = [("GET", EMBED_ROUTE, None), ("POST", EMBED_ROUTE + "/extra", {}), ("POST", "/api/garden/v1/embeddings/", {}),
                  ("POST", "/v1/embeddings", {"input": "x"}), ("GET", "/api/garden/board", None), ("GET", "/api/garden/worker/status", None),
                  ("GET", "/api/garden/residents", None), ("POST", "/api/garden/residents", {"resident": "x"}),
                  ("POST", "/api/garden/residents/x/pause", {}), ("GET", "/api/garden/posts/a:work:b/comments", None),
                  ("POST", "/api/garden/posts/a:work:b/comments", {"text": "x"}), ("POST", "/api/garden/window/refresh", {}),
                  ("POST", "/api/garden/sessions/start", {}), ("GET", "/api/garden/requests", None),
                  ("POST", "/api/garden/requests/ann.r00001/decide", {"status": "granted"}), ("POST", "/api/ha/service", {}), ("GET", "/api/config", None), ("GET", "/api/souls", None),
                  ("POST", "/api/cluster/chat", {"message": "hi"}), ("GET", "/api/health/all", None)]
        for method, path, body in routes:
            auth.THROTTLE._fails.clear()
            status, _h, payload = garden_call(method, path, body)
            self.assertEqual(status, 401, "%s %s gave %s" % (method, path, status))
        self.assertEqual(self.engine.requests, [])

    def test_both_garden_routes_and_no_third(self):
        self.assertEqual(garden_client.SCOPED_ROUTES, (garden_client.MODEL_ROUTE, garden_client.EMBED_ROUTE))
        self.assertEqual(garden_call("POST", CHAT_ROUTE, {"messages": [{"role": "user", "content": "hi"}]})[0], 200)   # authorised too
        h = type("H", (), {"headers": {"Authorization": "Bearer " + GARDEN_KEY}, "client_address": ("127.0.0.1", 1)})()
        self.assertEqual(auth.authenticate(h, server.config, *garden_client.EMBED_ROUTE), (garden_client.SCOPED, "grove-bearer"))
        self.assertEqual(auth.authenticate(h, server.config, "GET", EMBED_ROUTE)[0], None)

    def test_wrong_or_missing_keys_and_the_owner(self):
        self.assertEqual(garden_call("POST", EMBED_ROUTE, {"input": "x"}, key="nope")[0], 401)
        self.assertEqual(call_full("POST", EMBED_ROUTE, {"input": "x"}, auth=False)[0], 401)
        self.assertEqual(call_full("POST", EMBED_ROUTE, {"input": "x"})[0], 200)            # the owner's key may call it too
        with mock.patch.dict(server.config, {"garden": {"embed_url": self.engine.url}}):    # no garden key configured at all
            self.assertEqual(garden_call("POST", EMBED_ROUTE, {"input": "x"})[0], 401)

    def test_input_rules(self):
        bad = [{}, {"input": []}, {"input": [""]}, {"input": ["  "]}, {"input": [1]}, {"input": None}, {"input": {"a": 1}},
               {"input": ["x" * 950]}, {"input": ["ok", "y" * 2000]}, {"input": ["x"] * 65}]
        for body in bad:
            self.assertEqual(garden_call("POST", EMBED_ROUTE, body)[0], 400, str(body)[:60])
        self.assertEqual(garden_call("POST", EMBED_ROUTE, {"input": ["x" * 949]})[0], 200)
        self.assertEqual(garden_call("POST", EMBED_ROUTE, {"input": ["x"] * 64})[0], 200)
        self.assertEqual(garden_call("POST", EMBED_ROUTE, {"input": ["x" * 900] * 30})[0], 413)       # over max_body_bytes
        n = len(self.engine.requests)
        self.assertEqual(garden_call("POST", EMBED_ROUTE, {"input": ["x" * 950]})[0], 400)
        self.assertEqual(len(self.engine.requests), n, "a refused input must not reach the embedder")

    def test_engine_trouble_is_a_502(self):
        for fn in (lambda m, p, b: (500, {"error": "boom"}), lambda m, p, b: (200, {"data": [{"embedding": [1.0]}]}),
                   lambda m, p, b: (200, {"nope": 1})):
            eng = Serve(fn)
            self.addCleanup(eng.close)
            with mock.patch.dict(server.config, {"garden": {"model_key": GARDEN_KEY, "embed_url": eng.url}}):
                self.assertEqual(garden_call("POST", EMBED_ROUTE, {"input": ["a", "b"]})[0], 502)

    def test_the_default_url_is_the_embedder_endpoint(self):
        eng = Serve(lambda m, p, b: (200, vectors(b["input"])))
        self.addCleanup(eng.close)
        cluster = dict(server.config.get("cluster") or {}, embedder_url=eng.url + "/v1")
        with mock.patch.dict(server.config, {"garden": {"model_key": GARDEN_KEY}, "cluster": cluster}):
            self.assertEqual(garden_call("POST", EMBED_ROUTE, {"input": "x"})[0], 200)
        self.assertEqual(eng.requests[0]["path"], "/v1/embeddings")


# ---- the routes that talk to the worker ----------------------------------------------------------------------------------------
class FakeH:
    def __init__(self, principal=auth.OWNER, command="GET", raw=b""):
        self.principal, self.command, self._raw_body, self.wfile = principal, command, raw, io.BytesIO()
        self.status = None

    def send_response(self, s):
        self.status = s

    def send_header(self, *a):
        pass

    def end_headers(self):
        pass

    def answer(self):
        return json.loads(self.wfile.getvalue() or b"{}")


class FakeFeeder:
    def __init__(self):
        self.status, self.runs = {"weather": {"state": "pushed"}}, 0

    def run_once(self):
        self.runs += 1
        return {"weather": {"state": "pushed"}, "book": {"state": "skipped"}}

    def start(self):
        pass


class WorkerRouteTests(unittest.TestCase):
    def setUp(self):
        self.worker = Serve(self.handle)
        self.addCleanup(self.worker.close)
        self.config = {"garden": {"worker_url": self.worker.url, "control_key": CONTROL_KEY, "model": "m", "novelty_threshold": 0.85}}
        self.ctx = apps.AppContext(self.config, ROOT, lambda: tempfile.gettempdir())
        self.feeder = FakeFeeder()
        garden_client.install(self.ctx, feeder=self.feeder)
        self.reply = (200, {"ok": True})

    def handle(self, method, path, body):
        return self.reply

    def call(self, method, path, body=None, principal=auth.OWNER):
        import urllib.parse
        parsed = urllib.parse.urlparse(path)
        fn = self.ctx.routes.find(method, parsed.path)
        self.assertIsNotNone(fn, "%s %s is not a route" % (method, path))
        h = FakeH(principal, method)
        fn(h, parsed, body if body is not None else {})
        return h.status, h.answer()

    def test_every_new_route_is_owner_only(self):
        routes = [("GET", "/api/garden/worker/status"), ("GET", "/api/garden/residents"), ("POST", "/api/garden/residents"),
                  ("POST", "/api/garden/residents/x/pause"), ("GET", "/api/garden/posts/a:work:b/comments"),
                  ("POST", "/api/garden/posts/a:work:b/comments"), ("POST", "/api/garden/window/refresh")]
        for method, path in routes:
            for principal in (garden_client.SCOPED, None, "trusted:1.2.3.4"):
                self.assertEqual(self.call(method, path, {}, principal)[0], 401, (method, path, principal))
        self.assertEqual(self.worker.requests, [])
        self.assertEqual(self.feeder.runs, 0)

    def test_status_joins_the_workers_health_and_the_windows_state(self):
        self.reply = (200, {"ok": True, "running": "20260101T000000Z-abcd", "residents": ["a"]})
        status, out = self.call("GET", "/api/garden/worker/status")
        self.assertEqual((status, out["running"], out["window"]), (200, "20260101T000000Z-abcd", {"weather": {"state": "pushed"}}))
        self.assertEqual(self.worker.requests[0]["headers"]["Authorization"], "Bearer " + CONTROL_KEY)
        self.worker.close()
        status, out = self.call("GET", "/api/garden/worker/status")
        self.assertEqual((status, out["ok"]), (200, False))

    def test_residents_create_and_pause(self):
        self.reply = (200, {"ok": True, "residents": [{"id": "a"}]})
        self.assertEqual(self.call("GET", "/api/garden/residents")[1]["residents"], [{"id": "a"}])
        self.reply = (200, {"ok": True, "resident": "b", "created": True})
        self.assertEqual(self.call("POST", "/api/garden/residents", {"resident": "b"})[1]["created"], True)
        self.assertEqual(self.worker.requests[-1]["body"], {"resident": "b"})
        self.reply = (200, {"ok": True, "paused": True})
        self.call("POST", "/api/garden/residents/b/pause", {"paused": True})
        self.assertEqual((self.worker.requests[-1]["path"], self.worker.requests[-1]["body"]), ("/residents/b/pause", {"paused": True}))
        self.call("POST", "/api/garden/residents/b/pause", {"paused": False})
        self.assertEqual(self.worker.requests[-1]["body"], {"paused": False})
        n = len(self.worker.requests)
        self.assertEqual(self.call("POST", "/api/garden/residents/Bad%20Id/pause", {})[0], 404)
        self.assertEqual(self.call("POST", "/api/garden/residents/../x/pause", {})[0], 404)
        self.assertEqual(len(self.worker.requests), n)

    def test_comments_round_trip_with_the_post_id_quoted(self):
        pid = "resident-1:work:0001-a-poem"
        self.reply = (200, {"ok": True, "post": pid, "comments": [{"text": "hi"}], "stats": {}})
        self.assertEqual(self.call("GET", "/api/garden/posts/resident-1%3Awork%3A0001-a-poem/comments")[1]["comments"], [{"text": "hi"}])
        self.assertEqual(self.worker.requests[-1]["path"], "/posts/resident-1%3Awork%3A0001-a-poem/comments")
        self.reply = (200, {"ok": True, "comment": {"id": "c00001"}})
        status, out = self.call("POST", "/api/garden/posts/" + pid + "/comments", {"text": "Lovely", "rating": 5, "author": "evil"})
        self.assertEqual((status, out["comment"]["id"]), (200, "c00001"))
        self.assertEqual(self.worker.requests[-1]["body"], {"text": "Lovely", "rating": 5, "author": "user"})   # the author is not client-chosen
        self.call("POST", "/api/garden/posts/" + pid + "/comments", {"text": "x", "rating": "5"})
        self.assertIsNone(self.worker.requests[-1]["body"]["rating"])                                               # only whole numbers pass
        n = len(self.worker.requests)
        for bad in ("../x", "a:b:c", "resident-1:work:", "resident-1%3Awork%3A..%2F..%2Fx", "x"):
            self.assertIn(self.call("GET", "/api/garden/posts/%s/comments" % bad)[0], (400, 404), bad)
        self.assertEqual(len(self.worker.requests), n)

    def test_worker_errors_pass_through(self):
        self.reply = (409, {"ok": False, "error": "this resident is paused"})
        status, out = self.call("POST", "/api/garden/posts/resident-1:work:0001-x/comments", {"text": "x"})
        self.assertEqual(status, 409)
        self.assertIn("paused", out["error"])

    def test_start_passes_the_novelty_threshold_from_config(self):
        self.reply = (200, {"ok": True, "session": "s", "resident": "r"})
        client = garden_client.GardenClient(self.config)
        client.start_session({"resident": "r"})
        self.assertEqual(self.worker.requests[-1]["body"]["novelty_threshold"], 0.85)
        client.start_session({"resident": "r", "novelty_threshold": 0.7})
        self.assertEqual(self.worker.requests[-1]["body"]["novelty_threshold"], 0.7)
        client.config["garden"].pop("novelty_threshold")
        client.start_session({"resident": "r"})
        self.assertNotIn("novelty_threshold", self.worker.requests[-1]["body"])

    def test_refresh_runs_the_feeder_and_needs_a_configured_worker(self):
        status, out = self.call("POST", "/api/garden/window/refresh")
        self.assertEqual((status, out["window"]["book"]["state"]), (200, "skipped"))
        self.assertEqual(self.feeder.runs, 1)
        self.config["garden"] = {}
        self.assertEqual(self.call("POST", "/api/garden/window/refresh")[0], 503)
        self.assertEqual(self.feeder.runs, 1)


# ---- the window --------------------------------------------------------------------------------------------------------------
WEATHER = [{"entity_id": "light.x", "state": "on", "attributes": {}},
           {"entity_id": "weather.home", "state": "partlycloudy", "attributes": {"temperature": 71, "temperature_unit": "°F", "humidity": 40,
                                                                                    "wind_speed": 6, "wind_speed_unit": "mph"}},
           {"entity_id": "person.alex", "state": "home", "attributes": {"friendly_name": "Alex Smith"}},
           {"entity_id": "person.sam", "state": "away", "attributes": {"friendly_name": "Sam Jones"}}]
EVENTS = [{"label": "person", "camera": "kitchen_living_room", "start_time": NOW - 600, "sub_label": ["Alex", 0.9],
           "thumbnail": "BASE64IMAGEDATA", "data": {"top_score": 0.9}},
          {"label": "cat", "camera": "kitchen_living_room", "start_time": NOW - 3600},
          {"label": "person", "camera": "driveway_front_door", "start_time": NOW - 7200},
          {"label": "car", "camera": "driveway_front_door", "start_time": NOW - 30 * 3600}]
TRACE = [{"kind": "courage", "at": NOW - 100, "user": "Is the kitchen light on?", "outcome": "answered", "triggers": []},
         {"kind": "courage", "at": NOW - 200, "user": "Where is Rex?", "outcome": "step_cap", "triggers": ["step_cap", "repeated_call"]},
         {"at": NOW - 300, "user": "x" * 300, "outcome": "answered", "triggers": ["tool_error"]},
         {"kind": "boost", "at": NOW - 50, "outcome": "ok"}, {"kind": "courage", "at": NOW - 90000, "user": "old", "outcome": "answered"}]


class BuilderTests(unittest.TestCase):
    def test_weather(self):
        text = garden_window.build_weather(WEATHER)
        self.assertEqual(text, "Weather at home right now: partlycloudy, 71 °F, humidity 40%, wind 6 mph.")
        self.assertEqual(garden_window.build_weather([WEATHER[0]]), "")
        self.assertIn("partlycloudy", garden_window.build_weather(WEATHER, "weather.nowhere"))

    def test_cameras_are_labels_and_times_only(self):
        text = garden_window.build_cameras(EVENTS, NOW)
        self.assertIn("person x2 (driveway front door 1, kitchen living room 1)", text)
        self.assertIn("cat x1", text)
        self.assertNotIn("car", text)                                           # older than 24 h
        self.assertIn("10 min ago: person at kitchen living room", text)
        for leak in ("BASE64", "Alex", "0.9", "thumbnail"):
            self.assertNotIn(leak, text, "no pictures and no person names by default")
        self.assertIn("(Alex)", garden_window.build_cameras(EVENTS, NOW, allow_names=True))
        self.assertIn("nothing", garden_window.build_cameras([], NOW))

    def test_computer_trace_summary(self):
        text = garden_window.build_computer(TRACE, NOW)
        self.assertIn("3 turns (2 answered, 1 step_cap)", text)
        self.assertIn('"Is the kitchen light on?"', text)
        self.assertIn('"Where is Rex?" -> step_cap (step_cap, repeated_call)', text)
        self.assertNotIn("old", text.replace("household", ""))                   # outside the day
        self.assertNotIn("x" * 150, text)                                        # questions are cut short
        self.assertIn("not asked anything", garden_window.build_computer([], NOW))
        self.assertLessEqual(len(text), garden_window.MAX_CHARS)

    def test_headlines_and_feeds(self):
        rss = "<rss><channel><title>T</title><item><title>First &amp; best</title></item><item><title>Second</title></item></channel></rss>"
        atom = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Atom one</title></entry></feed>'
        self.assertEqual(garden_window._feed_titles(rss, "a.example"), [("a.example", "First & best"), ("a.example", "Second")])
        self.assertEqual(garden_window._feed_titles(atom, "b.example"), [("b.example", "Atom one")])
        bomb = '<!DOCTYPE x [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;">]><rss><channel><item><title>&b;</title></item></channel></rss>'
        self.assertEqual(garden_window._feed_titles(bomb, "evil"), [])
        self.assertEqual(garden_window._feed_titles("<not xml", "x"), [])
        text = garden_window.build_headlines([("a", "One"), ("b", "one"), ("c", "Two")])
        self.assertEqual(text, "Headlines:\n- [a] One\n- [c] Two")
        self.assertEqual(garden_window.build_headlines([]), "")

    def test_household_terms(self):
        config = {"garden": {"household_terms": ["Phantom"]}, "frigate": {"cameras": {"camera.k": "kitchen_living_room"}, "identities": {"dog": "rex"}},
                  "commentary": {"residents": ["Alex"]}}
        terms = garden_window.household_terms(config, WEATHER, ["Bella", "ab"])
        for t in ("Bella", "Phantom", "kitchen living room", "rex", "Alex", "Alex Smith", "Smith", "Sam Jones", "Jones"):
            self.assertIn(t, terms)
        self.assertNotIn("ab", terms)
        self.assertEqual(len({t.lower() for t in terms}), len(terms))

    def test_pick_passage(self):
        text = "\n\n".join("Paragraph number %d is long enough to count as a real paragraph of prose." % i for i in range(30))
        a = garden_window.pick_passage(text, random.Random(1))
        self.assertEqual(a, garden_window.pick_passage(text, random.Random(1)))
        self.assertGreater(len(a), 600)
        self.assertEqual(garden_window.pick_passage("tiny", random.Random(1)), "")
        self.assertEqual(garden_window.html_text("<h1>Title</h1><p>One  two</p><script>x()</script><p>Three</p>"), "Title\n\nOne two\n\nThree")


class FakeHttp:
    """http(method, url, headers, body, timeout, raw): answers by url prefix from a table; records calls."""

    def __init__(self, table):
        self.table, self.calls = table, []

    def __call__(self, method, url, headers=None, body=None, timeout=15.0, raw=False):
        self.calls.append((method, url, headers, body))
        for prefix, answer in self.table:
            if url.startswith(prefix):
                if isinstance(answer, Exception):
                    raise answer
                return answer(url) if callable(answer) else answer
        raise OSError("no route for " + url)


BASE_CONFIG = {"homeassistant": {"url": "http://ha.test:8123", "token": "t"}, "frigate": {"url": "http://frigate.test:5000"}}


class FeederTests(unittest.TestCase):
    def feeder(self, garden_cfg=None, table=None, **kw):
        cfg = dict(BASE_CONFIG, garden=garden_cfg or {})
        self.pushed = []
        http = FakeHttp(table if table is not None else [("http://ha.test:8123/api/states", WEATHER), ("http://frigate.test:5000/api/events", EVENTS)])
        f = garden_window.WindowFeeder(cfg, lambda k, t: self.pushed.append((k, t)), http=http, trace_recent=lambda n, kind: TRACE,
                                       extra_terms=lambda: ["Bella"], clock=lambda: NOW, **kw)
        f.http_fake = http
        return f

    def kinds(self):
        return [k for k, _t in self.pushed]

    def test_a_run_pushes_what_is_configured_and_skips_the_rest(self):
        f = self.feeder()
        status = f.run_once()
        self.assertEqual(self.kinds(), ["weather", "cameras", "computer", "household-terms"])
        self.assertEqual(status["weather"]["state"], "pushed")
        self.assertEqual(status["book"]["state"], "skipped")
        self.assertIn("not configured", status["book"]["detail"])
        self.assertEqual(status["headlines"]["state"], "skipped")
        text = dict(self.pushed)
        self.assertIn("partlycloudy", text["weather"])
        self.assertNotIn("BASE64", text["cameras"])
        self.assertIn("Bella", text["household-terms"].splitlines())
        self.assertIn("Alex Smith", text["household-terms"].splitlines())
        self.assertTrue(all(len(t) <= garden_window.MAX_CHARS for k, t in self.pushed if k != "household-terms"))
        ha_calls = [c for c in f.http_fake.calls if "/api/states" in c[1]]
        self.assertEqual(len(ha_calls), 1, "Home Assistant is read once per run")

    def test_each_kind_can_be_switched_off(self):
        f = self.feeder({"window": {"weather": False, "cameras": False, "computer": False, "terms": False}})
        status = f.run_once()
        self.assertEqual(self.pushed, [])
        self.assertTrue(all(status[k]["state"] in ("off", "skipped") for k in status))
        self.assertEqual(status["weather"]["state"], "off")
        self.assertEqual(status["household-terms"]["state"], "off")
        self.assertEqual([c for c in f.http_fake.calls if "ha.test" in c[1] or "frigate" in c[1]], [])      # an "off" source is not even asked

    def test_a_failing_source_does_not_stop_the_others(self):
        f = self.feeder(table=[("http://ha.test:8123", OSError("down")), ("http://frigate.test:5000/api/events", EVENTS)])
        status = f.run_once()
        self.assertEqual(status["weather"]["state"], "error")
        self.assertEqual(status["cameras"]["state"], "pushed")
        self.assertEqual(status["computer"]["state"], "pushed")
        self.assertIn("household-terms", self.kinds())                           # config names still go even when HA is down
        self.assertNotIn("weather", self.kinds())

    def test_unconfigured_sources_are_skipped_not_errors(self):
        cfg = {"garden": {}}
        pushed = []
        f = garden_window.WindowFeeder(cfg, lambda k, t: pushed.append(k), http=FakeHttp([]), clock=lambda: NOW)
        status = f.run_once()
        self.assertEqual({k: v["state"] for k, v in status.items() if k not in ("household-terms", "models")},
                         {"weather": "skipped", "cameras": "skipped", "book": "skipped", "headlines": "skipped", "computer": "skipped"})
        self.assertEqual(pushed, [])

    def test_kavita(self):
        html_page = "<html><body>" + "".join("<p>Sentence number %d of a long enough paragraph for the book page.</p>" % i for i in range(40)) + "</body></html>"
        table = [("http://kavita.test/api/Plugin/authenticate", {"token": "jwt"}),
                 ("http://kavita.test/api/Series/all-v2", [{"id": 7, "name": "Moby Dick", "format": 3}, {"id": 8, "name": "Comic", "format": 1}]),
                 ("http://kavita.test/api/Series/volumes", [{"chapters": [{"id": 70, "pages": 12}, {"id": 71, "pages": 0}]}]),
                 ("http://kavita.test/api/Book/70/book-page", json.dumps(html_page))]
        f = self.feeder({"kavita": {"url": "http://kavita.test", "key": "k"}}, table=table + [
            ("http://ha.test:8123", WEATHER), ("http://frigate.test:5000", EVENTS)], rng=random.Random(3))
        f.run_once()
        book = dict(self.pushed)["book"]
        self.assertTrue(book.startswith("From a book (Moby Dick), a passage:"))
        self.assertIn("Sentence number", book)
        auth_call = [c for c in f.http_fake.calls if "authenticate" in c[1]][0]
        self.assertIn("apiKey=k", auth_call[1])
        self.assertTrue(all(c[2] is None or "Authorization" in c[2] or "authenticate" in c[1] for c in f.http_fake.calls if "kavita" in c[1]))
        f2 = self.feeder({"kavita": {"url": "http://kavita.test", "key": "k"}}, table=[("http://kavita.test", OSError("down"))])
        self.assertEqual(f2.run_once()["book"]["state"], "error")

    def test_headlines_from_plain_rss(self):
        rss = "<rss><channel><item><title>Local news</title></item></channel></rss>"
        f = self.feeder({"rss": ["https://feed.example.org/rss", "ftp://nope", 5]}, table=[("https://feed.example.org/rss", rss)])
        f.run_once()
        self.assertEqual(dict(self.pushed)["headlines"], "Headlines:\n- [feed.example.org] Local news")

    def test_the_interval_has_a_floor(self):
        self.assertEqual(self.feeder().interval(), garden_window.DEFAULT_INTERVAL_S)
        self.assertEqual(self.feeder({"window": {"interval_s": 5}}).interval(), garden_window.MIN_INTERVAL_S)
        self.assertEqual(self.feeder({"window": {"interval_s": "x"}}).interval(), garden_window.DEFAULT_INTERVAL_S)
        self.assertEqual(self.feeder({"window": {"interval_s": 600}}).interval(), 600)

    def test_oversize_text_is_capped(self):
        long_events = [{"label": "l%d" % i, "camera": "cam" + "x" * 50, "start_time": NOW - i} for i in range(300)]
        text = garden_window.build_cameras(long_events, NOW)
        self.assertLessEqual(len(text), garden_window.MAX_CHARS)

    def test_install_starts_the_feeder_only_when_the_worker_is_configured(self):
        started = []
        for cfg, expect in (({"garden": {"worker_url": "http://w", "control_key": "k"}}, 1), ({"garden": {}}, 0)):
            ctx = apps.AppContext(cfg, ROOT, lambda: tempfile.gettempdir())
            garden_client.install(ctx)
            self.assertEqual(len(ctx._starters), expect)


if __name__ == "__main__":
    unittest.main()
