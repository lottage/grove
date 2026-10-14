"""Garden G4, StoneSage side (stonesage/garden_models.py, garden_client.py, garden_window.py): the model catalogue pushed to the worker, the
request routes, the phone buttons, the cross-reading settings sent with a start. No LAN: a fake worker on localhost, fake provenance.

Does not import `garden` (the worker): the lifecycle against the real worker store is in tests/test_garden_g4g5.py.
"""

import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from stonesage import apps, auth, server  # noqa: E402
from grove.stonesage_app import client as garden_client, models as garden_models, window as garden_window  # noqa: E402

CONTROL_KEY = "garden-control-key-for-tests-0123456789"
NOW = 1_700_000_000.0


def prov(file, sha, size_gb=5.0, loadable=("mainline",), arch="llama", lineage=None, error=None):
    return {"file": file, "sha256": sha, "size_gb": size_gb, "loadable_by": list(loadable), "architecture": arch,
            "lineage": lineage or {}, "error": error, "paths": ["/opt/models/" + file]}


SUMMARY = {"models": [
    prov("granite-4.0-h-tiny-q4_k_m.gguf", "a" * 64, 4.2, lineage={"name": "Granite 4.0 H Tiny", "size_label": "7B-A1B"}),
    prov("Ternary-Bonsai-2-27B-PTQ1_0.gguf", "b" * 64, 5.9, loadable=("prism",), lineage={"name": "Ternary Bonsai 2 27B", "size_label": "27B"}),
    prov("Hermes-4-14B-Q6_K.gguf", "c" * 64, 12.1, lineage={"name": "Hermes 4 14B", "size_label": "14B"}),
    prov("hermes-copy.gguf", "c" * 64, 12.1),                                       # same content under a second name: listed once
    prov("mmproj-qwen2.5-vl-7b-f16.gguf", "d" * 64, 1.4, arch="clip"),
    prov("bge-large-en-v1.5-f16.gguf", "e" * 64, 0.7, arch="bert"),
    prov("huge-70b-q4_k_m.gguf", "f" * 64, 40.0),
    prov("broken.gguf", "1" * 64, 3.0, error="bad header"),
    prov("nobuild.gguf", "2" * 64, 3.0, loadable=()),
    prov("no-hash.gguf", "", 3.0),
]}
CONFIG = {"garden": {"models": {"granite-cpu": {"layout": None, "unit": "llama-courage-lite.service", "engine_url": "http://e:8006/v1",
                                                "model_sha256": "a" * 64},
                                "bonsai": {"layout": "longctx_6600", "engine_url": "http://e:8005/v1", "model_sha256": "b" * 64, "tok_s": 22.5},
                                "mystery": {"layout": "longctx_6600", "engine_url": "http://e:8005/v1", "model_sha256": "9" * 64}}}}


class Serve:
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


# ---- the catalogue ------------------------------------------------------------------------------------------------------------
class CatalogueTests(unittest.TestCase):
    def test_configured_models_are_described_from_provenance(self):
        cat = garden_models.build_catalogue(CONFIG, SUMMARY, lambda name: 31.0 if name == "granite-cpu" else None, NOW)
        by = {m["name"]: m for m in cat["configured"]}
        self.assertEqual(list(by), ["bonsai", "granite-cpu", "mystery"])
        self.assertEqual(by["bonsai"]["description"], "Ternary Bonsai 2 27B · PTQ1_0 · build prism")
        self.assertEqual(by["bonsai"]["slot"], "engine layout longctx_6600")
        self.assertEqual(by["bonsai"]["tok_s"], 22.5, "a figure in config wins")
        self.assertEqual(by["granite-cpu"]["description"], "Granite 4.0 H Tiny · 7B-A1B · Q4_K_M · build mainline")
        self.assertEqual(by["granite-cpu"]["slot"], "CPU")
        self.assertEqual(by["granite-cpu"]["tok_s"], 31.0, "else the measured rate of its sessions")
        self.assertIn("no provenance record", by["mystery"]["description"])
        self.assertNotIn("tok_s", by["mystery"])

    def test_other_models_are_those_a_build_could_load_and_that_are_not_configured(self):
        cat = garden_models.build_catalogue(CONFIG, SUMMARY, None, NOW)
        names = [m["name"] for m in cat["other"]]
        self.assertEqual(names, ["Hermes-4-14B-Q6_K"])        # not: configured (granite, bonsai), clip/bert, too big, errored, no build, no hash, a duplicate
        self.assertEqual(cat["other"][0]["needs"], "the owner")
        named = garden_models.build_catalogue(dict(CONFIG, garden=dict(CONFIG["garden"], owner_label="Pat")), SUMMARY, None, NOW)
        self.assertEqual(named["other"][0]["needs"], "Pat")
        self.assertIn("Hermes 4 14B", cat["other"][0]["description"])

    def test_the_size_limit_is_configurable(self):
        conf = {"garden": dict(CONFIG["garden"], catalogue={"max_other_gb": 50})}
        names = [m["name"] for m in garden_models.build_catalogue(conf, SUMMARY, None, NOW)["other"]]
        self.assertIn("huge-70b-q4_k_m", names)

    def test_without_provenance_the_configured_models_are_still_listed(self):
        cat = garden_models.build_catalogue(CONFIG, None, None, NOW)
        self.assertEqual(len(cat["configured"]), 3)
        self.assertEqual(cat["other"], [])

    def test_the_catalogue_is_size_capped_and_each_line_is_short(self):
        many = {"models": [prov("m%04d-q4_k_m.gguf" % i, "%064x" % (i + 1000), lineage={"name": "N" * 400}) for i in range(300)]}
        text = garden_models.catalogue_text(CONFIG, many, None, NOW)
        self.assertLessEqual(len(text), garden_models.MAX_CATALOGUE_CHARS)
        cat = json.loads(text)
        self.assertEqual(len(cat["configured"]), 3, "the configured models are never dropped")
        self.assertLessEqual(len(cat["other"]), garden_models.MAX_OTHER)
        self.assertTrue(all(len(m["description"]) <= garden_models.MAX_DESCRIPTION for m in cat["other"]))

    def test_garden_models_that_are_not_objects_are_ignored(self):
        conf = {"garden": {"models": {"a": "nope", "b": {"model_sha256": "a" * 64, "layout": None}}}}
        self.assertEqual([m["name"] for m in garden_models.build_catalogue(conf, SUMMARY, None, NOW)["configured"]], ["b"])


# ---- pushed with the window snapshots ----------------------------------------------------------------------------------------------
class FeederPushTests(unittest.TestCase):
    def feeder(self, config, catalogue):
        self.pushed = {}
        off = {k: False for k in ("weather", "cameras", "book", "headlines", "computer", "terms")}
        conf = {"garden": dict(config.get("garden", {}), window=dict(off, **config.get("garden", {}).get("window", {})))}
        return garden_window.WindowFeeder(conf, lambda kind, text: self.pushed.__setitem__(kind, text), http=lambda *a, **k: {},
                                          catalogue=catalogue, clock=lambda: NOW)

    def test_the_catalogue_goes_out_as_the_models_snapshot(self):
        f = self.feeder(CONFIG, lambda: garden_models.catalogue_text(CONFIG, SUMMARY, None, NOW))
        status = f.run_once()
        self.assertEqual(status["models"]["state"], "pushed")
        self.assertEqual([m["name"] for m in json.loads(self.pushed["models"])["configured"]], ["bonsai", "granite-cpu", "mystery"])

    def test_it_is_not_clipped_like_a_window_item(self):
        big = json.dumps({"configured": [], "other": [{"name": "n%d" % i, "description": "d" * 150} for i in range(100)]})
        self.assertGreater(len(big), garden_window.MAX_CHARS)
        f = self.feeder(CONFIG, lambda: big)
        f.run_once()
        self.assertEqual(json.loads(self.pushed["models"])["other"][99]["name"], "n99")          # whole JSON, not cut at 6000

    def test_the_models_snapshot_can_be_switched_off_and_a_missing_source_is_skipped(self):
        f = self.feeder({"garden": {"window": {"models": False}}}, lambda: "{}")
        self.assertEqual(f.run_once()["models"]["state"], "off")
        self.assertNotIn("models", self.pushed)
        g = self.feeder(CONFIG, None)
        self.assertEqual(g.run_once()["models"]["state"], "skipped")

    def test_a_failing_provenance_does_not_stop_the_other_snapshots(self):
        def boom():
            raise ValueError("nope")

        f = self.feeder({"garden": {"window": {"terms": True}}}, boom)
        status = f.run_once()
        self.assertEqual(status["models"]["state"], "error")
        self.assertEqual(status["household-terms"]["state"], "empty", "the other kinds still ran")


# ---- the request routes ---------------------------------------------------------------------------------------------------------
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
    status = {}

    def run_once(self):
        return {}

    def start(self):
        pass


class FakeScheduler:
    def __init__(self):
        self.tick_hooks, self.sessions = [], None

    def holding(self):
        return None

    def note_manual_start(self, *a):
        pass


REQ = {"id": "ann.r00001", "resident": "ann", "name": "Ann", "model": "bonsai", "why": "<b>longer</b> thoughts", "status": "pending", "ts": 1.0}
REQ2 = {"id": "ann.r00002", "resident": "ann", "name": "Ann", "model": "Hermes-4-14B-Q6_K", "why": "curious", "status": "pending", "ts": 2.0}


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.pending = [dict(REQ), dict(REQ2)]
        self.decisions = []
        self.worker = Serve(self.handle)
        self.addCleanup(self.worker.close)
        self.config = {"garden": {"worker_url": self.worker.url, "control_key": CONTROL_KEY, "models": json.loads(json.dumps(CONFIG["garden"]["models"]))},
                       "engine_layouts": {"longctx_6600": {}, "tryout_6750": {}}}
        self.pushes, self.handlers, self.saved = [], [], []
        self.data = tempfile.mkdtemp(prefix="garden-models-")
        self.addCleanup(shutil.rmtree, self.data, True)
        self.ctx = apps.AppContext(self.config, ROOT, lambda: self.data, services={
            "add_phone_action_handler": self.handlers.append,
            "get_provenance": lambda: type("P", (), {"summary": lambda s: SUMMARY})(),
            "load_config": lambda: json.loads(json.dumps(self.config)), "save_config": self.saved.append})
        self.ctx.hooks["notify_phone"] = lambda title, message, data: self.pushes.append((title, message, data)) or {"ok": True}
        self.scheduler = FakeScheduler()
        garden_client.install(self.ctx, feeder=FakeFeeder(), scheduler=self.scheduler)

    def handle(self, method, path, body):
        if method == "GET" and path.startswith("/requests"):
            return 200, {"ok": True, "requests": [r for r in self.pending if r["status"] == "pending"]}
        if method == "POST" and path.startswith("/requests/") and path.endswith("/decide"):
            rid = urllib.parse.unquote(path.split("/")[2])
            hit = next((r for r in self.pending if r["id"] == rid and r["status"] == "pending"), None)
            if not hit:
                return 409, {"ok": False, "error": "that request was already granted"}
            hit["status"] = body["status"]
            self.decisions.append((rid, body))
            return 200, {"ok": True, "request": dict(hit, decided_by=body.get("by"))}
        if method == "GET" and path.startswith("/board"):
            return 200, {"ok": True, "posts": []}
        return 200, {"ok": True}

    def call(self, method, path, body=None, principal=auth.OWNER):
        parsed = urllib.parse.urlparse(path)
        fn = self.ctx.routes.find(method, parsed.path)
        self.assertIsNotNone(fn, "%s %s is not a route" % (method, path))
        h = FakeH(principal, method)
        fn(h, parsed, body if body is not None else {})
        return h.status, h.answer()

    def test_the_new_routes_are_owner_only(self):
        for method, path in (("GET", "/api/garden/requests"), ("POST", "/api/garden/requests/ann.r00001/decide")):
            for principal in (garden_client.SCOPED, None, "trusted:1.2.3.4"):
                self.assertEqual(self.call(method, path, {"status": "declined"}, principal)[0], 401, (method, path, principal))
        self.assertEqual(self.worker.requests, [])
        self.assertEqual(self.decisions, [])

    def test_the_list_says_which_models_have_a_slot_and_which_layouts_exist(self):
        status, out = self.call("GET", "/api/garden/requests")
        self.assertEqual(status, 200)
        rows = {r["id"]: r for r in out["requests"]}
        self.assertEqual((rows["ann.r00001"]["configured"], rows["ann.r00002"]["configured"]), (True, False))
        self.assertEqual(rows["ann.r00001"]["allowed"], False)
        self.assertEqual(out["layouts"], ["longctx_6600", "tryout_6750"])
        self.assertEqual(rows["ann.r00001"]["why"], "<b>longer</b> thoughts", "the reason reaches the page as data; the board renders text only")

    def test_the_owner_grants_a_configured_model_and_the_assignment_is_saved(self):
        status, out = self.call("POST", "/api/garden/requests/ann.r00001/decide", {"status": "granted", "note": "enjoy"})
        self.assertEqual(status, 200)
        self.assertEqual(self.decisions[0][1], {"status": "granted", "by": "owner", "note": "enjoy"})
        self.assertEqual(self.config["garden"]["residents"]["ann"]["model"], "bonsai")
        self.assertEqual(self.saved[-1]["garden"]["residents"]["ann"]["model"], "bonsai", "same save path as the board's model select")

    def test_a_model_with_no_slot_is_refused_until_the_layout_is_confirmed(self):
        for body in ({"status": "granted"}, {"status": "granted", "layout": "nope", "engine_url": "http://e:8011/v1"},
                     {"status": "granted", "layout": "tryout_6750"}):
            status, out = self.call("POST", "/api/garden/requests/ann.r00002/decide", body)
            self.assertEqual(status, 400, body)
        self.assertEqual(self.decisions, [])
        self.assertNotIn("Hermes-4-14B-Q6_K", self.config["garden"]["models"])
        status, out = self.call("POST", "/api/garden/requests/ann.r00002/decide",
                                {"status": "granted", "layout": "tryout_6750", "engine_url": "http://e:8011/v1"})
        self.assertEqual(status, 200)
        entry = self.config["garden"]["models"]["Hermes-4-14B-Q6_K"]
        self.assertEqual((entry["layout"], entry["engine_url"], entry["model_sha256"]), ("tryout_6750", "http://e:8011/v1", "c" * 64))
        self.assertEqual(self.config["garden"]["residents"]["ann"]["model"], "Hermes-4-14B-Q6_K")

    def test_declining_without_a_slot_gives_the_reason_and_changes_nothing(self):
        before = json.dumps(self.config["garden"].get("residents"))
        status, out = self.call("POST", "/api/garden/requests/ann.r00002/decide", {"status": "declined"})
        self.assertEqual(status, 200)
        self.assertEqual(self.decisions[0][1]["note"], garden_models.NO_SLOT)
        self.assertEqual(json.dumps(self.config["garden"].get("residents")), before)

    def test_bad_input(self):
        self.assertEqual(self.call("POST", "/api/garden/requests/ann.r00001/decide", {"status": "maybe"})[0], 400)
        self.assertEqual(self.call("POST", "/api/garden/requests/..%2Fx/decide", {"status": "granted"})[0], 400)
        self.assertEqual(self.call("POST", "/api/garden/requests/ann.r00077/decide", {"status": "granted"})[0], 404)
        self.assertEqual(self.decisions, [])

    def test_a_decided_request_cannot_be_decided_again(self):
        self.call("POST", "/api/garden/requests/ann.r00001/decide", {"status": "declined"})
        status, out = self.call("POST", "/api/garden/requests/ann.r00001/decide", {"status": "granted"})
        self.assertEqual(status, 404, "it is no longer pending")
        self.assertNotIn("residents", self.config["garden"])

    def test_the_phone_buttons_are_registered_and_work(self):
        self.assertEqual(len(self.handlers), 1)
        self.call("GET", "/api/garden/requests")
        self.assertEqual(len(self.pushes), 2, "the owner is asked once per waiting request")
        data = self.pushes[0][2]
        self.assertEqual([a["title"] for a in data["actions"]], ["Grant", "Decline"])
        self.assertEqual([a["title"] for a in self.pushes[1][2]["actions"]], ["Decline"])
        self.call("GET", "/api/garden/requests")
        self.assertEqual(len(self.pushes), 2)
        outcome = self.handlers[0]("GARDEN_GRANT_ann.r00001")
        self.assertIn("Granted bonsai to Ann", outcome)
        self.assertEqual(self.config["garden"]["residents"]["ann"]["model"], "bonsai")
        self.assertIsNone(self.handlers[0]("COURAGE_YES_1"))

    def test_the_board_read_and_the_scheduler_tick_pull_requests_throttled(self):
        self.assertEqual(len(self.scheduler.tick_hooks), 1)
        self.call("GET", "/api/garden/board")
        self.call("GET", "/api/garden/board")
        pulls = [r for r in self.worker.requests if r["path"].startswith("/requests")]
        self.assertEqual(len(pulls), 1, "a second board read inside the throttle window does not pull again")
        self.scheduler.tick_hooks[0]()
        self.assertEqual(len([r for r in self.worker.requests if r["path"].startswith("/requests")]), 1)

    def test_an_auto_grant_happens_on_a_board_read(self):
        self.config["garden"]["residents"] = {"ann": {"allowed_models": ["bonsai"]}}
        self.call("GET", "/api/garden/board")
        self.assertEqual(self.decisions[0][1]["by"], "auto")
        self.assertEqual(self.config["garden"]["residents"]["ann"]["model"], "bonsai")
        self.assertEqual(len(self.pushes), 1, "only the second request, which needs the owner, was pushed")

    def test_push_off_in_config(self):
        self.config["garden"]["grants"] = {"push": False}
        self.call("GET", "/api/garden/requests")
        self.assertEqual(self.pushes, [])


class StartBodyTests(unittest.TestCase):
    def test_cross_reading_settings(self):
        self.assertEqual(garden_client.cross_settings({}, "ann"), (True, []))
        conf = {"garden": {"residents": {"ben": {"cross_reading": False}, "ann": {"model": "x"}, "Bad Id": {"cross_reading": False}}}}
        self.assertEqual(garden_client.cross_settings(conf, "ann"), (True, ["ben"]))
        self.assertEqual(garden_client.cross_settings(conf, "ben"), (False, ["ben"]))
        off = {"garden": {"cross_reading": False}}
        self.assertEqual(garden_client.cross_settings(off, "ann"), (False, []))
        self.assertEqual(garden_client.cross_settings({"garden": {"cross_reading": True}}, "ann"), (True, []))

    def test_a_start_carries_them(self):
        worker = Serve(lambda m, p, b: (200, {"ok": True, "session": "20260101T000000Z-abcd", "resident": b.get("resident")}))
        self.addCleanup(worker.close)
        conf = {"garden": {"worker_url": worker.url, "control_key": CONTROL_KEY, "model": "m",
                           "residents": {"ben": {"cross_reading": False}}}}
        client = garden_client.GardenClient(conf)
        client.start_session({"resident": "ann"})
        client.start_session({"resident": "ben"})
        a, b = worker.requests[0]["body"], worker.requests[1]["body"]
        self.assertEqual((a["cross_reading"], a["hidden_residents"]), (True, ["ben"]))
        self.assertEqual((b["cross_reading"], b["hidden_residents"]), (False, ["ben"]))

    def test_the_new_client_calls(self):
        worker = Serve(lambda m, p, b: (200, {"ok": True, "requests": []}))
        self.addCleanup(worker.close)
        client = garden_client.GardenClient({"garden": {"worker_url": worker.url, "control_key": CONTROL_KEY}})
        client.requests()
        client.decide_request("ann.r00001", "granted", "auto", "ok")
        self.assertEqual(worker.requests[0]["path"], "/requests?status=pending")
        self.assertEqual((worker.requests[1]["path"], worker.requests[1]["body"]), ("/requests/ann.r00001/decide", {"status": "granted", "by": "auto", "note": "ok"}))
        with self.assertRaises(garden_client.GardenError):
            client.decide_request("../x", "granted")


class SchedulerHookTests(unittest.TestCase):
    def test_hooks_run_after_each_tick_and_a_failing_one_does_not_stop_the_thread(self):
        from grove.stonesage_app.scheduler import GardenScheduler

        ran, sleeps = [], []
        sched = GardenScheduler(lambda: {"garden": {}}, lambda fresh: {}, mock.Mock(), mock.Mock(), mock.Mock(), lambda: None, lambda: 0)
        sched.tick = lambda: ran.append("tick")
        sched.tick_hooks.append(lambda: ran.append("hook"))
        sched.tick_hooks.insert(0, lambda: 1 / 0)

        def sleep(s):
            sleeps.append(s)
            if len(sleeps) >= 2:
                raise SystemExit

        sched.sleep = sleep
        sched.start()
        sched._thread.join(3)
        self.assertEqual(ran[:4], ["tick", "hook", "tick", "hook"])


class PhoneHandlerRegistrationTests(unittest.TestCase):
    def test_a_handler_is_attached_now_and_when_the_agent_is_built(self):
        def handler(action):
            return None

        push = type("Push", (), {"extra_handlers": []})()
        agent = type("Agent", (), {"push": push})()
        with mock.patch.object(server, "_courage_agent", agent), mock.patch.object(server, "_phone_action_handlers", []) as registry:
            server._add_phone_action_handler(handler)
            server._add_phone_action_handler(handler)
            self.assertEqual(push.extra_handlers, [handler])
            self.assertEqual(registry, [handler])
        late = type("Push", (), {"extra_handlers": []})()
        with mock.patch.object(server, "_courage_agent", None), mock.patch.object(server, "_phone_action_handlers", []) as registry:
            server._add_phone_action_handler(handler)
            self.assertEqual(registry, [handler], "remembered for the agent that is built later")
        self.assertEqual(late.extra_handlers, [])


if __name__ == "__main__":
    unittest.main()
