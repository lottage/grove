"""The StoneSage side of the garden (stonesage/garden_client.py, auth.py): the scoped model proxy and the worker client.

The garden model key is valid on one route and refused everywhere else; the proxy forwards a chat body and nothing more; the
client talks to the worker with the control key. A fake engine and a fake worker run on localhost.
"""

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from route_harness import call_full, server_key  # noqa: E402
import grove_fixture  # noqa: E402
from stonesage import auth, server  # noqa: E402
from grove.stonesage_app import client as garden_client  # noqa: E402

grove_fixture.install()

GARDEN_KEY = "garden-model-key-for-tests-0123456789"
CONTROL_KEY = "garden-control-key-for-tests-0123456789"
ROUTE = "/api/garden/v1/chat/completions"
TOOL_ANSWER = {"id": "x1", "object": "chat.completion", "created": 1, "model": "m",
               "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                   "role": "assistant", "content": None,
                   "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "ha_call", "arguments": "{\"domain\": \"lock\"}"}}]}}],
               "usage": {"completion_tokens": 7}, "system_fingerprint": "secret-ish"}


class _Serve:
    """A localhost HTTP server whose handler is a function (method, path, headers, body) -> (status, json)."""

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


def garden_call(method, path, body=None, key=GARDEN_KEY):
    return call_full(method, path, body, headers={"Authorization": "Bearer " + key})


class KeyScopeTests(unittest.TestCase):
    def setUp(self):
        auth.THROTTLE._fails.clear()
        self.addCleanup(auth.THROTTLE._fails.clear)
        self.engine = _Serve(lambda m, p, b: (200, TOOL_ANSWER))
        self.addCleanup(self.engine.close)
        patcher = mock.patch.dict(server.config, {"garden": {"model_key": GARDEN_KEY, "engine_url": self.engine.url + "/v1",
                                                             "max_tokens_cap": 500}})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: os.environ.pop("STONESAGE_GARDEN_MODEL_KEY", None))
        os.environ.pop("STONESAGE_GARDEN_MODEL_KEY", None)
        self.owner = server_key()

    def chat_body(self):
        return {"messages": [{"role": "user", "content": "hi"}]}

    def test_garden_key_works_on_the_model_route(self):
        status, _h, payload = garden_call("POST", ROUTE, self.chat_body())
        self.assertEqual(status, 200, payload)
        self.assertEqual(json.loads(payload)["choices"][0]["finish_reason"], "tool_calls")

    def test_garden_key_is_refused_on_every_other_route(self):
        # a sample of real routes: Home Assistant calls, config, souls, chat, the agent API, secrets-bearing reads, the same path by GET
        routes = [("POST", "/api/ha/service", {"domain": "light", "service": "turn_on"}),
                  ("POST", "/api/ha/command", {"command": "unlock the door"}),
                  ("GET", "/api/ha/entities", None), ("GET", "/api/config", None), ("POST", "/api/config", {"garden": {}}),
                  ("GET", "/api/souls", None), ("GET", "/api/souls/courage-computer/export", None),
                  ("POST", "/api/souls/import", {}), ("POST", "/api/cluster/chat", {"message": "hi"}),
                  ("POST", "/v1/chat/completions", {"model": "computer", "messages": [{"role": "user", "content": "hi"}]}),
                  ("GET", "/v1/models", None), ("GET", "/api/garden/board", None),
                  ("POST", "/api/garden/sessions/start", {"resident": "a"}), ("POST", "/api/garden/sessions/x/stop", {}),
                  ("GET", ROUTE, None), ("POST", ROUTE + "/extra", {}), ("POST", "/api/engine/reload", {"service": "x"}),
                  ("GET", "/api/health/all", None), ("POST", "/api/auth/whoami", {})]
        for method, path, body in routes:
            auth.THROTTLE._fails.clear()
            status, _h, payload = garden_call(method, path, body)
            self.assertEqual(status, 401, "%s %s gave %s %r" % (method, path, status, payload[:200]))

    def test_garden_key_is_not_a_login(self):
        h = _FakeHeaders(GARDEN_KEY)
        self.assertEqual(auth.authenticate(h, server.config, "GET", "/api/config"), (None, "bad-key"))
        self.assertEqual(auth.authenticate(h, server.config, *garden_client.MODEL_ROUTE), (garden_client.SCOPED, "grove-bearer"))
        self.assertIsNone(auth.authenticate(h, server.config, "POST", "/api/garden/sessions/start")[0])

    def test_the_owner_key_can_call_the_model_route_too(self):
        status, _h, payload = call_full("POST", ROUTE, self.chat_body())          # bearer = the server key
        self.assertEqual(status, 200, payload)

    def test_the_owner_key_still_works_on_other_routes(self):
        status, _h, _p = call_full("GET", "/api/souls")
        self.assertEqual(status, 200)

    def test_a_wrong_key_or_no_key_is_refused_on_the_model_route(self):
        self.assertEqual(garden_call("POST", ROUTE, self.chat_body(), key="not-the-key")[0], 401)
        self.assertEqual(call_full("POST", ROUTE, self.chat_body(), auth=False)[0], 401)

    def test_no_garden_key_configured_means_no_garden_access(self):
        with mock.patch.dict(server.config, {"garden": {"engine_url": self.engine.url}}):
            self.assertEqual(garden_call("POST", ROUTE, self.chat_body())[0], 401)
            self.assertEqual(garden_call("POST", ROUTE, self.chat_body(), key="")[0], 401)

    def test_a_garden_key_equal_to_the_server_key_is_not_a_separate_key(self):
        with mock.patch.dict(server.config, {"garden": {"model_key": self.owner}}):
            self.assertEqual(auth.scoped_key_value(server.config, garden_client.SCOPED), "")

    def test_the_env_key_wins_over_config(self):
        os.environ["STONESAGE_GARDEN_MODEL_KEY"] = "env-garden-key-0123456789"
        self.assertEqual(garden_call("POST", ROUTE, self.chat_body(), key="env-garden-key-0123456789")[0], 200)
        self.assertEqual(garden_call("POST", ROUTE, self.chat_body(), key=GARDEN_KEY)[0], 401)

    def test_a_cookie_session_may_call_the_proxy_for_testing(self):
        cookie = auth.make_session(self.owner)
        status, _h, _p = call_full("POST", ROUTE, self.chat_body(), headers={
            "Cookie": "ss_session=" + cookie, "Origin": "http://localhost:8888"}, auth=False)
        self.assertEqual(status, 200)


class _FakeHeaders:
    """Just enough of a handler for auth.authenticate."""

    def __init__(self, bearer):
        self.headers = {"Authorization": "Bearer " + bearer}
        self.client_address = ("127.0.0.1", 1)


class ProxyTests(unittest.TestCase):
    def setUp(self):
        auth.THROTTLE._fails.clear()
        self.addCleanup(auth.THROTTLE._fails.clear)
        os.environ.pop("STONESAGE_GARDEN_MODEL_KEY", None)
        self.engine = _Serve(lambda m, p, b: (200, TOOL_ANSWER))
        self.addCleanup(self.engine.close)
        patcher = mock.patch.dict(server.config, {"garden": {"model_key": GARDEN_KEY, "engine_url": self.engine.url + "/v1",
                                                             "max_tokens_cap": 500, "max_body_bytes": 4000}})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_it_never_runs_a_tool_and_adds_nothing(self):
        tools = [{"type": "function", "function": {"name": "ha_call", "parameters": {"type": "object", "properties": {}}}}]
        body = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], "tools": tools,
                "temperature": 0.7, "seed": 5, "max_tokens": 100}
        with mock.patch("stonesage.courage.tools.CourageTools.execute", side_effect=AssertionError("a tool ran")) as ex:
            status, _h, payload = garden_call("POST", ROUTE, body)
        self.assertEqual(status, 200)
        ex.assert_not_called()
        self.assertEqual(len(self.engine.requests), 1, "the proxy must make one engine call and no follow-up")
        sent = self.engine.requests[0]["body"]
        self.assertEqual(sent["messages"], body["messages"])                       # no system prompt, memory or grounding added
        self.assertEqual(sent["tools"], tools)                                      # the garden's own schemas pass through
        self.assertEqual(sent["temperature"], 0.7)
        self.assertEqual(sent["seed"], 5)
        self.assertEqual(sent["max_tokens"], 100)
        answer = json.loads(payload)
        self.assertEqual(answer["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "ha_call")   # handed back unrun
        self.assertNotIn("system_fingerprint", answer)
        self.assertEqual(self.engine.requests[0]["path"], "/v1/chat/completions")

    def test_max_tokens_is_capped_and_defaulted(self):
        for asked in (100000, None, -5, "9999"):
            body = {"messages": [{"role": "user", "content": "u"}]}
            if asked is not None:
                body["max_tokens"] = asked
            garden_call("POST", ROUTE, body)
            self.assertEqual(self.engine.requests[-1]["body"]["max_tokens"], 500, asked)

    def test_unknown_fields_are_dropped_and_streaming_is_off(self):
        body = {"messages": [{"role": "user", "content": "u"}], "stream": True, "n": 5, "model": "computer", "grammar": "root ::= x",
                "cache_prompt": True, "logit_bias": {"1": 1}, "user": "x", "extra_body": {"a": 1}}
        garden_call("POST", ROUTE, body)
        sent = self.engine.requests[-1]["body"]
        self.assertEqual(sorted(sent), ["max_tokens", "messages", "n", "stream"])
        self.assertIs(sent["stream"], False)
        self.assertEqual(sent["n"], 1)

    def test_bad_bodies_are_refused_before_the_engine(self):
        for body in ({}, {"messages": []}, {"messages": "hi"}, {"messages": [{"role": "root", "content": "x"}]},
                     {"messages": [{"role": "user", "content": "x"}], "tools": "all"}):
            status, _h, _p = garden_call("POST", ROUTE, body)
            self.assertEqual(status, 400, body)
        self.assertEqual(self.engine.requests, [])

    def test_an_oversize_body_is_refused(self):
        status, _h, _p = garden_call("POST", ROUTE, {"messages": [{"role": "user", "content": "x" * 5000}]})
        self.assertEqual(status, 413)
        self.assertEqual(self.engine.requests, [])

    def test_engine_failures_become_a_502_without_detail(self):
        bad = _Serve(lambda m, p, b: (500, {"error": "boom at 192.0.2.5"}))
        self.addCleanup(bad.close)
        with mock.patch.dict(server.config, {"garden": {"model_key": GARDEN_KEY, "engine_url": bad.url}}):
            status, _h, payload = garden_call("POST", ROUTE, {"messages": [{"role": "user", "content": "u"}]})
        self.assertEqual(status, 502)
        self.assertNotIn(b"192.0.2", payload)

    def test_parse_tool_calls_passes_through_as_a_boolean_and_nothing_else_new_does(self):
        body = {"messages": [{"role": "user", "content": "u"}], "parse_tool_calls": False, "extra": 1}
        garden_call("POST", ROUTE, body)
        sent = self.engine.requests[-1]["body"]
        self.assertIs(sent["parse_tool_calls"], False)
        self.assertNotIn("extra", sent)
        garden_call("POST", ROUTE, {"messages": [{"role": "user", "content": "u"}], "parse_tool_calls": "no"})
        self.assertNotIn("parse_tool_calls", self.engine.requests[-1]["body"])

    def test_an_engine_parse_failure_is_a_502_that_carries_only_the_marker(self):
        bad = _Serve(lambda m, p, b: (500, {"error": "Failed to parse tool call arguments as JSON: invalid string at 192.0.2.5"}))
        self.addCleanup(bad.close)
        with mock.patch.dict(server.config, {"garden": {"model_key": GARDEN_KEY, "engine_url": bad.url}}):
            status, _h, payload = garden_call("POST", ROUTE, {"messages": [{"role": "user", "content": "u"}]})
        self.assertEqual(status, 502)
        self.assertIn(b"Failed to parse tool call", payload)
        self.assertNotIn(b"192.0.2", payload)

    def test_without_an_engine_url_it_answers_503(self):
        with mock.patch.dict(server.config, {"garden": {"model_key": GARDEN_KEY}}):
            self.assertEqual(garden_call("POST", ROUTE, {"messages": [{"role": "user", "content": "u"}]})[0], 503)

    def test_the_proxy_does_not_import_computers_loop(self):
        src = open(garden_client.__file__, encoding="utf-8").read()
        for name in ("courage", "hearth", "hass_client", "amem", "qdrant"):
            self.assertNotIn("import " + name, src)
            self.assertNotIn("from ." + name, src)


class ClientTests(unittest.TestCase):
    def setUp(self):
        auth.THROTTLE._fails.clear()
        self.addCleanup(auth.THROTTLE._fails.clear)
        os.environ.pop("STONESAGE_GARDEN_CONTROL_KEY", None)

        def worker(method, path, body):
            if path.startswith("/board"):
                return 200, {"ok": True, "posts": [{"kind": "work", "ts": 2}, {"kind": "wish", "ts": 1}]}
            if path == "/sessions/start":
                return 200, {"ok": True, "session": "20261013T120000Z-ab12", "resident": body["resident"]}
            if path.endswith("/stop"):
                return 200, {"ok": True, "ended": True}
            if method == "GET" and path.startswith("/sessions/20261013T120000Z-ab12"):
                return 200, {"ok": True, "session": "20261013T120000Z-ab12", "state": "ended", "reason": "rest", "turns": 4, "tokens": 310}
            return 404, {"ok": False, "error": "not found"}

        self.worker = _Serve(worker)
        self.addCleanup(self.worker.close)
        self.traces = []
        patcher = mock.patch.dict(server.config, {"garden": {
            "worker_url": self.worker.url, "control_key": CONTROL_KEY, "model": "tiny", "model_sha256": "a" * 64,
            "resident": "resident-1", "sampling": {"temperature": 0.8}}})
        patcher.start()
        self.addCleanup(patcher.stop)
        log = mock.Mock()
        log.write = self.traces.append
        patcher2 = mock.patch.dict(server.app_ctx.services, {"get_trace_log": lambda: log})
        patcher2.start()
        self.addCleanup(patcher2.stop)

    def test_board_goes_through_login_and_carries_the_control_key(self):
        status, _h, payload = call_full("GET", "/api/garden/board?since=1.5&resident=resident-1")
        self.assertEqual(status, 200, payload)
        self.assertEqual(len(json.loads(payload)["posts"]), 2)
        seen = self.worker.requests[-1]
        self.assertEqual(seen["headers"]["Authorization"], "Bearer " + CONTROL_KEY)
        self.assertIn("since=1.5", seen["path"])
        self.assertEqual(call_full("GET", "/api/garden/board", auth=False)[0], 401)

    def test_start_fills_in_the_configured_model_and_traces_it(self):
        status, _h, payload = call_full("POST", "/api/garden/sessions/start", {})
        self.assertEqual(status, 200, payload)
        sent = self.worker.requests[-1]["body"]
        self.assertEqual((sent["resident"], sent["model"], sent["model_sha256"]), ("resident-1", "tiny", "a" * 64))
        self.assertEqual(sent["sampling"], {"temperature": 0.8})
        rec = next(t for t in self.traces if t.get("event") == "start")
        self.assertEqual((rec["kind"], rec["outcome"], rec["session"]), ("garden", "started", "20261013T120000Z-ab12"))
        self.assertNotIn(CONTROL_KEY, json.dumps(self.traces))

    def test_stop_is_traced_and_the_id_is_checked(self):
        status, _h, payload = call_full("POST", "/api/garden/sessions/20261013T120000Z-ab12/stop", {"reason": "Computer needs the GPU"})
        self.assertEqual(status, 200, payload)
        self.assertEqual(self.worker.requests[-1]["body"], {"reason": "Computer needs the GPU"})
        rec = next(t for t in self.traces if t.get("event") == "stop")
        self.assertEqual((rec["kind"], rec["outcome"]), ("garden", "stopped"))
        status, _h, _p = call_full("POST", "/api/garden/sessions/..%2Fx/stop", {})
        self.assertIn(status, (400, 404))

    def test_a_finished_session_reads_back_through_stonesage(self):
        """2026-10-13 (rc.5 dry run): StoneSage had no GET route for a session, so every status read was a 404. Now it is passed to
        the worker, which answers for a finished session from its transcript's end record."""
        status, _h, payload = call_full("GET", "/api/garden/sessions/20261013T120000Z-ab12")
        self.assertEqual(status, 200, payload)
        out = json.loads(payload)
        self.assertEqual((out["state"], out["reason"], out["turns"], out["tokens"]), ("ended", "rest", 4, 310))
        self.assertEqual(self.worker.requests[-1]["method"], "GET")
        self.assertEqual(self.worker.requests[-1]["path"], "/sessions/20261013T120000Z-ab12")
        self.assertEqual(call_full("GET", "/api/garden/sessions/20261013T120000Z-ab12")[0], 200)
        self.assertIn(call_full("GET", "/api/garden/sessions/..%2Fx")[0], (400, 404))
        self.assertEqual(call_full("GET", "/api/garden/sessions/20261013T120000Z-ab12", auth=False)[0], 401)

    def test_a_worker_that_is_down_is_a_502_not_a_crash(self):
        with mock.patch.dict(server.config, {"garden": {"worker_url": "http://127.0.0.1:1", "control_key": CONTROL_KEY}}):
            status, _h, _p = call_full("GET", "/api/garden/board")
        self.assertEqual(status, 502)

    def test_unconfigured_is_a_503(self):
        with mock.patch.dict(server.config, {"garden": {}}):
            self.assertEqual(call_full("GET", "/api/garden/board")[0], 503)

    def test_a_refused_start_is_traced_as_refused(self):
        busy = _Serve(lambda m, p, b: (409, {"ok": False, "error": "a session is already running"}))
        self.addCleanup(busy.close)
        with mock.patch.dict(server.config, {"garden": {"worker_url": busy.url, "control_key": CONTROL_KEY, "model": "m"}}):
            status, _h, payload = call_full("POST", "/api/garden/sessions/start", {})
        self.assertEqual(status, 409, payload)
        self.assertEqual(next(t for t in self.traces if t.get("event") == "start")["outcome"], "refused")


if __name__ == "__main__":
    unittest.main()
