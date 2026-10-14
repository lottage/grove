"""Grove as an app of the real server (grove/app.py): listed in GET /api/apps with its tab, its board script served behind the login,
its model key scoped to two routes, its routes owned by it. The cockpit page itself names no app. No LAN."""

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from route_harness import call_full  # noqa: E402
import grove_fixture  # noqa: E402
from stonesage import auth, server  # noqa: E402
from grove.stonesage_app import client  # noqa: E402

CTX = grove_fixture.install()


class GroveAppTests(unittest.TestCase):
    def setUp(self):
        auth.THROTTLE._fails.clear()
        self.addCleanup(auth.THROTTLE._fails.clear)

    def test_it_is_listed_with_one_tab_that_points_at_the_board_script(self):
        status, _h, payload = call_full("GET", "/api/apps")
        self.assertEqual(status, 200)
        row = next(a for a in json.loads(payload)["apps"] if a["id"] == "grove")
        self.assertEqual(len(row["tabs"]), 1)
        tab = row["tabs"][0]
        self.assertEqual((tab["id"], tab["script"], tab["open"], tab["target"]), ("grove", "/apps/grove/board.js", "openGrove", "engine_console"))

    def test_the_board_script_is_served_to_the_login_only(self):
        status, hdrs, payload = call_full("GET", "/apps/grove/board.js")
        self.assertEqual(status, 200)
        self.assertIn(b"openGrove", payload)
        self.assertTrue(any(k.lower() == "content-type" and v.startswith("text/javascript") for k, v in hdrs))
        self.assertEqual(call_full("GET", "/apps/grove/board.js", auth=False)[0], 401)
        self.assertEqual(call_full("GET", "/apps/grove/%2e%2e/app.py")[0], 404)
        self.assertEqual(call_full("GET", "/apps/grove/stonesage_app/client.py")[0], 404)

    def test_the_manifest_needs_the_login(self):
        self.assertEqual(call_full("GET", "/api/apps", auth=False)[0], 401)

    def test_its_routes_are_owned_by_grove_and_the_scoped_key_is_declared(self):
        owners = {(r["method"], r["path"]): r["app"] for r in CTX.routes.table()}
        for route in client.SCOPED_ROUTES:
            self.assertEqual(owners[route], "grove")
        self.assertEqual(owners[("GET", "/api/garden/board")], "grove")
        self.assertEqual(auth.SCOPED_KEYS["grove"].routes, frozenset(client.SCOPED_ROUTES))

    def test_the_platform_names_no_grove_routes_of_its_own(self):
        src = open(auth.__file__, encoding="utf-8").read() + open(server.__file__, encoding="utf-8").read()
        self.assertNotIn("/api/garden/v1", src)
        self.assertNotIn("garden_model_key", src)

    def test_the_scheduler_gauges_reach_metrics_through_the_app(self):
        from unittest import mock
        sched = mock.Mock()
        sched.metrics.return_value = ["stonesage_garden_scheduler_enabled 0"]
        import tempfile
        with tempfile.TemporaryDirectory() as data, mock.patch.dict(os.environ, {"STONESAGE_DATA_DIR": data}),                 mock.patch.object(client, "SCHEDULER", sched):
            status, _h, payload = call_full("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertIn(b"stonesage_garden_scheduler_enabled 0", payload)

    def test_the_config_section_may_be_called_grove_or_garden(self):
        from grove.stonesage_app import section
        self.assertEqual(section.name({}), "grove")
        self.assertEqual(section.name({"garden": {"a": 1}}), "garden")
        self.assertEqual(section.name({"garden": {"a": 1}, "grove": {}}), "grove")
        self.assertEqual(section.section({"garden": {"worker_url": "x"}})["worker_url"], "x")
        cfg = {"garden": {}}
        section.writable(cfg)["k"] = 1
        self.assertEqual(cfg, {"garden": {"k": 1}})
        cfg = {}
        section.writable(cfg)["k"] = 1
        self.assertEqual(cfg, {"grove": {"k": 1}})
        self.assertEqual(client.model_key({"grove": {"model_key": "g"}}), "g")
        self.assertEqual(client.control_key({"garden": {"control_key": "c"}}), "c")
        self.assertEqual(section.env("MODEL_KEY", {"STONESAGE_GARDEN_MODEL_KEY": " old ", "STONESAGE_GROVE_MODEL_KEY": "new"}), "new")
        self.assertEqual(section.env("MODEL_KEY", {"STONESAGE_GARDEN_MODEL_KEY": " old "}), "old")


if __name__ == "__main__":
    unittest.main()
