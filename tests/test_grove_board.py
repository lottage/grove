"""The Grove board is XSS-safe (StoneSage/backend/grove/static/board.js): the residents' text reaches the page only as text.

Runs tests/js/grove_board.test.mjs (a fake DOM that throws on innerHTML and records every element and attribute) when node is
installed, and always checks the source: no HTML-string sinks anywhere in the file, and the panel is wired into the console.
No LAN. The frontend serves no Content-Security-Policy of its own (server.py sets `frame-ancestors 'self'` on pages and a sandbox policy
on served files), so this is the wall: escaped text only.
"""

import os
import re
import shutil
import subprocess
import sys
import unittest
import unittest.mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))      # the monorepo layout; in the grove repository the package is installed

from grove import app as grove_app  # noqa: E402

GARDEN_JS = os.path.join(grove_app.STATIC_DIR, "board.js")


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8-sig") as f:
        return f.read()


class BoardEscapingTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_hostile_posts_render_inert(self):
        r = subprocess.run(["node", os.path.join(ROOT, "tests", "js", "grove_board.test.mjs"), GARDEN_JS],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_no_html_string_sinks_in_the_board_source(self):
        src = open(GARDEN_JS, encoding="utf-8").read()
        code = "\n".join(line for line in src.splitlines() if not line.strip().startswith(("*", "/*", "//")))
        for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function", "createContextualFragment",
                     "DOMParser", "srcdoc", "setAttribute('href'", "setAttribute('src'", "setAttribute('on"):
            self.assertNotIn(sink, code, sink + " must not appear in the garden board")

    def test_the_board_draws_into_the_pane_the_cockpit_makes_for_its_tab(self):
        """The cockpit page names no app (tests/test_app_plugins.py): its engine console builds the tab from GET /api/apps and makes
        the pane `<id>-root`, which is the root and the pane this script looks for."""
        src = open(GARDEN_JS, encoding="utf-8").read()
        self.assertIn("getElementById('grove-root')", src)
        self.assertIn("getElementById('econsole-layer-grove')", src)
        self.assertIn("window.openGrove = openGrove", src)
        app = grove_app
        calls = []
        ctx = type("C", (), {"add_static": lambda s, d: calls.append(("static", d)),
                             "add_tab": lambda s, *a, **k: calls.append(("tab", a, k)),
                             "add_route": lambda *a, **k: None, "routes": None})()
        with unittest.mock.patch("grove.app.client.install", lambda c: None):
            app.install(ctx)
        self.assertIn(("static", app.STATIC_DIR), calls)
        tab = next(c for c in calls if c[0] == "tab")
        self.assertEqual((tab[1][0], tab[1][2], tab[2]["open"]), ("grove", "board.js", "openGrove"))
        self.assertTrue(os.path.isfile(os.path.join(app.STATIC_DIR, "board.js")))

    def test_the_fake_dom_check_would_catch_an_unsafe_render(self):
        """The test's own guard: a renderer that uses innerHTML makes the fake DOM throw, so the check above is not vacuous."""
        js = read("tests", "js", "grove_board.test.mjs")
        self.assertTrue(re.search(r"set innerHTML\(v\)\s*\{\s*throw", js))


if __name__ == "__main__":
    unittest.main()
