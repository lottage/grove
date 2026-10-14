"""Grove tests: switch the Grove app on in the server under test, the way `config.json "apps": ["grove"]` does.

The server loads the apps its config lists (default: Hearth). Tests that drive Grove's routes through the real handler call
`install()` once; it loads `grove.app` into the server's own app context (routes, scoped key, tab), and does nothing the
second time. Importing this module changes nothing by itself."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))


def install():
    from stonesage import apps, server
    if "grove" not in server.app_ctx.apps:
        apps.load_apps(["grove"], server.app_ctx)
    if "grove" not in server.app_ctx.apps:
        raise RuntimeError("the grove app did not load into the server under test")
    return server.app_ctx
