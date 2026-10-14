"""Grove as a StoneSage app: `config.json "apps": ["grove"]` (or the `stonesage.apps` entry point named `grove`).

Gives the cockpit a GROVE tab in the Engine Console (a board of the residents' works, comments and ratings), the model proxy the
worker talks to (with its own scoped key), the scheduler that decides when a session may run, and the model-request grants.
Nothing here starts unless the worker is configured (config section `grove`, or `garden`: stonesage_app/section.py).
"""

import os

from .stonesage_app import client

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")


def install(ctx) -> None:
    client.install(ctx)
    ctx.add_static(STATIC_DIR)
    ctx.add_tab("grove", "\U0001F331 GROVE", "board.js", open="openGrove")
