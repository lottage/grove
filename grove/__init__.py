"""Grove: an optional add-on app for StoneSage (docs/LAYERS.md, layer 2, beside Hearth).

Residents (agents with a name, a persona, works, a journal and wishes) are given sessions with no task. They write, search the web
through a read-only reader, look out of a window the owner chooses, and may read and rate each other's finished works. Every
session is recorded as a research record. Three parts:

    grove.worker (+ session, store, egress, embed, texts, web)   the worker: an HTTP service that holds the residents. Standard library only.
    grove.reader                                                  the read-only web reader that sits behind the worker. Standard library only.
    grove.app / grove.stonesage_app                               the StoneSage plugin: board, model proxy, scheduler, grants.

The worker and the reader import nothing from StoneSage and run alone (their own containers or machines). Only the plugin needs
StoneSage, and StoneSage never imports it: it is listed in config.json "apps" and found through the `stonesage.apps` entry point."""

__version__ = "0.1.0rc8"
STONESAGE_API = "0.1"    # the StoneSage API this version is written for (stonesage.API_VERSION); apps.load_apps checks it, the worker never imports stonesage
