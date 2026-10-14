"""Grove's side of StoneSage: the plugin that gives the cockpit a board, a model proxy for the worker and a scheduler.

client.py (routes, the proxy, the worker client), scheduler.py (when a session may run), models.py (model requests and grants),
window.py (what the residents see out of their window). Imports the platform (stonesage) and the foundation (lodestone); neither
imports it. `grove.app.install(ctx)` wires it in."""
