"""Where Grove keeps its settings in StoneSage's config.json: a section called `grove`, or `garden` (its older name).

A new install writes `grove`. An install that already has a `garden` section (made before the add-on got its
name) keeps it, unchanged: whichever section exists is the one that is read AND written. If both exist, `grove` wins and `garden`
is ignored. (The old watering block that was once also called `garden` is `irrigation` since config_migrate; see stonesage/config_migrate.py.)
"""

from typing import Any, Dict

SECTIONS = ("grove", "garden")
ENV_PREFIXES = ("STONESAGE_GROVE_", "STONESAGE_GARDEN_")         # the first name wins: STONESAGE_GROVE_MODEL_KEY, else STONESAGE_GARDEN_MODEL_KEY


def name(config: Dict[str, Any]) -> str:
    """The section in use: the first of `grove`, `garden` that is an object; `grove` when neither exists yet."""
    for n in SECTIONS:
        if isinstance((config or {}).get(n), dict):
            return n
    return SECTIONS[0]


def section(config: Dict[str, Any]) -> Dict[str, Any]:
    """The settings (read-only view; an empty dict when there are none)."""
    g = (config or {}).get(name(config))
    return g if isinstance(g, dict) else {}


def writable(config: Dict[str, Any]) -> Dict[str, Any]:
    """The settings to change in place (created when missing)."""
    return config.setdefault(name(config), {})


def env(suffix: str, environ: Any = None) -> str:
    """STONESAGE_GROVE_<suffix>, else STONESAGE_GARDEN_<suffix>, stripped; '' when neither is set."""
    import os
    environ = os.environ if environ is None else environ
    for prefix in ENV_PREFIXES:
        value = str(environ.get(prefix + suffix, "") or "").strip()
        if value:
            return value
    return ""
