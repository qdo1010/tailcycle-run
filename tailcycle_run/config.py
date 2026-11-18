"""Site configuration.

Every host, user, path, project and model location comes from ONE TOML file:
$TAILCYCLE_RUN_CONFIG, else ~/.config/tailcycle-run/config.toml. Start from
config.example.toml in the repo root. Nothing site-specific is hard-coded in
the package.

    from tailcycle_run.config import get, path
    LOGIN = get("cluster.login")
    CAL   = path("local.opencv_calib_root")
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:                                     # pragma: no cover
    import tomli as tomllib

_CFG: dict | None = None
_MISSING = object()


def config_path() -> Path:
    env = os.environ.get("TAILCYCLE_RUN_CONFIG")
    return Path(env).expanduser() if env else Path.home() / ".config" / "tailcycle-run" / "config.toml"


def load(p: Path | str | None = None) -> dict:
    """Load (and cache) the config. Raises with a clear message when missing."""
    global _CFG
    p = Path(p) if p else config_path()
    if not p.exists():
        raise SystemExit(f"tailcycle-run: no config at {p}. Copy config.example.toml "
                         f"there (or set TAILCYCLE_RUN_CONFIG) and fill in the <placeholders>.")
    with open(p, "rb") as fh:
        _CFG = tomllib.load(fh)
    return _CFG


def cfg() -> dict:
    return _CFG if _CFG is not None else load()


def get(key: str, default: Any = _MISSING) -> Any:
    """Dotted lookup, e.g. get("cluster.login"). A missing key without a
    default is an error naming the key."""
    node: Any = cfg()
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is _MISSING:
                raise SystemExit(f"tailcycle-run: config key '{key}' missing in {config_path()}")
            return default
        node = node[part]
    return node


def path(key: str, default: Any = _MISSING) -> Path:
    v = get(key, default)
    return Path(os.path.expanduser(str(v))) if v is not None else v


def sources() -> dict[str, dict]:
    """[sources.<tab>] tables: video_root, pred_root, animals, min_cams."""
    out = {}
    for tab, s in cfg().get("sources", {}).items():
        d = dict(s)
        for k in ("video_root", "pred_root"):
            if k in d:
                d[k] = Path(os.path.expanduser(d[k]))
        out[tab] = d
    return out


def sealed_animals() -> set[str]:
    return set(get("sealed_animals", []))
