"""YAML configuration loading with `base:` inheritance and deep merging."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "scenarios":
            out[k] = deep_merge(out[k], v)
        else:
            # scenario maps replace wholesale so a quick config can drop scenarios
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    return cfg
