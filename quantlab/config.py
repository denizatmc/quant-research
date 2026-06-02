"""Configuration loading.

I went back and forth on whether to use a heavier settings library (pydantic, hydra),
but for a project this size a thin dataclass wrapper over a YAML file is easier to read
and has no magic. The one nicety I kept is dotted-path access (`cfg.get("risk.var_confidence")`)
because it shows up everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Resolve paths relative to the repo root, not the caller's cwd. This is the kind of
# thing that quietly breaks notebooks if you don't pin it down early.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@dataclass
class Config:
    """A small wrapper around the parsed YAML.

    Keeps the raw dict around (so nothing is lost) but adds dotted-path lookups and
    a couple of convenience accessors for the paths I reach for constantly.
    """

    raw: dict[str, Any] = field(default_factory=dict)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # --- frequently used paths, resolved to absolute so callers don't have to care ---
    @property
    def cache_dir(self) -> Path:
        path = REPO_ROOT / self.get("data.cache_dir", "data/cache")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def sqlite_path(self) -> Path:
        return REPO_ROOT / self.get("database.sqlite_path", "data/cache/market.db")

    @property
    def universe(self) -> list[str]:
        return list(self.get("data.universe", []))

    @property
    def seed(self) -> int:
        return int(self.get("seed", 42))


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML config; falls back to the repo default if no path is given."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path}")
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(raw=raw)
