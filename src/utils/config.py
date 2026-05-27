"""Lightweight YAML config loader. Hydra-equivalent for our limited needs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class Config(dict):
    """dict subclass with attribute access for nested configs."""

    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(v) if isinstance(v, Mapping) else v


def load_yaml_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw)
