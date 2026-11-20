"""Загрузка конфигов (CONTRACT). Реализовано полностью, агентам менять не нужно."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
DEFAULT_CLASSES = ROOT / "configs" / "classes.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Optional[str | Path] = None, overrides: Optional[dict] = None) -> dict[str, Any]:
    """Загрузить default.yaml, поверх — необязательный свой файл и словарь overrides."""
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def load_classes(path: Optional[str | Path] = None) -> dict[str, Any]:
    """Загрузить classes.yaml -> {'classes': [...], 'defects': {...}}."""
    with open(path or DEFAULT_CLASSES, encoding="utf-8") as f:
        return yaml.safe_load(f)


def class_names(classes: Optional[dict] = None) -> list[str]:
    """Имена классов в порядке class_id."""
    classes = classes or load_classes()
    return [c["name"] for c in sorted(classes["classes"], key=lambda c: c["id"])]


def class_info(name: str, classes: Optional[dict] = None) -> dict[str, Any]:
    classes = classes or load_classes()
    for c in classes["classes"]:
        if c["name"] == name:
            return c
    raise KeyError(f"unknown class: {name}")


def resolve_path(p: str | Path) -> Path:
    """Путь из конфига -> абсолютный (относительно корня репозитория)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
