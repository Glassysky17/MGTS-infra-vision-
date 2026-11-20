"""Конфиги: согласованность classes.yaml и default.yaml. Должны проходить всегда."""
from __future__ import annotations

from mgts.config import class_info, class_names, load_config


def test_default_config_sections(cfg):
    for key in ("paths", "dataset", "preprocessing", "detection", "condition", "geo", "export", "pipeline"):
        assert key in cfg


def test_classes_ids_are_contiguous(classes):
    ids = sorted(c["id"] for c in classes["classes"])
    assert ids == list(range(len(ids)))


def test_all_class_defects_are_defined(classes):
    known = set(classes["defects"])
    for c in classes["classes"]:
        assert set(c["defects"]) <= known, f"{c['name']}: неизвестные дефекты"


def test_class_helpers(classes):
    assert class_names(classes)[0] == "cable_manhole"
    assert class_info("utility_pole", classes)["criticality"] > 0


def test_overrides():
    cfg = load_config(overrides={"detection": {"conf": 0.5}})
    assert cfg["detection"]["conf"] == 0.5 and cfg["detection"]["iou"] == 0.5
