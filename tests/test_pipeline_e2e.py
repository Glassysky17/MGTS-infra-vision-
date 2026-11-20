"""Сквозной тест: демо-данные -> пайплайн -> экспорт. Активируется, когда все агенты сдали свои части."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mgts.config import ROOT, load_config
from mgts.schemas import CSV_COLUMNS


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_data.py"), "--out", str(out), "--n", "16"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        if "NotImplementedError" in r.stderr:
            import os
            if os.environ.get("MGTS_STRICT") == "1":
                pytest.fail("scripts/make_demo_data.py не реализован (Агент 1, §0)")
            pytest.skip("not implemented yet: scripts/make_demo_data.py (Агент 1, §0)")
        pytest.fail(r.stderr)
    return out


def test_demo_data_layout(demo_dir):
    assert len(list((demo_dir / "images").glob("*.jpg"))) == 16
    assert (demo_dir / "frames.csv").exists() and list((demo_dir / "labels").glob("*.txt"))


def test_pipeline_end_to_end(demo_dir, tmp_path):
    from mgts.pipeline import run_pipeline
    cfg = load_config(overrides={"detection": {"backend": "dummy"}})
    records = run_pipeline(demo_dir / "images", tmp_path, cfg, track_path=demo_dir / "frames.csv")
    assert records, "должен быть найден хотя бы один объект"

    # файлы результата
    assert (tmp_path / "objects.csv").exists() and (tmp_path / "objects.geojson").exists()
    assert (tmp_path / "map.html").exists() and (tmp_path / "summary.json").exists()

    rows = list(csv.DictReader(open(tmp_path / "objects.csv", encoding="utf-8-sig")))
    assert set(CSV_COLUMNS) <= set(rows[0].keys())
    gj = json.loads((tmp_path / "objects.geojson").read_text(encoding="utf-8"))
    assert gj["type"] == "FeatureCollection" and gj["features"]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_objects"] == len(records) and summary["n_detections"] >= len(records)

    # дедупликация реально работает: объектов меньше, чем детекций (объекты видны в 2–3 соседних кадрах)
    assert len(records) < summary["n_detections"]
    # сортировка по приоритету
    order = ["critical", "high", "medium", "low"]
    idx = [order.index(r.priority.value) for r in records]
    assert idx == sorted(idx)


def test_cli_run(demo_dir, tmp_path):
    from mgts.cli import main
    rc = main(["run", "--input", str(demo_dir / "images"), "--output", str(tmp_path), "--track",
               str(demo_dir / "frames.csv"), "--backend", "dummy"])
    assert rc == 0 and (tmp_path / "objects.csv").exists()


def test_cli_missing_input_is_friendly(tmp_path):
    from mgts.cli import main
    assert main(["run", "--input", str(tmp_path / "nope"), "--output", str(tmp_path / "o")]) == 2
