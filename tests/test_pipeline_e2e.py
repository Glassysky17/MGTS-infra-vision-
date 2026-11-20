"""Сквозной тест: демо-данные -> пайплайн -> экспорт. Активируется, когда все агенты сдали свои части."""
from __future__ import annotations

import csv
import json

import pytest

from mgts.config import load_config
from mgts.schemas import CSV_COLUMNS


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    import cv2
    import numpy as np

    out = tmp_path_factory.mktemp("demo")
    images, labels = out / "images", out / "labels"
    images.mkdir()
    labels.mkdir()
    track_rows = ["frame_id,lat,lon,heading_deg,timestamp"]
    for index in range(16):
        frame_id = f"frame_{index:04d}"
        frame = np.random.default_rng(index).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        cv2.imwrite(str(images / f"{frame_id}.jpg"), frame)
        (labels / f"{frame_id}.txt").write_text("0 0.5 0.6 0.25 0.25\n", encoding="utf-8")
        track_rows.append(f"{frame_id},55.75,{37.6 + index * 0.00001},90,2026-01-01T10:00:{index:02d}")
    (out / "frames.csv").write_text("\n".join(track_rows), encoding="utf-8")
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
    assert {"input_scan", "setup", "gps", "inference", "assessment", "geolocation", "merge", "export"} <= summary["timing_ms"].keys()
    assert summary["total_elapsed_ms"] >= 0

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


def test_pipeline_continues_after_bad_frame_and_writes_artifacts(tmp_path, cfg, monkeypatch):
    import cv2
    import numpy as np

    from mgts import pipeline
    from mgts.schemas import BBox, ConditionAssessment, Detection, Severity

    input_dir = tmp_path / "frames"
    input_dir.mkdir()
    cv2.imwrite(str(input_dir / "good.jpg"), np.full((120, 160, 3), 90, np.uint8))
    (input_dir / "broken.jpg").write_bytes(b"not an image")
    progress = []

    class Detector:
        def detect(self, image, frame_id, image_path=None):
            return [Detection(frame_id, 0, "cable_manhole", 0.9, BBox(40, 40, 80, 80))]

    monkeypatch.setattr(pipeline, "build_detector", lambda config: Detector())
    monkeypatch.setattr(pipeline, "blur_score", lambda image: 100.0)
    monkeypatch.setattr(pipeline, "enhance", lambda image, config: image)
    monkeypatch.setattr(
        pipeline,
        "assess",
        lambda frame, detection, config, classifier, classes: ConditionAssessment(Severity.OK, 0.0),
    )
    monkeypatch.setattr(
        pipeline,
        "estimate_object_position",
        lambda meta, bbox, config: (meta.lat, meta.lon, meta.location_source),
    )
    run_cfg = {
        **cfg,
        "detection": {**cfg["detection"], "backend": "dummy"},
        "pipeline": {**cfg["pipeline"], "skip_blurry": False},
        "export": {**cfg["export"], "save_annotated_frames": True, "save_crops": True},
    }

    records = pipeline.run_pipeline(input_dir, tmp_path / "out", run_cfg, progress=lambda *args: progress.append(args))

    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert len(records) == 1
    assert summary["n_errors"] == 1 and summary["n_detections"] == 1
    assert len(progress) == 2
    assert (tmp_path / "out" / "objects.csv").is_file()
    assert (tmp_path / "out" / "objects.geojson").is_file()
    assert list((tmp_path / "out" / "annotated").glob("*.jpg"))
    assert list((tmp_path / "out" / "crops").glob("*.jpg"))


def test_pipeline_without_gps_keeps_objects_and_marks_location_unknown(tmp_path):
    import cv2
    import numpy as np

    from mgts.pipeline import run_pipeline

    dataset = tmp_path / "dataset"
    images, labels = dataset / "images", dataset / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    frame = np.random.default_rng(3).integers(0, 256, (64, 80, 3), dtype=np.uint8)
    assert cv2.imwrite(str(images / "frame.jpg"), frame)
    (labels / "frame.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    cfg = load_config(overrides={
        "detection": {"backend": "dummy"},
        "pipeline": {"skip_blurry": False},
        "geo": {"fallback_no_gps": "none"},
    })

    records = run_pipeline(images, tmp_path / "output", cfg)

    assert len(records) == 1
    assert (records[0].lat, records[0].lon, records[0].location_source) == (None, None, "none")
    assert "Неизвестно" in (tmp_path / "output" / "map.html").read_text(encoding="utf-8")


def test_export_keeps_unlocated_records_in_csv_and_geojson(tmp_path, classes):
    from mgts.export import to_csv, to_geojson, to_html_map
    from mgts.schemas import ObjectRecord, Priority, Severity

    record = ObjectRecord("OBJ-0001", "cable_manhole", Severity.OK, 0.0, False, Priority.LOW)
    csv_path, geojson_path, map_path = tmp_path / "objects.csv", tmp_path / "objects.geojson", tmp_path / "map.html"

    to_csv([record], csv_path)
    to_geojson([record], geojson_path)
    to_html_map([record], map_path)

    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "title_ru" in next(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    feature = json.loads(geojson_path.read_text(encoding="utf-8"))["features"][0]
    assert feature["geometry"] is None and feature["properties"]["object_id"] == "OBJ-0001"
    assert "Неизвестно" in map_path.read_text(encoding="utf-8")


def test_draw_annotated_does_not_mutate_source_frame():
    import numpy as np

    from mgts.export import draw_annotated
    from mgts.schemas import BBox, Severity

    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    result = draw_annotated(frame, [(BBox(5, 5, 30, 30), "люк", Severity.DEFECT, True)])

    assert result.shape == frame.shape and not np.any(frame) and np.any(result)
