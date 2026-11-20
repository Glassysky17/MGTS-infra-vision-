"""Агент 2: тайлинг, слияние, метрики, DummyDetector."""
from __future__ import annotations

import numpy as np
import pytest

from mgts.detection.evaluate import evaluate_detector, match_detections
from mgts.detection.tiling import detect_tiled, merge_detections, tile_slices
from mgts.schemas import BBox, Detection


def _d(cls, box, score=0.9, fid="f"):
    return Detection(fid, cls, f"c{cls}", score, BBox(*box))


def test_tiles_cover_whole_frame_and_stay_inside():
    w, h, t = 1920, 1080, 640
    cover = np.zeros((h, w), bool)
    for x1, y1, x2, y2 in tile_slices(w, h, t, 0.2):
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        cover[y1:y2, x1:x2] = True
    assert cover.all()


def test_tiles_no_duplicates_and_small_frame_single_window():
    wins = tile_slices(1920, 1080, 640, 0.2)
    assert len(wins) == len(set(wins))
    assert tile_slices(500, 400, 640, 0.2) == [(0, 0, 500, 400)]


def test_tiles_clamp_final_windows_to_image_edges():
    wins = tile_slices(901, 677, 400, 0.25)
    assert all(x2 <= 901 and y2 <= 677 for x1, y1, x2, y2 in wins)
    assert max(x2 for _, _, x2, _ in wins) == 901
    assert max(y2 for _, _, _, y2 in wins) == 677


def test_merge_keeps_best_of_overlapping_same_class():
    out = merge_detections([_d(0, (0, 0, 100, 100), 0.6), _d(0, (5, 5, 105, 105), 0.9)], 0.5)
    assert len(out) == 1 and out[0].score == 0.9


def test_merge_keeps_different_classes():
    out = merge_detections([_d(0, (0, 0, 100, 100)), _d(1, (0, 0, 100, 100))], 0.5)
    assert len(out) == 2


def test_merge_drops_box_cut_by_tile_border():
    big = _d(0, (0, 0, 200, 200), 0.8)
    cut = _d(0, (0, 0, 100, 195), 0.7)       # почти целиком внутри большой
    out = merge_detections([big, cut], 0.9)
    assert len(out) == 1 and out[0].bbox.area == big.bbox.area


def test_merge_containment_suppression_uses_fraction_of_small_box():
    outer = _d(0, (0, 0, 100, 100), 0.5)
    inner = _d(0, (10, 10, 20, 20), 0.99)
    assert merge_detections([inner, outer], 0.99) == [outer]


def test_detect_tiled_offsets_tile_boxes_and_runs_full_frame():
    class FakeDetector:
        calls = 0

        def detect_raw(self, img, frame_id):
            self.calls += 1
            if self.calls == 2:
                return [_d(0, (10, 10, 20, 20), fid=frame_id)]
            return []

    detector = FakeDetector()
    image = np.zeros((400, 800, 3), np.uint8)
    out = detect_tiled(detector, image, "frame", tile=400, overlap=0, iou_thr=0.5)
    assert detector.calls == 3
    assert len(out) == 1 and out[0].bbox == BBox(410, 10, 420, 20)


def test_match_detections_counts():
    gts = [_d(0, (0, 0, 50, 50)), _d(1, (100, 100, 150, 150))]
    preds = [_d(0, (2, 2, 52, 52), 0.9), _d(0, (300, 300, 350, 350), 0.8)]
    r = match_detections(preds, gts, 0.5)
    assert r == {"tp": 1, "fp": 1, "fn": 1}


def test_match_requires_same_class():
    r = match_detections([_d(1, (0, 0, 50, 50))], [_d(0, (0, 0, 50, 50))], 0.5)
    assert r == {"tp": 0, "fp": 1, "fn": 1}


def test_dummy_detector_reads_yolo_labels(tmp_path, cfg):
    import cv2
    from mgts.config import load_config
    from mgts.detection.infer import build_detector
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    img = np.zeros((200, 400, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "images" / "a.jpg"), img)
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.25 0.5\n")
    det = build_detector(load_config(overrides={"detection": {"backend": "dummy"}}))
    out = det.detect(img, "a", image_path=str(tmp_path / "images" / "a.jpg"))
    assert len(out) == 1 and out[0].class_id == 0
    assert out[0].class_name == "cable_manhole" and out[0].score == 0.9
    b = out[0].bbox
    assert abs(b.x1 - 150) < 1 and abs(b.x2 - 250) < 1 and abs(b.y1 - 50) < 1 and abs(b.y2 - 150) < 1


def test_dummy_detector_missing_or_invalid_labels_return_empty(tmp_path, cfg):
    from mgts.detection.infer import build_detector

    detector = build_detector({**cfg, "detection": {**cfg["detection"], "backend": "dummy"}})
    image = np.zeros((100, 100, 3), np.uint8)
    assert detector.detect(image, "missing", str(tmp_path / "images" / "missing.jpg")) == []
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "bad.txt").write_text("bad line\n99 0.5 0.5 0.5 0.5\n")
    assert detector.detect(image, "bad", str(tmp_path / "images" / "bad.jpg")) == []


def test_build_detector_rejects_unknown_backend(cfg):
    from mgts.detection.infer import build_detector

    bad_cfg = {**cfg, "detection": {**cfg["detection"], "backend": "unknown"}}
    with pytest.raises(ValueError, match="backend"):
        build_detector(bad_cfg)


def test_yolo_detector_uses_local_model_and_clips_predictions(monkeypatch, cfg, tmp_path):
    import sys
    import types

    weights = tmp_path / "weights.pt"
    weights.touch()
    observed = {}

    class Box:
        xyxy = np.array([[-5, 10, 50, 120]], dtype=float)
        conf = np.array([0.8])
        cls = np.array([0])

    class Model:
        names = {0: "wrong-name"}

        def predict(self, image, **kwargs):
            observed.update(kwargs)
            return [types.SimpleNamespace(boxes=Box())]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=lambda path: Model()))
    from mgts.detection.infer import build_detector

    yolo_cfg = {
        **cfg,
        "detection": {
            **cfg["detection"], "weights": str(weights), "base_model": "missing.pt", "device": "cpu",
            "tiling": {"enabled": False},
        },
    }
    detector = build_detector(yolo_cfg)
    out = detector.detect(np.zeros((100, 80, 3), np.uint8), "f")
    assert len(out) == 1 and out[0].class_name == "cable_manhole"
    assert out[0].bbox == BBox(0, 10, 50, 100)
    assert observed["device"] == "cpu" and observed["verbose"] is False
    assert observed["imgsz"] == cfg["detection"]["imgsz"]
    assert observed["conf"] == cfg["detection"]["conf"] and observed["iou"] == cfg["detection"]["iou"]


def test_yolo_detector_missing_local_weights_fails_without_download(cfg, tmp_path):
    from mgts.detection.infer import build_detector

    bad_cfg = {
        **cfg,
        "detection": {**cfg["detection"], "weights": str(tmp_path / "absent.pt"), "base_model": "not-downloaded.pt"},
    }
    with pytest.raises(RuntimeError, match="train-det"):
        build_detector(bad_cfg)


def test_yolo_detector_does_not_use_generic_base_weights_as_infrastructure_detector(cfg, tmp_path):
    from mgts.detection.infer import build_detector

    base_model = tmp_path / "base.pt"
    base_model.touch()
    bad_cfg = {
        **cfg,
        "detection": {
            **cfg["detection"], "weights": str(tmp_path / "untrained.pt"), "base_model": str(base_model),
        },
    }
    with pytest.raises(RuntimeError, match="обученных весов"):
        build_detector(bad_cfg)


def test_match_claims_best_iou_gt_once_and_includes_iou_boundary():
    gt = _d(0, (0, 0, 10, 10))
    preds = [_d(0, (0, 0, 10, 10), 0.9), _d(0, (0, 0, 10, 10), 0.8)]
    assert match_detections(preds, [gt], 1.0) == {"tp": 1, "fp": 1, "fn": 0}


def test_train_detector_fails_cleanly_when_data_or_local_weights_missing(cfg, tmp_path):
    from mgts.detection.train import train_detector

    train_cfg = {
        **cfg,
        "paths": {**cfg["paths"], "models_dir": str(tmp_path)},
        "detection": {
            **cfg["detection"], "base_model": "not-downloaded.pt", "weights": str(tmp_path / "detector.pt"),
            "device": "cpu",
        },
    }
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\ntrain: train.txt\nval: val.txt\n")
    with pytest.raises(RuntimeError, match="локальных весов"):
        train_detector(train_cfg, data_yaml)


def test_training_data_repeats_rare_class_frames(tmp_path):
    from pathlib import Path

    import yaml

    from mgts.detection.train import _prepare_balanced_training_data

    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "labels").mkdir()
    common_image, rare_image = dataset / "images" / "common.jpg", dataset / "images" / "rare.jpg"
    common_image.touch()
    rare_image.touch()
    (dataset / "labels" / "common.txt").write_text("0 0.5 0.5 0.1 0.1\n" * 40)
    (dataset / "labels" / "rare.txt").write_text("1 0.5 0.5 0.1 0.1\n")
    train_file, val_file = dataset / "train.txt", dataset / "val.txt"
    train_file.write_text(f"{common_image}\n{rare_image}\n")
    val_file.write_text(f"{rare_image}\n")
    data_file = dataset / "data.yaml"
    data_file.write_text(yaml.safe_dump({"path": str(dataset), "train": "train.txt", "val": "val.txt",
                                         "names": ["common", "rare"]}))
    work_dir = tmp_path / "work"

    balanced_yaml = _prepare_balanced_training_data(data_file, work_dir, rare_ratio=0.05, max_repeats=5)
    balanced_data = yaml.safe_load(balanced_yaml.read_text())
    balanced_lines = Path(balanced_data["train"]).read_text().splitlines()

    assert balanced_lines.count(str(rare_image)) > 1
    assert balanced_data["val"] == "val.txt"
    assert train_file.read_text() == f"{common_image}\n{rare_image}\n"


def test_evaluate_detector_writes_metrics_and_error_artifacts(tmp_path, cfg, monkeypatch):
    import cv2
    import yaml

    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "labels").mkdir()
    image = np.zeros((100, 120, 3), np.uint8)
    cv2.imwrite(str(dataset / "images" / "frame.jpg"), image)
    (dataset / "labels" / "frame.txt").write_text("0 0.5 0.5 0.25 0.5\n")
    (dataset / "val.txt").write_text(str(dataset / "images" / "frame.jpg") + "\n")
    (dataset / "data.yaml").write_text(yaml.safe_dump({"path": str(dataset), "val": "val.txt", "names": ["cable_manhole"]}))
    eval_cfg = {
        **cfg,
        "paths": {**cfg["paths"], "processed_dir": str(dataset)},
        "detection": {
            **cfg["detection"], "backend": "dummy", "tiling": {**cfg["detection"]["tiling"], "enabled": False},
        },
    }
    class FalsePositiveDetector:
        def detect(self, image, frame_id, image_path=None):
            return [_d(0, (0, 0, 10, 10), 0.7, frame_id)]

    monkeypatch.setattr("mgts.detection.infer.build_detector", lambda detector_cfg: FalsePositiveDetector())
    out_dir = tmp_path / "eval"
    metrics = evaluate_detector(eval_cfg, out_dir=out_dir)
    assert metrics["per_class"]["cable_manhole"]["recall"] == 0.0
    assert metrics["map50"] == 0.0 and metrics["recall_small"] == 0.0
    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "errors" / "fn_01_frame.jpg").is_file()
    assert (out_dir / "errors" / "fp_01_frame.jpg").is_file()
