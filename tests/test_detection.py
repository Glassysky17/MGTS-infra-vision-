"""Агент 2: тайлинг, слияние, метрики, DummyDetector."""
from __future__ import annotations

import numpy as np

from mgts.detection.evaluate import match_detections
from mgts.detection.tiling import merge_detections, tile_slices
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
    b = out[0].bbox
    assert abs(b.x1 - 150) < 1 and abs(b.x2 - 250) < 1 and abs(b.y1 - 50) < 1 and abs(b.y2 - 150) < 1
