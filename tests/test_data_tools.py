"""Агент 1: аугментации, синтез дефектов, разбиение, конвертация."""
from __future__ import annotations

import numpy as np

from mgts.data.augment import add_glare, add_motion_blur, add_night, add_rain, add_shadow, perspective_warp
from mgts.data.synth_defects import synth_cracks, synth_displaced_cover, synth_rust, synth_tilt
from mgts.schemas import BBox

from helpers import draw_hatch, draw_pole


def _img():
    return draw_hatch()[0]


def test_augmentations_keep_shape_dtype_and_change_image(rng):
    img = _img()
    for fn in (lambda x: add_motion_blur(x, 15, 20), lambda x: add_rain(x, 0.6, rng), lambda x: add_glare(x, 0.7, rng),
               lambda x: add_shadow(x, 0.6, rng), lambda x: add_night(x, 0.7, rng)):
        out = fn(img.copy())
        assert out.shape == img.shape and out.dtype == np.uint8
        assert not np.array_equal(out, img)


def test_motion_blur_reduces_sharpness():
    import cv2
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    lap = lambda x: cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert lap(add_motion_blur(img, 21, 0)) < lap(img) / 3


def test_perspective_warp_transforms_boxes(rng):
    img, box = draw_hatch()
    out, boxes = perspective_warp(img, [BBox(*box)], 0.08, rng)
    assert out.shape == img.shape and len(boxes) == 1
    assert 0 <= boxes[0].x1 < boxes[0].x2 <= img.shape[1] + 1


def test_synth_cracks_and_rust_only_modify_inside_bbox(rng):
    img, box = draw_hatch()
    b = BBox(*box)
    for fn in (lambda x: synth_cracks(x, b, rng), lambda x: synth_rust(x, b, rng)):
        out = fn(img.copy())
        diff = np.any(out != img, axis=2)
        assert diff.sum() > 50
        ys, xs = np.nonzero(diff)
        assert xs.min() >= b.x1 - 3 and xs.max() <= b.x2 + 3 and ys.min() >= b.y1 - 3 and ys.max() <= b.y2 + 3


def test_synth_tilt_returns_new_box_and_changes_geometry():
    img, box = draw_pole(tilt_deg=0)
    out, nb = synth_tilt(img, BBox(*box), 12.0)
    assert out.shape == img.shape and nb.width > BBox(*box).width       # наклонённый объект шире по x


def test_synth_displaced_cover_changes_image(rng):
    img, box = draw_hatch()
    assert not np.array_equal(synth_displaced_cover(img, BBox(*box), 0.15, rng), img)


def test_make_split_no_group_leak(tmp_path):
    import cv2
    from mgts.data.split import make_split
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for route in ("r1", "r2", "r3", "r4", "r5"):
        for i in range(10):
            n = f"{route}_{i:04d}"
            cv2.imwrite(str(tmp_path / "images" / f"{n}.jpg"), np.zeros((20, 20, 3), np.uint8))
            (tmp_path / "labels" / f"{n}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "data.yaml").write_text("names: [cable_manhole]\n")
    s = make_split(tmp_path, 0.2, 42, group_fn=lambda name: name.split("_")[0])
    group = lambda p: p.split("/")[-1].split("_")[0] if "/" in p else p.split("_")[0]
    assert {group(p) for p in s["train"]}.isdisjoint({group(p) for p in s["val"]})
    assert s["val"] and s["train"]
    s2 = make_split(tmp_path, 0.2, 42, group_fn=lambda name: name.split("_")[0])
    assert s == s2, "сплит должен быть детерминированным"


def test_convert_coco_to_yolo(tmp_path):
    import json
    import cv2
    from mgts.data.convert import convert_to_yolo
    src = tmp_path / "src"
    src.mkdir()
    cv2.imwrite(str(src / "a.jpg"), np.zeros((100, 200, 3), np.uint8))
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 200, "height": 100}],
            "categories": [{"id": 7, "name": "manhole"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [50, 25, 100, 50]}]}
    (src / "annotations.json").write_text(json.dumps(coco))
    out = convert_to_yolo(src, tmp_path / "out", "coco", class_map={"manhole": "cable_manhole"})
    line = (out / "labels" / "a.txt").read_text().split()
    assert line[0] == "0" and abs(float(line[1]) - 0.5) < 1e-3 and abs(float(line[2]) - 0.5) < 1e-3
    assert abs(float(line[3]) - 0.5) < 1e-3 and abs(float(line[4]) - 0.5) < 1e-3
    assert (out / "data.yaml").exists()
