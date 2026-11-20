"""Агент 1: аугментации, синтез дефектов, разбиение, конвертация."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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


def test_analyze_dataset_reports_objects_empty_and_broken_frames(tmp_path):
    import cv2
    from mgts.data.analyze import analyze_dataset

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "a.jpg"), np.zeros((100, 200, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "images" / "b.jpg"), np.zeros((100, 200, 3), np.uint8))
    (tmp_path / "images" / "duplicate.jpg").write_bytes((tmp_path / "images" / "a.jpg").read_bytes())
    (tmp_path / "images" / "bad.jpg").write_bytes(b"broken")
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.5 0.5\n")
    (tmp_path / "labels" / "duplicate.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.5 0.5\n")
    (tmp_path / "data.yaml").write_text("names: [cable_manhole, utility_pole]\n")
    (tmp_path / "annotations.csv").write_text("filename,class\na.jpg,manhole\n")
    report = tmp_path / "report.md"

    stats = analyze_dataset(tmp_path, report)

    assert stats["n_images"] == 3
    assert stats["n_objects"] == 4
    assert stats["per_class"] == {"cable_manhole": 2, "utility_pole": 2}
    assert stats["small_ratio"] == 0.5
    assert stats["empty_frames"] == 1
    assert stats["broken_files"] == [str(tmp_path / "images" / "bad.jpg")]
    assert stats["has_gps"] is False
    assert stats["duplicate_groups"] == [[str(tmp_path / "images" / name) for name in ("a.jpg", "b.jpg", "duplicate.jpg")]]
    assert "Выводы для дня 0" in report.read_text(encoding="utf-8")
    assert (tmp_path / "dataset_samples.jpg").is_file()

    (tmp_path / "frames.csv").write_text("frame_id,lat,lon\na,55.75,37.6\n")
    (tmp_path / "gt_defects.csv").write_text("defect_code\ncover_broken\n")
    gps_stats = analyze_dataset(tmp_path, tmp_path / "gps-report.md")
    assert gps_stats["has_gps"] is True
    assert gps_stats["defect_codes"] == ["cover_broken"]


def test_analyze_dataset_missing_directory_has_clear_error(tmp_path):
    from mgts.data.analyze import analyze_dataset

    with pytest.raises(FileNotFoundError, match="датасет"):
        analyze_dataset(tmp_path / "missing", tmp_path / "report.md")


def test_convert_csv_clips_boxes_and_records_unknown_classes(tmp_path):
    import csv
    import cv2
    from mgts.data.convert import convert_to_yolo

    src = tmp_path / "src"
    src.mkdir()
    cv2.imwrite(str(src / "a.jpg"), np.zeros((100, 200, 3), np.uint8))
    (src / "labels.csv").write_text(
        "filename,class,xmin,ymin,xmax,ymax\n"
        "a.jpg,manhole,-10,10,110,90\n"
        "a.jpg,unknown,20,20,30,30\n"
        "a.jpg,manhole,30,30,30,50\n"
    )

    out = convert_to_yolo(src, tmp_path / "out", class_map={"manhole": "cable_manhole"})

    label = (out / "labels" / "a.txt").read_text().split()
    assert label[0] == "0"
    assert np.allclose([float(value) for value in label[1:]], [0.275, 0.5, 0.55, 0.8])
    assert (out / "unmapped_classes.txt").read_text().splitlines() == ["unknown"]
    assert "invalid" in (out / "conversion.log").read_text(encoding="utf-8").lower()


def test_convert_voc_annotation_to_yolo(tmp_path):
    import cv2
    from mgts.data.convert import convert_to_yolo

    src = tmp_path / "src"
    src.mkdir()
    cv2.imwrite(str(src / "a.jpg"), np.zeros((100, 200, 3), np.uint8))
    (src / "a.xml").write_text(
        "<annotation><filename>a.jpg</filename><size><width>200</width><height>100</height></size>"
        "<object><name>cable_manhole</name><bndbox><xmin>51</xmin><ymin>26</ymin>"
        "<xmax>150</xmax><ymax>75</ymax></bndbox></object></annotation>"
    )

    out = convert_to_yolo(src, tmp_path / "out", fmt="auto")

    fields = (out / "labels" / "a.txt").read_text().split()
    assert fields[0] == "0"
    assert np.allclose([float(value) for value in fields[1:]], [0.5025, 0.505, 0.495, 0.49])


def test_convert_yolo_copies_images_and_is_idempotent(tmp_path):
    import cv2
    from mgts.data.convert import convert_to_yolo

    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir()
    cv2.imwrite(str(src / "images" / "frame.jpg"), np.zeros((80, 120, 3), np.uint8))
    (src / "labels" / "frame.txt").write_text("0 0.5 0.5 0.25 0.5\n")
    (src / "data.yaml").write_text("names: [cable_manhole]\n")
    out = tmp_path / "out"

    convert_to_yolo(src, out)
    first_label = (out / "labels" / "frame.txt").read_bytes()
    convert_to_yolo(src, out)

    assert (out / "images" / "frame.jpg").read_bytes() == (src / "images" / "frame.jpg").read_bytes()
    assert (out / "labels" / "frame.txt").read_bytes() == first_label


def test_make_split_default_groups_frame_sequences_and_writes_yaml(tmp_path):
    import cv2
    import yaml
    from mgts.data.split import make_split

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for route in ("north", "south", "east", "west", "central"):
        for frame in range(4):
            name = f"{route}_{frame:06d}"
            cv2.imwrite(str(tmp_path / "images" / f"{name}.jpg"), np.zeros((20, 20, 3), np.uint8))
            (tmp_path / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "data.yaml").write_text("names: [cable_manhole]\n")

    split = make_split(tmp_path, 0.2, 13)

    train_routes = {Path(path).stem.rsplit("_", 1)[0] for path in split["train"]}
    val_routes = {Path(path).stem.rsplit("_", 1)[0] for path in split["val"]}
    assert train_routes.isdisjoint(val_routes)
    assert split["val"] and split["train"]
    assert yaml.safe_load((tmp_path / "data.yaml").read_text())["val"] == "val.txt"


def test_extract_crops_writes_unlabeled_index_and_uses_split(tmp_path):
    import csv
    import cv2
    from mgts.data.crops import extract_crops

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    img = np.full((100, 120, 3), 80, np.uint8)
    cv2.imwrite(str(tmp_path / "images" / "frame_0.jpg"), img)
    (tmp_path / "labels" / "frame_0.txt").write_text("0 0.5 0.5 0.3 0.4\n1 0.5 0.5 0.05 0.05\n")
    (tmp_path / "data.yaml").write_text("names: [cable_manhole, utility_pole]\n")
    (tmp_path / "train.txt").write_text(str(tmp_path / "images" / "frame_0.jpg") + "\n")

    out = extract_crops(tmp_path, tmp_path / "crops", padding=0.1, min_size=10)

    rows = list(csv.DictReader((out / "index.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    assert (out / "train" / "cable_manhole" / "unlabeled" / "frame_0_0.jpg").exists()
    assert rows[0]["split"] == "train" and rows[0]["label"] == "unlabeled"


def test_build_train_augmentations_keeps_box_class_alignment():
    from mgts.data.augment import build_train_augmentations

    augment = build_train_augmentations({"dataset": {"augment": {"perspective_warp": 1.0}}})
    boxes = [BBox(80, 80, 220, 220)]
    out, transformed, classes = augment(_img(), boxes, [4], np.random.default_rng(5))

    assert out.shape == _img().shape and len(transformed) == len(classes) == 1
    assert classes == [4]


def test_augment_cli_adds_copies_for_rare_classes(tmp_path):
    import cv2
    import os
    import subprocess
    import sys

    from mgts.config import ROOT

    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "labels").mkdir()
    frame = np.full((100, 120, 3), 100, np.uint8)
    for name, labels in (("common", "0 0.3 0.5 0.3 0.4\n0 0.7 0.5 0.3 0.4\n"),
                         ("rare", "1 0.5 0.5 0.4 0.5\n")):
        cv2.imwrite(str(dataset / "images" / f"{name}.jpg"), frame)
        (dataset / "labels" / f"{name}.txt").write_text(labels)
    (dataset / "data.yaml").write_text("names: [cable_manhole, utility_pole]\n")
    output = tmp_path / "balanced"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))

    subprocess.run([sys.executable, "-m", "mgts.data.augment", "--dataset", str(dataset), "--out", str(output), "--factor", "1"],
                   check=True, capture_output=True, text=True, env=env)

    assert (output / "images" / "rare_aug000.jpg").is_file()
    assert (output / "labels" / "rare_aug000.txt").is_file()
    assert not (output / "images" / "common_aug000.jpg").exists()


def test_generate_synthetic_defects_marks_rows_synthetic(tmp_path):
    import csv
    import cv2
    from mgts.data.synth_defects import generate_synthetic_defects

    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "labels").mkdir()
    img, _ = draw_hatch()
    cv2.imwrite(str(dataset / "images" / "hatch.jpg"), img)
    (dataset / "labels" / "hatch.txt").write_text("0 0.5 0.5 0.45 0.45\n")
    (dataset / "data.yaml").write_text("names: [cable_manhole]\n")

    index = generate_synthetic_defects(str(dataset), str(tmp_path / "synthetic"), per_class=2, seed=19)

    rows = list(csv.DictReader(open(index, encoding="utf-8")))
    assert len(rows) == 2
    assert all(row["synthetic"] == "1" and row["label"] == "cover_broken" for row in rows)
    assert all((Path(index).parent / row["path"]).is_file() for row in rows)


def test_make_demo_data_is_repeatable_and_contains_repeated_defective_objects(tmp_path):
    import csv
    import subprocess
    import sys

    from mgts.config import ROOT

    output_dirs = [tmp_path / "first", tmp_path / "second"]
    for out in output_dirs:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_data.py"), "--out", str(out), "--n", "16", "--seed", "23"],
                       check=True, capture_output=True, text=True)
    first, second = output_dirs
    generated_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    assert generated_files == sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    assert all((first / path).read_bytes() == (second / path).read_bytes() for path in generated_files)
    assert len(list((first / "images").glob("*.jpg"))) == 16

    object_frames: dict[int, list[int]] = {0: [], 1: []}
    for label_path in sorted((first / "labels").glob("*.txt")):
        frame_index = int(label_path.stem.split("_")[-1])
        for line in label_path.read_text().splitlines():
            if line.strip():
                object_frames[int(line.split()[0])].append(frame_index)
    object_groups = []
    for class_id, indices in object_frames.items():
        runs: list[list[int]] = []
        for index in indices:
            if not runs or index > runs[-1][-1] + 1:
                runs.append([index])
            else:
                runs[-1].append(index)
        object_groups.extend((class_id, run) for run in runs)
    assert len(object_groups) >= 3
    assert all(len(frames) == 3 for _, frames in object_groups)
    with (first / "gt_defects.csv").open(encoding="utf-8", newline="") as stream:
        defect_rows = list(csv.DictReader(stream))
    defect_groups = {(row["class_name"], int(row["frame_id"].split("_")[-1]) // 4) for row in defect_rows}
    assert len(defect_groups) / len(object_groups) >= 0.25

    with (first / "frames.csv").open(encoding="utf-8", newline="") as stream:
        tracks = list(csv.DictReader(stream))
    steps_m = [(float(b["lat"]) - float(a["lat"])) * 111320 for a, b in zip(tracks, tracks[1:])]
    assert all(3.9 <= step <= 4.1 for step in steps_m)
