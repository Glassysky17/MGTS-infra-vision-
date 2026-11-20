"""Агент 3: геометрия, скоринг, приоритеты."""
from __future__ import annotations

import copy
import cv2
import numpy as np

from mgts.condition.geometry import (
    analyze_hatch, corrosion_ratio, crack_score, estimate_pole_tilt, segment_foreground,
    surface_subsidence_score, vertical_reference_angle,
)
from mgts.condition.classifier import DefectClassifier, train_classifier
from mgts.condition.scoring import assess, priority_for
from mgts.schemas import BBox, ConditionAssessment, DefectFinding, Detection, Priority, Severity

from helpers import draw_hatch, draw_pole


def _pole_crop(tilt_deg: float = 0) -> np.ndarray:
    image, (x1, y1, x2, y2) = draw_pole(tilt_deg=tilt_deg)
    return image[max(0, y1 - 20):min(image.shape[0], y2 + 20),
                 max(0, x1 - 20):min(image.shape[1], x2 + 20)]


def test_tilt_estimation_straight_vs_tilted():
    img0, img10 = _pole_crop(tilt_deg=0), _pole_crop(tilt_deg=10)
    r0, r10 = estimate_pole_tilt(img0), estimate_pole_tilt(img10)
    assert abs(r0.angle_deg) < 2.0
    assert abs(abs(r10.angle_deg) - 10) < 1.5
    assert r10.confidence > 0.3


def test_tilt_sign_right_is_positive():
    img = _pole_crop(tilt_deg=12)
    assert estimate_pole_tilt(img).angle_deg > 0
    img = _pole_crop(tilt_deg=-12)
    assert estimate_pole_tilt(img).angle_deg < 0


def test_tilt_non_elongated_mask_has_zero_confidence():
    crop = np.full((80, 80, 3), 120, np.uint8)
    mask = np.zeros((80, 80), np.uint8)
    mask[30:50, 30:50] = 255
    assert estimate_pole_tilt(crop, mask).confidence == 0.0


def test_segment_foreground_nonempty_for_pole():
    img = _pole_crop()
    m = segment_foreground(img)
    assert m.shape == img.shape[:2] and m.dtype == np.uint8
    assert 0.02 < (m > 0).mean() < 0.5


def test_segment_foreground_rejects_degenerate_and_invalid_crops():
    for img in (np.full((40, 40, 3), 120, np.uint8), np.zeros((0, 0, 3), np.uint8)):
        mask = segment_foreground(img)
        assert mask.shape == img.shape[:2] and not np.any(mask)


def test_segment_foreground_rejects_masks_below_five_percent(monkeypatch):
    labels = np.zeros((100, 100), np.int32)
    labels[40:60, 40:60] = 1
    stats = np.array([[0, 0, 100, 100, 9600], [40, 40, 20, 20, 400]])
    monkeypatch.setattr(cv2, "connectedComponentsWithStats", lambda *_args, **_kwargs: (2, labels, stats, None))

    mask = segment_foreground(np.full((100, 100, 3), 120, np.uint8))

    assert not np.any(mask)


def test_vertical_reference_angle_uses_dominant_near_vertical_lines():
    img = np.full((240, 240, 3), 120, np.uint8)
    for x in range(30, 211, 30):
        cv2.line(img, (x, 220), (x + 25, 40), (20, 20, 20), 2)
    assert 6.0 <= vertical_reference_angle(img) <= 10.0
    assert vertical_reference_angle(np.zeros_like(img)) == 0.0


def test_vertical_reference_angle_ignores_horizontal_edges():
    img = np.full((240, 240, 3), 120, np.uint8)
    for x in range(30, 211, 30):
        cv2.line(img, (x, 220), (x + 25, 40), (20, 20, 20), 2)
    for y in range(20, 221, 20):
        cv2.line(img, (5, y), (235, y), (40, 40, 40), 1)
    assert 6.0 <= vertical_reference_angle(img) <= 10.0


def test_round_hatch_is_found_and_circular():
    img, _ = draw_hatch()
    r = analyze_hatch(img)
    assert r.found and r.circularity > 0.8


def test_hatch_analysis_returns_not_found_for_empty_or_tiny_crop():
    for crop in (np.zeros((0, 0, 3), np.uint8), np.zeros((8, 8, 3), np.uint8)):
        result = analyze_hatch(crop)
        assert not result.found and result.circularity == 0.0


def test_crack_score_detects_dark_thin_lines_and_is_neutral_on_blank():
    blank = np.full((100, 100, 3), 160, np.uint8)
    cracked = blank.copy()
    cv2.line(cracked, (15, 20), (85, 78), (15, 15, 15), 2)
    assert crack_score(blank) == 0.0
    assert crack_score(cracked) > 0.05


def test_corrosion_ratio_counts_rust_color_only_inside_object_mask():
    crop = np.full((40, 40, 3), 120, np.uint8)
    crop[10:30, 10:30] = (30, 70, 150)
    mask = np.zeros((40, 40), np.uint8)
    mask[10:30, 10:30] = 255
    assert corrosion_ratio(crop, mask) > 0.9
    assert corrosion_ratio(crop, np.zeros_like(mask)) == 0.0


def test_surface_subsidence_score_is_bounded_and_neutral_on_uniform_surface():
    frame = np.full((100, 100, 3), 120, np.uint8)
    score = surface_subsidence_score(frame, (40, 40, 60, 60))
    assert 0.0 <= score <= 1.0 and score == 0.0


def _det(cls_id, name, box, score=0.9):
    return Detection("f", cls_id, name, score, BBox(*box))


def test_assess_flags_tilted_pole(cfg, classes):
    img, box = draw_pole(tilt_deg=14)
    a = assess(img, _det(1, "utility_pole", box), cfg, classifier=None, classes=classes)
    assert any(f.code == "pole_tilt" for f in a.findings)
    assert a.severity in (Severity.DEFECT, Severity.UNCERTAIN) and a.score >= cfg["condition"]["scoring"]["warning_threshold"]
    ev = next(f for f in a.findings if f.code == "pole_tilt")
    assert ev.evidence, "дефект должен нести объяснение"


def test_assess_straight_pole_is_not_defect(cfg, classes):
    img, box = draw_pole(tilt_deg=0)
    a = assess(img, _det(1, "utility_pole", box), cfg, classifier=None, classes=classes)
    assert a.severity in (Severity.OK, Severity.WARNING)
    assert not any(f.code == "pole_tilt" and f.score > 0.6 for f in a.findings)


def test_assess_intact_round_hatch_has_no_break_or_displacement_findings(cfg, classes):
    frame, box = draw_hatch()
    result = assess(frame, _det(0, "cable_manhole", box), cfg, classes=classes)

    assert not {finding.code for finding in result.findings} & {"cover_broken", "cover_displaced"}


def test_assess_tiny_crop_needs_review(cfg, classes):
    img, _ = draw_pole()
    a = assess(img, _det(1, "utility_pole", (10, 10, 18, 20)), cfg, classes=classes)
    assert a.needs_review
    assert a.severity is Severity.UNCERTAIN and a.score == 0.0


def test_assess_low_detection_score_needs_review(cfg, classes):
    img, box = draw_hatch()
    a = assess(img, _det(0, "cable_manhole", box, score=0.1), cfg, classes=classes)
    assert a.needs_review


def test_assess_never_raises_on_garbage(cfg, classes):
    img = np.zeros((50, 50, 3), np.uint8)
    a = assess(img, _det(0, "cable_manhole", (0, 0, 50, 50)), cfg, classes=classes)
    assert isinstance(a, ConditionAssessment) and 0.0 <= a.score <= 1.0


def test_priority_monotonic_in_score(cfg, classes):
    order = [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.CRITICAL]
    prev = -1
    for s in (0.0, 0.3, 0.6, 0.9, 1.0):
        sev = Severity.DEFECT if s >= 0.6 else (Severity.WARNING if s >= 0.35 else Severity.OK)
        a = ConditionAssessment(sev, s, [DefectFinding("pole_tilt", s)] if s else [], False)
        idx = order.index(priority_for("cable_manhole", a, cfg, classes))
        assert idx >= prev
        prev = idx


def test_priority_ok_is_low(cfg, classes):
    assert priority_for("cable_manhole", ConditionAssessment(Severity.OK, 0.0, [], False), cfg, classes) is Priority.LOW


def test_priority_missing_cover_is_at_least_high(cfg, classes):
    a = ConditionAssessment(Severity.DEFECT, 0.65, [DefectFinding("cover_missing", 0.9)], False)
    assert priority_for("cable_manhole", a, cfg, classes) in (Priority.HIGH, Priority.CRITICAL)


def test_assess_adds_only_applicable_classifier_findings_with_noisy_or(cfg, classes):
    class Classifier:
        is_ready = True

        def predict(self, crop_bgr, class_name):
            return {"cover_missing": 0.8, "pole_tilt": 0.99}

    frame = np.full((100, 100, 3), 120, np.uint8)
    result = assess(frame, _det(0, "cable_manhole", (20, 20, 80, 80)), cfg, Classifier(), classes)

    assert [(finding.code, finding.source, finding.score) for finding in result.findings] == [
        ("cover_missing", "classifier", 0.8),
    ]
    assert result.score == 0.8 and result.severity is Severity.UNCERTAIN and result.needs_review


def test_assess_flags_classifier_geometry_disagreement(cfg, classes):
    class Classifier:
        is_ready = True

        def predict(self, crop_bgr, class_name):
            return {"pole_tilt": 0.0}

    frame, box = draw_pole(tilt_deg=14)
    result = assess(frame, _det(1, "utility_pole", box), cfg, Classifier(), classes)

    assert result.needs_review and result.severity is Severity.UNCERTAIN


def test_assess_flags_classifier_positive_without_geometric_support(cfg, classes):
    class Classifier:
        is_ready = True

        def predict(self, crop_bgr, class_name):
            return {"cover_missing": 0.8}

    cfg = copy.deepcopy(cfg)
    cfg["condition"]["scoring"]["warning_threshold"] = 0.95
    cfg["condition"]["scoring"]["defect_threshold"] = 0.99
    frame = np.full((100, 100, 3), 120, np.uint8)
    result = assess(frame, _det(0, "cable_manhole", (20, 20, 80, 80)), cfg, Classifier(), classes)

    assert result.needs_review


def test_classifier_without_weights_is_not_ready_and_returns_no_predictions(cfg, classes, tmp_path):
    cfg = copy.deepcopy(cfg)
    cfg["condition"]["classifier"]["weights"] = str(tmp_path / "missing-model.pkl")
    classifier = DefectClassifier(cfg, classes)

    assert not classifier.is_ready
    assert classifier.predict(np.zeros((32, 32, 3), np.uint8), "cable_manhole") == {}


def test_train_classifier_reports_missing_labeled_data(cfg, tmp_path):
    cfg = copy.deepcopy(cfg)
    cfg["paths"]["crops_dir"] = str(tmp_path / "empty-crops")
    try:
        train_classifier(cfg)
    except ValueError as error:
        assert "index.csv" in str(error)
    else:
        raise AssertionError("training without index.csv must explain the missing data")


def test_train_classifier_fits_cpu_fallback_and_predicts_only_applicable_codes(cfg, classes, tmp_path):
    import csv

    cfg = copy.deepcopy(cfg)
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    rows = []
    for index in range(8):
        crop = np.full((48, 48, 3), 120, np.uint8)
        label = "corrosion" if index < 4 else "ok"
        if label == "corrosion":
            crop[10:38, 10:38] = (30, 70, 150)
        image_path = crops_dir / f"frame_{index}.png"
        assert cv2.imwrite(str(image_path), crop)
        rows.append({"path": str(image_path), "class_name": "utility_pole", "label": label,
                     "frame_id": f"frame_{index}", "synthetic": "0"})
    with (crops_dir / "index.csv").open("w", newline="", encoding="utf-8") as index_file:
        writer = csv.DictWriter(index_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    cfg["paths"]["crops_dir"] = str(crops_dir)
    cfg["paths"]["models_dir"] = str(tmp_path / "models")
    cfg["condition"]["classifier"]["weights"] = str(tmp_path / "condition.pkl")
    path = train_classifier(cfg)
    classifier = DefectClassifier(cfg, classes)
    predictions = classifier.predict(cv2.imread(rows[0]["path"]), "utility_pole")

    assert path.is_file() and classifier.is_ready
    assert "corrosion" in predictions and "cover_missing" not in predictions
    assert (tmp_path / "models" / "condition_metrics.json").is_file()


def test_train_classifier_does_not_use_training_samples_as_validation(cfg, classes, tmp_path):
    import csv
    import json

    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    rows = []
    for label, frame_id, color in (("corrosion", "positive", (30, 70, 150)),
                                   ("ok", "negative", (120, 120, 120))):
        for index in range(4):
            crop = np.full((48, 48, 3), color, np.uint8)
            image_path = crops_dir / f"{frame_id}_{index}.png"
            assert cv2.imwrite(str(image_path), crop)
            rows.append({"path": str(image_path), "class_name": "utility_pole", "label": label,
                         "frame_id": frame_id, "synthetic": "0"})
    with (crops_dir / "index.csv").open("w", newline="", encoding="utf-8") as index_file:
        writer = csv.DictWriter(index_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    cfg = copy.deepcopy(cfg)
    cfg["paths"]["models_dir"] = str(tmp_path / "models")
    cfg["condition"]["classifier"]["weights"] = str(tmp_path / "condition.pkl")
    train_classifier(cfg, crops_dir)
    metrics = json.loads((tmp_path / "models" / "condition_metrics.json").read_text())

    assert metrics["per_defect"]["corrosion"] == {
        "precision": None, "recall": None, "threshold": 0.5, "validation_available": False,
    }
