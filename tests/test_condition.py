"""Агент 3: геометрия, скоринг, приоритеты."""
from __future__ import annotations

import numpy as np

from mgts.condition.geometry import analyze_hatch, estimate_pole_tilt, segment_foreground
from mgts.condition.scoring import assess, priority_for
from mgts.schemas import BBox, ConditionAssessment, DefectFinding, Detection, Priority, Severity

from helpers import draw_hatch, draw_pole


def test_tilt_estimation_straight_vs_tilted():
    img0, _ = draw_pole(tilt_deg=0)
    img10, _ = draw_pole(tilt_deg=10)
    r0, r10 = estimate_pole_tilt(img0), estimate_pole_tilt(img10)
    assert abs(r0.angle_deg) < 2.0
    assert abs(abs(r10.angle_deg) - 10) < 1.5
    assert r10.confidence > 0.3


def test_tilt_sign_right_is_positive():
    img, _ = draw_pole(tilt_deg=12)
    assert estimate_pole_tilt(img).angle_deg > 0
    img, _ = draw_pole(tilt_deg=-12)
    assert estimate_pole_tilt(img).angle_deg < 0


def test_segment_foreground_nonempty_for_pole():
    img, _ = draw_pole()
    m = segment_foreground(img)
    assert m.shape == img.shape[:2] and m.dtype == np.uint8
    assert 0.02 < (m > 0).mean() < 0.5


def test_round_hatch_is_found_and_circular():
    img, _ = draw_hatch()
    r = analyze_hatch(img)
    assert r.found and r.circularity > 0.8


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


def test_assess_tiny_crop_needs_review(cfg, classes):
    img, _ = draw_pole()
    a = assess(img, _det(1, "utility_pole", (10, 10, 18, 20)), cfg, classes=classes)
    assert a.needs_review


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
