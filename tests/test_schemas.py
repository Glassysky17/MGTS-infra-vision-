"""Контракты данных: эти тесты ДОЛЖНЫ проходить всегда."""
from __future__ import annotations

import json

from mgts.schemas import (CSV_COLUMNS, BBox, ConditionAssessment, DefectFinding, Detection, ObjectRecord,
                          Observation, Priority, Severity, assessment_from_dict, object_record_from_dict,
                          observation_from_dict, to_jsonable)


def test_bbox_basic():
    b = BBox(10, 20, 110, 220)
    assert b.width == 100 and b.height == 200 and b.area == 20000
    assert b.center == (60, 120) and b.bottom_center == (60, 220)


def test_bbox_iou():
    a, b = BBox(0, 0, 10, 10), BBox(5, 5, 15, 15)
    assert abs(a.iou(b) - 25 / 175) < 1e-9
    assert a.iou(a) == 1.0
    assert a.iou(BBox(20, 20, 30, 30)) == 0.0


def test_bbox_yolo_roundtrip():
    b = BBox(100, 50, 300, 250)
    xc, yc, w, h = b.to_xywhn(1000, 500)
    b2 = BBox.from_xywhn(xc, yc, w, h, 1000, 500)
    assert all(abs(p - q) < 1e-6 for p, q in zip(b.to_list(), b2.to_list()))


def test_bbox_clip_and_pad():
    b = BBox(-5, -5, 50, 50).clip(40, 40)
    assert b.to_list() == [0, 0, 40, 40]
    p = BBox(10, 10, 20, 20).pad(0.5, 100, 100)
    assert p.to_list() == [5, 5, 25, 25]


def _record() -> ObjectRecord:
    f = DefectFinding("pole_tilt", 0.8, "geometry", "наклон 9.3° > 8°")
    return ObjectRecord(
        object_id="OBJ-0001", class_name="utility_pole", severity=Severity.DEFECT, condition_score=0.8,
        needs_review=False, priority=Priority.HIGH, findings=[f], lat=55.75, lon=37.61, location_source="exif",
        confidence=0.9, n_observations=3, best_frame_id="f1", best_bbox=BBox(1, 2, 3, 4), frame_ids=["f1", "f2"],
    )


def test_record_row_matches_csv_columns():
    row = _record().to_row()
    assert list(row.keys()) == CSV_COLUMNS
    assert row["defects"] == "pole_tilt" and "9.3" in row["description"]


def test_record_json_roundtrip():
    r = _record()
    d = json.loads(json.dumps(to_jsonable(r)))
    r2 = object_record_from_dict(d)
    assert r2.priority is Priority.HIGH and r2.severity is Severity.DEFECT
    assert r2.best_bbox == BBox(1, 2, 3, 4) and r2.findings[0].code == "pole_tilt"


def test_observation_roundtrip():
    det = Detection("f1", 1, "utility_pole", 0.8, BBox(0, 0, 10, 20))
    a = ConditionAssessment(Severity.WARNING, 0.4, [DefectFinding("corrosion", 0.4)], True)
    o = Observation("f1", det, a, sharpness=120.0, est_lat=1.0, est_lon=2.0, location_source="track")
    o2 = observation_from_dict(json.loads(json.dumps(to_jsonable(o))))
    assert o2.detection.bbox == det.bbox and o2.assessment.needs_review and o2.est_lon == 2.0
    assert assessment_from_dict(to_jsonable(a)).top_finding.code == "corrosion"
