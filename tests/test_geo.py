"""Агент 4: проекция в землю, GPS, дедупликация."""
from __future__ import annotations

import csv

import pytest

from mgts.geo.dedup import merge_observations
from mgts.geo.gps import build_frame_meta, interpolate_position, load_track_csv
from mgts.geo.project import estimate_object_position, pixel_to_ground_offset
from mgts.schemas import BBox, ConditionAssessment, Detection, FrameMeta, Observation, Severity
from mgts.utils import haversine_m

CAM = {"height_m": 1.4, "pitch_deg": 0.0, "hfov_deg": 90.0}
W, H = 1280, 720


def test_ground_projection_known_values():
    f, r = pixel_to_ground_offset(640, 360 + 200, W, H, CAM)
    assert abs(f - 4.48) < 0.1 and abs(r) < 0.05
    f2, r2 = pixel_to_ground_offset(640 + 100, 360 + 200, W, H, CAM)
    assert abs(r2 - 0.7) < 0.05
    _, r3 = pixel_to_ground_offset(640 - 100, 360 + 200, W, H, CAM)
    assert abs(r3 + r2) < 1e-6


def test_ground_projection_lower_pixel_is_closer():
    f_low, _ = pixel_to_ground_offset(640, 600, W, H, CAM)
    f_high, _ = pixel_to_ground_offset(640, 480, W, H, CAM)
    assert f_low < f_high


def test_ground_projection_above_horizon_is_none():
    assert pixel_to_ground_offset(640, 100, W, H, CAM) is None
    assert pixel_to_ground_offset(640, 360, W, H, CAM) is None


def test_object_position_follows_heading(cfg):
    cfg = {**cfg, "geo": {**cfg["geo"], "camera": {**CAM, "mount_offset_m": [0.0, 0.0]}}}
    box = BBox(600, 500, 680, 560)       # основание в (640, 560) -> ~4.5 м вперёд
    north = FrameMeta("f", "f.jpg", W, H, 55.75, 37.6, 0.0, None, "exif")
    lat, lon, src = estimate_object_position(north, box, cfg)
    assert lat > 55.75 and abs(haversine_m(55.75, 37.6, lat, lon) - 4.48) < 0.5 and src == "exif"
    east = FrameMeta("f", "f.jpg", W, H, 55.75, 37.6, 90.0, None, "exif")
    lat, lon, _ = estimate_object_position(east, box, cfg)
    assert lon > 37.6 and abs(lat - 55.75) < 1e-4


def test_object_position_without_gps_is_none(cfg):
    fm = FrameMeta("f", "f.jpg", W, H, None, None, None, None, "none")
    assert estimate_object_position(fm, BBox(0, 0, 10, 10), cfg)[:2] == (None, None)


def test_interpolate_heading_shortest_arc():
    track = [
        {"timestamp": "2026-01-01T10:00:00", "lat": 55.0, "lon": 37.0, "heading_deg": 350.0},
        {"timestamp": "2026-01-01T10:00:10", "lat": 55.001, "lon": 37.001, "heading_deg": 10.0},
    ]
    p = interpolate_position(track, "2026-01-01T10:00:05")
    assert abs(p["lat"] - 55.0005) < 1e-6
    assert min(abs(p["heading_deg"] - 0.0), abs(p["heading_deg"] - 360.0)) < 1.0
    assert interpolate_position(track, "2026-01-01T10:10:00") is None


def test_load_track_csv_tolerant(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("frame_id;latitude;longitude\na;55.1;37.1\nb;55.2;37.2\n", encoding="utf-8")
    tr = load_track_csv(p)
    assert len(tr) == 2 and tr[0]["lat"] == 55.1 and tr[1]["lon"] == 37.2


def test_build_frame_meta_synthetic_fallback(tmp_path, cfg):
    import cv2
    import numpy as np
    img = tmp_path / "x.jpg"
    cv2.imwrite(str(img), np.zeros((48, 64, 3), np.uint8))
    fm = build_frame_meta(img, 3, cfg, track=None)
    assert fm.width == 64 and fm.height == 48
    assert fm.location_source in ("synthetic", "none")
    if cfg["geo"]["fallback_no_gps"] == "synthetic":
        assert fm.lat is not None


def _obs(fid, cls, lat, lon, score=0.9, sharp=100.0, sev=Severity.OK, s=0.1):
    det = Detection(fid, 0 if cls == "cable_manhole" else 1, cls, score, BBox(10, 10, 60, 60))
    return Observation(fid, det, ConditionAssessment(sev, s, [], False), sharp, lat, lon, "exif")


def _dedup_cfg(cfg):
    return {**cfg, "geo": {**cfg["geo"], "dedup": {**cfg["geo"]["dedup"], "use_appearance": False}}}


def test_dedup_merges_nearby_same_class(cfg, classes):
    obs = [_obs("a", "cable_manhole", 55.75000, 37.6), _obs("b", "cable_manhole", 55.750015, 37.6)]   # ~1.7 м
    recs = merge_observations(obs, _dedup_cfg(cfg), classes)
    assert len(recs) == 1 and recs[0].n_observations == 2 and set(recs[0].frame_ids) == {"a", "b"}


def test_dedup_keeps_far_objects_separate(cfg, classes):
    obs = [_obs("a", "cable_manhole", 55.75000, 37.6), _obs("b", "cable_manhole", 55.75050, 37.6)]    # ~55 м
    assert len(merge_observations(obs, _dedup_cfg(cfg), classes)) == 2


def test_dedup_does_not_merge_different_classes(cfg, classes):
    obs = [_obs("a", "cable_manhole", 55.75, 37.6), _obs("b", "utility_pole", 55.75, 37.6)]
    assert len(merge_observations(obs, _dedup_cfg(cfg), classes)) == 2


def test_dedup_sorted_by_priority_and_ids(cfg, classes):
    bad = _obs("a", "cable_manhole", 55.75, 37.6, sev=Severity.DEFECT, s=0.9)
    good = _obs("b", "cable_manhole", 55.76, 37.6, sev=Severity.OK, s=0.0)
    recs = merge_observations([good, bad], _dedup_cfg(cfg), classes)
    assert recs[0].condition_score > recs[1].condition_score
    assert [r.object_id for r in recs] == ["OBJ-0001", "OBJ-0002"]


def test_dedup_median_position_robust_to_outlier(cfg, classes):
    obs = [_obs(str(i), "cable_manhole", 55.75 + d, 37.6) for i, d in enumerate([0, 0.000004, 0.000002, 0.00003])]
    r = merge_observations(obs, _dedup_cfg(cfg), classes)
    assert len(r) == 1 and abs(r[0].lat - 55.750003) < 5e-6
