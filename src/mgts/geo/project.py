"""Проекция пикселя на землю -> координаты объекта (Агент 4, задача 2).

Используем модель камеры-обскуры (pinhole) и плоскость земли — классическая геометрия компьютерной графики.
"""
from __future__ import annotations

import math
from typing import Optional

from ..schemas import BBox, FrameMeta
from ..utils import offset_latlon


def pixel_to_ground_offset(  # CONTRACT
    u: float, v: float, img_w: int, img_h: int, cam: dict
) -> Optional[tuple[float, float]]:
    """Луч через пиксель (u, v) -> точка пересечения с плоскостью земли.
    cam: {'height_m', 'pitch_deg' (вниз +), 'hfov_deg'}. Фокус: f = (img_w/2) / tan(hfov/2), главная точка — центр кадра.
    Возврат: (forward_m, right_m) в системе автомобиля, или None, если луч не попадает в землю
    (v выше горизонта) либо расстояние > 60 м (слишком неточно).
    Проверки (юнит-тест): пиксель в центре кадра при pitch=0 -> луч параллелен земле -> None;
    пиксель ниже центра -> forward тем меньше, чем ниже v; симметрия по u."""
    height = float(cam["height_m"])
    pitch = math.radians(float(cam["pitch_deg"]))
    hfov = float(cam["hfov_deg"])
    if img_w <= 0 or img_h <= 0 or height <= 0 or not 0 < hfov < 180:
        return None
    focal = (img_w / 2) / math.tan(math.radians(hfov / 2))
    dx = (u - img_w / 2) / focal
    dy = (v - img_h / 2) / focal
    down = dy * math.cos(pitch) + math.sin(pitch)
    forward_ray = -dy * math.sin(pitch) + math.cos(pitch)
    if down <= 1e-9:
        return None
    scale = height / down
    forward, right = scale * forward_ray, scale * dx
    if forward <= 0 or forward > float(cam.get("max_distance_m", 60.0)):
        return None
    return forward, right


def estimate_object_position(  # CONTRACT
    frame: FrameMeta, bbox: BBox, cfg: dict
) -> tuple[Optional[float], Optional[float], str]:
    """Координаты ОБЪЕКТА: берём точку основания bbox (bottom_center; для колодцев — центр bbox),
    pixel_to_ground_offset -> (forward, right) -> повернуть на heading_deg -> (север, восток) ->
    utils.offset_latlon от позиции автомобиля (с учётом geo.camera.mount_offset_m).
    Возврат (lat, lon, location_source). Если у кадра нет lat/lon -> (None, None, 'none').
    Если проекция невозможна (None) — вернуть позицию автомобиля и source + '_approx'
    (например 'exif_approx'): оператор поймёт, что точность ±несколько метров."""
    if frame.lat is None or frame.lon is None:
        return None, None, "none"
    camera = cfg.get("geo", {}).get("camera", {})
    pixel = bbox.bottom_center
    offset = pixel_to_ground_offset(*pixel, frame.width, frame.height, camera)
    source = frame.location_source or "none"
    if offset is None:
        return frame.lat, frame.lon, f"{source}_approx"
    forward, right = offset
    mount = camera.get("mount_offset_m", [0.0, 0.0])
    forward += float(mount[0])
    right += float(mount[1])
    heading = math.radians(float(frame.heading_deg or 0.0))
    north = forward * math.cos(heading) - right * math.sin(heading)
    east = forward * math.sin(heading) + right * math.cos(heading)
    lat, lon = offset_latlon(frame.lat, frame.lon, north, east)
    return lat, lon, source
