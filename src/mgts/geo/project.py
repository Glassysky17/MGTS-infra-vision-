"""Проекция пикселя на землю -> координаты объекта (Агент 4, задача 2).

Используем модель камеры-обскуры (pinhole) и плоскость земли — классическая геометрия компьютерной графики.
"""
from __future__ import annotations

from typing import Optional

from ..schemas import BBox, FrameMeta


def pixel_to_ground_offset(  # CONTRACT
    u: float, v: float, img_w: int, img_h: int, cam: dict
) -> Optional[tuple[float, float]]:
    """Луч через пиксель (u, v) -> точка пересечения с плоскостью земли.
    cam: {'height_m', 'pitch_deg' (вниз +), 'hfov_deg'}. Фокус: f = (img_w/2) / tan(hfov/2), главная точка — центр кадра.
    Возврат: (forward_m, right_m) в системе автомобиля, или None, если луч не попадает в землю
    (v выше горизонта) либо расстояние > 60 м (слишком неточно).
    Проверки (юнит-тест): пиксель в центре кадра при pitch=0 -> луч параллелен земле -> None;
    пиксель ниже центра -> forward тем меньше, чем ниже v; симметрия по u."""
    raise NotImplementedError("Агент 4: tasks/agent_4_integration.md §2")


def estimate_object_position(  # CONTRACT
    frame: FrameMeta, bbox: BBox, cfg: dict
) -> tuple[Optional[float], Optional[float], str]:
    """Координаты ОБЪЕКТА: берём точку основания bbox (bottom_center; для колодцев — центр bbox),
    pixel_to_ground_offset -> (forward, right) -> повернуть на heading_deg -> (север, восток) ->
    utils.offset_latlon от позиции автомобиля (с учётом geo.camera.mount_offset_m).
    Возврат (lat, lon, location_source). Если у кадра нет lat/lon -> (None, None, 'none').
    Если проекция невозможна (None) — вернуть позицию автомобиля и source + '_approx'
    (например 'exif_approx'): оператор поймёт, что точность ±несколько метров."""
    raise NotImplementedError("Агент 4: tasks/agent_4_integration.md §2")
