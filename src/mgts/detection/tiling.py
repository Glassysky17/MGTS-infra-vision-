"""Тайловая нарезка для мелких объектов — собственная реализация идеи SAHI (Агент 2, задача 3)."""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from ..schemas import BBox, Detection


def tile_slices(img_w: int, img_h: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:  # CONTRACT
    """Сетка окон (x1, y1, x2, y2) размером tile×tile с перекрытием overlap (0..1).
    Свойства (проверяются тестами): окна покрывают ВЕСЬ кадр; последний ряд/столбец прижат к краю
    (не выходит за кадр); если кадр меньше tile — одно окно на весь кадр; без дубликатов."""
    if img_w <= 0 or img_h <= 0:
        return []
    if tile <= 0:
        raise ValueError("tile must be positive")
    if not 0 <= overlap <= 1:
        raise ValueError("overlap must be in [0, 1]")
    if img_w <= tile and img_h <= tile:
        return [(0, 0, img_w, img_h)]
    step = max(1, int(tile * (1 - overlap)))

    def starts(side: int) -> list[int]:
        if side <= tile:
            return [0]
        offsets = list(range(0, side - tile + 1, step))
        if offsets[-1] + tile < side:
            offsets.append(side - tile)
        return offsets

    return list(dict.fromkeys(
        (x, y, min(x + tile, img_w), min(y + tile, img_h))
        for y in starts(img_h)
        for x in starts(img_w)
    ))


def merge_detections(dets: list[Detection], iou_thr: float = 0.5) -> list[Detection]:  # CONTRACT
    """Объединить детекции со всех тайлов + полнокадровой: NMS внутри класса по IoU, плюс слияние
    рамок, обрезанных границей тайла (если одна рамка почти целиком (>80% площади) лежит внутри другой
    того же класса — оставить большую). Вернуть отсортированно по score убыванию."""
    by_class: dict[int, list[Detection]] = {}
    for detection in dets:
        by_class.setdefault(detection.class_id, []).append(detection)

    merged: list[Detection] = []
    for candidates in by_class.values():
        ordered = sorted(candidates, key=lambda detection: detection.score, reverse=True)
        not_contained = [
            small for small in ordered
            if not any(
                large.bbox.area > small.bbox.area
                and small.bbox.area > 0
                and _intersection_area(small, large) / small.bbox.area >= 0.8
                for large in ordered
            )
        ]
        kept: list[Detection] = []
        for candidate in not_contained:
            if all(candidate.bbox.iou(previous.bbox) <= iou_thr for previous in kept):
                kept.append(candidate)
        merged.extend(kept)
    return sorted(merged, key=lambda detection: detection.score, reverse=True)


def _intersection_area(first: Detection, second: Detection) -> float:
    x1, y1 = max(first.bbox.x1, second.bbox.x1), max(first.bbox.y1, second.bbox.y1)
    x2, y2 = min(first.bbox.x2, second.bbox.x2), min(first.bbox.y2, second.bbox.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def detect_tiled(detector, img_bgr: np.ndarray, frame_id: str, tile: int, overlap: float, iou_thr: float) -> list[Detection]:  # CONTRACT
    """Прогнать detector.detect_raw(crop) по тайлам + полный кадр (уменьшенный), перевести рамки в координаты
    исходного кадра (прибавить смещение тайла), вызвать merge_detections."""
    height, width = img_bgr.shape[:2]
    if height <= 0 or width <= 0:
        return []
    detections: list[Detection] = []
    for x1, y1, x2, y2 in tile_slices(width, height, tile, overlap):
        for detection in detector.detect_raw(img_bgr[y1:y2, x1:x2], frame_id):
            bbox = detection.bbox
            detections.append(replace(
                detection,
                bbox=BBox(bbox.x1 + x1, bbox.y1 + y1, bbox.x2 + x1, bbox.y2 + y1).clip(width, height),
            ))

    scale = min(1.0, tile / max(height, width))
    if scale < 1.0:
        reduced = cv2.resize(
            img_bgr, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
        )
    else:
        reduced = img_bgr
    scale_x = reduced.shape[1] / width
    scale_y = reduced.shape[0] / height
    for detection in detector.detect_raw(reduced, frame_id):
        bbox = detection.bbox
        detections.append(replace(
            detection,
            bbox=BBox(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y).clip(width, height),
        ))
    return merge_detections(detections, iou_thr)
