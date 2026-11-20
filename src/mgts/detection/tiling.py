"""Тайловая нарезка для мелких объектов — собственная реализация идеи SAHI (Агент 2, задача 3)."""
from __future__ import annotations

import numpy as np

from ..schemas import Detection


def tile_slices(img_w: int, img_h: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:  # CONTRACT
    """Сетка окон (x1, y1, x2, y2) размером tile×tile с перекрытием overlap (0..1).
    Свойства (проверяются тестами): окна покрывают ВЕСЬ кадр; последний ряд/столбец прижат к краю
    (не выходит за кадр); если кадр меньше tile — одно окно на весь кадр; без дубликатов."""
    raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §3")


def merge_detections(dets: list[Detection], iou_thr: float = 0.5) -> list[Detection]:  # CONTRACT
    """Объединить детекции со всех тайлов + полнокадровой: NMS внутри класса по IoU, плюс слияние
    рамок, обрезанных границей тайла (если одна рамка почти целиком (>80% площади) лежит внутри другой
    того же класса — оставить большую). Вернуть отсортированно по score убыванию."""
    raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §3")


def detect_tiled(detector, img_bgr: np.ndarray, frame_id: str, tile: int, overlap: float, iou_thr: float) -> list[Detection]:  # CONTRACT
    """Прогнать detector.detect_raw(crop) по тайлам + полный кадр (уменьшенный), перевести рамки в координаты
    исходного кадра (прибавить смещение тайла), вызвать merge_detections."""
    raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §3")
