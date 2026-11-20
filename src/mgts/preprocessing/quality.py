"""Оценка качества кадров (Агент 2, задача 1)."""
from __future__ import annotations

import numpy as np


def blur_score(img_bgr: np.ndarray) -> float:  # CONTRACT
    """Мера резкости: дисперсия лапласиана серого изображения (больше = резче).
    Для устойчивости к разрешению считать на уменьшенной копии (длинная сторона 640).
    Пустое/чёрное изображение -> 0.0."""
    raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §1")


def is_sharp(img_bgr: np.ndarray, threshold: float) -> bool:  # CONTRACT
    """blur_score(img) >= threshold."""
    raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §1")


def brightness_stats(img_bgr: np.ndarray) -> dict[str, float]:  # CONTRACT
    """{'mean': .., 'p5': .., 'p95': .., 'underexposed': bool, 'overexposed': bool} по яркости (0..255).
    Нужно для решения, применять ли гамма-коррекцию и для пометки кадров как «плохих»."""
    raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §1")


def select_frames(paths: list[str], threshold: float, max_keep: int | None = None) -> list[str]:  # CONTRACT
    """Отфильтровать размытые кадры. Если подряд идущие кадры почти идентичны (разность гистограмм
    ниже порога) — оставить самый резкий из группы. Порядок сохранить."""
    raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §1")


def calibrate_blur_threshold(paths: list[str], percentile: float = 15.0) -> float:  # CONTRACT
    """Подобрать порог резкости по набору реальных кадров (перцентиль распределения blur_score).
    Использовать на день 0 и записать значение в configs/default.yaml -> preprocessing.blur_threshold."""
    raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §1")
