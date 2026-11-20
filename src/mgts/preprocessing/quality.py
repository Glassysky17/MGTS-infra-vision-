"""Оценка качества кадров (Агент 2, задача 1)."""
from __future__ import annotations

import cv2
import numpy as np

from ..utils import read_image


def blur_score(img_bgr: np.ndarray) -> float:  # CONTRACT
    """Мера резкости: дисперсия лапласиана серого изображения (больше = резче).
    Для устойчивости к разрешению считать на уменьшенной копии (длинная сторона 640).
    Пустое/чёрное изображение -> 0.0."""
    if img_bgr.size == 0 or img_bgr.ndim < 2 or min(img_bgr.shape[:2]) == 0:
        return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    height, width = gray.shape[:2]
    longest = max(height, width)
    if longest > 640:
        scale = 640 / longest
        gray = cv2.resize(gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_sharp(img_bgr: np.ndarray, threshold: float) -> bool:  # CONTRACT
    """blur_score(img) >= threshold."""
    return blur_score(img_bgr) >= threshold


def brightness_stats(img_bgr: np.ndarray) -> dict[str, float]:  # CONTRACT
    """{'mean': .., 'p5': .., 'p95': .., 'underexposed': bool, 'overexposed': bool} по яркости (0..255).
    Нужно для решения, применять ли гамма-коррекцию и для пометки кадров как «плохих»."""
    if img_bgr.size == 0 or img_bgr.ndim < 2 or min(img_bgr.shape[:2]) == 0:
        return {"mean": 0.0, "p5": 0.0, "p95": 0.0, "underexposed": True, "overexposed": False}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    mean, p5, p95 = (float(v) for v in (gray.mean(), *np.percentile(gray, (5, 95))))
    return {"mean": mean, "p5": p5, "p95": p95, "underexposed": p95 < 70, "overexposed": p5 > 200}


def select_frames(paths: list[str], threshold: float, max_keep: int | None = None) -> list[str]:  # CONTRACT
    """Отфильтровать размытые кадры. Если подряд идущие кадры почти идентичны (разность гистограмм
    ниже порога) — оставить самый резкий из группы. Порядок сохранить."""
    candidates: list[tuple[int, str, float, np.ndarray]] = []
    for index, path in enumerate(paths):
        try:
            image = read_image(path)
        except (OSError, ValueError):
            continue
        score = blur_score(image)
        if score >= threshold:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist)
            candidates.append((index, path, score, hist))

    kept: list[tuple[str, float]] = []
    group: list[tuple[int, str, float, np.ndarray]] = []
    for candidate in candidates:
        if group and candidate[0] == group[-1][0] + 1 and cv2.compareHist(
            group[-1][3], candidate[3], cv2.HISTCMP_CORREL
        ) > 0.98:
            group.append(candidate)
            continue
        if group:
            best = max(group, key=lambda item: item[2])
            kept.append((best[1], best[2]))
        group = [candidate]
    if group:
        best = max(group, key=lambda item: item[2])
        kept.append((best[1], best[2]))

    if max_keep is not None:
        if max_keep <= 0:
            return []
        if len(kept) > max_keep:
            indexes = np.linspace(0, len(kept) - 1, max_keep, dtype=int)
            kept = [kept[i] for i in indexes]
    return [path for path, _ in kept]


def calibrate_blur_threshold(paths: list[str], percentile: float = 15.0) -> float:  # CONTRACT
    """Подобрать порог резкости по набору реальных кадров (перцентиль распределения blur_score).
    Использовать на день 0 и записать значение в configs/default.yaml -> preprocessing.blur_threshold."""
    scores: list[float] = []
    for path in paths:
        try:
            scores.append(blur_score(read_image(path)))
        except (OSError, ValueError):
            continue
    return float(np.percentile(scores, percentile)) if scores else 0.0
