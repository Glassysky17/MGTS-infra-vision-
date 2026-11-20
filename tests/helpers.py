"""Генераторы синтетических изображений для тестов (без внешних файлов)."""
from __future__ import annotations

import math

import cv2
import numpy as np


def noisy_background(w: int = 400, h: int = 400, seed: int = 0, level: int = 120) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), level, np.uint8)
    noise = rng.normal(0, 6, (h, w, 1)).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def draw_pole(w: int = 400, h: int = 400, tilt_deg: float = 0.0, length: int = 260, width: int = 22, seed: int = 0) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Тёмная опора, основание в (w/2, h-40); наклон вправо (вершина смещается вправо) при tilt_deg > 0.
    Возвращает (изображение, bbox xyxy)."""
    img = noisy_background(w, h, seed)
    bx, by = w / 2, h - 40
    t = math.radians(tilt_deg)
    dx, dy = math.sin(t), -math.cos(t)          # направление от основания к вершине
    nx, ny = math.cos(t), math.sin(t)           # нормаль
    hw = width / 2
    pts = np.array([
        [bx - nx * hw, by - ny * hw], [bx + nx * hw, by + ny * hw],
        [bx + dx * length + nx * hw, by + dy * length + ny * hw],
        [bx + dx * length - nx * hw, by + dy * length - ny * hw],
    ], np.float32)
    cv2.fillPoly(img, [pts.astype(np.int32)], (45, 50, 60))
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return img, (int(x1), int(y1), int(x2), int(y2))


def draw_hatch(w: int = 400, h: int = 400, seed: int = 0) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Круглый люк на асфальте. Возвращает (изображение, bbox)."""
    img = noisy_background(w, h, seed, level=110)
    c, r = (w // 2, h // 2), 90
    cv2.circle(img, c, r, (60, 62, 66), -1)
    cv2.circle(img, c, r, (30, 30, 30), 4)
    cv2.circle(img, c, r - 25, (75, 78, 82), 3)
    return img, (c[0] - r, c[1] - r, c[0] + r, c[1] + r)
