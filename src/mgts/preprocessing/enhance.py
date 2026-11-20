"""Улучшение изображения перед детекцией (Агент 2, задача 2)."""
from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(img_bgr: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:  # CONTRACT
    """CLAHE по каналу L в пространстве LAB; цвета не искажать."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def auto_gamma(img_bgr: np.ndarray, target_mean: float = 110.0) -> np.ndarray:  # CONTRACT
    """Подобрать гамму так, чтобы средняя яркость приблизилась к target_mean (ограничить 0.5..2.0),
    применить через LUT. Для уже нормальных кадров — почти тождественное преобразование."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    gamma = 2.0 if mean <= 1.0 else np.log(target_mean / 255.0) / np.log(mean / 255.0)
    gamma = float(np.clip(gamma, 0.5, 2.0))
    lookup = np.clip(255.0 * (np.arange(256) / 255.0) ** gamma, 0, 255).astype(np.uint8)
    return cv2.LUT(img_bgr, lookup)


def denoise(img_bgr: np.ndarray, strength: int = 5) -> np.ndarray:  # CONTRACT
    """Шумоподавление (cv2.fastNlMeansDenoisingColored или билатеральный фильтр). Включается только по конфигу."""
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, strength, strength, 7, 21)


def enhance(img_bgr: np.ndarray, cfg: dict) -> np.ndarray:  # CONTRACT
    """Применить цепочку по cfg['preprocessing']['enhance'] (clahe, gamma, denoise).
    Вход не мутировать; вернуть новый массив того же размера и dtype uint8."""
    options = cfg.get("preprocessing", {}).get("enhance", {})
    out = img_bgr.copy()
    if options.get("clahe", False):
        out = apply_clahe(out, options.get("clahe_clip", 2.0), options.get("clahe_grid", 8))
    gamma = options.get("gamma")
    if gamma == "auto":
        out = auto_gamma(out, options.get("gamma_target_mean", 110.0))
    elif gamma is not None:
        lut = np.clip(255.0 * (np.arange(256) / 255.0) ** float(gamma), 0, 255).astype(np.uint8)
        out = cv2.LUT(out, lut)
    if options.get("denoise", False):
        out = denoise(out, options.get("denoise_strength", 5))
    return out
