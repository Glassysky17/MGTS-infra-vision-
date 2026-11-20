"""Улучшение изображения перед детекцией (Агент 2, задача 2)."""
from __future__ import annotations

import numpy as np


def apply_clahe(img_bgr: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:  # CONTRACT
    """CLAHE по каналу L в пространстве LAB; цвета не искажать."""
    raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §2")


def auto_gamma(img_bgr: np.ndarray, target_mean: float = 110.0) -> np.ndarray:  # CONTRACT
    """Подобрать гамму так, чтобы средняя яркость приблизилась к target_mean (ограничить 0.5..2.0),
    применить через LUT. Для уже нормальных кадров — почти тождественное преобразование."""
    raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §2")


def denoise(img_bgr: np.ndarray, strength: int = 5) -> np.ndarray:  # CONTRACT
    """Шумоподавление (cv2.fastNlMeansDenoisingColored или билатеральный фильтр). Включается только по конфигу."""
    raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §2")


def enhance(img_bgr: np.ndarray, cfg: dict) -> np.ndarray:  # CONTRACT
    """Применить цепочку по cfg['preprocessing']['enhance'] (clahe, gamma, denoise).
    Вход не мутировать; вернуть новый массив того же размера и dtype uint8."""
    raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §2")
