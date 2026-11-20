"""Инференс детектора (Агент 2, задачи 4–5).

Единый интерфейс для всех бэкендов — остальной код (pipeline) знает только `build_detector(cfg)`.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..schemas import Detection


class DetectorProtocol(Protocol):  # CONTRACT
    def detect(self, img_bgr: np.ndarray, frame_id: str, image_path: str | None = None) -> list[Detection]:
        """Детекции на кадре в АБСОЛЮТНЫХ пикселях исходного изображения."""
        ...


class YoloDetector:
    """Обёртка над ultralytics.YOLO (задача 4).

    __init__(cfg): загрузить веса cfg['detection']['weights'] (если файла нет — base_model),
        device по cfg ('auto' -> cuda если есть, иначе cpu), классы из configs/classes.yaml
        (если имена в модели расходятся с classes.yaml — приоритет у classes.yaml по class_id, залогировать).
    detect(): при detection.tiling.enabled и большой картинке (min(w,h) >= min_image_side) — detect_tiled,
        иначе обычный прогон. Порог conf/iou — из конфига. Результат: list[Detection], bbox клипнуть по кадру.
    detect_raw(img): один прогон без тайлинга (используется detect_tiled).
    """

    def __init__(self, cfg: dict):
        raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §4")

    def detect_raw(self, img_bgr: np.ndarray, frame_id: str = "") -> list[Detection]:
        raise NotImplementedError

    def detect(self, img_bgr: np.ndarray, frame_id: str, image_path: str | None = None) -> list[Detection]:
        raise NotImplementedError


class DummyDetector:
    """«Детектор» для интеграции ДО обучения модели (СДЕЛАТЬ ПЕРВЫМ, 10 минут).

    Читает GT-разметку YOLO рядом с картинкой (…/images/x.jpg -> …/labels/x.txt; путь передаётся
    в image_path), переводит в Detection со score=0.9. Если разметки нет — возвращает [].
    Нужен, чтобы Агенты 3 и 4 тестировали свои части, не дожидаясь обучения.
    """

    def __init__(self, cfg: dict):
        raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §0")

    def detect(self, img_bgr: np.ndarray, frame_id: str, image_path: str | None = None) -> list[Detection]:
        raise NotImplementedError


def build_detector(cfg: dict) -> DetectorProtocol:  # CONTRACT
    """Фабрика: cfg['detection']['backend'] == 'yolo' -> YoloDetector, 'dummy' -> DummyDetector.
    Если backend='yolo', но весов нет и base_model недоступна — понятная ошибка с подсказкой
    «обучите модель: mgts train-det» (а не трейсбек из ultralytics)."""
    raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §0/§4")
