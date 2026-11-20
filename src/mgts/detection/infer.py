"""Инференс детектора (Агент 2, задачи 4–5).

Единый интерфейс для всех бэкендов — остальной код (pipeline) знает только `build_detector(cfg)`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..config import class_names, load_classes, resolve_path
from ..schemas import BBox, Detection
from .tiling import detect_tiled

logger = logging.getLogger(__name__)


class DetectorProtocol(Protocol):  # CONTRACT
    def detect(self, img_bgr: np.ndarray, frame_id: str, image_path: str | None = None) -> list[Detection]:
        """Детекции на кадре в АБСОЛЮТНЫХ пикселях исходного изображения."""
        ...


class YoloDetector:
    """Обёртка над ultralytics.YOLO (задача 4).

    __init__(cfg): загрузить обученные веса cfg['detection']['weights']; base_model используется только при обучении,
        device по cfg ('auto' -> cuda если есть, иначе cpu), классы из configs/classes.yaml
        (если имена в модели расходятся с classes.yaml — приоритет у classes.yaml по class_id, залогировать).
    detect(): при detection.tiling.enabled и большой картинке (min(w,h) >= min_image_side) — detect_tiled,
        иначе обычный прогон. Порог conf/iou — из конфига. Результат: list[Detection], bbox клипнуть по кадру.
    detect_raw(img): один прогон без тайлинга (используется detect_tiled).
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["detection"]
        weights = resolve_path(self.cfg.get("weights", ""))
        if not weights.is_file():
            raise RuntimeError(f"Нет обученных весов детектора: {weights}; сначала обучите модель: mgts train-det")
        self.weights_path = weights
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Для backend=yolo установите ultralytics") from exc
        self.model = YOLO(str(self.weights_path))
        classes_path = cfg.get("paths", {}).get("classes_file")
        self.names = class_names(load_classes(resolve_path(classes_path)) if classes_path else None)
        self.device = self.cfg.get("device", "auto")
        if self.device == "auto":
            try:
                import torch
                self.device = "0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        model_names = getattr(self.model, "names", {})
        for class_id, expected in enumerate(self.names):
            if isinstance(model_names, dict):
                actual = model_names.get(class_id)
            else:
                actual = model_names[class_id] if class_id < len(model_names) else None
            if actual is not None and actual != expected:
                logger.warning("Класс %s модели заменён на имя из classes.yaml: %s", actual, expected)

    def detect_raw(self, img_bgr: np.ndarray, frame_id: str = "") -> list[Detection]:
        """Получить сырые детекции на полном кадре или тайле и обрезать рамки по изображению."""
        if img_bgr.size == 0:
            return []
        result = self.model.predict(
            img_bgr,
            imgsz=self.cfg.get("imgsz", 640),
            conf=self.cfg.get("conf", 0.25),
            iou=self.cfg.get("iou", 0.5),
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None:
            return []
        coords = _as_numpy(boxes.xyxy)
        scores = _as_numpy(boxes.conf)
        class_ids = _as_numpy(boxes.cls)
        height, width = img_bgr.shape[:2]
        detections: list[Detection] = []
        for coords_one, score, class_id_value in zip(coords, scores, class_ids):
            class_id = int(class_id_value)
            bbox = BBox(*(float(value) for value in coords_one)).clip(width, height)
            if bbox.area <= 0:
                continue
            name = self.names[class_id] if 0 <= class_id < len(self.names) else f"class_{class_id}"
            detections.append(Detection(frame_id, class_id, name, float(score), bbox))
        return detections

    def detect(self, img_bgr: np.ndarray, frame_id: str, image_path: str | None = None) -> list[Detection]:
        """Выполнить инференс с тайлингом для больших кадров, если он включён в конфиге."""
        tiling = self.cfg.get("tiling", {})
        height, width = img_bgr.shape[:2]
        if tiling.get("enabled", False) and min(height, width) >= tiling.get("min_image_side", 1280):
            return detect_tiled(
                self, img_bgr, frame_id, tiling.get("tile_size", 640),
                tiling.get("overlap", 0.2), self.cfg.get("iou", 0.5),
            )
        return self.detect_raw(img_bgr, frame_id)


class DummyDetector:
    """«Детектор» для интеграции ДО обучения модели (СДЕЛАТЬ ПЕРВЫМ, 10 минут).

    Читает GT-разметку YOLO рядом с картинкой (…/images/x.jpg -> …/labels/x.txt; путь передаётся
    в image_path), переводит в Detection со score=0.9. Если разметки нет — возвращает [].
    Нужен, чтобы Агенты 3 и 4 тестировали свои части, не дожидаясь обучения.
    """

    def __init__(self, cfg: dict) -> None:
        classes_path = cfg.get("paths", {}).get("classes_file")
        self.names = class_names(load_classes(resolve_path(classes_path)) if classes_path else None)

    def detect(self, img_bgr: np.ndarray, frame_id: str, image_path: str | None = None) -> list[Detection]:
        """Преобразовать YOLO-разметку кадра в контрактные детекции."""
        if not image_path:
            return []
        image_path_obj = Path(image_path)
        label_path = image_path_obj.parent.parent / "labels" / f"{image_path_obj.stem}.txt"
        if not label_path.is_file():
            return []
        height, width = img_bgr.shape[:2]
        detections: list[Detection] = []
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            logger.warning("Не удалось прочитать YOLO-разметку %s", label_path, exc_info=True)
            return []
        for line in lines:
            fields = line.split()
            if len(fields) != 5:
                continue
            try:
                class_id = int(fields[0])
                xc, yc, box_w, box_h = map(float, fields[1:])
            except ValueError:
                continue
            if class_id < 0 or class_id >= len(self.names) or not np.isfinite([xc, yc, box_w, box_h]).all() or box_w <= 0 or box_h <= 0:
                continue
            bbox = BBox.from_xywhn(xc, yc, box_w, box_h, width, height).clip(width, height)
            if bbox.area <= 0:
                continue
            name = self.names[class_id]
            detections.append(Detection(frame_id, class_id, name, 0.9, bbox))
        return detections


def build_detector(cfg: dict) -> DetectorProtocol:  # CONTRACT
    """Фабрика: cfg['detection']['backend'] == 'yolo' -> YoloDetector, 'dummy' -> DummyDetector.
    Если backend='yolo', но весов нет и base_model недоступна — понятная ошибка с подсказкой
    «обучите модель: mgts train-det» (а не трейсбек из ultralytics)."""
    backend = cfg.get("detection", {}).get("backend", "yolo")
    if backend == "dummy":
        return DummyDetector(cfg)
    if backend == "yolo":
        return YoloDetector(cfg)
    raise ValueError(f"Неизвестный detection.backend: {backend}")


def _as_numpy(values: Any) -> np.ndarray:
    """Преобразовать tensor или массив Ultralytics в NumPy без обязательной зависимости от torch."""
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values)
