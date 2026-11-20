"""Обучение детектора (Агент 2, задача 5)."""
from __future__ import annotations

from pathlib import Path


def train_detector(cfg: dict, data_yaml: str | Path | None = None) -> Path:  # CONTRACT
    """Дообучить YOLO (transfer learning) на data/processed и скопировать лучшие веса в
    cfg['detection']['weights'] (models/detector.pt). Вернуть путь к весам.

    Детали:
      - base_model из конфига; imgsz, epochs, batch, device — из конфига;
      - device='auto': GPU если есть; на CPU автоматически уменьшить imgsz до 640, epochs до 15 и взять yolov8n;
      - аугментации ultralytics оставить включёнными (mosaic, hsv, fliplr — НО fliplr можно, flipud отключить);
      - дисбаланс классов: при сильном перекосе (редкий класс < 5% объектов) увеличить cls-loss weight
        или дублировать кадры с редким классом (через список train.txt);
      - ранняя остановка (patience=15), seed фиксирован, логи в runs/;
      - после обучения сразу вызвать evaluate (см. evaluate.py) и сохранить metrics в models/detector_metrics.json.
    """
    raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §5")
