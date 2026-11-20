"""Классификатор дефектов по вырезкам (Агент 3, задача 2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class DefectClassifier:  # CONTRACT
    """Мультиметочный классификатор дефектов объекта: вырезка -> {код_дефекта: вероятность}.

    Бэкенды (cfg['condition']['classifier']['backbone']):
      - 'efficientnet_b0' | 'resnet18': torchvision, предобученные на ImageNet, голова под N кодов дефектов
        (мультиметка: BCEWithLogits); дообучение: сначала только голова (2–3 эпохи), затем всё с lr/10;
      - 'features_lr': FALLBACK без GPU — признаки (гистограммы HSV, LBP/края, выход geometry.*) +
        LogisticRegression/GradientBoosting из scikit-learn.
    Если весов нет — predict() возвращает {} (а scoring использует только геометрию), без исключений.
    """

    def __init__(self, cfg: dict, classes: dict | None = None):
        raise NotImplementedError("Агент 3: tasks/agent_3_condition.md §2")

    @property
    def is_ready(self) -> bool:
        """True, если веса загружены и можно вызывать predict."""
        raise NotImplementedError

    def predict(self, crop_bgr: np.ndarray, class_name: str) -> dict[str, float]:
        """Вероятности только для дефектов, применимых к class_name (classes.yaml -> defects у класса)."""
        raise NotImplementedError("Агент 3: tasks/agent_3_condition.md §2")


def train_classifier(cfg: dict, crops_dir: str | Path | None = None) -> Path:  # CONTRACT
    """Обучить классификатор на data/crops (+ синтетика Агента 1), сохранить веса в
    cfg['condition']['classifier']['weights'], вернуть путь. Требования:
      - разбиение train/val по ИСХОДНЫМ кадрам (без утечек); синтетику не класть в val, если есть реальные дефекты;
      - взвешенная функция потерь / sampler против дисбаланса; аугментации (flip, цвет, небольшой поворот);
      - вывести precision/recall по каждому коду дефекта, сохранить в models/condition_metrics.json;
      - выбрать порог по каждому коду дефекта, максимизирующий F2 (recall важнее), записать в тот же json;
      - уложиться во время: <= 10 минут на CPU при <= 5000 вырезок (маленький input_size, мало эпох).
    """
    raise NotImplementedError("Агент 3: tasks/agent_3_condition.md §2")
