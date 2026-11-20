"""Классификатор дефектов по вырезкам (Агент 3, задача 2)."""
from __future__ import annotations

import csv
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import class_info, load_classes, resolve_path
from .geometry import analyze_hatch, corrosion_ratio, crack_score

logger = logging.getLogger(__name__)

class DefectClassifier:  # CONTRACT
    """Мультиметочный классификатор дефектов объекта: вырезка -> {код_дефекта: вероятность}.

    Бэкенды (cfg['condition']['classifier']['backbone']):
      - 'efficientnet_b0' | 'resnet18': torchvision, предобученные на ImageNet, голова под N кодов дефектов
        (мультиметка: BCEWithLogits); дообучение: сначала только голова (2–3 эпохи), затем всё с lr/10;
      - 'features_lr': FALLBACK без GPU — признаки (гистограммы HSV, LBP/края, выход geometry.*) +
        LogisticRegression/GradientBoosting из scikit-learn.
    Если весов нет — predict() возвращает {} (а scoring использует только геометрию), без исключений.
    """

    def __init__(self, cfg: dict, classes: dict | None = None) -> None:
        self._cfg = cfg.get("condition", {}).get("classifier", {})
        self._classes = classes or load_classes()
        self._model: dict[str, Any] | None = None
        weights = self._cfg.get("weights")
        if not weights:
            return
        try:
            with resolve_path(weights).open("rb") as model_file:
                model = pickle.load(model_file)
            if isinstance(model, dict) and isinstance(model.get("models"), dict) and model["models"]:
                self._model = model
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError, ModuleNotFoundError):
            logger.info("Не удалось загрузить веса классификатора %s; используются геометрические признаки", weights)
            logger.debug("Ошибка загрузки весов классификатора", exc_info=True)

    @property
    def is_ready(self) -> bool:
        """True, если веса загружены и можно вызывать predict."""
        return self._model is not None

    def predict(self, crop_bgr: np.ndarray, class_name: str) -> dict[str, float]:
        """Вероятности только для дефектов, применимых к class_name (classes.yaml -> defects у класса)."""
        if not self.is_ready or crop_bgr is None or crop_bgr.size == 0:
            return {}
        try:
            features = _extract_features(crop_bgr).reshape(1, -1)
            applicable = set(class_info(class_name, self._classes).get("defects", []))
            if "generic_damage" in self._model["models"]:
                applicable.add("generic_damage")
            output: dict[str, float] = {}
            for code in applicable:
                model = self._model["models"].get(code)
                if model is None:
                    continue
                probability = float(model.predict_proba(features)[0, 1])
                output[code] = float(np.clip(probability, 0.0, 1.0))
            return output
        except (cv2.error, ValueError, TypeError, KeyError, IndexError, AttributeError):
            logger.warning("Ошибка классификации объекта %s", class_name, exc_info=True)
            return {}

    def threshold_for(self, code: str, default: float = 0.5) -> float:
        """Порог дефекта, подобранный при обучении, или переданное значение по умолчанию."""
        if not self.is_ready:
            return default
        return float(self._model.get("thresholds", {}).get(code, default))


def _extract_features(crop_bgr: np.ndarray) -> np.ndarray:
    """Построить фиксированный вектор цветовых и геометрических признаков."""
    crop = cv2.resize(crop_bgr, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return np.concatenate((
        hist,
        np.array([edges.mean() / 255.0, gray.mean() / 255.0, gray.std() / 255.0,
                  crack_score(crop), corrosion_ratio(crop), analyze_hatch(crop).circularity], dtype=np.float32),
    )).astype(np.float32)


def train_classifier(cfg: dict, crops_dir: str | Path | None = None) -> Path:  # CONTRACT
    """Обучить классификатор на data/crops (+ синтетика Агента 1), сохранить веса в
    cfg['condition']['classifier']['weights'], вернуть путь. Требования:
      - разбиение train/val по ИСХОДНЫМ кадрам (без утечек); синтетику не класть в val, если есть реальные дефекты;
      - взвешенная функция потерь / sampler против дисбаланса; аугментации (flip, цвет, небольшой поворот);
      - вывести precision/recall по каждому коду дефекта, сохранить в models/condition_metrics.json;
      - выбрать порог по каждому коду дефекта, максимизирующий F2 (recall важнее), записать в тот же json;
      - уложиться во время: <= 10 минут на CPU при <= 5000 вырезок (маленький input_size, мало эпох).
    """
    classes = load_classes()
    root = Path(crops_dir or cfg.get("paths", {}).get("crops_dir", "data/crops"))
    index_path = root if root.is_file() else root / "index.csv"
    if not index_path.is_file():
        raise ValueError(f"Не найден файл разметки вырезок: {index_path} (ожидается index.csv)")
    with index_path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    samples = _load_labeled_samples(rows, index_path.parent, classes)
    if not samples:
        raise ValueError(f"В {index_path} нет размеченных и читаемых вырезок (нужны метки ok или код дефекта)")

    codes = sorted({code for item in samples for code in item["targets"]})
    models: dict[str, Any] = {}
    metrics: dict[str, dict[str, Any]] = {}
    seed = int(cfg.get("dataset", {}).get("seed", 42))
    for code in codes:
        code_samples = [item for item in samples if code in item["targets"]]
        labels = np.array([item["targets"][code] for item in code_samples], dtype=np.uint8)
        if len(np.unique(labels)) < 2:
            continue
        features = np.stack([item["features"] for item in code_samples])
        groups = np.array([item["frame_id"] for item in code_samples])
        validation_mask = np.zeros(len(labels), dtype=bool)
        real_indices = np.array([i for i, item in enumerate(code_samples) if not item["synthetic"]], dtype=int)
        if len(np.unique(groups[real_indices])) > 1:
            _, val_idx = next(GroupShuffleSplit(test_size=0.2, n_splits=1,
                                                 random_state=seed).split(real_indices, labels[real_indices],
                                                                         groups[real_indices]))
            validation_mask[real_indices[val_idx]] = True
        train_mask = ~validation_mask
        validation_available = bool(validation_mask.any() and len(np.unique(labels[train_mask])) >= 2)
        if not validation_available:
            logger.warning("Для %s нет независимой validation-выборки; метрики не рассчитываются", code)
            train_mask = np.ones(len(labels), dtype=bool)
            validation_mask = np.zeros(len(labels), dtype=bool)
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500,
                                                                    random_state=seed))
        model.fit(features[train_mask], labels[train_mask])
        if validation_available:
            probabilities = model.predict_proba(features[validation_mask])[:, 1]
            threshold, precision, recall = _best_f2_threshold(labels[validation_mask], probabilities)
        else:
            threshold, precision, recall = 0.5, None, None
        models[code] = model
        metrics[code] = {"precision": precision, "recall": recall, "threshold": threshold,
                         "validation_available": validation_available}
    if not models:
        raise ValueError("Для обучения нужны хотя бы один положительный и один отрицательный пример дефекта")

    bundle = {"version": 1, "models": models,
              "thresholds": {code: data["threshold"] for code, data in metrics.items()}}
    weights_path = resolve_path(cfg.get("condition", {}).get("classifier", {}).get("weights", "models/condition_clf.pkl"))
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    with weights_path.open("wb") as target:
        pickle.dump(bundle, target, protocol=pickle.HIGHEST_PROTOCOL)
    models_dir = resolve_path(cfg.get("paths", {}).get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    with (models_dir / "condition_metrics.json").open("w", encoding="utf-8") as target:
        json.dump({"per_defect": metrics}, target, ensure_ascii=False, indent=2)
    return weights_path


def _load_labeled_samples(rows: list[dict[str, str]], index_dir: Path, classes: dict) -> list[dict[str, Any]]:
    """Прочитать только пригодные вырезки с метками состояния и собрать их признаки."""
    class_codes = {item["name"]: set(item.get("defects", [])) for item in classes.get("classes", [])}
    samples = []
    for row in rows:
        label = (row.get("label") or "").strip()
        class_name = (row.get("class_name") or "").strip()
        if not label or label.lower() in {"unlabeled", "unknown"} or class_name not in class_codes:
            continue
        image_path = Path(row.get("path") or "")
        if not image_path.is_absolute():
            local_path = index_dir / image_path
            image_path = local_path if local_path.is_file() else resolve_path(image_path)
        crop = cv2.imread(str(image_path))
        if crop is None or crop.size == 0:
            continue
        codes = class_codes[class_name]
        if label == "ok":
            targets = {code: 0 for code in codes}
            targets["generic_damage"] = 0
        elif label in codes:
            targets = {code: int(code == label) for code in codes}
            targets["generic_damage"] = 1
        elif label == "generic_damage" or label == "defect":
            targets = {"generic_damage": 1}
        else:
            continue
        samples.append({"features": _extract_features(crop), "targets": targets,
                        "frame_id": row.get("frame_id") or image_path.stem,
                        "synthetic": str(row.get("synthetic", "0")).strip().lower() in {"1", "true", "yes"}})
    return samples


def _best_f2_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, float]:
    """Подобрать порог по F2 (вес recall выше precision)."""
    best = (0.5, 0.0, 0.0, -1.0)
    for threshold in np.arange(0.05, 1.0, 0.05):
        predictions = probabilities >= threshold
        precision = float(precision_score(labels, predictions, zero_division=0))
        recall = float(recall_score(labels, predictions, zero_division=0))
        f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
        if (f2, recall) > (best[3], best[2]):
            best = (float(threshold), precision, recall, f2)
    return best[0], best[1], best[2]
