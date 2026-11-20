"""Обучение детектора (Агент 2, задача 5)."""
from __future__ import annotations

import json
import logging
import math
import shutil
import tempfile
from pathlib import Path

import yaml

from ..config import resolve_path

logger = logging.getLogger(__name__)


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
    data_path = resolve_path(data_yaml) if data_yaml is not None else resolve_path(cfg["paths"]["processed_dir"]) / "data.yaml"
    if not data_path.is_file():
        raise FileNotFoundError(f"YOLO data config not found: {data_path}")
    det_cfg = cfg["detection"]
    target = resolve_path(det_cfg.get("weights", "models/detector.pt"))
    configured = resolve_path(det_cfg.get("base_model", ""))
    existing_weights = resolve_path(det_cfg.get("weights", ""))
    source = existing_weights if existing_weights.is_file() else configured
    device = det_cfg.get("device", "auto")
    auto_device = device == "auto"
    if auto_device:
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if auto_device and device == "cpu":
        imgsz, epochs = 640, 15
        nano_weights = resolve_path("models/yolov8n.pt")
        if nano_weights.is_file() and not existing_weights.is_file():
            source = nano_weights
    else:
        imgsz, epochs = det_cfg.get("imgsz", 640), det_cfg.get("epochs", 15)
    if not source.is_file():
        raise RuntimeError(f"Нет локальных весов для обучения: {source}; скачайте веса отдельно")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Для обучения установите optional dependency ultralytics") from exc
    model = YOLO(str(source))
    with tempfile.TemporaryDirectory(prefix="mgts-train-") as temporary_dir:
        training_data = _prepare_balanced_training_data(
            data_path, Path(temporary_dir), float(det_cfg.get("rare_class_threshold", 0.05)),
            int(det_cfg.get("rare_class_max_repeats", 5)),
        )
        model.train(
            data=str(training_data),
            imgsz=imgsz,
            epochs=epochs,
            batch=det_cfg.get("batch", 16),
            device=device,
            patience=det_cfg.get("patience", 15),
            seed=det_cfg.get("seed", 42),
            fliplr=det_cfg.get("fliplr", 0.5),
            flipud=0.0,
            project=str(target.parent / "runs"),
            name="detector",
            exist_ok=True,
        )
    trainer = model.trainer
    best_value = getattr(trainer, "best", None)
    best = Path(best_value) if best_value else Path(trainer.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Обучение завершилось без best.pt: {best}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target)

    from .evaluate import evaluate_detector
    eval_cfg = {
        **cfg,
        "paths": {**cfg["paths"], "processed_dir": str(data_path.parent)},
        "detection": {**det_cfg, "weights": str(target), "backend": "yolo"},
    }
    metrics = evaluate_detector(eval_cfg, out_dir=target.parent / "evaluation")
    metric_path = target.parent / "detector_metrics.json"
    metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Модель сохранена: %s", target)
    return target


def _prepare_balanced_training_data(data_path: Path, work_dir: Path, rare_ratio: float,
                                    max_repeats: int) -> Path:
    """Подготовить временный YOLO-конфиг с повторами кадров редких классов."""
    data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    dataset_root = Path(data.get("path", data_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_path.parent / dataset_root
    train_value = data.get("train")
    if not isinstance(train_value, str) or Path(train_value).suffix.lower() != ".txt":
        logger.warning("Балансировка классов пропущена: train должен ссылаться на train.txt")
        return data_path
    train_path = Path(train_value)
    if not train_path.is_absolute():
        train_path = dataset_root / train_path
    if not train_path.is_file():
        logger.warning("Балансировка классов пропущена: не найден список обучения %s", train_path)
        return data_path

    rows, class_counts = _read_train_rows(train_path)
    total_objects = sum(class_counts.values())
    rare_classes = {class_id: count for class_id, count in class_counts.items()
                    if total_objects and count / total_objects < rare_ratio}
    if not rare_classes:
        return data_path
    # ponytail: максимум копий ограничен конфигом, увеличить по замерам recall и длительности обучения
    repeats = {class_id: min(max_repeats, max(1, math.ceil(rare_ratio * total_objects / count) - 1))
               for class_id, count in rare_classes.items()}
    balanced_lines = [line for line, _ in rows]
    for line, class_ids in rows:
        copies = max((repeats[class_id] for class_id in class_ids if class_id in repeats), default=0)
        balanced_lines.extend([line] * copies)
    work_dir.mkdir(parents=True, exist_ok=True)
    balanced_train = work_dir / "train.txt"
    balanced_train.write_text("\n".join(balanced_lines) + "\n", encoding="utf-8")
    data["path"] = str(dataset_root.resolve())
    data["train"] = str(balanced_train.resolve())
    balanced_yaml = work_dir / "data.yaml"
    balanced_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger.info("Балансировка train split: редкие классы %s, кадров %d → %d",
                sorted(rare_classes), len(rows), len(balanced_lines))
    return balanced_yaml


def _read_train_rows(train_path: Path) -> tuple[list[tuple[str, set[int]]], dict[int, int]]:
    """Прочитать классы объектов для каждого кадра train-списка."""
    rows: list[tuple[str, set[int]]] = []
    class_counts: dict[int, int] = {}
    for line in train_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        image_path = Path(line)
        if not image_path.is_absolute():
            image_path = train_path.parent / image_path
        label_path = _label_path(image_path)
        class_ids: set[int] = set()
        if label_path.is_file():
            for label in label_path.read_text(encoding="utf-8").splitlines():
                fields = label.split()
                if fields:
                    try:
                        class_id = int(fields[0])
                    except ValueError:
                        continue
                    class_ids.add(class_id)
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
        rows.append((line, class_ids))
    return rows, class_counts


def _label_path(image_path: Path) -> Path:
    """Найти YOLO-разметку для кадра в каталоге images."""
    images_dir = next((parent for parent in image_path.parents if parent.name == "images"), None)
    if images_dir is not None:
        return images_dir.parent / "labels" / image_path.relative_to(images_dir).with_suffix(".txt")
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
