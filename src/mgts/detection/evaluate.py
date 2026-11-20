"""Оценка качества детектора (Агент 2, задача 6)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from ..config import class_names, load_classes, resolve_path
from ..schemas import BBox, Detection
from ..utils import read_image, write_image

def match_detections(  # CONTRACT
    preds: list[Detection], gts: list[Detection], iou_thr: float = 0.5
) -> dict[str, int]:
    """Жадное сопоставление по IoU с учётом класса (по убыванию score). Вернуть {'tp','fp','fn'}.
    Чистая функция, без файлов — покрывается юнит-тестом."""
    unmatched = set(range(len(gts)))
    true_positives = 0
    false_positives = 0
    for prediction in sorted(preds, key=lambda detection: detection.score, reverse=True):
        candidates = [
            (index, prediction.bbox.iou(gts[index].bbox))
            for index in sorted(unmatched)
            if gts[index].class_id == prediction.class_id
        ]
        if candidates:
            best_index, best_iou = max(candidates, key=lambda item: item[1])
        else:
            best_index, best_iou = -1, 0.0
        if best_index >= 0 and best_iou >= iou_thr:
            unmatched.remove(best_index)
            true_positives += 1
        else:
            false_positives += 1
    return {"tp": true_positives, "fp": false_positives, "fn": len(gts) - true_positives}


def evaluate_detector(cfg: dict, split: str = "val", out_dir: str | Path = "outputs/eval_det") -> dict:  # CONTRACT
    """Прогнать детектор (через build_detector — со всем тайлингом!) по split и посчитать:
      precision, recall, F1 по классам; mAP@0.5 (по AP-кривым, можно через ultralytics val для базовой цифры);
      отдельно recall по МЕЛКИМ объектам (bbox < 32 px) — проверяет эффект тайлинга.
    Сохранить: out_dir/metrics.json, out_dir/errors/ (топ-20 FN и топ-20 FP — кадры с рамками GT зелёным,
    предсказаниями красным) и вернуть словарь метрик. Выписать выводы в docs/DETECTION_REPORT.md.
    Сравнить «без тайлинга» и «с тайлингом» — цифры в отчёт.
    """
    dataset_dir = resolve_path(cfg["paths"]["processed_dir"])
    data_file = dataset_dir / "data.yaml"
    if not data_file.is_file():
        raise FileNotFoundError(f"YOLO config not found: {data_file}")
    with data_file.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    dataset_root = Path(data.get("path", dataset_dir))
    if not dataset_root.is_absolute():
        dataset_root = data_file.parent / dataset_root
    image_paths = _split_images(data.get(split), dataset_root)
    if not image_paths:
        raise FileNotFoundError(f"Нет изображений для split '{split}' в {data_file}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    classes_file = cfg.get("paths", {}).get("classes_file")
    names = class_names(load_classes(resolve_path(classes_file)) if classes_file else None)
    if cfg.get("detection", {}).get("tiling", {}).get("enabled", False):
        no_tiling_cfg = {**cfg, "detection": {**cfg["detection"], "tiling": {**cfg["detection"]["tiling"], "enabled": False}}}
        result = _evaluate_pass(_build_for_evaluation(cfg), image_paths, data, names, out_path)
        untiled = _evaluate_pass(_build_for_evaluation(no_tiling_cfg), image_paths, data, names, None)
        result["tiling_comparison"] = {"without_tiling": _summary(untiled), "with_tiling": _summary(result)}
    else:
        result = _evaluate_pass(_build_for_evaluation(cfg), image_paths, data, names, out_path)
    with (out_path / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def _build_for_evaluation(cfg: dict) -> Any:
    from .infer import build_detector
    return build_detector(cfg)


def _split_images(split_value: Any, dataset_root: Path) -> list[Path]:
    if split_value is None:
        return []
    entries = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for entry in entries:
        path = Path(entry)
        path = path if path.is_absolute() else dataset_root / path
        if path.suffix.lower() == ".txt":
            images.extend(Path(line.strip()) if Path(line.strip()).is_absolute() else path.parent / line.strip()
                          for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        elif path.is_dir():
            images.extend(sorted(item for item in path.rglob("*") if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}))
        elif path.is_file():
            images.append(path)
    return sorted(dict.fromkeys(images))


def _label_path(image_path: Path) -> Path:
    images_dirs = [parent for parent in image_path.parents if parent.name == "images"]
    if images_dirs:
        images_dir = images_dirs[0]
        return images_dir.parent / "labels" / image_path.relative_to(images_dir).with_suffix(".txt")
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def _read_ground_truth(image_path: Path, image: np.ndarray, data: dict, names: list[str], frame_id: str) -> list[Detection]:
    label_path = _label_path(image_path)
    if not label_path.is_file():
        return []
    height, width = image.shape[:2]
    data_names = data.get("names", names)
    if isinstance(data_names, dict):
        data_names = [data_names[key] for key in sorted(data_names, key=lambda key: int(key))]
    gts: list[Detection] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            continue
        if class_id < 0 or class_id >= len(data_names) or not np.isfinite(values).all():
            continue
        bbox = BBox.from_xywhn(*values, width, height).clip(width, height)
        if bbox.area:
            gts.append(Detection(frame_id, class_id, str(data_names[class_id]), 1.0, bbox))
    return gts


def _evaluate_pass(detector: Any, image_paths: list[Path], data: dict, names: list[str], out_dir: Path | None) -> dict:
    records: list[tuple[list[Detection], list[Detection], Path]] = []
    elapsed = 0.0
    for image_path in image_paths:
        image = read_image(image_path)
        frame_id = str(image_path)
        gts = _read_ground_truth(image_path, image, data, names, frame_id)
        started = time.perf_counter()
        predictions = detector.detect(image, frame_id, str(image_path))
        elapsed += time.perf_counter() - started
        records.append((predictions, gts, image_path))

    class_ids = sorted(set(range(len(names))) | {item.class_id for preds, gts, _ in records for item in (*preds, *gts)})
    class_results: dict[str, dict[str, float]] = {}
    small_tp = small_total = 0
    fn_images: list[tuple[int, Path, list[Detection], list[Detection]]] = []
    fp_images: list[tuple[float, Path, list[Detection], list[Detection]]] = []
    for class_id in class_ids:
        class_preds = [pred for preds, _, _ in records for pred in preds if pred.class_id == class_id]
        class_gts = [gt for _, gts, _ in records for gt in gts if gt.class_id == class_id]
        counts = {"tp": 0, "fp": 0, "fn": 0}
        for predictions, gts, _ in records:
            frame_counts = match_detections(
                [pred for pred in predictions if pred.class_id == class_id],
                [gt for gt in gts if gt.class_id == class_id],
            )
            for key in counts:
                counts[key] += frame_counts[key]
        precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
        recall = counts["tp"] / max(len(class_gts), 1)
        ap50 = _average_precision(class_preds, class_gts)
        name = names[class_id] if class_id < len(names) else f"class_{class_id}"
        class_results[name] = {"precision": precision, "recall": recall, "f1": _f1(precision, recall), "ap50": ap50}
        for predictions, gts, image_path in records:
            class_small = [gt for gt in gts if gt.class_id == class_id and min(gt.bbox.width, gt.bbox.height) < 32]
            if class_small:
                small_total += len(class_small)
                small_tp += match_detections([p for p in predictions if p.class_id == class_id], class_small)["tp"]

    for predictions, gts, image_path in records:
        counts = match_detections(predictions, gts)
        if counts["fn"]:
            fn_images.append((counts["fn"], image_path, predictions, gts))
        if counts["fp"]:
            score = max((pred.score for pred in predictions), default=0.0)
            fp_images.append((score, image_path, predictions, gts))
    classes_with_ground_truth = [
        values["ap50"] for class_id, values in zip(class_ids, class_results.values())
        if any(gt.class_id == class_id for _, gts, _ in records for gt in gts)
    ]
    metrics = {
        "per_class": class_results,
        "map50": float(np.mean(classes_with_ground_truth)) if classes_with_ground_truth else 0.0,
        "recall_small": small_tp / small_total if small_total else 0.0,
        "ms_per_frame": elapsed * 1000 / len(image_paths),
    }
    if out_dir is not None:
        (out_dir / "errors").mkdir(parents=True, exist_ok=True)
        _save_error_frames(out_dir / "errors", fn_images, fp_images)
    return metrics


def _average_precision(preds: list[Detection], gts: list[Detection]) -> float:
    if not gts:
        return 0.0
    ordered = sorted(preds, key=lambda detection: detection.score, reverse=True)
    matched: dict[str, set[int]] = {}
    tp: list[int] = []
    fp: list[int] = []
    for prediction in ordered:
        same_frame = [i for i, gt in enumerate(gts) if gt.frame_id == prediction.frame_id and gt.class_id == prediction.class_id]
        used = matched.setdefault(prediction.frame_id, set())
        candidates = [(i, prediction.bbox.iou(gts[i].bbox)) for i in same_frame if i not in used]
        best = max(candidates, key=lambda item: item[1], default=(-1, 0.0))
        if best[1] >= 0.5:
            used.add(best[0])
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    recall = np.cumsum(tp) / len(gts)
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _summary(metrics: dict) -> dict:
    return {key: metrics[key] for key in ("per_class", "map50", "recall_small", "ms_per_frame")}


def _save_error_frames(errors_dir: Path, fn_images: list, fp_images: list) -> None:
    for kind, items in (("fn", sorted(fn_images, reverse=True, key=lambda item: item[0])[:20]),
                        ("fp", sorted(fp_images, reverse=True, key=lambda item: item[0])[:20])):
        for index, (_, path, predictions, gts) in enumerate(items, start=1):
            image = read_image(path)
            annotated = image.copy()
            for detection in gts:
                box = detection.bbox
                cv2.rectangle(annotated, (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2)), (0, 200, 0), 2)
            for detection in predictions:
                box = detection.bbox
                cv2.rectangle(annotated, (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2)), (0, 0, 230), 2)
            write_image(errors_dir / f"{kind}_{index:02d}_{path.name}", annotated)
