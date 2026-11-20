"""Аугментации, имитирующие съёмку с автомобиля (Агент 1, задача 5).

Идея: используем знания компьютерной графики — физически правдоподобные искажения,
а не просто случайный шум. Все функции принимают и возвращают BGR uint8, не меняют размер кадра
(кроме perspective_warp, которая возвращает и преобразованные bbox).
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

from ..schemas import BBox
from ..utils import list_images, read_image

logger = logging.getLogger(__name__)


def add_motion_blur(img: np.ndarray, length: int = 15, angle_deg: float = 0.0) -> np.ndarray:  # CONTRACT
    """Размытие в движении: свернуть с линейным ядром заданной длины и направления."""
    size = max(1, int(length))
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center, radius = size // 2, (size - 1) / 2
    angle = np.deg2rad(angle_deg)
    dx, dy = np.cos(angle) * radius, np.sin(angle) * radius
    cv2.line(kernel, (round(center - dx), round(center - dy)), (round(center + dx), round(center + dy)), 1.0, 1)
    total = float(kernel.sum())
    if total:
        kernel /= total
    return cv2.filter2D(img, -1, kernel)


def add_rain(img: np.ndarray, intensity: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:  # CONTRACT
    """Дождь: полупрозрачные наклонные штрихи + лёгкое размытие и снижение контраста."""
    generator = rng or np.random.default_rng()
    strength = float(np.clip(intensity, 0.0, 1.0))
    height, width = img.shape[:2]
    layer = np.zeros_like(img)
    count = int(strength * 0.002 * height * width)
    for _ in range(count):
        x = int(generator.integers(0, width))
        y = int(generator.integers(0, height))
        length = int(generator.integers(10, 31))
        angle = np.deg2rad(float(generator.uniform(70, 80)))
        endpoint = (int(np.clip(x + length * np.cos(angle), 0, width - 1)),
                    int(np.clip(y + length * np.sin(angle), 0, height - 1)))
        shade = int(generator.integers(175, 256))
        cv2.line(layer, (x, y), endpoint, (shade, shade, shade), 1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    result = img.astype(np.float32) * (1.0 - 0.3 * strength) + layer.astype(np.float32) * (0.5 * strength)
    return np.clip(result, 0, 255).astype(np.uint8)


def add_glare(img: np.ndarray, strength: float = 0.6, rng: np.random.Generator | None = None) -> np.ndarray:  # CONTRACT
    """Блик/засветка: гауссово световое пятно + bloom (размытие ярких областей) на случайной позиции."""
    generator = rng or np.random.default_rng()
    amount = float(np.clip(strength, 0.0, 1.0))
    height, width = img.shape[:2]
    cx, cy = int(generator.integers(width)), int(generator.integers(height))
    sigma = 0.15 * max(height, width)
    y, x = np.ogrid[:height, :width]
    gaussian = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    result = img.astype(np.float32) + amount * 255 * gaussian[..., None]
    bright = np.where(np.max(img, axis=2) > 200, 255, 0).astype(np.uint8)
    bloom = cv2.GaussianBlur(bright, (61, 61), 0)
    result += bloom[..., None].astype(np.float32) * (amount * 0.35)
    return np.clip(result, 0, 255).astype(np.uint8)


def add_shadow(img: np.ndarray, strength: float = 0.5, rng: np.random.Generator | None = None) -> np.ndarray:  # CONTRACT
    """Тень от дома/дерева: затемнить случайный многоугольник с мягким краем."""
    generator = rng or np.random.default_rng()
    height, width = img.shape[:2]
    vertices = int(generator.integers(4, 7))
    polygon = np.column_stack((generator.integers(0, width, vertices), generator.integers(0, height, vertices))).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    mask = cv2.GaussianBlur(mask, (51, 51), 0).astype(np.float32) / 255.0
    darkness = float(np.clip(strength, 0.0, 1.0))
    factor = 1.0 - darkness * mask
    return np.clip(img.astype(np.float32) * factor[..., None], 0, 255).astype(np.uint8)


def add_night(img: np.ndarray, level: float = 0.6, rng: np.random.Generator | None = None) -> np.ndarray:  # CONTRACT
    """Сумерки/ночь: гамма + снижение яркости + сдвиг к синему + шум матрицы (пуассон+гаусс)."""
    generator = rng or np.random.default_rng()
    amount = float(np.clip(level, 0.0, 1.0))
    result = 255.0 * (img.astype(np.float32) / 255.0) ** (1.0 + amount)
    result *= 1.0 - 0.5 * amount
    result[..., 0] *= 1.1
    result[..., 2] *= 0.9
    result += generator.normal(0, 4 + 8 * amount, img.shape)
    return np.clip(result, 0, 255).astype(np.uint8)


def perspective_warp(  # CONTRACT
    img: np.ndarray, boxes: list[BBox], max_shift: float = 0.08, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, list[BBox]]:
    """Случайная перспективная трансформация (гомография) + пересчёт рамок через ту же матрицу."""
    generator = rng or np.random.default_rng()
    height, width = img.shape[:2]
    _, matrix = _random_homography(width, height, max_shift, generator)
    warped = cv2.warpPerspective(img, matrix, (width, height))
    transformed, _ = _transform_boxes(boxes, matrix, width, height)
    return warped, transformed


def build_train_augmentations(cfg: dict):  # CONTRACT
    """Вернуть вызываемый объект `aug(img, boxes, class_ids) -> (img, boxes, class_ids)`,
    комбинирующий функции выше со случайными вероятностями (albumentations допускается).
    Параметры вероятностей — в cfg['dataset']['augment'] (добавить секцию в default.yaml).
    Также реализовать CLI-режим: `python -m mgts.data.augment --dataset ... --out ... --factor 2`,
    который офлайн дописывает аугментированные копии только редких классов (для балансировки).
    """
    probabilities = cfg.get("dataset", {}).get("augment", {})

    def augment(img: np.ndarray, boxes: list[BBox], class_ids: list[int],
                rng: np.random.Generator | None = None) -> tuple[np.ndarray, list[BBox], list[int]]:
        generator = rng or np.random.default_rng()
        result, result_boxes, result_classes = img.copy(), list(boxes), list(class_ids)
        choices = {
            "motion_blur": lambda frame: add_motion_blur(frame, int(generator.integers(7, 22)), float(generator.uniform(-15, 15))),
            "rain": lambda frame: add_rain(frame, 0.6, generator),
            "glare": lambda frame: add_glare(frame, 0.6, generator),
            "shadow": lambda frame: add_shadow(frame, 0.6, generator),
            "night": lambda frame: add_night(frame, 0.6, generator),
        }
        for name, operation in choices.items():
            probability = float(np.clip(probabilities.get(name, 0.0), 0.0, 1.0))
            if generator.random() < probability:
                result = operation(result)
        probability = float(np.clip(probabilities.get("perspective_warp", 0.0), 0.0, 1.0))
        if generator.random() < probability:
            height, width = result.shape[:2]
            _, matrix = _random_homography(width, height, float(cfg.get("dataset", {}).get("perspective_max_shift", 0.08)), generator)
            result = cv2.warpPerspective(result, matrix, (width, height))
            result_boxes, kept = _transform_boxes(result_boxes, matrix, width, height)
            result_classes = [class_id for index, class_id in enumerate(result_classes) if index in kept]
        return result, result_boxes, result_classes

    return augment


def _random_homography(width: int, height: int, max_shift: float,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Создать гомографию для углов изображения."""
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    offsets = rng.uniform(-max(0.0, max_shift), max(0.0, max_shift), source.shape).astype(np.float32)
    offsets *= np.float32([width, height])
    return source, cv2.getPerspectiveTransform(source, source + offsets)


def _transform_boxes(boxes: list[BBox], matrix: np.ndarray, width: int, height: int) -> tuple[list[BBox], list[int]]:
    """Преобразовать bbox, обрезать к кадру и сохранить исходные индексы."""
    output: list[BBox] = []
    kept: list[int] = []
    for index, box in enumerate(boxes):
        corners = np.float32([[[box.x1, box.y1], [box.x2, box.y1], [box.x2, box.y2], [box.x1, box.y2]]])
        points = cv2.perspectiveTransform(corners, matrix)[0]
        transformed = BBox(float(points[:, 0].min()), float(points[:, 1].min()),
                           float(points[:, 0].max()), float(points[:, 1].max()))
        clipped = transformed.clip(width, height)
        if clipped.area > 0 and clipped.area >= box.area * 0.5:
            output.append(clipped)
            kept.append(index)
    return output, kept


def main() -> None:
    """Офлайн аугментировать кадры с классами, встречающимися реже среднего."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--factor", type=int, default=2)
    args = parser.parse_args()
    _augment_rare_dataset(Path(args.dataset), Path(args.out), max(0, args.factor))


def _augment_rare_dataset(dataset: Path, out: Path, factor: int) -> None:
    """Скопировать исходный YOLO-набор и добавить копии кадров редких классов."""
    source_images, source_labels = dataset / "images", dataset / "labels"
    images, labels = list_images(source_images, recursive=False), source_labels
    counts: dict[str, int] = {}
    for image in images:
        label = labels / f"{image.stem}.txt"
        if label.is_file():
            for line in label.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if fields:
                    counts[fields[0]] = counts.get(fields[0], 0) + 1
    if not counts:
        raise ValueError("Нет YOLO-разметки для балансировки")
    mean_count = sum(counts.values()) / len(counts)
    rare = {class_id for class_id, count in counts.items() if count < mean_count}
    out_images, out_labels = out / "images", out / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    from ..config import load_config
    augment = build_train_augmentations(load_config())
    for image in images:
        label_path = labels / f"{image.stem}.txt"
        lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.is_file() else []
        shutil.copy2(image, out_images / image.name)
        if label_path.is_file():
            shutil.copy2(label_path, out_labels / label_path.name)
        if not any(line.split() and line.split()[0] in rare for line in lines):
            continue
        frame = read_image(image)
        boxes, class_ids = [], []
        for line in lines:
            fields = line.split()
            if len(fields) == 5:
                try:
                    class_id = int(fields[0])
                    box = BBox.from_xywhn(*(float(value) for value in fields[1:]), frame.shape[1], frame.shape[0])
                except ValueError:
                    logger.warning("Пропускается некорректная YOLO-разметка %s: %s", label_path, line)
                    continue
                class_ids.append(class_id)
                boxes.append(box)
        for copy_index in range(factor):
            augmented, warped, kept_classes = augment(frame, boxes, class_ids, np.random.default_rng(copy_index))
            stem = f"{image.stem}_aug{copy_index:03d}"
            cv2.imwrite(str(out_images / f"{stem}{image.suffix}"), augmented)
            out_lines = []
            for class_id, box in zip(kept_classes, warped):
                values = box.to_xywhn(augmented.shape[1], augmented.shape[0])
                out_lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in values))
            (out_labels / f"{stem}.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    source_yaml = dataset / "data.yaml"
    if source_yaml.is_file():
        shutil.copy2(source_yaml, out / "data.yaml")
    for name in ("train.txt", "val.txt"):
        split_file = dataset / name
        if split_file.is_file():
            shutil.copy2(split_file, out / name)


if __name__ == "__main__":
    main()
