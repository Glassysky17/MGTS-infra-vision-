"""Синтез дефектов на исправных объектах (Агент 1, задача 6).

Нужен, если в датасете нет/мало дефектных примеров. Генерируемые дефекты ДОЛЖНЫ иметь
метку (код из configs/classes.yaml), чтобы Агент 3 мог обучать и проверять классификатор/геометрию.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

from ..schemas import BBox
from ..utils import list_images, read_image

logger = logging.getLogger(__name__)


def synth_cracks(img: np.ndarray, bbox: BBox, rng: np.random.Generator | None = None, n: int = 3) -> np.ndarray:  # CONTRACT
    """Нарисовать n реалистичных трещин внутри bbox: случайные ветвящиеся полилинии
    (случайное блуждание с ветвлением), тёмный цвет с мягкими краями и текстурным шумом, прозрачность 0.6–0.9.
    Метка: cover_broken / pole_crack."""
    generator = rng or np.random.default_rng()
    height, width = img.shape[:2]
    box = bbox.clip(width, height)
    x1, y1, x2, y2 = (int(round(value)) for value in box.to_list())
    if x2 <= x1 or y2 <= y1:
        return img.copy()
    layer = np.zeros((height, width), dtype=np.uint8)
    for _ in range(max(0, n)):
        x = int(generator.integers(x1, max(x1 + 1, x2)))
        y = int(generator.integers(y1, max(y1 + 1, y2)))
        angle = float(generator.uniform(-np.pi, np.pi))
        previous = (x, y)
        for _ in range(int(generator.integers(20, 61))):
            angle += float(generator.uniform(-0.4, 0.4))
            length = float(generator.uniform(3, 7))
            x = int(np.clip(x + np.cos(angle) * length, x1, x2 - 1))
            y = int(np.clip(y + np.sin(angle) * length, y1, y2 - 1))
            current = (x, y)
            cv2.line(layer, previous, current, 255, int(generator.integers(1, 4)), cv2.LINE_AA)
            if generator.random() < 0.08:
                branch_angle = angle + float(generator.choice([-1, 1])) * float(generator.uniform(0.3, 0.9))
                branch_end = (int(np.clip(x + np.cos(branch_angle) * generator.uniform(5, 15), x1, x2 - 1)),
                              int(np.clip(y + np.sin(branch_angle) * generator.uniform(5, 15), y1, y2 - 1)))
                cv2.line(layer, current, branch_end, 210, int(generator.integers(1, 3)), cv2.LINE_AA)
            previous = current
    layer = cv2.GaussianBlur(layer, (3, 3), 0).astype(np.float32) / 255.0
    result = img.copy()
    region = result[y1:y2, x1:x2].astype(np.float32)
    alpha = layer[y1:y2, x1:x2, None] * float(generator.uniform(0.6, 0.9))
    result[y1:y2, x1:x2] = np.clip(region * (1.0 - alpha) + np.array([20, 20, 20]) * alpha, 0, 255).astype(np.uint8)
    return result


def synth_rust(img: np.ndarray, bbox: BBox, rng: np.random.Generator | None = None, coverage: float = 0.25) -> np.ndarray:  # CONTRACT
    """Ржавчина: пятна (шум Перлина/gaussian-blob) оранжево-коричневых оттенков, смешанные по маске
    внутри bbox с покрытием ~coverage площади. Метка: corrosion."""
    generator = rng or np.random.default_rng()
    height, width = img.shape[:2]
    box = bbox.clip(width, height)
    x1, y1, x2, y2 = (int(round(value)) for value in box.to_list())
    if x2 <= x1 or y2 <= y1:
        return img.copy()
    region = img[y1:y2, x1:x2].copy()
    noise = generator.random(region.shape[:2], dtype=np.float32)
    scales = [max(3, ((min(region.shape[:2]) // factor) | 1)) for factor in (2, 4, 8)]
    texture = np.zeros_like(noise)
    for kernel in scales:
        texture += cv2.GaussianBlur(noise, (kernel, kernel), 0)
    texture /= len(scales)
    amount = float(np.clip(coverage, 0.0, 1.0))
    threshold = np.quantile(texture, 1.0 - amount) if amount else 1.0
    mask = (texture >= threshold).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)[..., None]
    luminance = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
    rust = np.concatenate((np.full_like(luminance, 30), np.full_like(luminance, 70), np.full_like(luminance, 150)), axis=2)
    rust += generator.normal(0, 14, region.shape).astype(np.float32)
    # Смешивание с оригиналом сохраняет исходную фактуру под ржавым налётом.
    rust = rust * 0.7 + luminance * 0.3
    blend = np.clip(mask * 0.85, 0, 0.85)
    result = img.copy()
    result[y1:y2, x1:x2] = np.clip(region.astype(np.float32) * (1 - blend) + rust * blend, 0, 255).astype(np.uint8)
    return result


def synth_tilt(img: np.ndarray, bbox: BBox, angle_deg: float) -> tuple[np.ndarray, BBox]:  # CONTRACT
    """Наклонить объект (опору) на angle_deg вокруг основания (нижний центр bbox):
    вырезать область, повернуть аффинным преобразованием с альфа-маской, вклеить на закрытый фон
    (inpaint cv2.INPAINT_TELEA). Вернуть новое изображение и новую рамку. Метка: pole_tilt."""
    height, width = img.shape[:2]
    box = bbox.clip(width, height)
    margin_x, margin_y = max(8, int(box.width * 0.5)), max(8, int(box.height * 0.1))
    x1 = max(0, int(box.x1) - margin_x)
    y1 = max(0, int(box.y1) - margin_y)
    x2 = min(width, int(box.x2) + margin_x)
    y2 = min(height, int(box.y2) + margin_y)
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    object_mask = np.where(gray <= threshold + 8, 255, 0).astype(np.uint8)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if not np.any(object_mask):
        object_mask[int(box.y1) - y1:int(box.y2) - y1, int(box.x1) - x1:int(box.x2) - x1] = 255
    local_bbox = BBox(box.x1 - x1, box.y1 - y1, box.x2 - x1, box.y2 - y1)
    pivot = ((local_bbox.x1 + local_bbox.x2) / 2, local_bbox.y2)
    matrix = cv2.getRotationMatrix2D(pivot, -float(angle_deg), 1.0)
    background = cv2.inpaint(crop, object_mask, 5, cv2.INPAINT_TELEA)
    rotated = cv2.warpAffine(crop, matrix, (crop.shape[1], crop.shape[0]))
    rotated_mask = cv2.warpAffine(object_mask, matrix, (crop.shape[1], crop.shape[0]))
    output = img.copy()
    alpha = rotated_mask.astype(np.float32)[..., None] / 255.0
    output[y1:y2, x1:x2] = np.clip(rotated * alpha + background * (1.0 - alpha), 0, 255).astype(np.uint8)
    corners = np.float32([[[local_bbox.x1, local_bbox.y1], [local_bbox.x2, local_bbox.y1],
                           [local_bbox.x2, local_bbox.y2], [local_bbox.x1, local_bbox.y2]]])
    transformed = cv2.transform(corners, matrix)[0]
    new_box = BBox(float(transformed[:, 0].min() + x1), float(transformed[:, 1].min() + y1),
                   float(transformed[:, 0].max() + x1), float(transformed[:, 1].max() + y1)).clip(width, height)
    return output, new_box


def synth_displaced_cover(img: np.ndarray, bbox: BBox, shift_frac: float = 0.15, rng: np.random.Generator | None = None) -> np.ndarray:  # CONTRACT
    """Смещённая крышка люка: сдвинуть содержимое эллипса люка на shift_frac, оставив тёмную щель.
    Метка: cover_displaced."""
    generator = rng or np.random.default_rng()
    height, width = img.shape[:2]
    box = bbox.clip(width, height)
    x1, y1, x2, y2 = (int(round(value)) for value in box.to_list())
    if x2 <= x1 or y2 <= y1:
        return img.copy()
    crop = img[y1:y2, x1:x2]
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    center = ((crop.shape[1] - 1) // 2, (crop.shape[0] - 1) // 2)
    axes = (max(1, crop.shape[1] // 2 - 2), max(1, crop.shape[0] // 2 - 2))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    dx = int(round(crop.shape[1] * float(shift_frac) * float(generator.choice([-1, 1]))))
    dy = int(round(crop.shape[0] * float(shift_frac) * float(generator.choice([-1, 1])) * 0.25))
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(crop, matrix, (crop.shape[1], crop.shape[0]))
    moved_mask = cv2.warpAffine(mask, matrix, (crop.shape[1], crop.shape[0]))
    result = img.copy()
    output_crop = result[y1:y2, x1:x2]
    output_crop[mask > 0] = (35, 35, 35)
    output_crop[moved_mask > 0] = moved[moved_mask > 0]
    return result


def generate_synthetic_defects(  # CONTRACT
    dataset_dir: str, out_dir: str, per_class: int = 100, seed: int = 0
) -> str:
    """Для исправных объектов датасета сгенерировать per_class дефектных вариантов на класс.
    Сохранить вырезки в out_dir/<class>/<defect_code>/ + index.csv (path, class_name, label, synthetic=1).
    Реальные и синтетические данные ДОЛЖНЫ быть различимы в index.csv (колонка synthetic),
    а валидацию классификатора считать только на реальных примерах, если они есть.
    Вернуть путь к index.csv.
    """
    root, destination = Path(dataset_dir), Path(out_dir)
    images_dir, labels_dir = root / "images", root / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Папка images не найдена: {images_dir}")
    data_path = root / "data.yaml"
    data = yaml.safe_load(data_path.read_text(encoding="utf-8")) if data_path.exists() else {}
    names = (data or {}).get("names", [])
    if isinstance(names, dict):
        names = [name for _, name in sorted(names.items(), key=lambda pair: int(pair[0]))]
    names = [str(name) for name in names]
    candidates: dict[str, list[tuple[np.ndarray, BBox, str]]] = {}
    known_defects = _known_defects(root / "gt_defects.csv")
    for image_path in list_images(images_dir, recursive=False):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        try:
            image = read_image(image_path)
        except (FileNotFoundError, OSError, cv2.error):
            logger.warning("Кадр пропущен при синтезе: %s", image_path)
            continue
        height, width = image.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            try:
                class_id = int(fields[0])
                box = BBox.from_xywhn(*(float(value) for value in fields[1:]), width, height).clip(width, height)
            except ValueError:
                continue
            class_name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
            if any(marked_class == class_name and marked_box.iou(box) >= 0.5
                   for marked_class, marked_box in known_defects.get(image_path.stem, [])):
                continue
            x1, y1, x2, y2 = (int(round(value)) for value in box.to_list())
            candidates.setdefault(class_name, []).append((image[y1:y2, x1:x2].copy(), BBox(0, 0, x2 - x1, y2 - y1), image_path.stem))
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    rng = np.random.default_rng(seed)
    for class_name, objects in sorted(candidates.items()):
        generators = _defect_generators(class_name)
        for index in range(max(0, per_class)):
            crop, box, frame_id = objects[int(rng.integers(len(objects)))]
            defect_code, operation = generators[index % len(generators)]
            variant_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
            if defect_code == "pole_tilt":
                generated, _ = operation(crop, box, 12.0)
            else:
                generated = operation(crop, box, variant_rng)
            relative = Path(class_name) / defect_code / f"{frame_id}_{index:04d}.jpg"
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), generated):
                continue
            rows.append({"path": relative.as_posix(), "class_name": class_name, "label": defect_code,
                         "synthetic": "1", "frame_id": frame_id})
    index_path = destination / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "class_name", "label", "synthetic", "frame_id"])
        writer.writeheader()
        writer.writerows(rows)
    return str(index_path)


def _known_defects(path: Path) -> dict[str, list[tuple[str, BBox]]]:
    """Собрать bbox уже размеченных дефектов, чтобы не синтезировать их повторно."""
    result: dict[str, list[tuple[str, BBox]]] = {}
    if not path.is_file():
        return result
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    box = BBox(*(float(value) for value in row["bbox"].replace(";", ",").split(",")))
                except (KeyError, TypeError, ValueError):
                    continue
                result.setdefault(Path(row.get("frame_id", "")).stem, []).append((row.get("class_name", ""), box))
    except OSError:
        logger.warning("Не удалось прочитать метки дефектов %s", path, exc_info=True)
    return result


def _defect_generators(class_name: str) -> list[tuple[str, Callable[..., np.ndarray | tuple[np.ndarray, BBox]]]]:
    """Вернуть подходящие синтетические операции по имени класса."""
    if class_name == "cable_manhole":
        return [("cover_broken", synth_cracks)]
    if class_name == "utility_pole":
        return [("pole_tilt", synth_tilt), ("pole_crack", synth_cracks), ("corrosion", synth_rust)]
    if class_name == "cable_cabinet":
        return [("corrosion", synth_rust)]
    return [("generic_damage", synth_cracks)]
