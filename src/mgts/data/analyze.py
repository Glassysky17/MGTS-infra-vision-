"""Анализ датасета -> docs/DATASET_REPORT.md (Агент 1, задача 1)."""
from __future__ import annotations

import csv
import hashlib
import logging
import re
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from ..schemas import BBox
from ..utils import list_images, read_image

logger = logging.getLogger(__name__)

def analyze_dataset(dataset_dir: str | Path, out_md: str | Path = "docs/DATASET_REPORT.md") -> dict:  # CONTRACT
    """Собрать статистику датасета и записать отчёт в Markdown.

    Что посчитать (YOLO-разметка: images/ + labels/, либо сначала convert_to_yolo):
      - число изображений, число объектов, объектов на кадр;
      - распределение по классам (абс. и доли) -> выявить дисбаланс;
      - распределение размеров изображений и размеров bbox (в px и в долях кадра);
        доля «мелких» объектов (bbox < 32x32 px или < 0.5% площади кадра);
      - кадры без объектов; битые/нечитаемые файлы; дубликаты (по хэшу);
      - есть ли в данных GPS (EXIF/CSV) и дефектные классы/атрибуты;
      - 12 примеров кадров с нарисованными рамками -> docs/dataset_samples.jpg.
    Вернуть словарь статистики (ключи: n_images, n_objects, per_class, small_ratio, has_gps, ...).
    В отчёт добавить раздел «Выводы для дня 0»: какие классы реально есть, какие дефекты видны,
    что делать с дисбалансом.
    """
    root = Path(dataset_dir)
    images_dir, labels_dir = root / "images", root / "labels"
    if not root.is_dir() or not images_dir.is_dir():
        raise FileNotFoundError(f"Папка датасета или images не найдена: {root}")
    names = _class_names(root / "data.yaml")
    images = list_images(images_dir, recursive=False)
    broken_files: list[str] = []
    per_class: dict[str, int] = {}
    dimensions: list[tuple[int, int]] = []
    samples: list[tuple[np.ndarray, list[tuple[BBox, str]]]] = []
    hashes: dict[str, list[str]] = {}
    bbox_sizes: list[tuple[float, float]] = []
    empty_frames = 0
    small_objects = 0
    n_objects = 0

    for image_path in images:
        try:
            image = read_image(image_path)
        except (FileNotFoundError, cv2.error, OSError) as exc:
            logger.warning("Не удалось прочитать %s: %s", image_path, exc)
            broken_files.append(str(image_path))
            continue
        height, width = image.shape[:2]
        dimensions.append((width, height))
        try:
            hashes.setdefault(hashlib.sha256(image_path.read_bytes()).hexdigest(), []).append(str(image_path))
        except OSError:
            logger.warning("Не удалось посчитать хэш %s", image_path, exc_info=True)
        parsed: list[tuple[BBox, str]] = []
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                fields = line.split()
                if len(fields) != 5:
                    logger.warning("Некорректная YOLO-строка %s:%d", label_path, line_no)
                    continue
                try:
                    class_id = int(fields[0])
                    box = BBox.from_xywhn(*(float(value) for value in fields[1:]), width, height).clip(width, height)
                except (ValueError, OverflowError):
                    logger.warning("Некорректная YOLO-строка %s:%d", label_path, line_no)
                    continue
                if box.area <= 0:
                    continue
                class_name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
                parsed.append((box, class_name))
                bbox_sizes.append((box.width, box.height))
                per_class[class_name] = per_class.get(class_name, 0) + 1
                n_objects += 1
                if min(box.width, box.height) < 32 or box.area < 0.005 * width * height:
                    small_objects += 1
        if not parsed:
            empty_frames += 1
        if len(samples) < 12:
            samples.append((image, parsed))

    has_gps = _has_gps(root, images)
    duplicate_groups = sorted((sorted(paths) for paths in hashes.values() if len(paths) > 1), key=lambda paths: paths[0])
    small_ratio = small_objects / n_objects if n_objects else 0.0
    sizes = np.asarray(bbox_sizes, dtype=float)
    bbox_size_stats = ({"min": [float(value) for value in sizes.min(axis=0)],
                        "median": [float(value) for value in np.median(sizes, axis=0)],
                        "max": [float(value) for value in sizes.max(axis=0)]} if len(sizes) else {})
    defect_codes = _defect_codes(root / "gt_defects.csv")
    has_defect_labels = bool(defect_codes) or any(
        re.search(r"defect|broken|crack|rust|tilt|corrosion", name, re.I) for name in per_class)
    stats = {"n_images": len(images) - len(broken_files), "n_objects": n_objects, "per_class": dict(sorted(per_class.items())),
             "small_ratio": small_ratio, "empty_frames": empty_frames, "broken_files": broken_files, "has_gps": has_gps,
             "duplicate_groups": duplicate_groups, "bbox_size_px": bbox_size_stats, "has_defect_labels": has_defect_labels,
             "defect_codes": defect_codes}
    output = Path(out_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_path = output.parent / "dataset_samples.jpg"
    _write_samples(samples, sample_path)
    _write_report(output, root, stats, dimensions, sample_path)
    return stats


def _class_names(path: Path) -> list[str]:
    """Прочитать имена классов в порядке YOLO class_id."""
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(name) for _, name in sorted(names.items(), key=lambda pair: int(pair[0]))]
    return [str(name) for name in names]


def _has_gps(root: Path, images: list[Path]) -> bool:
    """Проверить EXIF GPS и sidecar-файлы CSV/GPX."""
    if any(path.suffix.lower() == ".gpx" for path in root.rglob("*")):
        return True
    for path in root.rglob("*.csv"):
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                headers = {name.strip().lower() for name in (csv.DictReader(stream).fieldnames or [])}
            if headers & {"lat", "latitude"} and headers & {"lon", "lng", "longitude"}:
                return True
        except OSError:
            continue
    for path in images:
        try:
            exif = Image.open(path).getexif()
            if exif.get(34853) or exif.get_ifd(34853):
                return True
        except (OSError, AttributeError, KeyError):
            continue
    return False


def _defect_codes(path: Path) -> list[str]:
    """Прочитать коды дефектов из необязательного sidecar CSV."""
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return sorted({(row.get("defect_code") or row.get("label") or "").strip()
                           for row in csv.DictReader(stream) if (row.get("defect_code") or row.get("label"))})
    except OSError:
        logger.warning("Не удалось прочитать метки дефектов %s", path, exc_info=True)
        return []


def _write_samples(samples: list[tuple[np.ndarray, list[tuple[BBox, str]]]], path: Path) -> None:
    """Сохранить первые 12 кадров в сетку с нарисованными YOLO-рамками."""
    tiles: list[np.ndarray] = []
    for image, boxes in samples:
        copy = image.copy()
        for box, name in boxes:
            p1 = (round(box.x1), round(box.y1))
            p2 = (round(box.x2), round(box.y2))
            cv2.rectangle(copy, p1, p2, (0, 220, 0), 2)
            cv2.putText(copy, name, (p1[0], max(18, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        tiles.append(cv2.resize(copy, (320, 180)))
    while len(tiles) < 12:
        tiles.append(np.full((180, 320, 3), 245, dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[start:start + 4]) for start in range(0, 12, 4)])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid)


def _write_report(path: Path, root: Path, stats: dict, dimensions: list[tuple[int, int]], sample_path: Path) -> None:
    """Записать сводку и выводы для первичного осмотра данных."""
    n_objects = stats["n_objects"]
    low_classes = [name for name, count in stats["per_class"].items() if n_objects and count / n_objects < 0.05]
    scenario = "A" if stats["has_defect_labels"] else "B" if n_objects else "C"
    size_counts: dict[tuple[int, int], int] = {}
    for size in dimensions:
        size_counts[size] = size_counts.get(size, 0) + 1
    lines = [f"# Отчёт по датасету `{root}`", "", f"**Выбранный сценарий: {scenario}.** " +
             ("Есть классы дефектов с метками." if scenario == "A" else "Есть разметка объектов, явные метки состояния не обнаружены." if scenario == "B" else "Разметки объектов не обнаружено."),
             "", "## Сводка", "", "| Показатель | Значение |", "|---|---:|",
             f"| Читаемых изображений | {stats['n_images']} |", f"| Объектов | {n_objects} |",
             f"| Пустых кадров | {stats['empty_frames']} |", f"| Битых файлов | {len(stats['broken_files'])} |",
             f"| Доля мелких рамок | {stats['small_ratio']:.1%} |", f"| GPS/трек | {'есть' if stats['has_gps'] else 'нет'} |",
             f"| Групп точных дубликатов | {len(stats['duplicate_groups'])} |",
             f"| Метки состояния/дефектов | {'есть' if stats['has_defect_labels'] else 'нет'} |",
             "", "## Классы", "", "| Класс | Объектов | Доля |", "|---|---:|---:|"]
    for name, count in stats["per_class"].items():
        lines.append(f"| {name} | {count} | {count / n_objects:.1%} |")
    lines.extend(["", "## Размеры изображений", "", "| Размер | Кадров |", "|---|---:|"])
    lines.extend(f"| {width}×{height} | {count} |" for (width, height), count in sorted(size_counts.items()))
    lines.extend(["", "## Выводы для дня 0", ""])
    if scenario == "C":
        lines.append("Разметки нет: подготовить ручную разметку в CVAT/Label Studio; текущие метрики по классам неприменимы.")
    elif scenario == "B":
        lines.append("Дефектные состояния не размечены: использовать геометрические признаки и синтетические дефекты, не считать их реальными метками.")
    else:
        codes = ", ".join(stats["defect_codes"])
        lines.append("Разделить классы объектов и дефектные метки перед обучением классификатора состояния." +
                     (f" Найденные коды: {codes}." if codes else ""))
    lines.append("Дисбаланс (<5% объектов): " + (", ".join(low_classes) if low_classes else "не обнаружен по порогу 5%"))
    if stats["bbox_size_px"]:
        size = stats["bbox_size_px"]
        lines.append(f"Размер bbox (min/медиана/max): {size['min']}/{size['median']}/{size['max']} px.")
    if stats["duplicate_groups"]:
        lines.append(f"Обнаружено точных дубликатов кадров: {sum(len(group) - 1 for group in stats['duplicate_groups'])}.")
    lines.extend(["", f"Примеры кадров с рамками: [{sample_path.name}]({sample_path.name}).", ""])
    if stats["broken_files"]:
        lines.extend(["## Битые файлы", "", *[f"- `{name}`" for name in stats["broken_files"]], ""])
    path.write_text("\n".join(lines), encoding="utf-8")
