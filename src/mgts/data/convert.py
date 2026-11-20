"""Конвертация разметки организатора -> YOLO (Агент 1, задача 2)."""
from __future__ import annotations

import csv
import json
import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
import yaml

from ..config import class_names as configured_class_names
from ..utils import list_images, read_image

logger = logging.getLogger(__name__)

def convert_to_yolo(  # CONTRACT
    src_dir: str | Path,
    out_dir: str | Path,
    fmt: str = "auto",
    class_map: Optional[dict[str, str]] = None,
) -> Path:
    """Привести датасет к структуре YOLO и вернуть путь к out_dir.

    Результат:
        out_dir/images/*.jpg
        out_dir/labels/*.txt      строки: "<class_id> <xc> <yc> <w> <h>" (норм. 0..1)
        out_dir/data.yaml         names в порядке configs/classes.yaml

    fmt: 'auto' | 'yolo' | 'coco' | 'voc' | 'csv'. При 'auto' определить по содержимому
         (*.json с "annotations" -> coco, *.xml -> voc, *.txt рядом с картинками -> yolo).
    class_map: {имя_в_разметке_организатора: имя_из_classes.yaml}; None -> совпадение по имени.
               Неизвестные классы не молча теряем, а пишем в out_dir/unmapped_classes.txt и в лог.
    Требования: не портить исходники (копировать), битые bbox (w/h<=0, вне кадра) клипать/отбрасывать
    с логированием, идемпотентность (повторный запуск даёт тот же результат).
    """
    source, destination = Path(src_dir), Path(out_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Исходный датасет не найден: {source}")
    images = list_images(source)
    if not images:
        raise FileNotFoundError(f"В датасете нет изображений: {source}")
    selected_fmt = _detect_format(source, images) if fmt == "auto" else fmt.lower()
    if selected_fmt not in {"yolo", "coco", "voc", "csv"}:
        raise ValueError(f"Неизвестный формат разметки: {fmt}")
    target_names = configured_class_names()
    if not target_names:
        raise ValueError("В configs/classes.yaml не заданы классы")
    names_by_stem = _source_yolo_names(source)
    if not names_by_stem:
        names_by_stem = target_names
    annotations = _read_annotations(source, images, selected_fmt, names_by_stem)
    destination.mkdir(parents=True, exist_ok=True)
    image_out, label_out = destination / "images", destination / "labels"
    image_out.mkdir(exist_ok=True)
    label_out.mkdir(exist_ok=True)
    mapped_ids = {name: index for index, name in enumerate(target_names)}
    mapping = class_map or {}
    unmapped: set[str] = set()
    diagnostics: list[str] = []
    image_by_name = {path.name: path for path in images}
    for image_name, source_path in image_by_name.items():
        image = read_image(source_path)
        height, width = image.shape[:2]
        shutil.copy2(source_path, image_out / source_path.name)
        lines: list[str] = []
        for class_name, x1, y1, x2, y2 in annotations.get(image_name, []):
            target = mapping.get(class_name, class_name)
            if target not in mapped_ids:
                unmapped.add(class_name)
                diagnostics.append(f"unknown class {class_name} in {image_name}")
                continue
            original = (x1, y1, x2, y2)
            x1, y1 = max(0.0, min(float(width), x1)), max(0.0, min(float(height), y1))
            x2, y2 = max(0.0, min(float(width), x2)), max(0.0, min(float(height), y2))
            if x2 <= x1 or y2 <= y1:
                diagnostics.append(f"invalid bbox {original} in {image_name}")
                logger.warning("Отброшена некорректная рамка %s в %s", original, image_name)
                continue
            box_values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                          (x2 - x1) / width, (y2 - y1) / height)
            lines.append(f"{mapped_ids[target]} " + " ".join(f"{value:.6f}" for value in box_values))
        (label_out / f"{source_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    data = {"path": ".", "train": "images", "val": "images", "names": {i: name for i, name in enumerate(target_names)}}
    (destination / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    unmapped_path = destination / "unmapped_classes.txt"
    if unmapped:
        unmapped_path.write_text("\n".join(sorted(unmapped)) + "\n", encoding="utf-8")
    else:
        unmapped_path.unlink(missing_ok=True)
    log_path = destination / "conversion.log"
    if diagnostics:
        log_path.write_text("\n".join(diagnostics) + "\n", encoding="utf-8")
    else:
        log_path.unlink(missing_ok=True)
    return destination


def _detect_format(source: Path, images: list[Path]) -> str:
    """Определить формат по содержимому аннотаций, а не по имени файла."""
    for path in source.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if all(key in data for key in ("images", "annotations", "categories")):
            return "coco"
    if list(source.rglob("*.xml")):
        return "voc"
    for path in source.rglob("*.csv"):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            headers = {cell.strip().lower() for cell in next(csv.reader(stream), [])}
        if {"filename", "class", "xmin", "ymin", "xmax", "ymax"} <= headers:
            return "csv"
    stems = {path.stem for path in images}
    if any((source / "labels" / f"{stem}.txt").is_file() or (source / f"{stem}.txt").is_file() for stem in stems):
        return "yolo"
    raise ValueError(f"Не удалось определить формат разметки в {source}")


def _source_yolo_names(source: Path) -> list[str]:
    """Загрузить имена исходных YOLO-классов, если рядом есть data.yaml."""
    path = source / "data.yaml"
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(name) for _, name in sorted(names.items(), key=lambda pair: int(pair[0]))]
    return [str(name) for name in names]


def _read_annotations(source: Path, images: list[Path], fmt: str, names: list[str]) -> dict[str, list[tuple[str, float, float, float, float]]]:
    """Разобрать выбранный формат в общие координаты пикселей."""
    result: dict[str, list[tuple[str, float, float, float, float]]] = {}
    if fmt == "coco":
        for path in sorted(source.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not all(key in data for key in ("images", "annotations", "categories")):
                continue
            image_names = {item["id"]: Path(item["file_name"]).name for item in data["images"]}
            categories = {item["id"]: item["name"] for item in data["categories"]}
            for item in data["annotations"]:
                try:
                    x, y, width, height = (float(value) for value in item["bbox"])
                    image_name = image_names[item["image_id"]]
                    category = categories.get(item["category_id"], str(item["category_id"]))
                except (KeyError, TypeError, ValueError):
                    logger.warning("Некорректная COCO-аннотация: %s", item)
                    continue
                result.setdefault(image_name, []).append((category, x, y, x + width, y + height))
            break
    elif fmt == "voc":
        by_name = {path.name for path in images}
        for xml_path in sorted(source.rglob("*.xml")):
            try:
                root = ET.parse(xml_path).getroot()
            except (ET.ParseError, OSError):
                continue
            image_name = Path(root.findtext("filename", xml_path.stem + ".jpg")).name
            if image_name not in by_name:
                continue
            for obj in root.findall("object"):
                bounds = obj.find("bndbox")
                if bounds is not None:
                    try:
                        values = tuple(float(bounds.findtext(key, "0")) for key in ("xmin", "ymin", "xmax", "ymax"))
                    except ValueError:
                        logger.warning("Некорректный VOC bbox в %s", xml_path)
                        continue
                    result.setdefault(image_name, []).append((obj.findtext("name", ""), *values))
    elif fmt == "csv":
        for path in sorted(source.rglob("*.csv")):
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not {"filename", "class", "xmin", "ymin", "xmax", "ymax"} <= set(reader.fieldnames or []):
                    continue
                for row in reader:
                    try:
                        values = tuple(float(row[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
                    except (TypeError, ValueError):
                        logger.warning("Некорректный CSV bbox в %s: %s", path, row)
                        continue
                    result.setdefault(Path(row["filename"]).name, []).append((row["class"], *values))
            break
    else:
        for image in images:
            label = source / "labels" / f"{image.stem}.txt"
            if not label.is_file():
                label = source / f"{image.stem}.txt"
            if not label.is_file():
                continue
            try:
                image_array = read_image(image)
            except (OSError, cv2.error):
                logger.warning("Не удалось прочитать %s", image, exc_info=True)
                continue
            height, width = image_array.shape[:2]
            for line in label.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) != 5:
                    continue
                try:
                    class_id = int(fields[0])
                    name = names[class_id] if 0 <= class_id < len(names) else fields[0]
                    xc, yc, bw, bh = (float(value) for value in fields[1:])
                    result.setdefault(image.name, []).append((name, (xc - bw / 2) * width, (yc - bh / 2) * height,
                                                               (xc + bw / 2) * width, (yc + bh / 2) * height))
                except (ValueError, OSError, cv2.error):
                    logger.warning("Не удалось разобрать строку %s: %s", label, line)
    return result
