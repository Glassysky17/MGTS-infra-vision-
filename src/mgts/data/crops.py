"""Вырезки объектов для обучения классификатора состояния (Агент 1, задача 4)."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2
import yaml

from ..schemas import BBox
from ..utils import list_images, read_image

logger = logging.getLogger(__name__)

def extract_crops(  # CONTRACT
    dataset_dir: str | Path,
    out_dir: str | Path,
    padding: float = 0.15,
    min_size: int = 24,
) -> Path:
    """Вырезать каждый размеченный объект (с полями padding) и разложить по папкам.

    Структура: out_dir/<split>/<class_name>/<label>/<frame>_<idx>.jpg, где
      label = 'ok' | 'defect' | <код дефекта>, если дефектная метка есть в разметке;
      иначе всё кладётся в 'unlabeled' (тогда Агент 3 использует синтетические дефекты и геометрию).
    Также записать out_dir/index.csv: path, split, class_name, label, frame_id, x1,y1,x2,y2.
    Объекты меньше min_size по короткой стороне пропускать (писать в лог количество).
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
    split_by_path: dict[str, str] = {}
    for split in ("train", "val"):
        split_file = root / f"{split}.txt"
        if split_file.is_file():
            for line in split_file.read_text(encoding="utf-8").splitlines():
                split_by_path[str(Path(line.strip()).resolve())] = split
    defect_rows = _defect_labels(root / "gt_defects.csv")
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []
    skipped = 0
    for image_path in list_images(images_dir, recursive=False):
        try:
            image = read_image(image_path)
        except (FileNotFoundError, OSError, cv2.error) as exc:
            logger.warning("Кадр пропущен %s: %s", image_path, exc)
            continue
        height, width = image.shape[:2]
        split = split_by_path.get(str(image_path.resolve()), "train")
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            fields = line.split()
            if len(fields) != 5:
                skipped += 1
                continue
            try:
                class_id = int(fields[0])
                box = BBox.from_xywhn(*(float(value) for value in fields[1:]), width, height).clip(width, height)
            except (ValueError, OverflowError):
                skipped += 1
                continue
            padded = box.pad(padding, width, height)
            x1, y1, x2, y2 = (int(round(value)) for value in padded.to_list())
            crop = image[y1:y2, x1:x2]
            if crop.size == 0 or min(crop.shape[:2]) < min_size:
                skipped += 1
                continue
            class_name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
            state_label = _match_defect(defect_rows.get(image_path.stem, []), class_name, box) or "unlabeled"
            relative = Path(split) / class_name / state_label / f"{image_path.stem}_{index}.jpg"
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), crop):
                logger.warning("Не удалось сохранить кроп %s", target)
                skipped += 1
                continue
            rows.append({"path": relative.as_posix(), "split": split, "class_name": class_name, "label": state_label,
                         "frame_id": image_path.stem, "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2})
    fields = ["path", "split", "class_name", "label", "frame_id", "x1", "y1", "x2", "y2"]
    with (destination / "index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Сохранено %d кропов; пропущено %d объектов меньше порога/с некорректной разметкой", len(rows), skipped)
    return destination


def _defect_labels(path: Path) -> dict[str, list[dict[str, str]]]:
    """Прочитать необязательную разметку состояния из gt_defects.csv."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {frame: items for frame, items in _group_csv_rows(csv.DictReader(stream)).items()}


def _group_csv_rows(reader: csv.DictReader) -> dict[str, list[dict[str, str]]]:
    """Сгруппировать метки дефектов по frame_id."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in reader:
        frame = Path(row.get("frame_id", "")).stem
        if frame:
            grouped.setdefault(frame, []).append({key: str(value or "") for key, value in row.items()})
    return grouped


def _match_defect(rows: list[dict[str, str]], class_name: str, box: BBox) -> str | None:
    """Вернуть код дефекта совпадающего класса и bbox, если он задан."""
    for row in rows:
        if row.get("class_name") != class_name:
            continue
        raw = row.get("bbox", "")
        if raw:
            try:
                values = [float(value) for value in raw.replace(";", ",").split(",")]
                if len(values) == 4 and BBox(*values).iou(box) < 0.5:
                    continue
            except ValueError:
                continue
        label = row.get("defect_code") or row.get("label")
        if label:
            return label
    return None
