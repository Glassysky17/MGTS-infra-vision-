"""train/val-разбиение без утечек (Агент 1, задача 3)."""
from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Callable, Optional

import yaml

from ..utils import list_images

logger = logging.getLogger(__name__)

def make_split(  # CONTRACT
    dataset_dir: str | Path,
    val_fraction: float = 0.2,
    seed: int = 42,
    group_fn: Optional[Callable[[str], str]] = None,
) -> dict[str, list[str]]:
    """Разбить кадры на train/val и записать списки + обновить data.yaml.

    Выход: {'train': [пути], 'val': [пути]}; файлы dataset_dir/train.txt, val.txt;
    в data.yaml прописать `train: train.txt`, `val: val.txt`.
    ВАЖНО: кадры одной серии/маршрута (соседние кадры одного видео) почти идентичны -> их нельзя
    делить между train и val (утечка). group_fn(имя_файла) -> ключ группы; по умолчанию —
    префикс имени до последнего номера кадра (например 'route12_000345' -> 'route12') или блок из N
    последовательных кадров. Стратификация: редкие классы должны присутствовать в val.
    Детерминированно при фиксированном seed.
    """
    root = Path(dataset_dir)
    images_dir = root / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Папка images не найдена: {images_dir}")
    images = list_images(images_dir, recursive=False)
    if not images:
        raise ValueError(f"В датасете нет изображений: {images_dir}")
    fraction = min(max(float(val_fraction), 0.0), 1.0)
    groups: dict[str, list[Path]] = {}
    for image in images:
        group = group_fn(image.stem) if group_fn else _default_group(image.stem)
        groups.setdefault(group, []).append(image)
    if len(groups) == 1 and len(images) > 1:
        ordered = sorted(images)
        groups = {f"block_{start // 30:05d}": ordered[start:start + 30] for start in range(0, len(ordered), 30)}

    yaml_path = root / "data.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) if yaml_path.exists() else {}
    data = data or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [name for _, name in sorted(names.items(), key=lambda pair: int(pair[0]))]
    group_classes = {key: _group_classes(files, root / "labels", names) for key, files in groups.items()}
    shuffled = sorted(groups)
    random.Random(seed).shuffle(shuffled)
    val_keys: list[str] = []
    val_count = 0
    target_count = int(len(images) * fraction)
    max_val_groups = max(0, len(groups) - 1)
    class_group_counts: dict[str, int] = {}
    for class_set in group_classes.values():
        for name in class_set:
            class_group_counts[name] = class_group_counts.get(name, 0) + 1
    while val_count < target_count and len(val_keys) < max_val_groups:
        covered = {name for key in val_keys for name in group_classes[key]}
        uncovered = {name for class_set in group_classes.values() for name in class_set} - covered
        candidates = [key for key in shuffled if key not in val_keys]
        choice = max(candidates, key=lambda key: (sum(1 / class_group_counts[name] for name in group_classes[key] & uncovered), -shuffled.index(key)))
        val_keys.append(choice)
        val_count += len(groups[choice])
    val_paths = [str(path.resolve()) for key in val_keys for path in groups[key]]
    train_paths = [str(path.resolve()) for key in groups if key not in val_keys for path in groups[key]]
    (root / "train.txt").write_text("\n".join(train_paths) + ("\n" if train_paths else ""), encoding="utf-8")
    (root / "val.txt").write_text("\n".join(val_paths) + ("\n" if val_paths else ""), encoding="utf-8")
    data.update({"train": "train.txt", "val": "val.txt"})
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger.info("Split: %d train, %d val frames across %d groups", len(train_paths), len(val_paths), len(groups))
    return {"train": train_paths, "val": val_paths}


def _default_group(stem: str) -> str:
    """Убрать последний блок номера кадра из имени серии."""
    return re.sub(r"[_-]?\d+$", "", stem) or stem


def _group_classes(files: list[Path], labels_dir: Path, names: list[str]) -> set[str]:
    """Собрать множество классов объектов, встречающихся в группе кадров."""
    found: set[str] = set()
    for image in files:
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            continue
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            try:
                index = int(fields[0])
            except ValueError:
                continue
            found.add(str(names[index]) if 0 <= index < len(names) else fields[0])
    return found
