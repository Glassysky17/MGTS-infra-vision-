"""train/val-разбиение без утечек (Агент 1, задача 3)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional


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
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §3")
