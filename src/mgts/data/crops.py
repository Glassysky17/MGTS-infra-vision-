"""Вырезки объектов для обучения классификатора состояния (Агент 1, задача 4)."""
from __future__ import annotations

from pathlib import Path


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
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §4")
