"""Анализ датасета -> docs/DATASET_REPORT.md (Агент 1, задача 1)."""
from __future__ import annotations

from pathlib import Path


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
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §1")
