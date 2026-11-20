"""Конвертация разметки организатора -> YOLO (Агент 1, задача 2)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


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
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §2")
