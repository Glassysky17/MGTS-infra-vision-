"""Оценка качества детектора (Агент 2, задача 6)."""
from __future__ import annotations

from pathlib import Path

from ..schemas import Detection


def match_detections(  # CONTRACT
    preds: list[Detection], gts: list[Detection], iou_thr: float = 0.5
) -> dict[str, int]:
    """Жадное сопоставление по IoU с учётом класса (по убыванию score). Вернуть {'tp','fp','fn'}.
    Чистая функция, без файлов — покрывается юнит-тестом."""
    raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §6")


def evaluate_detector(cfg: dict, split: str = "val", out_dir: str | Path = "outputs/eval_det") -> dict:  # CONTRACT
    """Прогнать детектор (через build_detector — со всем тайлингом!) по split и посчитать:
      precision, recall, F1 по классам; mAP@0.5 (по AP-кривым, можно через ultralytics val для базовой цифры);
      отдельно recall по МЕЛКИМ объектам (bbox < 32 px) — проверяет эффект тайлинга.
    Сохранить: out_dir/metrics.json, out_dir/errors/ (топ-20 FN и топ-20 FP — кадры с рамками GT зелёным,
    предсказаниями красным) и вернуть словарь метрик. Выписать выводы в docs/DETECTION_REPORT.md.
    Сравнить «без тайлинга» и «с тайлингом» — цифры в отчёт.
    """
    raise NotImplementedError("Агент 2: tasks/agent_2_detection.md §6")
