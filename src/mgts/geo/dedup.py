"""Объединение повторных наблюдений одного объекта (Агент 4, задача 3)."""
from __future__ import annotations

from ..schemas import Observation, ObjectRecord


def merge_observations(  # CONTRACT
    observations: list[Observation], cfg: dict, classes: dict | None = None
) -> list[ObjectRecord]:
    """Кластеризовать наблюдения в уникальные объекты и собрать ObjectRecord.

    Правила кластеризации:
      - объединять только наблюдения ОДНОГО класса;
      - расстояние между est-координатами <= cfg.geo.dedup.radius_m (haversine) -> связаны (union-find / DBSCAN);
      - если use_appearance: дополнительно гистограммы HSV вырезок (корреляция) >= appearance_min_sim,
        иначе близкие, но разные объекты (два соседних колодца) сливаются неправильно;
      - наблюдения без координат (location_source='none'): группировать по frame_id-соседству
        (подряд идущие кадры с перекрывающимся bbox, IoU > 0.3), иначе — каждое своим объектом.
    Агрегация кластера в ObjectRecord:
      - лучшее наблюдение: max(detection.score * normalized_sharpness);
        crop_path/frame_path/best_bbox/best_frame_id — оттуда;
      - координаты: медиана est_lat/est_lon по кластеру (устойчиво к выбросам);
      - findings: объединение по code, score = max; severity: если >= 50% наблюдений DEFECT или лучшее — DEFECT
        -> DEFECT; иначе максимальная по серьёзности; needs_review = any(needs_review) ИЛИ расхождение наблюдений;
        condition_score = max score по кластеру, взвешенный частотой (не брать единичный выброс как истину, но и не терять);
      - priority = condition.scoring.priority_for(...) — импортировать и применить к агрегированной оценке;
      - object_id: 'OBJ-0001'… в порядке убывания priority, затем condition_score;
      - n_observations, frame_ids заполнить.
    Результат отсортирован по priority (CRITICAL первым), затем condition_score по убыванию."""
    raise NotImplementedError("Агент 4: tasks/agent_4_integration.md §3")
