"""Сквозной пайплайн (Агент 4, задача 4)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .schemas import ObjectRecord


def run_pipeline(  # CONTRACT
    input_dir: str | Path,
    output_dir: str | Path,
    cfg: dict,
    track_path: Optional[str | Path] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> list[ObjectRecord]:
    """Картинки -> перечень объектов + файлы результата.

    Шаги (на каждом кадре; ошибка одного кадра не должна ронять запуск — логировать и идти дальше):
      1. list_images(input_dir); при pipeline.max_frames — ограничить;
      2. загрузить трек (csv/gpx) если track_path; build_frame_meta для каждого кадра;
      3. read_image -> blur_score; если pipeline.skip_blurry и кадр размыт — пропустить (счётчик в сводку);
      4. enhance(img, cfg) — только для детекции; оценка состояния идёт по ОРИГИНАЛЬНОМУ кадру (цвета/текстуры!);
      5. detector = build_detector(cfg) (создать один раз); dets = detector.detect(...);
      6. для каждой детекции: condition.assess(...) -> Observation; кроп сохранить (если export.save_crops),
         позицию объекта оценить geo.project.estimate_object_position;
      7. merge_observations(...) -> list[ObjectRecord];
      8. export: CSV, GeoJSON, HTML-карта, аннотированные кадры (export.py) в output_dir;
      9. вернуть records.
    progress(i, n, frame_id) вызывать после каждого кадра (для UI).
    Также записать output_dir/summary.json: n_frames, n_skipped_blurry, n_detections, n_objects,
    по severity/priority, время по этапам (ms/кадр) — пригодится для презентации."""
    raise NotImplementedError("Агент 4: tasks/agent_4_integration.md §4")
