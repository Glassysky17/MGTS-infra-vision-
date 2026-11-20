"""Сквозной пайплайн (Агент 4, задача 4)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .condition.classifier import DefectClassifier
from .condition.scoring import assess
from .config import class_names, load_classes, resolve_path
from .detection.infer import build_detector
from .export import draw_annotated, to_csv, to_geojson, to_html_map
from .geo.dedup import merge_observations
from .geo.gps import build_frame_meta, load_track_csv, load_track_gpx
from .geo.project import estimate_object_position
from .observability import current_run_id, log_context, new_run_id, trace_span
from .preprocessing.enhance import enhance
from .preprocessing.quality import blur_score
from .schemas import BBox, ConditionAssessment, Detection, Observation, ObjectRecord, Severity
from .utils import list_images, read_image, write_image

logger = logging.getLogger(__name__)


class _LabelFileDetector:
    """Минимальный DummyDetector для интеграции, пока модуль A2 остаётся заглушкой."""

    def __init__(self, names: list[str]) -> None:
        """Сохранить порядок имён классов для YOLO label-файлов."""
        self.names = names

    def detect(self, image: np.ndarray, frame_id: str, image_path: str | None = None) -> list[Detection]:
        """Прочитать соседний label-файл и вернуть рамки в пикселях."""
        if image_path is None:
            return []
        label_path = Path(image_path).parents[1] / "labels" / f"{Path(image_path).stem}.txt"
        if not label_path.is_file():
            return []
        height, width = image.shape[:2]
        results = []
        with label_path.open(encoding="utf-8") as file:
            for line in file:
                parts = line.split()
                if len(parts) != 5:
                    continue
                try:
                    class_id = int(parts[0])
                    xc, yc, box_w, box_h = map(float, parts[1:])
                    name = self.names[class_id]
                except (ValueError, IndexError):
                    continue
                results.append(Detection(frame_id, class_id, name, 0.9,
                                         BBox.from_xywhn(xc, yc, box_w, box_h, width, height).clip(width, height)))
        return results


def run_pipeline(  # CONTRACT
    input_dir: str | Path,
    output_dir: str | Path,
    cfg: dict,
    track_path: Optional[str | Path] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> list[ObjectRecord]:
    """Обработать кадры, записать результаты и сводку по этапам."""
    run_id = current_run_id() or new_run_id()
    with log_context(run_id=run_id):
        with trace_span(logger, "pipeline", input_dir=str(input_dir), output_dir=str(output_dir),
                        backend=cfg.get("detection", {}).get("backend", "default")):
            return _run_pipeline(input_dir, output_dir, cfg, track_path, progress)


def _run_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    cfg: dict,
    track_path: Optional[str | Path],
    progress: Optional[Callable[[int, int, str], None]],
) -> list[ObjectRecord]:
    """Внутренний запуск: контекст и общий span задаются run_pipeline."""
    pipeline_started = time.perf_counter()
    source_dir, output_dir = Path(input_dir), Path(output_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Папка с кадрами не найдена: {source_dir}")
    max_frames = cfg.get("pipeline", {}).get("max_frames")
    stage_seconds = {name: 0.0 for name in (
        "input_scan", "setup", "gps", "image_quality", "inference", "assessment", "geolocation", "merge", "export"
    )}
    with trace_span(logger, "pipeline.input_scan", timings=stage_seconds, timing_key="input_scan",
                    input_dir=str(source_dir), max_frames=max_frames):
        images = list_images(source_dir)
        if max_frames is not None:
            images = images[:int(max_frames)]
    if not images:
        raise FileNotFoundError(f"В папке нет изображений: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with trace_span(logger, "pipeline.setup", timings=stage_seconds, timing_key="setup",
                    frames=len(images), track_path=str(track_path) if track_path else None):
        classes_path = cfg.get("paths", {}).get("classes_file")
        classes = load_classes(resolve_path(classes_path) if classes_path else None)
        names = class_names(classes)
        try:
            detector = build_detector(cfg)
        except NotImplementedError:
            if cfg.get("detection", {}).get("backend") != "dummy":
                raise
            logger.warning("DummyDetector из A2 пока не реализован; использую чтение YOLO labels напрямую")
            detector = _LabelFileDetector(names)
        classifier = DefectClassifier(cfg, classes) if cfg.get("condition", {}).get("classifier", {}).get("enabled", False) else None
        track = None
        if track_path:
            track_file = Path(track_path)
            if not track_file.is_file():
                raise FileNotFoundError(f"GPS-трек не найден: {track_file}")
            track = load_track_gpx(track_file) if track_file.suffix.lower() == ".gpx" else load_track_csv(track_file)

    observations: list[Observation] = []
    n_skipped_blurry = n_errors = n_detections = 0
    progress_interval = max(1, (len(images) + 19) // 20)
    for index, image_path in enumerate(images):
        frame_id = image_path.stem
        with log_context(frame_id=frame_id):
            try:
                with trace_span(logger, "frame.process", level=logging.DEBUG, frame_index=index, image=str(image_path)):
                    with trace_span(logger, "frame.gps_metadata", level=logging.DEBUG, log_traceback=False,
                                    timings=stage_seconds, timing_key="gps"):
                        meta = build_frame_meta(image_path, index, cfg, track)
                    with trace_span(logger, "frame.image_quality", level=logging.DEBUG, log_traceback=False,
                                    timings=stage_seconds, timing_key="image_quality"):
                        image = read_image(image_path)
                        sharpness = blur_score(image)
                    if cfg.get("pipeline", {}).get("skip_blurry", False) and sharpness < float(cfg.get("preprocessing", {}).get("blur_threshold", 0)):
                        n_skipped_blurry += 1
                        logger.debug("frame.skipped_blurry sharpness=%.2f", sharpness, extra={"stage": "frame.skip_blurry"})
                        continue
                    with trace_span(logger, "frame.inference", level=logging.DEBUG, log_traceback=False,
                                    timings=stage_seconds, timing_key="inference"):
                        detections = detector.detect(enhance(image, cfg), meta.frame_id, str(image_path))
                    n_detections += len(detections)
                    logger.debug("frame.detections count=%d sharpness=%.2f", len(detections), sharpness,
                                 extra={"stage": "frame.detections"})
                    annotated_items = []
                    for detection_index, detection in enumerate(detections):
                        with trace_span(logger, "detection.assessment", level=logging.DEBUG, log_traceback=False,
                                        timings=stage_seconds, timing_key="assessment", detection_index=detection_index,
                                        class_name=detection.class_name, confidence=detection.score):
                            try:
                                assessment = assess(image, detection, cfg, classifier, classes)
                            except NotImplementedError:
                                logger.warning("condition.assess пока не реализован; сохраняю детекцию без оценки")
                                assessment = ConditionAssessment(Severity.OK, 0.0)
                        with trace_span(logger, "detection.geolocation", level=logging.DEBUG, log_traceback=False,
                                        timings=stage_seconds, timing_key="geolocation", detection_index=detection_index):
                            lat, lon, location_source = estimate_object_position(meta, detection.bbox, cfg)
                            crop_path = None
                            if cfg.get("export", {}).get("save_crops", False):
                                clipped = detection.bbox.clip(image.shape[1], image.shape[0])
                                x1, y1, x2, y2 = (int(value) for value in clipped.to_list())
                                crop = image[y1:y2, x1:x2]
                                if crop.size:
                                    crop_file = output_dir / "crops" / f"{frame_id}_{detection_index}.jpg"
                                    write_image(crop_file, crop)
                                    crop_path = str(crop_file)
                            observations.append(Observation(meta.frame_id, detection, assessment, sharpness, lat, lon,
                                                            location_source, crop_path, str(image_path)))
                            annotated_items.append((detection.bbox, detection.class_name, assessment.severity,
                                                   assessment.needs_review))
                    if cfg.get("export", {}).get("save_annotated_frames", False):
                        with trace_span(logger, "frame.annotated_export", level=logging.DEBUG, log_traceback=False):
                            write_image(output_dir / "annotated" / f"{frame_id}.jpg",
                                        draw_annotated(image, annotated_items))
            except Exception:
                n_errors += 1
            finally:
                if progress:
                    progress(index + 1, len(images), frame_id)
                completed = index + 1
                if completed == len(images) or (len(images) >= 20 and completed % progress_interval == 0):
                    elapsed = time.perf_counter() - pipeline_started
                    logger.info("pipeline.progress completed=%d/%d errors=%d skipped_blurry=%d detections=%d rate_fps=%.2f",
                                completed, len(images), n_errors, n_skipped_blurry, n_detections,
                                completed / elapsed if elapsed else 0.0,
                                extra={"stage": "pipeline.progress"})

    with trace_span(logger, "pipeline.merge_observations", timings=stage_seconds, timing_key="merge",
                    observations=len(observations)):
        records = merge_observations(observations, cfg, classes)
    with trace_span(logger, "pipeline.export", timings=stage_seconds, timing_key="export",
                    output_dir=str(output_dir), objects=len(records)):
        to_csv(records, output_dir / "objects.csv")
        to_geojson(records, output_dir / "objects.geojson")
        to_html_map(records, output_dir / "map.html")
    elapsed_ms = sum(stage_seconds.values()) * 1000
    summary = {
        "n_frames": len(images), "n_processed": len(images) - n_skipped_blurry - n_errors,
        "n_skipped_blurry": n_skipped_blurry, "n_errors": n_errors, "n_detections": n_detections,
        "n_objects": len(records),
        "severity": {severity.value: sum(record.severity == severity for record in records) for severity in Severity},
        "priority": {priority: sum(record.priority.value == priority for record in records) for priority in ("critical", "high", "medium", "low")},
        "timing_ms": {name: round(seconds * 1000, 2) for name, seconds in stage_seconds.items()},
        "ms_per_frame": round(elapsed_ms / len(images), 2) if images else 0.0,
        "total_elapsed_ms": round((time.perf_counter() - pipeline_started) * 1000, 2),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("pipeline.completed frames=%d processed=%d skipped_blurry=%d errors=%d detections=%d objects=%d timing_ms=%s",
                summary["n_frames"], summary["n_processed"], n_skipped_blurry, n_errors, n_detections, len(records),
                summary["timing_ms"], extra={"stage": "pipeline.summary"})
    return records
