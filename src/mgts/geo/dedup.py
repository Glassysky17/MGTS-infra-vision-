"""Объединение повторных наблюдений одного объекта (Агент 4, задача 3)."""
from __future__ import annotations

import re
import statistics

import cv2

from ..condition.scoring import priority_for
from ..schemas import ConditionAssessment, ObjectRecord, Observation, Priority, Severity
from ..utils import haversine_m

_SEVERITY_ORDER = {Severity.OK: 0, Severity.WARNING: 1, Severity.UNCERTAIN: 2, Severity.DEFECT: 3}
_PRIORITY_ORDER = {Priority.LOW: 0, Priority.MEDIUM: 1, Priority.HIGH: 2, Priority.CRITICAL: 3}


def _appearance_similarity(first: Observation, second: Observation) -> float | None:
    """Корреляция HSV-гистограмм двух доступных кропов."""
    if not first.crop_path or not second.crop_path:
        return None
    image_a, image_b = cv2.imread(first.crop_path), cv2.imread(second.crop_path)
    if image_a is None or image_b is None:
        return None
    hsv_a, hsv_b = cv2.cvtColor(image_a, cv2.COLOR_BGR2HSV), cv2.cvtColor(image_b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def _adjacent(first: str, second: str) -> bool:
    """Проверить соседство нумерованных идентификаторов кадров."""
    numbers = [re.search(r"(\d+)$", value) for value in (first, second)]
    return bool(numbers[0] and numbers[1] and abs(int(numbers[0].group(1)) - int(numbers[1].group(1))) == 1)


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
    if not observations:
        return []
    geo_cfg = cfg.get("geo", {}).get("dedup", {})
    radius = float(geo_cfg.get("radius_m", 6.0))
    use_appearance = bool(geo_cfg.get("use_appearance", False))
    appearance_min = float(geo_cfg.get("appearance_min_sim", 0.5))
    parents = list(range(len(observations)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for i, first in enumerate(observations):
        for j in range(i + 1, len(observations)):
            second = observations[j]
            if first.detection.class_name != second.detection.class_name:
                continue
            has_coords = first.est_lat is not None and first.est_lon is not None and second.est_lat is not None and second.est_lon is not None
            if has_coords:
                if haversine_m(first.est_lat, first.est_lon, second.est_lat, second.est_lon) > radius:
                    continue
                if use_appearance:
                    similarity = _appearance_similarity(first, second)
                    if similarity is not None and similarity < appearance_min:
                        continue
            elif not (_adjacent(first.frame_id, second.frame_id) and
                      first.detection.bbox.iou(second.detection.bbox) > float(geo_cfg.get("missing_coord_iou_threshold", 0.3))):
                continue
            left, right = root(i), root(j)
            if left != right:
                parents[right] = left

    clusters: dict[int, list[Observation]] = {}
    for index, observation in enumerate(observations):
        clusters.setdefault(root(index), []).append(observation)
    records = []
    for cluster in clusters.values():
        max_sharpness = max((item.sharpness for item in cluster), default=0.0)
        best = max(cluster, key=lambda item: item.detection.score * (item.sharpness / max_sharpness if max_sharpness else 1.0))
        latitudes = [item.est_lat for item in cluster if item.est_lat is not None]
        longitudes = [item.est_lon for item in cluster if item.est_lon is not None]
        findings_by_code = {}
        for item in cluster:
            for finding in item.assessment.findings:
                if finding.code not in findings_by_code or finding.score > findings_by_code[finding.code].score:
                    findings_by_code[finding.code] = finding
        findings = list(findings_by_code.values())
        severity = max((item.assessment.severity for item in cluster), key=_SEVERITY_ORDER.get)
        defects = sum(item.assessment.severity == Severity.DEFECT for item in cluster)
        if defects >= len(cluster) / 2 or best.assessment.severity == Severity.DEFECT:
            severity = Severity.DEFECT
        needs_review = any(item.assessment.needs_review for item in cluster)
        scores = [item.assessment.score for item in cluster]
        condition_score = max(scores, default=0.0)
        assessment = ConditionAssessment(severity, condition_score, findings, needs_review)
        priority = priority_for(best.detection.class_name, assessment, cfg, classes)
        records.append(ObjectRecord(
            object_id="", class_name=best.detection.class_name, severity=severity,
            condition_score=condition_score, needs_review=needs_review, priority=priority,
            findings=findings, lat=statistics.median(latitudes) if latitudes else None,
            lon=statistics.median(longitudes) if longitudes else None, location_source=best.location_source,
            confidence=best.detection.score, n_observations=len(cluster), best_frame_id=best.frame_id,
            best_bbox=best.detection.bbox, frame_ids=list(dict.fromkeys(item.frame_id for item in cluster)),
            crop_path=best.crop_path, frame_path=best.frame_path,
        ))
    records.sort(key=lambda record: (-_PRIORITY_ORDER[record.priority], -record.condition_score, record.best_frame_id))
    for index, record in enumerate(records, start=1):
        record.object_id = f"OBJ-{index:04d}"
    return records
