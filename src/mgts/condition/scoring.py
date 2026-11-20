"""Итоговая оценка состояния и приоритет (Агент 3, задача 3). Это «точка входа» модуля для pipeline."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..config import class_info, load_classes
from ..schemas import ConditionAssessment, DefectFinding, Detection, Priority, Severity
from .geometry import (
    analyze_hatch, corrosion_ratio, crack_score, estimate_pole_tilt,
    segment_foreground, surface_subsidence_score, vertical_reference_angle,
)

logger = logging.getLogger(__name__)


def assess(  # CONTRACT
    frame_bgr: np.ndarray,
    detection: Detection,
    cfg: dict,
    classifier: Any = None,
    classes: dict | None = None,
) -> ConditionAssessment:
    """Оценить состояние ОДНОГО обнаруженного объекта.

    Шаги:
      1. Вырезать crop по detection.bbox с полями (cfg.dataset.crop_padding), клипнуть по кадру.
         Слишком маленький crop (<24 px по короткой стороне) -> severity=UNCERTAIN, needs_review=True.
      2. По классу объекта (classes.yaml -> defects) запустить применимые геометрические проверки:
           cable_manhole: analyze_hatch (ellipse_residual, dark_gap_ratio), crack_score, surface_subsidence_score
           utility_pole : estimate_pole_tilt (минус vertical_reference_angle), crack_score, corrosion_ratio
           cable_cabinet: corrosion_ratio, crack_score
         и, если classifier.is_ready, добавить вероятности классификатора.
      3. Каждый сработавший признак -> DefectFinding(code, score 0..1, source, evidence «наклон 9.3° > порога 8°»).
         Нормировка score признака: линейно от warn-порога (0.35) до defect-порога (0.6) и выше — см. cfg.condition.
      4. Агрегация: итоговый score = 1 - Π(1 - w_i * s_i) (noisy-OR) по findings с весами defects[code].weight.
         severity: DEFECT если score >= defect_threshold; WARNING если >= warning_threshold; иначе OK.
      5. needs_review=True, если: |score - порог| < uncertain_band, ИЛИ detection.score < min_det_score,
         ИЛИ геометрия и классификатор противоречат друг другу (один >0.7, другой <0.2),
         ИЛИ confidence геометрии низкая при наличии подозрения. В этом случае severity=UNCERTAIN, если score рядом с порогом.
    Функция не должна падать: любое исключение внутри признака логируется, признак пропускается.
    """
    classes = classes or load_classes()
    height, width = frame_bgr.shape[:2] if isinstance(frame_bgr, np.ndarray) and frame_bgr.ndim >= 2 else (0, 0)
    if not width or not height or frame_bgr.size == 0:
        return ConditionAssessment(Severity.UNCERTAIN, 0.0, [], True)
    padding = float(cfg.get("dataset", {}).get("crop_padding", 0.15))
    box = detection.bbox.pad(padding, width, height)
    x1, y1, x2, y2 = (int(round(value)) for value in box.to_list())
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < 24:
        return ConditionAssessment(Severity.UNCERTAIN, 0.0, [], True)

    try:
        applicable = set(class_info(detection.class_name, classes).get("defects", []))
    except (KeyError, TypeError):
        applicable = set()
    findings, geometry_confidence, geometry_suspected = _geometry_findings(
        frame_bgr, crop, box.to_list(), detection.class_name, cfg, applicable
    )
    classifier_scores: dict[str, float] = {}
    if classifier is not None:
        try:
            if classifier.is_ready:
                classifier_scores = classifier.predict(crop, detection.class_name) or {}
                if "generic_damage" in classifier_scores:
                    applicable.add("generic_damage")
        except Exception:
            logger.exception("Классификатор состояния отказал для %s", detection.frame_id)
    classifier_high = False
    classifier_low = False
    probability_threshold = float(cfg.get("condition", {}).get("classifier", {}).get("probability_threshold", 0.5))
    for code, probability in classifier_scores.items():
        if code not in applicable:
            continue
        score = float(np.clip(probability, 0.0, 1.0))
        classifier_high |= score > 0.7
        classifier_low |= score < 0.2
        try:
            threshold_for = getattr(classifier, "threshold_for", None)
            threshold = float(threshold_for(code, probability_threshold)) if callable(threshold_for) else probability_threshold
        except (TypeError, ValueError, KeyError):
            threshold = probability_threshold
        if score >= threshold:
            findings.append(DefectFinding(code, score, "classifier", f"вероятность {score:.2f} ≥ порога {threshold:.2f}"))
    geometry_score = max((finding.score for finding in findings if finding.source == "geometry"), default=0.0)

    defect_cfg = classes.get("defects", {})
    complement = 1.0
    for finding in findings:
        weight = float(defect_cfg.get(finding.code, {}).get("weight", 1.0))
        complement *= 1.0 - float(np.clip(weight * finding.score, 0.0, 1.0))
    score = float(np.clip(1.0 - complement, 0.0, 1.0))
    scoring_cfg = cfg.get("condition", {}).get("scoring", {})
    defect_threshold = float(scoring_cfg.get("defect_threshold", 0.6))
    warning_threshold = float(scoring_cfg.get("warning_threshold", 0.35))
    if score >= defect_threshold:
        severity = Severity.DEFECT
    elif score >= warning_threshold:
        severity = Severity.WARNING
    else:
        severity = Severity.OK
    band = float(scoring_cfg.get("uncertain_band", 0.1))
    needs_review = (min(abs(score - warning_threshold), abs(score - defect_threshold)) < band or
                    detection.score < float(scoring_cfg.get("min_det_score", 0.3)) or
                    (geometry_score > 0.7 and classifier_low) or
                    (classifier_high and geometry_score < 0.2) or
                    (geometry_suspected and geometry_confidence < float(scoring_cfg.get("min_geometry_confidence", 0.3))))
    if needs_review and severity is not Severity.OK:
        severity = Severity.UNCERTAIN
    return ConditionAssessment(severity, score, findings, needs_review)


def _threshold_score(value: float, warning: float, defect: float) -> float:
    """Нормировать выраженность признака: warn → 0.35, defect → 1.0."""
    if value < warning:
        return 0.0
    fraction = (value - warning) / max(defect - warning, 1e-6)
    return float(np.clip(fraction, 0.0, 1.0) * 0.65 + 0.35)


def _geometry_findings(
    frame_bgr: np.ndarray,
    crop_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    class_name: str,
    cfg: dict,
    applicable: set[str],
) -> tuple[list[DefectFinding], float, bool]:
    """Выполнить геометрические проверки класса; одна ошибка пропускает только свой признак."""
    geometry = cfg.get("condition", {}).get("geometry", {})
    findings: list[DefectFinding] = []
    confidence, suspected = 1.0, False
    if class_name == "utility_pole":
        try:
            tilt = estimate_pole_tilt(crop_bgr)
            angle = tilt.angle_deg - vertical_reference_angle(frame_bgr)
            threshold = float(geometry.get("tilt_warn_deg", 4.0))
            suspected = abs(angle) >= threshold
            if suspected and "pole_tilt" in applicable:
                score = _threshold_score(abs(angle), threshold, float(geometry.get("tilt_defect_deg", 8.0)))
                findings.append(DefectFinding("pole_tilt", score, "geometry",
                                              f"наклон {abs(angle):.1f}° > порога {threshold:.1f}°"))
            confidence = float(np.clip(tilt.confidence, 0.0, 1.0))
        except Exception:
            logger.exception("Не удалось оценить наклон опоры")
    if class_name in {"utility_pole", "cable_manhole", "cable_cabinet"}:
        _try_add_crack(crop_bgr, geometry, applicable, findings)
    if class_name in {"utility_pole", "cable_cabinet"}:
        _try_add_corrosion(crop_bgr, geometry, applicable, findings)
    if class_name == "cable_manhole":
        _try_add_hatch(crop_bgr, geometry, applicable, findings)
        try:
            value = surface_subsidence_score(frame_bgr, bbox)
            warn = float(geometry.get("subsidence_warn", 0.3))
            suspected |= value >= warn
            if value >= warn and "subsidence_around" in applicable:
                score = _threshold_score(value, warn, float(geometry.get("subsidence_defect", 0.6)))
                findings.append(DefectFinding("subsidence_around", score, "geometry",
                                              f"оценка просадки {value:.2f} ≥ порога {warn:.2f}"))
        except Exception:
            logger.exception("Не удалось оценить просадку покрытия")
    return findings, confidence, suspected


def _try_add_crack(crop: np.ndarray, geometry: dict, applicable: set[str], findings: list[DefectFinding]) -> None:
    """Добавить finding трещин, если признак превышает порог."""
    try:
        value = crack_score(crop, segment_foreground(crop))
        warn = float(geometry.get("crack_score_warn", 0.25))
        if value >= warn:
            code = "cover_broken" if "cover_broken" in applicable else "pole_crack"
            if code in applicable:
                defect = float(geometry.get("crack_score_defect", 0.55))
                findings.append(DefectFinding(code, _threshold_score(value, warn, defect), "geometry",
                                              f"оценка трещин {value:.2f} ≥ порога {warn:.2f}"))
    except Exception:
        logger.exception("Не удалось оценить трещины")


def _try_add_corrosion(crop: np.ndarray, geometry: dict, applicable: set[str], findings: list[DefectFinding]) -> None:
    """Добавить finding коррозии, если доля ржавчины превышает порог."""
    try:
        value = corrosion_ratio(crop)
        warn = float(geometry.get("corrosion_ratio_warn", 0.1))
        if value >= warn and "corrosion" in applicable:
            defect = float(geometry.get("corrosion_ratio_defect", 0.25))
            findings.append(DefectFinding("corrosion", _threshold_score(value, warn, defect), "geometry",
                                          f"доля ржавчины {value:.2f} ≥ порога {warn:.2f}"))
    except Exception:
        logger.exception("Не удалось оценить коррозию")


def _try_add_hatch(crop: np.ndarray, geometry: dict, applicable: set[str], findings: list[DefectFinding]) -> None:
    """Добавить finding смещённой крышки по форме и щели."""
    try:
        hatch = analyze_hatch(crop)
        circularity_limit = float(geometry.get("hatch_circularity_min", 0.85))
        gap_limit = float(geometry.get("hatch_dark_gap_warn", 0.2))
        if hatch.found and (hatch.circularity < circularity_limit or hatch.dark_gap_ratio >= gap_limit):
            score = max((circularity_limit - hatch.circularity) / max(circularity_limit, 1e-6),
                        hatch.dark_gap_ratio / max(gap_limit, 1e-6))
            evidence = (f"округлость {hatch.circularity:.2f} < {circularity_limit:.2f}; "
                        f"тёмная щель {hatch.dark_gap_ratio:.2f}")
            if "cover_displaced" in applicable:
                findings.append(DefectFinding("cover_displaced", float(np.clip(0.35 + 0.65 * score, 0, 1)),
                                              "geometry", evidence))
    except Exception:
        logger.exception("Не удалось оценить форму крышки люка")


def priority_for(class_name: str, assessment: ConditionAssessment, cfg: dict, classes: dict | None = None) -> Priority:  # CONTRACT
    """Приоритет проверки/ремонта: критичность класса (classes.yaml) × score состояния.
    p = criticality * score (+0.15, если среди findings cover_missing / cable_sag — угроза безопасности).
    CRITICAL: p >= 0.75; HIGH: >= 0.5; MEDIUM: >= 0.25 или needs_review; иначе LOW.
    Объекты severity=OK с needs_review=False -> LOW."""
    if assessment.severity is Severity.OK and not assessment.needs_review:
        return Priority.LOW
    classes = classes or load_classes()
    try:
        criticality = float(class_info(class_name, classes).get("criticality", 0.0))
    except (KeyError, TypeError, ValueError):
        criticality = 0.0
    codes = {finding.code for finding in assessment.findings}
    safety_bonus = float(cfg.get("condition", {}).get("scoring", {}).get("safety_bonus", 0.15))
    priority_score = criticality * assessment.score + (safety_bonus if codes & {"cover_missing", "cable_sag"} else 0.0)
    scoring = cfg.get("condition", {}).get("scoring", {})
    if priority_score >= float(scoring.get("priority_critical", 0.75)):
        return Priority.CRITICAL
    if priority_score >= float(scoring.get("priority_high", 0.5)):
        return Priority.HIGH
    if priority_score >= float(scoring.get("priority_medium", 0.25)) or assessment.needs_review:
        return Priority.MEDIUM
    return Priority.LOW
