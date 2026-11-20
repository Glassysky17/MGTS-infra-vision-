"""Итоговая оценка состояния и приоритет (Агент 3, задача 3). Это «точка входа» модуля для pipeline."""
from __future__ import annotations

import numpy as np

from ..schemas import ConditionAssessment, Detection, Priority


def assess(  # CONTRACT
    frame_bgr: np.ndarray,
    detection: Detection,
    cfg: dict,
    classifier=None,
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
    raise NotImplementedError("Агент 3: tasks/agent_3_condition.md §3")


def priority_for(class_name: str, assessment: ConditionAssessment, cfg: dict, classes: dict | None = None) -> Priority:  # CONTRACT
    """Приоритет проверки/ремонта: критичность класса (classes.yaml) × score состояния.
    p = criticality * score (+0.15, если среди findings cover_missing / cable_sag — угроза безопасности).
    CRITICAL: p >= 0.75; HIGH: >= 0.5; MEDIUM: >= 0.25 или needs_review; иначе LOW.
    Объекты severity=OK с needs_review=False -> LOW."""
    raise NotImplementedError("Агент 3: tasks/agent_3_condition.md §3")
