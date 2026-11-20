"""Синтез дефектов на исправных объектах (Агент 1, задача 6).

Нужен, если в датасете нет/мало дефектных примеров. Генерируемые дефекты ДОЛЖНЫ иметь
метку (код из configs/classes.yaml), чтобы Агент 3 мог обучать и проверять классификатор/геометрию.
"""
from __future__ import annotations

import numpy as np

from ..schemas import BBox


def synth_cracks(img: np.ndarray, bbox: BBox, rng: np.random.Generator | None = None, n: int = 3) -> np.ndarray:  # CONTRACT
    """Нарисовать n реалистичных трещин внутри bbox: случайные ветвящиеся полилинии
    (случайное блуждание с ветвлением), тёмный цвет с мягкими краями и текстурным шумом, прозрачность 0.6–0.9.
    Метка: cover_broken / pole_crack."""
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §6")


def synth_rust(img: np.ndarray, bbox: BBox, rng: np.random.Generator | None = None, coverage: float = 0.25) -> np.ndarray:  # CONTRACT
    """Ржавчина: пятна (шум Перлина/gaussian-blob) оранжево-коричневых оттенков, смешанные по маске
    внутри bbox с покрытием ~coverage площади. Метка: corrosion."""
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §6")


def synth_tilt(img: np.ndarray, bbox: BBox, angle_deg: float) -> tuple[np.ndarray, BBox]:  # CONTRACT
    """Наклонить объект (опору) на angle_deg вокруг основания (нижний центр bbox):
    вырезать область, повернуть аффинным преобразованием с альфа-маской, вклеить на закрытый фон
    (inpaint cv2.INPAINT_TELEA). Вернуть новое изображение и новую рамку. Метка: pole_tilt."""
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §6")


def synth_displaced_cover(img: np.ndarray, bbox: BBox, shift_frac: float = 0.15, rng: np.random.Generator | None = None) -> np.ndarray:  # CONTRACT
    """Смещённая крышка люка: сдвинуть содержимое эллипса люка на shift_frac, оставив тёмную щель.
    Метка: cover_displaced."""
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §6")


def generate_synthetic_defects(  # CONTRACT
    dataset_dir: str, out_dir: str, per_class: int = 100, seed: int = 0
) -> str:
    """Для исправных объектов датасета сгенерировать per_class дефектных вариантов на класс.
    Сохранить вырезки в out_dir/<class>/<defect_code>/ + index.csv (path, class_name, label, synthetic=1).
    Реальные и синтетические данные ДОЛЖНЫ быть различимы в index.csv (колонка synthetic),
    а валидацию классификатора считать только на реальных примерах, если они есть.
    Вернуть путь к index.csv.
    """
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §6")
