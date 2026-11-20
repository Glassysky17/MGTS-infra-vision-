"""Геометрические и «классические» признаки дефектов (Агент 3, задача 1).

Здесь работают алгоритмы компьютерной графики/обработки изображений без обучения: они объяснимы
и не требуют большого датасета. Все функции принимают ВЫРЕЗКУ объекта (BGR uint8, с небольшими полями).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TiltResult:  # CONTRACT
    angle_deg: float        # отклонение от вертикали, знак: + = наклон вправо по кадру
    confidence: float       # 0..1: насколько выражена вытянутая вертикальная структура
    method: str             # 'pca_mask' | 'hough' | 'none'


@dataclass
class HatchResult:  # CONTRACT
    circularity: float      # 4πA/P² контура люка (1 = идеальная окружность; в перспективе — эллипс, см. ellipse_residual)
    ellipse_residual: float # средняя невязка контура относительно вписанного эллипса, нормированная на размер (0 = идеально)
    dark_gap_ratio: float   # доля «тёмной щели» вдоль контура (признак смещённой/неплотной крышки)
    found: bool             # удалось ли выделить контур люка


def segment_foreground(crop_bgr: np.ndarray) -> np.ndarray:  # CONTRACT
    """Грубая маска объекта внутри вырезки (uint8 0/255): GrabCut с прямоугольником по центру вырезки
    ИЛИ адаптивный порог + морфология (opening/closing) + крупнейшая компонента. Использовать как основу
    для остальных функций. Если сегментация вырождена (маска <5% или >95%) — вернуть пустую маску."""
    raise NotImplementedError("Агент 3: tasks/agent_3_condition.md §1")


def estimate_pole_tilt(crop_bgr: np.ndarray, mask: np.ndarray | None = None) -> TiltResult:  # CONTRACT
    """Угол наклона опоры.
    Метод 1 (основной): PCA по координатам пикселей маски -> главная ось; угол оси к вертикали.
    Метод 2 (проверка): Canny + HoughLinesP, взять длинные почти вертикальные отрезки, взвешенное по длине
        среднее углов. Если методы расходятся >3° — снизить confidence.
    Для не вытянутых масок (отношение собственных значений < 3) вернуть confidence ≈ 0.
    Тест: нарисованный прямоугольник 20×200 px, повёрнутый на 10°, даёт |angle| ≈ 10° ± 1.5°."""
    raise NotImplementedError("Агент 3: tasks/agent_3_condition.md §1")


def vertical_reference_angle(frame_bgr: np.ndarray) -> float:  # CONTRACT
    """Крен камеры: по доминирующим вертикальным линиям ВСЕГО кадра (углы зданий, столбы) оценить
    поворот, который надо вычесть из угла опоры. Вернуть градусы (0.0, если надёжных линий мало).
    Использовать медиану углов отрезков Хафа в диапазоне ±15° от вертикали."""
    raise NotImplementedError("Агент 3: tasks/agent_3_condition.md §1")


def analyze_hatch(crop_bgr: np.ndarray) -> HatchResult:  # CONTRACT
    """Анализ люка/крышки: края (Canny) -> контуры -> самый крупный почти замкнутый контур по центру ->
    circularity, вписанный эллипс (cv2.fitEllipse) и невязка, тёмная щель (доля тёмных пикселей в кольце
    шириной ~8% вдоль контура относительно медианной яркости)."""
    raise NotImplementedError("Агент 3: tasks/agent_3_condition.md §1")


def crack_score(crop_bgr: np.ndarray, mask: np.ndarray | None = None) -> float:  # CONTRACT
    """0..1 — выраженность трещин/сколов: тёмные тонкие линии (black-hat по серому + порог) ->
    скелетизация -> суммарная длина / периметр объекта, нормировка в [0,1] через насыщающую функцию.
    Регулярные детали (рёбра люка, логотип) не должны давать высокий балл: сравнивать с ожидаемым
    «чистым» паттерном (низкая плотность связных ветвящихся линий)."""
    raise NotImplementedError("Агент 3: tasks/agent_3_condition.md §1")


def corrosion_ratio(crop_bgr: np.ndarray, mask: np.ndarray | None = None) -> float:  # CONTRACT
    """Доля пикселей маски с «ржавым» цветом: HSV-диапазон оранжево-коричневого (H≈5–25, S>80, V 40–200),
    после морфологической фильтрации мелких пятен. 0..1."""
    raise NotImplementedError("Агент 3: tasks/agent_3_condition.md §1")


def surface_subsidence_score(frame_bgr: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> float:  # CONTRACT
    """0..1 — признак просадки/разрушения покрытия вокруг колодца: кольцо вокруг bbox (ширина ~0.5 bbox),
    сравнить текстуру/края кольца с «фоновой» полосой дальше от объекта: рост плотности краёв и дисперсии
    локальной яркости, наличие тёмной полосы/ямы. Нормировать в [0,1]."""
    raise NotImplementedError("Агент 3: tasks/agent_3_condition.md §1")
