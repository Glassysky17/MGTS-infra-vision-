"""Геометрические и «классические» признаки дефектов (Агент 3, задача 1).

Здесь работают алгоритмы компьютерной графики/обработки изображений без обучения: они объяснимы
и не требуют большого датасета. Все функции принимают ВЫРЕЗКУ объекта (BGR uint8, с небольшими полями).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
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
    if crop_bgr is None or crop_bgr.ndim < 2:
        return np.zeros((0, 0), dtype=np.uint8)
    height, width = crop_bgr.shape[:2]
    empty = np.zeros((height, width), dtype=np.uint8)
    if height < 31 or width < 31 or crop_bgr.size == 0:
        return empty
    try:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 5)
        if np.mean(mask > 0) > 0.5:
            mask = cv2.bitwise_not(mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        cx, cy = width // 2, height // 2
        center_label = int(labels[cy, cx])
        if center_label == 0:
            candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_LEFT] <= cx <
                          stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] and
                          stats[i, cv2.CC_STAT_TOP] <= cy <
                          stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]]
            center_label = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA], default=0)
        result = np.where(labels == center_label, 255, 0).astype(np.uint8) if center_label else empty
        ratio = np.count_nonzero(result) / result.size
        return result if 0.05 <= ratio <= 0.95 else empty
    except (cv2.error, ValueError, TypeError):
        return empty


def estimate_pole_tilt(crop_bgr: np.ndarray, mask: np.ndarray | None = None) -> TiltResult:  # CONTRACT
    """Угол наклона опоры.
    Метод 1 (основной): PCA по координатам пикселей маски -> главная ось; угол оси к вертикали.
    Метод 2 (проверка): Canny + HoughLinesP, взять длинные почти вертикальные отрезки, взвешенное по длине
        среднее углов. Если методы расходятся >3° — снизить confidence.
    Для не вытянутых масок (отношение собственных значений < 3) вернуть confidence ≈ 0.
    Тест: нарисованный прямоугольник 20×200 px, повёрнутый на 10°, даёт |angle| ≈ 10° ± 1.5°."""
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.ndim < 2:
        return TiltResult(0.0, 0.0, "none")
    height = crop_bgr.shape[0]
    foreground = segment_foreground(crop_bgr) if mask is None else np.asarray(mask)
    if foreground.ndim != 2 or foreground.shape != crop_bgr.shape[:2]:
        return TiltResult(0.0, 0.0, "none")
    points = np.column_stack(np.nonzero(foreground > 0))[:, ::-1].astype(np.float64)
    if len(points) < 2:
        return TiltResult(0.0, 0.0, "none")
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points, rowvar=False))
    ratio = float(eigenvalues[-1] / max(eigenvalues[0], 1e-6))
    vx, vy = eigenvectors[:, -1]
    if vy > 0:
        vx, vy = -vx, -vy
    angle = float(np.degrees(np.arctan2(vx, -vy)))
    confidence = float(np.clip((ratio - 3.0) / 10.0, 0.0, 1.0)) if ratio >= 3.0 else 0.0
    if confidence == 0.0:
        return TiltResult(angle, 0.0, "pca_mask")
    try:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
        lines = cv2.HoughLinesP(cv2.Canny(gray, 50, 150), 1, np.pi / 180,
                                threshold=max(10, int(0.1 * height)),
                                minLineLength=max(1, int(0.4 * height)), maxLineGap=10)
        hough_angles: list[tuple[float, float]] = []
        if lines is not None:
            for line in np.asarray(lines).reshape(-1, 4):
                x1, y1, x2, y2 = (float(v) for v in line)
                dx, dy = x2 - x1, y2 - y1
                if dy > 0:
                    dx, dy = -dx, -dy
                line_angle = float(np.degrees(np.arctan2(dx, -dy)))
                length = float(np.hypot(dx, dy))
                if abs(line_angle) < 30.0:
                    hough_angles.append((line_angle, length))
        if hough_angles:
            hough_angle = sum(a * length for a, length in hough_angles) / sum(length for _, length in hough_angles)
            if abs(angle - hough_angle) > 3.0:
                confidence *= 0.5
    except (cv2.error, ValueError, TypeError):
        pass
    return TiltResult(angle, confidence, "pca_mask")


def vertical_reference_angle(frame_bgr: np.ndarray) -> float:  # CONTRACT
    """Крен камеры: по доминирующим вертикальным линиям ВСЕГО кадра (углы зданий, столбы) оценить
    поворот, который надо вычесть из угла опоры. Вернуть градусы (0.0, если надёжных линий мало).
    Использовать медиану углов отрезков Хафа в диапазоне ±15° от вертикали."""
    if frame_bgr is None or frame_bgr.size == 0 or frame_bgr.ndim < 2:
        return 0.0
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        height = gray.shape[0]
        lines = cv2.HoughLinesP(cv2.Canny(gray, 50, 150), 1, np.pi / 180,
                                threshold=max(10, int(0.05 * height)),
                                minLineLength=max(1, int(0.1 * height)), maxLineGap=8)
        if lines is None:
            return 0.0
        angles = []
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            dx, dy = float(x2 - x1), float(y2 - y1)
            if dy > 0:
                dx, dy = -dx, -dy
            angle = float(np.degrees(np.arctan2(dx, -dy)))
            if abs(angle) <= 15.0:
                angles.append(angle)
        return float(np.median(angles)) if len(angles) >= 5 else 0.0
    except (cv2.error, ValueError, TypeError):
        return 0.0


def analyze_hatch(crop_bgr: np.ndarray) -> HatchResult:  # CONTRACT
    """Анализ люка/крышки: края (Canny) -> контуры -> самый крупный почти замкнутый контур по центру ->
    circularity, вписанный эллипс (cv2.fitEllipse) и невязка, тёмная щель (доля тёмных пикселей в кольце
    шириной ~8% вдоль контура относительно медианной яркости)."""
    empty = HatchResult(0.0, 0.0, 0.0, False)
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.ndim < 2:
        return empty
    height, width = crop_bgr.shape[:2]
    if min(height, width) < 12:
        return empty
    try:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        center_contours = [contour for contour in contours if
                           width / 3 <= cv2.moments(contour).get("m10", 0.0) /
                           max(cv2.moments(contour).get("m00", 0.0), 1e-6) <= 2 * width / 3 and
                           height / 3 <= cv2.moments(contour).get("m01", 0.0) /
                           max(cv2.moments(contour).get("m00", 0.0), 1e-6) <= 2 * height / 3]
        if not center_contours:
            return empty
        contour = max(center_contours, key=cv2.contourArea)
        area, perimeter = float(cv2.contourArea(contour)), float(cv2.arcLength(contour, True))
        if area <= 0 or perimeter <= 0:
            return empty
        circularity = float(4 * np.pi * area / perimeter ** 2)
        residual = 0.0
        if len(contour) >= 5:
            (cx, cy), (axis_a, axis_b), rotation = cv2.fitEllipse(contour)
            a, b = max(axis_a / 2.0, 1e-6), max(axis_b / 2.0, 1e-6)
            theta = np.radians(rotation)
            points = contour.reshape(-1, 2).astype(np.float64) - (cx, cy)
            x_rot = points[:, 0] * np.cos(theta) + points[:, 1] * np.sin(theta)
            y_rot = -points[:, 0] * np.sin(theta) + points[:, 1] * np.cos(theta)
            residual = float(np.mean(np.abs((x_rot / a) ** 2 + (y_rot / b) ** 2 - 1.0)))
        ring = np.zeros((height, width), np.uint8)
        thickness = max(1, int(round(0.08 * min(height, width))))
        cv2.drawContours(ring, [contour], -1, 255, thickness)
        pixels = gray[ring > 0]
        median = float(np.median(gray))
        gap_ratio = float(np.mean(pixels < 0.6 * median)) if pixels.size and median > 0 else 0.0
        return HatchResult(float(np.clip(circularity, 0.0, 1.0)), residual, gap_ratio, True)
    except (cv2.error, ValueError, TypeError, ZeroDivisionError):
        return empty


def crack_score(crop_bgr: np.ndarray, mask: np.ndarray | None = None) -> float:  # CONTRACT
    """0..1 — выраженность трещин/сколов: тёмные тонкие линии (black-hat по серому + порог) ->
    скелетизация -> суммарная длина / периметр объекта, нормировка в [0,1] через насыщающую функцию.
    Регулярные детали (рёбра люка, логотип) не должны давать высокий балл: сравнивать с ожидаемым
    «чистым» паттерном (низкая плотность связных ветвящихся линий)."""
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.ndim < 2:
        return 0.0
    height, width = crop_bgr.shape[:2]
    if min(height, width) < 3:
        return 0.0
    try:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((15, 15), np.uint8))
        _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if mask is not None:
            if mask.shape != binary.shape:
                return 0.0
            binary = cv2.bitwise_and(binary, binary, mask=np.where(mask > 0, 255, 0).astype(np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        line_mask = np.zeros_like(binary)
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] < 3:
                continue
            x, y, box_w, box_h = stats[component, :4]
            local = labels[y:y + box_h, x:x + box_w] == component
            points = np.column_stack(np.nonzero(local))
            if len(points) >= 3:
                eigenvalues = np.linalg.eigvalsh(np.cov(points, rowvar=False))
                if eigenvalues[-1] / max(eigenvalues[0], 1e-6) >= 3.0:
                    line_mask[y:y + box_h, x:x + box_w][local] = 255
        skeleton = _skeletonize(line_mask > 0)
        length = int(np.count_nonzero(skeleton))
        return float(np.clip(1.0 - np.exp(-3.0 * length / (2 * (height + width))), 0.0, 1.0))
    except (cv2.error, ValueError, TypeError):
        return 0.0


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """Истончить бинарную маску методом Zhang–Suen без необязательных пакетов."""
    image = np.pad(binary.astype(np.uint8), 1)
    while True:
        changed = False
        for phase in (0, 1):
            p2, p3, p4 = image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:]
            p5, p6, p7 = image[2:, 2:], image[2:, 1:-1], image[2:, :-2]
            p8, p9 = image[1:-1, :-2], image[:-2, :-2]
            center = image[1:-1, 1:-1]
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(neighbors)
            transitions = sum((a == 0) & (b == 1) for a, b in zip(neighbors, neighbors[1:] + neighbors[:1]))
            condition = ((p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0) if phase == 0 else
                         (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0))
            remove = (center == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & condition
            if np.any(remove):
                center[remove] = 0
                changed = True
        if not changed:
            return image[1:-1, 1:-1].astype(bool)


def corrosion_ratio(crop_bgr: np.ndarray, mask: np.ndarray | None = None) -> float:  # CONTRACT
    """Доля пикселей маски с «ржавым» цветом: HSV-диапазон оранжево-коричневого (H≈5–25, S>80, V 40–200),
    после морфологической фильтрации мелких пятен. 0..1."""
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.ndim < 2:
        return 0.0
    try:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        rust = cv2.inRange(hsv, np.array((5, 80, 40), np.uint8), np.array((25, 255, 200), np.uint8))
        rust = cv2.morphologyEx(rust, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        if mask is not None:
            if mask.shape != rust.shape:
                return 0.0
            object_mask = mask > 0
        else:
            object_mask = np.ones(rust.shape, dtype=bool)
        area = int(np.count_nonzero(object_mask))
        return float(np.count_nonzero((rust > 0) & object_mask) / area) if area else 0.0
    except (cv2.error, ValueError, TypeError):
        return 0.0


def surface_subsidence_score(frame_bgr: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> float:  # CONTRACT
    """0..1 — признак просадки/разрушения покрытия вокруг колодца: кольцо вокруг bbox (ширина ~0.5 bbox),
    сравнить текстуру/края кольца с «фоновой» полосой дальше от объекта: рост плотности краёв и дисперсии
    локальной яркости, наличие тёмной полосы/ямы. Нормировать в [0,1]."""
    if frame_bgr is None or frame_bgr.size == 0 or frame_bgr.ndim < 2:
        return 0.0
    height, width = frame_bgr.shape[:2]
    try:
        x1, y1, x2, y2 = (int(round(value)) for value in bbox_xyxy)
        bw, bh = max(x2 - x1, 0), max(y2 - y1, 0)
        if not bw or not bh:
            return 0.0
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ring_box = (max(0, int(cx - bw)), max(0, int(cy - bh)),
                    min(width, int(cx + bw)), min(height, int(cy + bh)))
        outer_box = (max(0, int(cx - bw * 1.5)), max(0, int(cy - bh * 1.5)),
                     min(width, int(cx + bw * 1.5)), min(height, int(cy + bh * 1.5)))
        x1r, y1r, x2r, y2r = ring_box
        x1o, y1o, x2o, y2o = outer_box
        ring_mask = np.zeros((height, width), np.uint8)
        bg_mask = np.zeros_like(ring_mask)
        ring_mask[y1r:y2r, x1r:x2r] = 1
        bg_mask[y1o:y2o, x1o:x2o] = 1
        bg_mask[y1r:y2r, x1r:x2r] = 0
        # Убрать собственно рамку объекта из сравнительного кольца.
        x1b, y1b, x2b, y2b = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        ring_mask[y1b:y2b, x1b:x2b] = 0
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        edges = cv2.Canny(gray, 50, 150) > 0
        ring_pixels, bg_pixels = ring_mask > 0, bg_mask > 0
        if not np.any(ring_pixels) or not np.any(bg_pixels):
            return 0.0
        d_ring, d_bg = float(np.mean(edges[ring_pixels])), float(np.mean(edges[bg_pixels]))
        v_ring = float(np.var(gray[ring_pixels]))
        v_bg = float(np.var(gray[bg_pixels]))
        edge_ratio = d_ring / d_bg if d_bg > 0 else 0.0
        variance_ratio = v_ring / v_bg if v_bg > 0 else 0.0
        return float(np.clip(0.5 * (edge_ratio - 1.0) + 0.5 * (variance_ratio - 1.0), 0.0, 1.0))
    except (cv2.error, ValueError, TypeError, ZeroDivisionError):
        return 0.0
