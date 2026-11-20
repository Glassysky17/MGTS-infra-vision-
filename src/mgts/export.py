"""Экспорт результатов (Агент 4, задача 5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .schemas import ObjectRecord


def to_csv(records: list[ObjectRecord], path: str | Path) -> Path:  # CONTRACT
    """CSV (UTF-8 с BOM, чтобы Excel открывал кириллицу) с колонками schemas.CSV_COLUMNS,
    + колонка title_ru (русское название класса из classes.yaml). Порядок строк как в records."""
    raise NotImplementedError("Агент 4: tasks/agent_4_integration.md §5")


def to_geojson(records: list[ObjectRecord], path: str | Path) -> Path:  # CONTRACT
    """FeatureCollection; Point [lon, lat] (ВНИМАНИЕ: порядок lon, lat!), properties = ObjectRecord.to_row().
    Объекты без координат НЕ включать в geometry (geometry: null допустим) — но не терять в CSV."""
    raise NotImplementedError("Агент 4: tasks/agent_4_integration.md §5")


def to_html_map(records: list[ObjectRecord], path: str | Path, title: str = "МГТС: объекты для проверки") -> Path:  # CONTRACT
    """Автономный HTML (folium): маркеры по приоритету (critical красный, high оранжевый, medium жёлтый, low зелёный),
    всплывающее окно: тип, состояние, дефекты (description), уверенность, миниатюра кропа (base64, <= 160 px),
    метка «⚠ ручная проверка» при needs_review; легенда; фильтр слоёв по priority.
    Если нет ни одной координаты — карта-заглушка с таблицей объектов. Если folium не установлен — простая HTML-таблица."""
    raise NotImplementedError("Агент 4: tasks/agent_4_integration.md §5")


def draw_annotated(frame_bgr: np.ndarray, items: list[tuple]) -> np.ndarray:  # CONTRACT
    """Нарисовать рамки на копии кадра. items: [(BBox, label_str, Severity, needs_review)].
    Цвет по severity (ok зелёный, warning жёлтый, defect красный, uncertain серый/фиолетовый),
    подпись с фоном для читаемости, толщина линии пропорциональна размеру кадра."""
    raise NotImplementedError("Агент 4: tasks/agent_4_integration.md §5")
