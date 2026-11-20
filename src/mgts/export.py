"""Экспорт результатов (Агент 4, задача 5)."""
from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np

from .config import load_classes
from .schemas import CSV_COLUMNS, ObjectRecord

_PRIORITY_COLORS = {"critical": "red", "high": "orange", "medium": "#d8a600", "low": "green"}
_SEVERITY_COLORS = {"ok": (0, 170, 0), "warning": (0, 200, 255), "defect": (0, 0, 230), "uncertain": (180, 60, 160)}


def _title_ru(class_name: str) -> str:
    """Получить русское название класса из справочника."""
    return next((item["title_ru"] for item in load_classes()["classes"] if item["name"] == class_name), class_name)


def _table_html(records: list[ObjectRecord], title: str) -> str:
    """Сформировать автономную таблицу для карты без координат/folium."""
    rows = []
    for record in records:
        row = record.to_row()
        if record.lat is None or record.lon is None:
            row["location_source"] = "Неизвестно"
        cells = "".join(f"<td>{html.escape(str(value if value is not None else ''))}</td>" for value in row.values())
        rows.append(f"<tr>{cells}</tr>")
    return ("<!doctype html><html><meta charset='utf-8'><title>" + html.escape(title) +
            "</title><h1>" + html.escape(title) +
            "</h1><table border='1'><thead><tr>" + "".join(f"<th>{html.escape(column)}</th>" for column in CSV_COLUMNS) +
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></html>")


def to_csv(records: list[ObjectRecord], path: str | Path) -> Path:  # CONTRACT
    """CSV (UTF-8 с BOM, чтобы Excel открывал кириллицу) с колонками schemas.CSV_COLUMNS,
    + колонка title_ru (русское название класса из classes.yaml). Порядок строк как в records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS + ["title_ru"])
        writer.writeheader()
        for record in records:
            row = record.to_row()
            row["title_ru"] = _title_ru(record.class_name)
            writer.writerow(row)
    return path


def to_geojson(records: list[ObjectRecord], path: str | Path) -> Path:  # CONTRACT
    """FeatureCollection; Point [lon, lat] (ВНИМАНИЕ: порядок lon, lat!), properties = ObjectRecord.to_row().
    Объекты без координат НЕ включать в geometry (geometry: null допустим) — но не терять в CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features = []
    for record in records:
        geometry = None
        if record.lat is not None and record.lon is not None:
            geometry = {"type": "Point", "coordinates": [record.lon, record.lat]}
        features.append({"type": "Feature", "geometry": geometry, "properties": record.to_row()})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def to_html_map(records: list[ObjectRecord], path: str | Path, title: str = "МГТС: объекты для проверки") -> Path:  # CONTRACT
    """Автономный HTML (folium): маркеры по приоритету (critical красный, high оранжевый, medium жёлтый, low зелёный),
    всплывающее окно: тип, состояние, дефекты (description), уверенность, миниатюра кропа (base64, <= 160 px),
    метка «⚠ ручная проверка» при needs_review; легенда; фильтр слоёв по priority.
    Если нет ни одной координаты — карта-заглушка с таблицей объектов. Если folium не установлен — простая HTML-таблица."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    located = [record for record in records if record.lat is not None and record.lon is not None]
    if not located:
        path.write_text(_table_html(records, title), encoding="utf-8")
        return path
    try:
        import folium
    except ImportError:
        path.write_text(_table_html(records, title), encoding="utf-8")
        return path
    center = [sum(record.lat for record in located) / len(located), sum(record.lon for record in located) / len(located)]
    map_view = folium.Map(location=center, zoom_start=16, control_scale=True)
    groups = {}
    for priority in ("critical", "high", "medium", "low"):
        groups[priority] = folium.FeatureGroup(name=priority.title(), show=True)
        groups[priority].add_to(map_view)
    for record in located:
        priority = record.priority.value
        lines = [f"<b>{html.escape(record.class_name)}</b>", f"Состояние: {html.escape(record.severity.value)}",
                 f"Уверенность: {record.confidence:.2f}"]
        if record.description:
            lines.append(f"Дефекты: {html.escape(record.description)}")
        if record.needs_review:
            lines.append("<strong>⚠ ручная проверка</strong>")
        if record.location_source.startswith("synthetic"):
            lines.append("<strong>Координаты синтетические</strong>")
        if record.crop_path:
            crop = cv2.imread(str(record.crop_path))
            if crop is not None:
                scale = min(160 / max(crop.shape[:2]), 1.0)
                thumb = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".jpg", thumb)
                if ok:
                    image = base64.b64encode(encoded).decode("ascii")
                    lines.append(f'<img width="160" src="data:image/jpeg;base64,{image}">')
        popup = folium.Popup("<br>".join(lines), max_width=280)
        folium.CircleMarker(location=[record.lat, record.lon], radius=7, color=_PRIORITY_COLORS.get(priority, "gray"),
                            fill=True, fill_opacity=0.85, popup=popup,
                            tooltip=f"{record.object_id} · {record.priority.value}").add_to(groups.get(priority, groups["low"]))
    folium.LayerControl(collapsed=False).add_to(map_view)
    legend = "<div style='position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px'>" + "".join(
        f"<div><span style='color:{color}'>●</span> {priority}</div>" for priority, color in _PRIORITY_COLORS.items()
    ) + "</div>"
    map_view.get_root().html.add_child(folium.Element(legend))
    map_view.save(str(path))
    return path


def draw_annotated(frame_bgr: np.ndarray, items: list[tuple]) -> np.ndarray:  # CONTRACT
    """Нарисовать рамки на копии кадра. items: [(BBox, label_str, Severity, needs_review)].
    Цвет по severity (ok зелёный, warning жёлтый, defect красный, uncertain серый/фиолетовый),
    подпись с фоном для читаемости, толщина линии пропорциональна размеру кадра."""
    output = frame_bgr.copy()
    thickness = max(1, round(max(frame_bgr.shape[:2]) / 500))
    font_scale = max(0.4, max(frame_bgr.shape[:2]) / 1200)
    for bbox, label, severity, needs_review in items:
        color = _SEVERITY_COLORS.get(getattr(severity, "value", str(severity)), _SEVERITY_COLORS["uncertain"])
        x1, y1, x2, y2 = (round(value) for value in bbox.to_list())
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        caption = str(label) + (" ⚠" if needs_review else "")
        (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        top = max(0, y1 - text_h - baseline - 4)
        cv2.rectangle(output, (x1, top), (min(output.shape[1] - 1, x1 + text_w + 6), y1), color, -1)
        cv2.putText(output, caption, (x1 + 3, max(text_h, y1 - baseline - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return output
