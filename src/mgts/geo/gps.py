"""Источники координат кадров (Агент 4, задача 1)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..schemas import FrameMeta


def read_exif_gps(image_path: str | Path) -> Optional[dict]:  # CONTRACT
    """Прочитать GPS из EXIF (Pillow): {'lat','lon','heading_deg','timestamp'} или None.
    Учесть ref N/S/E/W, рациональные числа, GPSImgDirection, DateTimeOriginal. Не падать на файлах без EXIF."""
    raise NotImplementedError("Агент 4: tasks/agent_4_integration.md §1")


def load_track_csv(path: str | Path) -> list[dict]:  # CONTRACT
    """CSV-трек: колонки frame_id | timestamp, lat, lon, [heading_deg]. Вернуть список записей,
    отсортированный по времени/порядку. Терпимо относиться к разделителям (',' ';') и названиям (latitude/lng)."""
    raise NotImplementedError("Агент 4: tasks/agent_4_integration.md §1")


def load_track_gpx(path: str | Path) -> list[dict]:  # CONTRACT
    """GPX-трек (gpxpy или xml.etree): [{'timestamp','lat','lon'}], курс вычислить по соседним точкам."""
    raise NotImplementedError("Агент 4: tasks/agent_4_integration.md §1")


def interpolate_position(track: list[dict], timestamp: str) -> Optional[dict]:  # CONTRACT
    """Линейная интерполяция позиции и курса (курс — по кратчайшей дуге!) на момент timestamp.
    Вне диапазона трека более чем на 5 с -> None."""
    raise NotImplementedError("Агент 4: tasks/agent_4_integration.md §1")


def build_frame_meta(  # CONTRACT
    image_path: str | Path,
    index: int,
    cfg: dict,
    track: Optional[list[dict]] = None,
) -> FrameMeta:
    """Собрать FrameMeta для кадра. Приоритет источников:
      1) запись трека с frame_id == имя файла (location_source='csv');
      2) EXIF GPS ('exif');
      3) интерполяция трека по timestamp из EXIF/имени ('track');
      4) cfg.geo.fallback_no_gps == 'synthetic': псевдо-позиция по индексу кадра вдоль прямой
         (шаг 4 м от фиксированной точки, курс 90°), location_source='synthetic' — в UI/CSV явно помечать!;
      5) иначе lat/lon=None, 'none'.
    width/height заполнить (быстро, через Pillow без полного декодирования)."""
    raise NotImplementedError("Агент 4: tasks/agent_4_integration.md §1")
