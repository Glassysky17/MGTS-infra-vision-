"""Источники координат кадров (Агент 4, задача 1)."""
from __future__ import annotations

import csv
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from ..schemas import FrameMeta


def _number(value: Any) -> float:
    """Преобразовать число Pillow/CSV в float."""
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        return float(value.numerator) / float(value.denominator)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def _timestamp(value: Any) -> Optional[datetime]:
    """Разобрать ISO-время и привести его к сопоставимому UTC."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(str(value).strip(), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: Any) -> Optional[str]:
    """Вернуть дату-время в ISO 8601, если строка распознана."""
    parsed = _timestamp(value)
    return parsed.isoformat() if parsed else None


def _dms(value: Any) -> float:
    """Перевести EXIF-координату из градусов, минут и секунд."""
    return sum(_number(part) / divisor for part, divisor in zip(value, (1, 60, 3600)))


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Вычислить начальный азимут между точками WGS84."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    east = math.sin(delta_lon) * math.cos(p2)
    north = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(delta_lon)
    return math.degrees(math.atan2(east, north)) % 360


def read_exif_gps(image_path: str | Path) -> Optional[dict]:  # CONTRACT
    """Прочитать GPS из EXIF (Pillow): {'lat','lon','heading_deg','timestamp'} или None.
    Учесть ref N/S/E/W, рациональные числа, GPSImgDirection, DateTimeOriginal. Не падать на файлах без EXIF."""
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            gps = exif.get_ifd(0x8825) if exif else {}
            if not gps or 2 not in gps or 4 not in gps:
                return None
            lat = _dms(gps[2]) * (-1 if str(gps.get(1, "N")).upper() == "S" else 1)
            lon = _dms(gps[4]) * (-1 if str(gps.get(3, "E")).upper() == "W" else 1)
            heading = _number(gps[17]) % 360 if 17 in gps else None
            raw_timestamp = exif.get(36867) or exif.get(306)
            return {"lat": lat, "lon": lon, "heading_deg": heading, "timestamp": _iso(raw_timestamp)}
    except (OSError, ValueError, TypeError, ZeroDivisionError, AttributeError, KeyError):
        return None


def load_track_csv(path: str | Path) -> list[dict]:  # CONTRACT
    """CSV-трек: колонки frame_id | timestamp, lat, lon, [heading_deg]. Вернуть список записей,
    отсортированный по времени/порядку. Терпимо относиться к разделителям (',' ';') и названиям (latitude/lng)."""
    with open(path, encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(file, dialect=dialect)
        if not reader.fieldnames:
            return []
        columns = {name.strip().lower(): name for name in reader.fieldnames}
        lat_key = next((columns[k] for k in ("lat", "latitude") if k in columns), None)
        lon_key = next((columns[k] for k in ("lon", "lng", "longitude") if k in columns), None)
        time_key = next((columns[k] for k in ("time", "timestamp") if k in columns), None)
        frame_key = next((columns[k] for k in ("frame", "frame_id", "filename") if k in columns), None)
        heading_key = next((columns[k] for k in ("heading_deg", "heading", "course") if k in columns), None)
        if lat_key is None or lon_key is None:
            raise ValueError("CSV-трек должен содержать колонки lat/lon (или latitude/longitude)")
        records = []
        for row in reader:
            try:
                item = {"lat": float(row[lat_key]), "lon": float(row[lon_key])}
                item["timestamp"] = _iso(row.get(time_key)) if time_key else None
                item["frame_id"] = (row.get(frame_key) or "").strip() if frame_key else ""
                item["heading_deg"] = float(row[heading_key]) % 360 if heading_key and row.get(heading_key) else None
            except (TypeError, ValueError):
                continue
            item["_order"] = len(records)
            records.append(item)
    records.sort(key=lambda item: (_timestamp(item["timestamp"]) is None, _timestamp(item["timestamp"]) or datetime.max.replace(tzinfo=timezone.utc), item["_order"]))
    for item in records:
        item.pop("_order", None)
    return records


def load_track_gpx(path: str | Path) -> list[dict]:  # CONTRACT
    """GPX-трек (gpxpy или xml.etree): [{'timestamp','lat','lon'}], курс вычислить по соседним точкам."""
    root = ET.parse(path).getroot()
    records = []
    for point in root.iter():
        if point.tag.rsplit("}", 1)[-1] != "trkpt":
            continue
        time_node = next((node for node in point if node.tag.rsplit("}", 1)[-1] == "time"), None)
        if time_node is None or not time_node.text:
            continue
        try:
            records.append({"timestamp": _iso(time_node.text), "lat": float(point.attrib["lat"]), "lon": float(point.attrib["lon"])})
        except (KeyError, ValueError):
            continue
    for index, item in enumerate(records):
        other = records[index + 1] if index + 1 < len(records) else records[index - 1] if index else None
        if other is not None:
            start, end = (item, other) if index + 1 < len(records) else (other, item)
            item["heading_deg"] = _bearing(start["lat"], start["lon"], end["lat"], end["lon"])
        else:
            item["heading_deg"] = None
    return records


def interpolate_position(track: list[dict], timestamp: str) -> Optional[dict]:  # CONTRACT
    """Линейная интерполяция позиции и курса (курс — по кратчайшей дуге!) на момент timestamp.
    Вне диапазона трека более чем на 5 с -> None."""
    target = _timestamp(timestamp)
    points = sorted(((_timestamp(row.get("timestamp")), row) for row in track), key=lambda pair: pair[0] or datetime.max.replace(tzinfo=timezone.utc))
    points = [(time, row) for time, row in points if time is not None]
    if target is None or not points:
        return None
    if target < points[0][0]:
        if (points[0][0] - target).total_seconds() > 5:
            return None
        left = right = points[0]
    elif target > points[-1][0]:
        if (target - points[-1][0]).total_seconds() > 5:
            return None
        left = right = points[-1]
    else:
        left = right = points[0]
        for candidate in points[1:]:
            if candidate[0] >= target:
                right = candidate
                break
            left = candidate
    t0, p0 = left
    t1, p1 = right
    fraction = (target - t0).total_seconds() / (t1 - t0).total_seconds() if t1 != t0 else 0.0
    lat = float(p0["lat"]) + fraction * (float(p1["lat"]) - float(p0["lat"]))
    lon = float(p0["lon"]) + fraction * (float(p1["lon"]) - float(p0["lon"]))
    h0, h1 = p0.get("heading_deg"), p1.get("heading_deg")
    heading = None
    if h0 is not None and h1 is not None:
        delta = (float(h1) - float(h0) + 180) % 360 - 180
        heading = (float(h0) + fraction * delta) % 360
    elif h0 is not None or h1 is not None:
        heading = float(h0 if h0 is not None else h1) % 360
    return {"lat": lat, "lon": lon, "heading_deg": heading, "timestamp": _iso(timestamp)}


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
      4) если трек есть, но кадр не удалось точно сопоставить — ближайшая по времени или по индексу
         точка трека (location_source='track_approx');
      5) cfg.geo.fallback_no_gps == 'synthetic': псевдо-позиция по индексу кадра на север
           (шаг 4 м от фиксированной точки, курс 0°), location_source='synthetic' — в UI/CSV явно помечать!;
      6) иначе lat/lon=None, 'none'.
    width/height заполнить (быстро, через Pillow без полного декодирования)."""
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        width, height = image.size
        exif = image.getexif()
        exif_timestamp = _iso(exif.get(36867) or exif.get(306)) if exif else None
    exif = read_exif_gps(image_path)
    if exif and not exif_timestamp:
        exif_timestamp = exif.get("timestamp")
    frame_id = image_path.stem
    track_rows = track or []
    matched = next((row for row in track_rows if row.get("frame_id") and Path(str(row["frame_id"])).name in (image_path.name, frame_id)), None)
    timestamp_match = re.search(r"(\d{4}-\d\d-\d\d)[T_ ](\d\d)[-:]?(\d\d)[-:]?(\d\d)", image_path.stem)
    filename_timestamp = f"{timestamp_match[1]}T{timestamp_match[2]}:{timestamp_match[3]}:{timestamp_match[4]}" if timestamp_match else None
    timestamp = exif_timestamp or (matched or {}).get("timestamp") or _iso(filename_timestamp)
    if matched:
        lat, lon = float(matched["lat"]), float(matched["lon"])
        heading, source = matched.get("heading_deg"), "csv"
    elif exif:
        lat, lon = exif["lat"], exif["lon"]
        heading, source = exif.get("heading_deg"), "exif"
    else:
        interpolated = interpolate_position(track_rows, timestamp) if timestamp else None
        if interpolated:
            lat, lon = interpolated["lat"], interpolated["lon"]
            heading, source = interpolated.get("heading_deg"), "track"
        elif track_rows:
            target = _timestamp(timestamp)
            timed_rows = [(moment, row) for row in track_rows if (moment := _timestamp(row.get("timestamp"))) is not None]
            if target and timed_rows:
                _, point = min(timed_rows, key=lambda item: abs((item[0] - target).total_seconds()))
            else:
                point = track_rows[min(index, len(track_rows) - 1)]
            lat, lon = float(point["lat"]), float(point["lon"])
            heading, source = point.get("heading_deg"), "track_approx"
        elif cfg.get("geo", {}).get("fallback_no_gps", "none") == "synthetic":
            lat, lon, heading, source = 55.75 + index * 4.0 / 111_320.0, 37.6, 0.0, "synthetic"
        else:
            lat, lon, heading, source = None, None, None, "none"
    return FrameMeta(frame_id, str(image_path), width, height, lat, lon, heading, timestamp, source)
