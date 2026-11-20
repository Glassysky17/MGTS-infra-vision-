"""Общие утилиты (CONTRACT, реализовано). Дополнять можно, существующие сигнатуры не менять."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: str | Path, recursive: bool = True) -> list[Path]:
    """Отсортированный список изображений в папке."""
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(p for p in it if p.suffix.lower() in IMAGE_EXTS)


def read_image(path: str | Path) -> np.ndarray:
    """Прочитать изображение как BGR uint8. Поддерживает юникод-пути. Бросает FileNotFoundError."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def write_image(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"cannot encode image: {path}")
    buf.tofile(str(path))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние между двумя точками WGS84, метры."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def offset_latlon(lat: float, lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    """Сдвинуть точку на (север, восток) метров (локальная плоская аппроксимация)."""
    dlat = north_m / 111_320.0
    dlon = east_m / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return lat + dlat, lon + dlon


def save_json(obj, path: str | Path) -> None:
    from .schemas import to_jsonable
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def chunked(seq: Iterable, n: int) -> Iterator[list]:
    buf: list = []
    for x in seq:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
