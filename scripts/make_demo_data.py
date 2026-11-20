"""Генератор синтетического демо-набора с псевдо-GPS (Агент 1, задача 0 — СДЕЛАТЬ ПЕРВЫМ).

Зачем: чтобы остальные агенты и `make e2e` работали ДО появления реального датасета (и как страховка).

Требования:
    python scripts/make_demo_data.py --out data/demo --n 40 [--seed 0]
Результат:
    data/demo/images/frame_0000.jpg ...      вид «улица»: асфальт, тротуар, небо, здания-прямоугольники;
                                             на асфальте — эллипсы-люки (классы cable_manhole),
                                             у тротуара — вертикальные прямоугольники-опоры (utility_pole),
                                             часть объектов «дефектна»: трещины на люке, наклон опоры > 8°.
    data/demo/labels/frame_0000.txt          YOLO-разметка GT (class_id xc yc w h)
    data/demo/frames.csv                     frame_id,lat,lon,heading_deg,timestamp  — трек по прямой/ломаной,
                                             шаг ~3–5 м между кадрами; одни и те же объекты видны в 2–3 соседних кадрах
                                             (для проверки дедупликации)
    data/demo/gt_defects.csv                 frame_id,bbox,class_name,defect_code  — что реально «сломано»
Разрешение 1280x720. Детерминировано по seed. Использовать numpy/cv2, без внешних данных.
"""
from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    """Создать детерминированный набор кадров улицы с YOLO-метками и треком."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.n < 1:
        ap.error("--n должен быть положительным")
    out = Path(args.out)
    images_dir, labels_dir = out / "images", out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for generated_file in (*images_dir.glob("*.jpg"), *labels_dir.glob("*.txt")):
        generated_file.unlink()
    rng = np.random.default_rng(args.seed)
    tracks: list[dict[str, str | float]] = []
    defects: list[dict[str, str]] = []

    for i in range(args.n):
        frame_id = f"frame_{i:04d}"
        image = _street_scene(rng)
        labels: list[str] = []
        for object_id, class_id, start, defect_code in _demo_objects(args.n):
            if not start <= i < start + 3:
                continue
            step = i - start
            if class_id == 0:
                box = _draw_manhole(image, 770 + step * 12, 552 + step * 14, 28 + step * 4, defect_code == "cover_broken", rng)
            else:
                box = _draw_pole(image, 940 + step * 9, 650, 215 + step * 8, 11.0 if defect_code == "pole_tilt" else 1.5)
            x1, y1, x2, y2 = box
            xc, yc = (x1 + x2) / 2 / 1280, (y1 + y2) / 2 / 720
            width, height = (x2 - x1) / 1280, (y2 - y1) / 720
            labels.append(f"{class_id} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}")
            if defect_code:
                defects.append({"frame_id": frame_id, "bbox": f"{x1},{y1},{x2},{y2}",
                                "class_name": "cable_manhole" if class_id == 0 else "utility_pole",
                                "defect_code": defect_code})
        if not cv2.imwrite(str(images_dir / f"{frame_id}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(f"Не удалось сохранить кадр {frame_id}")
        (labels_dir / f"{frame_id}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        tracks.append({"frame_id": frame_id, "lat": f"{55.75 + i * 4 / 111320:.8f}", "lon": "37.60000000",
                       "heading_deg": "0", "timestamp": (datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=i)).isoformat()})

    _write_csv(out / "frames.csv", ["frame_id", "lat", "lon", "heading_deg", "timestamp"], tracks)
    _write_csv(out / "gt_defects.csv", ["frame_id", "bbox", "class_name", "defect_code"], defects)
    (out / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnames:\n  0: cable_manhole\n  1: utility_pole\n", encoding="utf-8"
    )


def _demo_objects(n: int) -> list[tuple[str, int, int, str]]:
    """Вернуть разные физические объекты с короткими соседними сериями наблюдений."""
    if n < 3:
        return [("manhole_a", 0, 0, "cover_broken")]
    starts = [0, max(0, n // 4), max(0, n // 2), max(0, 3 * n // 4)]
    return [("manhole_a", 0, starts[0], "cover_broken"),
            ("pole_a", 1, starts[1], ""),
            ("manhole_b", 0, starts[2], ""),
            ("pole_b", 1, starts[3], "pole_tilt")]


def _street_scene(rng: np.random.Generator) -> np.ndarray:
    """Нарисовать простую перспективную городскую улицу 1280×720."""
    height, width = 720, 1280
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        if y < 288:
            t = y / 288
            image[y] = (int(190 - 20 * t), int(170 - 16 * t), int(135 - 10 * t))
        else:
            t = (y - 288) / (height - 288)
            image[y] = (int(65 + 20 * t), int(70 + 22 * t), int(75 + 25 * t))
    cv2.rectangle(image, (0, 170), (280, 390), (112, 120, 125), -1)
    cv2.rectangle(image, (35, 125), (225, 175), (90, 100, 105), -1)
    cv2.rectangle(image, (1010, 145), (1279, 390), (105, 113, 120), -1)
    for x in (70, 145, 205, 1060, 1140, 1220):
        cv2.rectangle(image, (x, 215), (x + 35, 275), (175, 190, 195), -1)
    cv2.fillPoly(image, [np.array([[0, 350], [410, 288], [870, 288], [1280, 350], [1280, 430], [0, 430]], np.int32)], (125, 125, 125))
    cv2.line(image, (0, 430), (1280, 430), (190, 190, 190), 5)
    cv2.line(image, (0, 288), (1280, 288), (95, 95, 95), 3)
    cv2.line(image, (635, 330), (635, 720), (180, 180, 180), 4)
    cv2.line(image, (645, 330), (645, 720), (180, 180, 180), 4)
    # Небольшая фактура асфальта сохраняет естественный вид после JPEG.
    noise = rng.normal(0, 2.5, (432, width, 1))
    image[288:] = np.clip(image[288:].astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return image


def _draw_manhole(image: np.ndarray, cx: int, cy: int, radius: int, cracked: bool,
                  rng: np.random.Generator) -> tuple[int, int, int, int]:
    """Нарисовать крышку и при необходимости трещины; вернуть охватывающую рамку."""
    cv2.ellipse(image, (cx, cy), (radius, max(8, radius // 2)), -5, 0, 360, (42, 45, 49), -1)
    cv2.ellipse(image, (cx, cy), (radius, max(8, radius // 2)), -5, 0, 360, (25, 28, 31), 3)
    cv2.ellipse(image, (cx, cy), (max(4, radius - 7), max(5, radius // 2 - 5)), -5, 0, 360, (70, 72, 74), 2)
    if cracked:
        points = [(cx - radius // 3, cy - 2), (cx - 2, cy + 3), (cx + radius // 4, cy - 4),
                  (cx + radius // 2, cy + 2)]
        for start, end in zip(points, points[1:]):
            cv2.line(image, start, end, (12, 13, 14), 2, cv2.LINE_AA)
    return cx - radius - 2, cy - radius // 2 - 2, cx + radius + 2, cy + radius // 2 + 2


def _draw_pole(image: np.ndarray, base_x: int, base_y: int, length: int,
               angle_deg: float) -> tuple[int, int, int, int]:
    """Нарисовать опору с заданным наклоном и вернуть рамку."""
    angle = math.radians(angle_deg)
    top_x, top_y = base_x + int(length * math.sin(angle)), base_y - int(length * math.cos(angle))
    normal = (int(7 * math.cos(angle)), int(7 * math.sin(angle)))
    polygon = np.array([[base_x - normal[0], base_y - normal[1]], [base_x + normal[0], base_y + normal[1]],
                        [top_x + normal[0], top_y + normal[1]], [top_x - normal[0], top_y - normal[1]]], np.int32)
    cv2.fillPoly(image, [polygon], (45, 52, 60))
    cv2.line(image, tuple(polygon[0]), tuple(polygon[3]), (83, 86, 88), 2, cv2.LINE_AA)
    return int(polygon[:, 0].min()), int(polygon[:, 1].min()), int(polygon[:, 0].max()), int(polygon[:, 1].max())


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str | float]]) -> None:
    """Записать CSV с устойчивым порядком полей и строк."""
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
