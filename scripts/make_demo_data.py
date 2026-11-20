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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.parse_args()
    raise NotImplementedError("Агент 1: tasks/agent_1_data.md §0")


if __name__ == "__main__":
    main()
