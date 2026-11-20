# Агент 2 — Предобработка и детекция

**Миссия:** надёжно находить объекты (в том числе мелкие) на кадрах с автомобиля и дать единый интерфейс детектора
остальной системе.

**Ветка:** `agent2-detection`. **Владеешь:** `src/mgts/preprocessing/`, `src/mgts/detection/`, секции `preprocessing`,
`detection`, `docs/DETECTION_REPORT.md`.
**Тесты:** `tests/test_preprocessing.py`, `tests/test_detection.py`.

> Работай по TDD (`tasks/TDD.md`). Алгоритмы по шагам — `docs/API_SPEC.md` §2.

## Порядок задач

### §0. `DummyDetector` и `build_detector` — ПЕРВЫМ (≤ 10 минут)
Читает GT-разметку YOLO рядом с картинкой (`images/x.jpg → labels/x.txt`), отдаёт `Detection` со `score=0.9`.
Благодаря ему A3 и A4 отлаживаются, не дожидаясь обучения. `build_detector(cfg)` — фабрика по `detection.backend`.
Тест: `test_dummy_detector_reads_yolo_labels`.

### §1. `preprocessing/quality.py`
`blur_score` (дисперсия лапласиана на уменьшенной копии), `is_sharp`, `brightness_stats`, `select_frames`,
`calibrate_blur_threshold`. Потом, как придёт датасет, откалибруй порог и запиши в конфиг.
Тесты: `test_blur_*`, `test_is_sharp_*`, `test_brightness_*`.

### §2. `preprocessing/enhance.py`
CLAHE по каналу L (LAB), авто-гамма через LUT, шумоподавление по флагу, `enhance(img, cfg)` — цепочка.
**Важно:** вход не мутировать, размер/тип сохранять. Enhance применяется ТОЛЬКО перед детекцией
(оценка состояния идёт по оригиналу — цвет и текстура нужны для ржавчины и трещин).
Тесты: `test_enhance_*`, `test_auto_gamma_*`, `test_clahe_*`.

### §3. `detection/tiling.py` — тайлинг (собственный SAHI)
`tile_slices`: сетка окон с перекрытием, последний ряд и столбец прижимаются к краю; `merge_detections`: NMS по классу +
удаление рамок, обрезанных границей тайла; `detect_tiled`: тайлы + уменьшенный полный кадр (крупные объекты не
помещаются в тайл), перевод координат в исходный кадр.
Тесты: `test_tiles_*`, `test_merge_*`.

### §4. `detection/infer.py` — `YoloDetector`
Загрузка весов (`weights` → иначе `base_model`), device auto, `detect_raw`, `detect` (тайлинг если `tiling.enabled` и
`min(w, h) >= min_image_side`). Имена классов брать из `classes.yaml`. Рамки клипнуть по кадру.
Понятная ошибка, если ни весов, ни base_model нет.

### §5. `detection/train.py`
`train_detector`: transfer learning через ultralytics. На CPU автоматически: `yolov8n`, `imgsz=640`, `epochs=15`.
**Как только `data/processed` будет готов (≈ T+40), запускай обучение в фоне** и дальше работай над оценкой.
Дисбаланс классов: дублирование кадров редких классов в `train.txt`. После обучения — `models/detector.pt`
и `models/detector_metrics.json`.

### §6. `detection/evaluate.py`
`match_detections` (чистая функция, жадное сопоставление по IoU+классу), `evaluate_detector` (precision/recall/F1 по
классам, mAP@0.5, **recall по мелким объектам**, топ-20 FN и FP в `out_dir/errors/`).
Прогоняй **через build_detector с тайлингом**, и сравни «без тайлинга / с тайлингом» — цифры в `docs/DETECTION_REPORT.md`.
Тесты: `test_match_*`.

### §7. Подбор порога
Подбери `detection.conf` под приоритет recall (на val: найди порог, где recall по объектам ≥ 0.9 при приемлемой
precision), запиши в конфиг и в отчёт с графиком PR-кривой.

## Критерии приёмки
- [ ] `build_detector(cfg).detect(...)` работает с `backend=yolo` и `backend=dummy`.
- [ ] Все тесты в твоих файлах зелёные.
- [ ] Обучен детектор, `models/detector.pt` лежит (не коммитить в git — в `.gitignore`; передать A4 путь/ссылку).
- [ ] `docs/DETECTION_REPORT.md`: метрики с тайлингом и без, список типовых ошибок, выбранный `conf`.
- [ ] Скорость: мс/кадр на CPU/GPU записана (нужно для презентации).

## Fallback
- Нет GPU: `yolov8n`, `imgsz=640`, 15 эпох, на ≤ 1500 кадрах; тайлинг только на инференсе.
- Ultralytics не ставится/не качает веса: `torchvision` Faster R-CNN MobileNet (`fasterrcnn_mobilenet_v3_large_fpn`) с
  дообучением головы; интерфейс `detect` тот же.
- Обучение не успевает: брать лучший промежуточный `last.pt`, остальное время тратить на оценку и пороги.
- Мало разметки: сначала обучить на том, что есть; A1 пришлёт аугментированные копии редких классов.
