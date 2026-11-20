# Низкоуровневая спецификация API

Для каждой функции: **сигнатура → вход → выход → алгоритм по шагам → ошибки и крайние случаи**.
Всё, что здесь описано, уже лежит в коде как заглушка с той же сигнатурой. Ничего не придумывай заново:
бери сигнатуру из кода, поведение — отсюда, примеры — из `tests/`.

Общие соглашения:
- Изображение = `np.ndarray`, форма `(H, W, 3)`, `dtype=uint8`, порядок каналов **BGR** (как в OpenCV).
- Рамка = `BBox(x1, y1, x2, y2)` в **пикселях** исходного кадра; YOLO-строки — нормированные `xc yc w h`.
- Угол в градусах; координаты WGS84 в градусах; расстояния в метрах.
- Нет исключений наружу там, где написано «не падает»: внутри `try/except`, лог через `logging.getLogger(__name__)`.
- Все пороги — из `cfg` (словарь из `load_config()`), в коде чисел-«магии» нет.

---

## 1. `mgts.data` (Агент 1)

### `analyze.analyze_dataset(dataset_dir, out_md="docs/DATASET_REPORT.md") -> dict`
Вход: папка YOLO-датасета (`images/`, `labels/`, `data.yaml`). Выход: словарь
`{n_images, n_objects, per_class: {name: count}, small_ratio, empty_frames, broken_files, has_gps}`.
1. `list_images(dataset_dir/images)`; для каждого файла: `read_image` в try → при ошибке в `broken_files`.
2. Для каждого `labels/<stem>.txt`: разобрать строки `cls xc yc w h`; нет файла → кадр без объектов.
3. Посчитать счётчики по классам; `small_ratio` = доля рамок с `min(w_px, h_px) < 32` или площадью < 0.5% кадра.
4. `has_gps`: у любого кадра есть EXIF GPS (Pillow) или рядом лежит `*.csv/*.gpx`.
5. Записать Markdown: таблицы, выводы (дисбаланс: класс < 5% объектов → пометить), картинка `docs/dataset_samples.jpg` (12 кадров с рамками).
Ошибки: нет папки → `FileNotFoundError` с понятным текстом.

### `convert.convert_to_yolo(src_dir, out_dir, fmt="auto", class_map=None) -> Path`
1. Определить `fmt`: есть `*.json` с ключами `images/annotations/categories` → `coco`; `*.xml` с `<annotation>` → `voc`; `*.txt` рядом с картинками → `yolo`; `*.csv` с колонками `filename,class,xmin,ymin,xmax,ymax` → `csv`.
2. Прочитать аннотации в общий вид: `{image_path: [(class_name, x1, y1, x2, y2, img_w, img_h)]}`. COCO: `bbox=[x,y,w,h]`, имя класса через `categories`.
3. `class_map` применить; имя не из `classes.yaml` → пропустить и записать в `out_dir/unmapped_classes.txt`.
4. Рамку клипнуть по кадру; `w<=0` или `h<=0` после клипа → отбросить (лог).
5. Копировать картинку в `out_dir/images/`, писать `labels/<stem>.txt` (`class_id xc yc w h`, 6 знаков), `data.yaml` (`path`, `names` по порядку id).
6. Идемпотентно: повторный запуск перезаписывает тем же содержимым.
Вернуть `Path(out_dir)`.

### `split.make_split(dataset_dir, val_fraction=0.2, seed=42, group_fn=None) -> dict[str, list[str]]`
1. Список кадров; ключ группы: `group_fn(stem)` или (по умолчанию) стем без последнего числового блока (`route12_000345` → `route12`); если ключ у всех одинаков — блоки по 30 подряд идущих кадров.
2. Для каждой группы: множество классов в ней. Отсортировать группы, перемешать `random.Random(seed)`.
3. Жадно набирать val, пока доля кадров < `val_fraction`; сначала группы, содержащие ещё не покрытые val редкие классы.
4. Записать `train.txt`/`val.txt` (абсолютные пути картинок, по строке), обновить `data.yaml` (`train: train.txt`, `val: val.txt`).
Вернуть `{'train': [...], 'val': [...]}`. Один и тот же вход → один и тот же выход.

### `crops.extract_crops(dataset_dir, out_dir, padding=0.15, min_size=24) -> Path`
Для каждой рамки: `BBox.pad(padding, W, H)` → вырезать → если `min(h, w) < min_size` пропустить и посчитать → сохранить в `out_dir/<split>/<class>/<label>/<stem>_<idx>.jpg`
(`split` берётся из `train.txt/val.txt`, `label` = `unlabeled`, если меток состояния нет) → строка в `index.csv`. Вернуть `Path(out_dir)`.

### `augment.*` (все: вход BGR uint8 → выход BGR uint8 того же размера)
| Функция | Алгоритм |
|---|---|
| `add_motion_blur(img, length, angle_deg)` | ядро `k=zeros((length,length))`, нарисовать линию через центр под углом, `k/=k.sum()`, `cv2.filter2D(img, -1, k)` |
| `add_rain(img, intensity, rng)` | слой нулей; `n=int(intensity*0.002*H*W)` штрихов длиной 10–30 px под 70–80° (`cv2.line`, серо-белые); `GaussianBlur(layer,(3,3))`; `out=img*(1-0.3*intensity)+layer*0.5*intensity`; клип 0..255 |
| `add_glare(img, strength, rng)` | случайный центр; 2D-гаусс с σ=0.15·max(H,W), нормировать в 0..1; `out=img+ strength*255*gauss[...,None]`; плюс bloom: яркая маска `>200` → `GaussianBlur(61)` → добавить |
| `add_shadow(img, strength, rng)` | случайный полигон из 4–6 вершин → маска 0/1 → `GaussianBlur(maska, 51)` → `out=img*(1 - strength*mask)` |
| `add_night(img, level, rng)` | `out=255*(img/255)**(1+level)`; `out*= (1-0.5*level)`; канал B ×1.1, R ×0.9; шум `rng.normal(0, 4+8*level)` |
| `perspective_warp(img, boxes, max_shift, rng)` | 4 угла + случайные смещения ≤ `max_shift*размер`; `M=cv2.getPerspectiveTransform`; `cv2.warpPerspective`; рамки: 4 угла каждой через `M` → охватывающий прямоугольник → клип; оставить рамки с площадью ≥ 50% исходной |
| `build_train_augmentations(cfg)` | вернуть замыкание `aug(img, boxes, class_ids)`; каждую функцию применять с вероятностью `cfg['dataset']['augment'][name]`; возвращает `(img, boxes, class_ids)` с согласованными списками |

### `synth_defects.*`
| Функция | Алгоритм |
|---|---|
| `synth_cracks(img, bbox, rng, n)` | `n` раз: старт в случайной точке внутри bbox; шагами по 3–6 px блуждание с дрейфом направления ±0.4 рад, 20–60 шагов; на каждом шаге с p=0.08 ответвление; рисовать на отдельном слое `cv2.line` толщиной 1–3, цвет (20,20,20); слой → `GaussianBlur(3)`; смешать `out=img*(1-a)+layer_color*a`, `a∈[0.6,0.9]` только внутри bbox |
| `synth_rust(img, bbox, rng, coverage)` | шум: `rng.random((h,w))` → blur σ=h/8 → нормализовать; порог по квантилю `1-coverage`; маска → blur(5); цвет ржавчины `(30,70,150)`±вариации BGR; смешать с яркостью оригинала (`V` сохранить) внутри bbox |
| `synth_tilt(img, bbox, angle_deg)` | вырезать bbox; `cv2.getRotationMatrix2D((w/2, h), -angle_deg, 1)` (центр = основание); `warpAffine` с альфа-маской; исходную область закрыть `cv2.inpaint(..., TELEA)`; вклеить повёрнутый кусок; новая рамка = охват повёрнутых углов |
| `synth_displaced_cover(img, bbox, shift_frac, rng)` | эллипс в bbox → маска; сдвинуть содержимое маски на `shift_frac*ширина` в случайном направлении; освободившийся серп залить тёмным (35,35,35) |
| `generate_synthetic_defects(dataset_dir, out_dir, per_class, seed)` | по кропам исправных объектов случайно выбирать применимый дефект класса, генерировать, писать `index.csv` (`synthetic=1`, `label=<код>`) |

---

## 2. `mgts.preprocessing` и `mgts.detection` (Агент 2)

### `quality.blur_score(img_bgr) -> float`
1. Серый `cv2.cvtColor`. 2. Если `max(H,W) > 640` — уменьшить до 640 (`INTER_AREA`). 3. `float(cv2.Laplacian(gray, cv2.CV_64F).var())`.
Пустой/нулевой массив → `0.0`.

### `quality.is_sharp(img_bgr, threshold) -> bool` → `blur_score(img) >= threshold`.

### `quality.brightness_stats(img_bgr) -> dict`
Серый; `mean`, `p5`, `p95` (`np.percentile`); `underexposed = p95 < 70`; `overexposed = p5 > 200`.

### `quality.select_frames(paths, threshold, max_keep=None) -> list[str]`
Для каждого пути: `read_image`, `blur_score` (нечитаемые пропустить) → оставить `>= threshold`; соседние кадры с корреляцией гистограмм > 0.98 образуют группу — оставить самый резкий; сохранить исходный порядок; `max_keep` → равномерная выборка.

### `quality.calibrate_blur_threshold(paths, percentile=15.0) -> float` → `np.percentile([blur_score(...)], percentile)`.

### `enhance.apply_clahe(img, clip, grid)`: BGR→LAB; `cv2.createCLAHE(clip, (grid,grid))` на L; LAB→BGR.
### `enhance.auto_gamma(img, target_mean)`: `m=mean(gray)`; `gamma = log(target/255)/log(m/255)` (при `m<=1` → 2.0); клип в `[0.5, 2.0]`; LUT `255*(i/255)**(gamma)`, применять `cv2.LUT`. (gamma<1 осветляет — проверь знак тестом `test_auto_gamma_brightens_dark_image`.)
### `enhance.denoise(img, strength)`: `cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)`.
### `enhance.enhance(img, cfg)`: копия → по флагам `clahe`, `gamma` (`auto`/число/`null`), `denoise` → результат; вход не менять.

### `tiling.tile_slices(img_w, img_h, tile, overlap) -> list[(x1,y1,x2,y2)]`
1. Если `img_w<=tile and img_h<=tile` → `[(0,0,img_w,img_h)]`.
2. `step = max(1, int(tile*(1-overlap)))`.
3. `xs = list(range(0, max(img_w-tile,0)+1, step))`; если последний `x+tile < img_w` → добавить `img_w-tile`. Аналогично `ys`. Для сторон меньше `tile` — один старт 0 и размер `min(tile, side)`.
4. Окна = все пары `(x, y, min(x+tile,img_w), min(y+tile,img_h))`; убрать дубликаты (`dict.fromkeys`).

### `tiling.merge_detections(dets, iou_thr) -> list[Detection]`
1. Группировать по `class_id`. 2. Внутри: сортировка по `score` убыв.; жадный NMS по `BBox.iou > iou_thr`.
3. Дополнительно: если рамка A (меньшая) лежит ≥ 80% своей площади внутри рамки B того же класса → удалить A.
4. Склеить, отсортировать по `score` убыв.

### `tiling.detect_tiled(detector, img, frame_id, tile, overlap, iou_thr)`
Для каждого окна: `crop=img[y1:y2,x1:x2]` → `detector.detect_raw(crop, frame_id)` → сдвинуть рамки на `(x1,y1)`.
Плюс полный кадр, уменьшенный до `tile` по длинной стороне → рамки умножить на коэффициент масштаба. Всё → `merge_detections`.

### `infer.DummyDetector` / `YoloDetector` / `build_detector`
- `DummyDetector.detect(img, frame_id, image_path)`: `label=Path(image_path).parents[1]/'labels'/(stem+'.txt')`; нет файла → `[]`; иначе для каждой строки `Detection(frame_id, cls, name, 0.9, BBox.from_xywhn(..., W, H))`. Имя класса из `class_names()`.
- `YoloDetector.__init__`: `from ultralytics import YOLO`; путь весов = `resolve_path(cfg.detection.weights)` если существует, иначе `base_model`; `self.names = class_names()`.
- `YoloDetector.detect_raw(img, frame_id)`: `res = model.predict(img, imgsz, conf, iou, device, verbose=False)[0]`; для каждого `box`: `xyxy`, `conf`, `cls` → `Detection`; рамку клипнуть.
- `YoloDetector.detect`: если тайлинг включён и `min(H,W) >= min_image_side` → `detect_tiled`, иначе `detect_raw`.
- `build_detector(cfg)`: `backend` → класс; неизвестный → `ValueError`; yolo без весов и без интернета → `RuntimeError("обучите модель: mgts train-det")`.

### `evaluate.match_detections(preds, gts, iou_thr) -> {'tp','fp','fn'}`
`preds` по `score` убыв.; для каждого — среди несопоставленных GT **того же `class_id`** взять с максимальным IoU; `>= iou_thr` → tp (GT занят), иначе fp; `fn = len(gts) - tp`.

### `evaluate.evaluate_detector` и `train.train_detector`
Описаны в докстрингах файлов; формат `metrics.json`: `{"per_class": {name: {"precision":..,"recall":..,"f1":..,"ap50":..}}, "map50":.., "recall_small":.., "ms_per_frame":..}`.

---

## 3. `mgts.condition` (Агент 3)

Типы: `TiltResult(angle_deg, confidence, method)`, `HatchResult(circularity, ellipse_residual, dark_gap_ratio, found)` — уже в файле.

### `geometry.segment_foreground(crop) -> mask uint8 (H,W) 0/255`
1. Серый + `GaussianBlur(5)`. 2. `cv2.adaptiveThreshold(..., ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 31, 5)` — объект тёмный; если доля белого >50% → инвертировать.
3. `morphologyEx` OPEN 3×3 затем CLOSE 9×9. 4. `connectedComponentsWithStats` → оставить компоненту с наибольшей площадью, чей bbox пересекает центр кропа.
5. Доля маски <5% или >95% → `np.zeros`.

### `geometry.estimate_pole_tilt(crop, mask=None) -> TiltResult`
1. `mask = mask or segment_foreground(crop)`; пусто → `TiltResult(0.0, 0.0, 'none')`.
2. Координаты `pts = np.column_stack(np.nonzero(mask))[:, ::-1]` (x, y); центрировать; ковариация; `eigh`.
3. Вытянутость `r = λmax/max(λmin,1e-6)`; `r<3` → `confidence=0`.
4. Главный вектор `v=(vx,vy)`; привести к направлению вверх (`vy<0`); угол от вертикали `angle = degrees(atan2(vx, -vy))` (вправо = +).
5. Проверка Хафом: `Canny` по кропу → `HoughLinesP(minLineLength=0.4*H)`; взять отрезки с углом к вертикали <30°; средневзвешенный по длине угол `a_h`.
6. `confidence = clip((r-3)/10,0,1)`; если `|angle - a_h| > 3` → `confidence *= 0.5`. `method = 'pca_mask'`.

### `geometry.vertical_reference_angle(frame) -> float`
`Canny` → `HoughLinesP(minLineLength=0.1*H)`; углы к вертикали в `±15°`; нет ≥ 5 отрезков → `0.0`; иначе медиана.

### `geometry.analyze_hatch(crop) -> HatchResult`
1. Серый, blur, `Canny(50,150)`, `dilate`. 2. `findContours(RETR_EXTERNAL)`; взять контур с макс. площадью, центр которого в центральной трети кропа; нет → `found=False`.
3. `circularity = 4π·A/P²`. 4. `cv2.fitEllipse` (нужно ≥5 точек) → невязка: среднее по точкам контура `|(x'/a)²+(y'/b)²-1|` в системе эллипса.
5. Кольцо шириной 8% вдоль контура (`drawContours` толщиной) → доля пикселей с яркостью `< 0.6·медиана_кропа`.

### `geometry.crack_score(crop, mask=None) -> float`
Серый; `blackhat` с ядром 15×15; порог Otsu; маска объекта (если есть) применяется; скелет (Zhang–Suen или `skimage`); `length = число пикселей скелета`; `norm = 2·(H+W)`; `score = 1 - exp(-3·length/norm)`; клип 0..1.

### `geometry.corrosion_ratio(crop, mask=None) -> float`
HSV; `inRange(H 5–25, S 80–255, V 40–200)`; OPEN 3×3; `ratio = пикселей/пикселей_маски_или_всего`.

### `geometry.surface_subsidence_score(frame, bbox) -> float`
Кольцо = bbox, расширенный на 0.5·размер, минус bbox; фон = ещё одно кольцо шириной 0.5·размер снаружи. `d_ring`, `d_bg` — плотность Canny-краёв; `v_ring`, `v_bg` — дисперсия яркости. `score = clip(0.5·(d_ring/d_bg-1) + 0.5·(v_ring/v_bg-1), 0, 1)` (при нулевом знаменателе → 0).

### `classifier.DefectClassifier` и `train_classifier`
- `__init__`: прочитать `condition.classifier`; если файла весов нет → `self._model=None`.
- `is_ready` → `self._model is not None`.
- `predict(crop, class_name)`: `{}` если не готов; признаки/тензор (resize `input_size`, нормировка ImageNet) → вероятности по кодам, применимым к `class_name`.
- `train_classifier`: читать `index.csv`, split по `frame_id`, обучить, сохранить `weights` и `condition_metrics.json` (`{"per_defect": {code: {"precision","recall","threshold"}}}`).

### `scoring.assess(frame, detection, cfg, classifier, classes) -> ConditionAssessment`
```
bbox  = detection.bbox.pad(cfg.dataset.crop_padding, W, H)
crop  = frame[y1:y2, x1:x2];   min(h,w) < 24 → return Assessment(UNCERTAIN, 0.0, [], needs_review=True)
findings = []
по class_name:
  utility_pole:  tilt = estimate_pole_tilt(crop).angle_deg - vertical_reference_angle(frame)
                 |tilt| >= tilt_warn_deg → finding('pole_tilt', s, 'geometry', f'наклон {abs(tilt):.1f}° > порога {thr:.1f}°')
                 s = clip((|tilt| - warn) / (defect - warn), 0, 1) * 0.65 + 0.35   # warn→0.35, defect→1.0
  cable_manhole: hatch = analyze_hatch(crop); crack = crack_score(crop); subs = surface_subsidence_score(frame, bbox)
                 findings: cover_displaced (circularity < min или dark_gap высокий), cover_broken (crack ≥ порога), subsidence_around
  оба класса:    corrosion_ratio ≥ порога → 'corrosion'
classifier.is_ready → добавить findings из predict (source='classifier') для p >= порога кода
score = 1 - ∏(1 - weight[code] * finding.score)           # noisy-OR
severity = DEFECT if score >= defect_threshold else WARNING if score >= warning_threshold else OK
needs_review = any(
    abs(score - порог) < uncertain_band для ближайшего порога,
    detection.score < min_det_score,
    геометрия и классификатор расходятся (один >0.7, другой <0.2),
)
needs_review and severity != OK → severity = UNCERTAIN
```
Каждый блок признака — в `try/except`; любое исключение → признак пропускается. Возврат всегда `ConditionAssessment`, `0 <= score <= 1`.

### `scoring.priority_for(class_name, assessment, cfg, classes) -> Priority`
`crit = class_info(class_name)['criticality']`; `p = crit·score`; `+0.15` если среди `findings` есть `cover_missing` или `cable_sag`;
`p >= 0.75` → CRITICAL; `>= 0.5` → HIGH; `>= 0.25` или `needs_review` → MEDIUM; иначе LOW; `severity==OK and not needs_review` → LOW.

---

## 4. `mgts.geo`, `pipeline`, `export`, `cli` (Агент 4)

### `gps.read_exif_gps(path) -> dict | None`
`PIL.Image.open(path)._getexif()` / `getexif().get_ifd(0x8825)`; `lat = d+m/60+s/3600`, знак по `GPSLatitudeRef` (S → −), аналогично lon (W → −); `GPSImgDirection` → `heading_deg`; `DateTimeOriginal` → ISO. Нет GPS → `None`.

### `gps.load_track_csv(path)`; `gps.load_track_gpx(path)`
`csv.Sniffer` для разделителя; синонимы колонок: `lat|latitude`, `lon|lng|longitude`, `time|timestamp`, `frame|frame_id|filename`; числа → `float`. GPX: `<trkpt lat lon><time>`, курс = азимут на следующую точку.

### `gps.interpolate_position(track, timestamp) -> dict | None`
Найти соседние точки `p0 ≤ t ≤ p1`; `a=(t-t0)/(t1-t0)`; `lat,lon` линейно; курс: `d=((h1-h0+180)%360)-180; h=(h0+a*d)%360`; `t` дальше 5 с от границы → `None`.

### `gps.build_frame_meta(path, index, cfg, track=None) -> FrameMeta`
Размеры: `Image.open(path).size`. Источники по приоритету — см. докстринг. Синтетика: `lat=55.7500+index*4/111320`, `lon=37.6000`, `heading=0.0` (едем на север), `location_source='synthetic'`.

### `project.pixel_to_ground_offset(u, v, W, H, cam) -> (forward, right) | None`
```
f   = (W/2) / tan(radians(hfov/2));   cx, cy = W/2, H/2
d   = ((u-cx)/f, (v-cy)/f, 1.0)               # x вправо, y вниз, z вперёд
p   = radians(pitch)                           # вниз положительный
y'  =  d.y*cos(p) + d.z*sin(p)                 # вниз (мировая)
z'  = -d.y*sin(p) + d.z*cos(p)                 # вперёд (мировая)
y' <= 1e-9 → None                              # луч не идёт вниз
t   = height / y';   forward = t*z';  right = t*d.x
forward <= 0 или forward > 60 → None
```
Проверка: `u=cx, v=cy+200, pitch=0, H=1.4, W=1280, hfov=90` → `forward≈4.48`.

### `project.estimate_object_position(frame, bbox, cfg) -> (lat, lon, source)`
Точка: `bbox.bottom_center` (для люка допустимо `center`). `offset=pixel_to_ground_offset(...)`; нет → `(frame.lat, frame.lon, source+'_approx')`.
Сдвиг от камеры → от GPS-антенны: `forward += mount[0]; right += mount[1]`. Поворот: `h=radians(heading)`; `north = forward·cos h − right·sin h`; `east = forward·sin h + right·cos h`; `offset_latlon(frame.lat, frame.lon, north, east)`.

### `dedup.merge_observations(observations, cfg, classes) -> list[ObjectRecord]`
```
union-find по индексам наблюдений
для пар (i,j) одного class_name:
    оба с координатами: haversine ≤ radius_m [и сходство гистограмм ≥ appearance_min_sim, если use_appearance и оба кропа есть] → union
    без координат: соседние frame_id и bbox.iou > 0.3 → union
для каждого кластера:
    best = argmax(det.score * sharpness/max_sharp)
    lat,lon = медиана est_lat/est_lon (None-ы игнорировать)
    findings = по code: максимум score; evidence лучшего
    score = max(condition_score); severity = max по серьёзности (ok<warning<uncertain<defect)
    needs_review = any(...)
    priority = priority_for(class_name, ConditionAssessment(...), cfg, classes)
сортировка: priority (critical первым), затем condition_score убыв.; id = f"OBJ-{i:04d}" c 1
```

### `pipeline.run_pipeline(...)` — псевдокод
```
cfg-секции читать один раз; detector = build_detector(cfg); clf = DefectClassifier(cfg) (если condition.classifier.enabled)
track = load_track_* (если track_path)
observations = []
for i, path in enumerate(list_images(input_dir)[:max_frames]):
    try:
        meta = build_frame_meta(path, i, cfg, track); img = read_image(path)
        sharp = blur_score(img);  skip, если skip_blurry and sharp < blur_threshold  (n_skipped += 1)
        dets = detector.detect(enhance(img, cfg), meta.frame_id, str(path))
        for det in dets:
            a = assess(img, det, cfg, clf, classes)
            lat, lon, src = estimate_object_position(meta, det.bbox, cfg)
            crop → output/crops/<frame>_<k>.jpg (если export.save_crops)
            observations.append(Observation(...))
        annotated → output/annotated/<frame>.jpg
    except Exception: log.exception; n_errors += 1
    progress(i+1, n, meta.frame_id)
records = merge_observations(observations, cfg, classes)
to_csv / to_geojson / to_html_map; summary.json; return records
```
Имена файлов результата: `objects.csv`, `objects.geojson`, `map.html`, `summary.json`, `annotated/`, `crops/`.

### `export.*`
- `to_csv`: `csv.DictWriter`, `encoding='utf-8-sig'`, поля `CSV_COLUMNS + ['title_ru']`.
- `to_geojson`: `{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Point","coordinates":[lon,lat]},"properties":row}]}`; `ensure_ascii=False`.
- `to_html_map`: `folium.Map(location=средняя точка)`, `CircleMarker` на объект, цвет по приоритету, `Popup(html)`; `FeatureGroup` на каждый приоритет + `LayerControl`; `map.save(path)`.
- `draw_annotated`: копия; `cv2.rectangle`; подпись `f"{label}"` на залитом прямоугольнике; цвет по severity (BGR: ok (0,170,0), warning (0,200,255), defect (0,0,230), uncertain (180,60,160)).

### `cli.main(argv) -> int`
Парсер уже есть. Реализуй диспетчер по `args.cmd`; пользовательские ошибки (`FileNotFoundError`, `RuntimeError` из `build_detector`) → `print(msg, file=sys.stderr); return 2`; успех → `0`.
