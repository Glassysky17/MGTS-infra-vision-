# Контракты данных

Источник истины — `src/mgts/schemas.py`. Здесь — смысл полей и форматы файлов.

## Объекты в коде

| Тип | Назначение | Важные правила |
|---|---|---|
| `BBox(x1,y1,x2,y2)` | рамка в **абсолютных пикселях** исходного кадра | YOLO-формат нормированный: `to_xywhn / from_xywhn` |
| `FrameMeta` | метаданные кадра, позиция **автомобиля** | `location_source` ∈ exif / track / csv / synthetic / none |
| `Detection` | один объект на кадре от детектора | `score` — уверенность детектора, не состояния |
| `DefectFinding` | один признак дефекта | `code` из `classes.yaml → defects`, `evidence` обязателен для объяснимости |
| `ConditionAssessment` | оценка состояния объекта на кадре | `score` 0..1 («степень неисправности»); `UNCERTAIN` → `needs_review` |
| `Observation` | детекция + оценка + оценка позиции объекта | одно наблюдение, до дедупликации |
| `ObjectRecord` | уникальный объект итогового перечня | `to_row()` = строка CSV, колонки `CSV_COLUMNS` |

## Файлы на диске

### YOLO-датасет (`data/processed/`)
```
images/*.jpg   labels/*.txt   data.yaml   train.txt   val.txt
```
Строка разметки: `class_id xc yc w h` (0..1). `data.yaml → names` в порядке `configs/classes.yaml`.

### Вырезки (`data/crops/`)
`<split>/<class_name>/<label>/<frame>_<idx>.jpg` + `index.csv`
(`path, split, class_name, label, frame_id, x1, y1, x2, y2[, synthetic]`).

### Трек (`frames.csv`)
`frame_id,lat,lon,heading_deg,timestamp` (разделитель `,` или `;`; допустимы названия `latitude/longitude/lng`).

### Результаты запуска (`outputs/<run>/`)
| Файл | Содержимое |
|---|---|
| `objects.csv` | UTF-8 с BOM; колонки `CSV_COLUMNS` + `title_ru` |
| `objects.geojson` | `FeatureCollection`, `Point [lon, lat]`, `properties = ObjectRecord.to_row()` |
| `map.html` | автономная карта (folium) |
| `annotated/*.jpg` | кадры с рамками (цвет по severity) |
| `crops/*.jpg` | вырезки лучших наблюдений |
| `summary.json` | счётчики и время по этапам |

## Значения

- `Severity`: `ok`, `warning`, `defect`, `uncertain`
- `Priority`: `low`, `medium`, `high`, `critical` (порядок возрастания)
- Координаты: WGS84, градусы. Курс: 0 = север, 90 = восток.
- В GeoJSON порядок `[lon, lat]`, в CSV — отдельные колонки `lat`, `lon`.

## Как менять контракт
Строка в `tasks/CONTRACT_REQUESTS.md`: `кто | что | зачем`. Агент 4 решает и вносит изменение одним коммитом
вместе с обновлением `schemas.py`, тестов и этого файла.
