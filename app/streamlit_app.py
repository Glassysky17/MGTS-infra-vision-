"""Веб-демо на Streamlit (Агент 4, задача 7). Запуск: `streamlit run app/streamlit_app.py`.

Макет интерфейса:
  Боковая панель: загрузка нескольких изображений ИЛИ выбор папки data/demo; необязательный CSV/GPX трек;
                  переключатели: тайлинг, улучшение изображения, порог уверенности; кнопка «Анализировать».
  Основная часть (вкладки):
    1. «Перечень»: таблица ObjectRecord (приоритет цветом, фильтры по приоритету/классу/«требует проверки»),
       кнопки скачать CSV и GeoJSON.
    2. «Карта»: HTML-карта (components.html) с маркерами по приоритету.
    3. «Кадры»: галерея аннотированных кадров; клик/выбор объекта из таблицы показывает кроп и список дефектов
       с evidence («наклон 9.3° > порога 8°»).
    4. «Сводка»: метрики (кадров, объектов, critical/high, пропущено размытых), время обработки на кадр.
  Прогресс-бар через progress-callback пайплайна. Результаты кэшировать (st.cache_data / session_state).
  Если реальных координат нет — явная плашка «Координаты синтетические/отсутствуют».
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    """Запустить интерфейс загрузки, анализа и просмотра результатов MGTS."""
    import streamlit as st
    import streamlit.components.v1 as components

    from mgts.config import load_config, resolve_path
    from mgts.observability import configure_logging, log_context, new_run_id, trace_span
    from mgts.pipeline import run_pipeline

    st.set_page_config(page_title="МГТС · мониторинг инфраструктуры", layout="wide")
    st.title("Мониторинг инфраструктуры МГТС")
    st.caption("Анализ кадров служебных автомобилей с объяснимой оценкой состояния")
    with st.sidebar:
        st.header("Входные данные")
        uploads = st.file_uploader("Загрузить изображения", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"], accept_multiple_files=True)
        folder = st.text_input("Или папка с кадрами", value="data/demo/images")
        track_upload = st.file_uploader("GPS-трек (необязательно)", type=["csv", "gpx"])
        config_file = st.text_input("Конфиг (необязательно)")
        detector_ready = resolve_path("models/detector.pt").is_file()
        backend = st.selectbox(
            "Детектор", ["yolo", "dummy"], index=0 if detector_ready else 1,
            format_func={"yolo": "YOLO · обученная модель", "dummy": "Демо · готовая разметка"}.get,
        )
        if not detector_ready:
            st.info("Обученной YOLO-модели пока нет. Демо-режим читает готовые рамки из labels/ и сам фото не распознаёт.")
        run = st.button("Анализировать", type="primary", use_container_width=True)

    if run:
        configure_logging()
        with log_context(run_id=new_run_id(), command="streamlit"):
            try:
                with trace_span(logger, "streamlit.analysis", backend=backend,
                                input_folder=str(folder) if not uploads else f"uploads:{len(uploads)}",
                                has_track=bool(track_upload)):
                    working = Path(tempfile.mkdtemp(prefix="mgts-demo-"))
                    images_dir = working / "images"
                    images_dir.mkdir()
                    track_path = None
                    if uploads:
                        for uploaded in uploads:
                            (images_dir / Path(uploaded.name).name).write_bytes(uploaded.getvalue())
                    else:
                        images_dir = Path(folder).expanduser()
                    if track_upload:
                        track_path = working / Path(track_upload.name).name
                        track_path.write_bytes(track_upload.getvalue())
                    output_dir = working / "output"
                    cfg = load_config(config_file or None, overrides={"detection": {"backend": backend}})
                    bar = st.progress(0, text="Ожидание обработки")

                    def update_progress(current: int, total: int, frame_id: str) -> None:
                        bar.progress(current / total, text=f"Кадр {current}/{total}: {frame_id}")

                    records = run_pipeline(images_dir, output_dir, cfg, track_path=track_path, progress=update_progress)
                    st.session_state["mgts_result"] = {
                        "records": records,
                        "output": str(output_dir),
                        "summary": json.loads((output_dir / "summary.json").read_text(encoding="utf-8")),
                        "backend": backend,
                    }
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                st.error(str(error))
            except Exception:
                st.error("Не удалось обработать входные данные. Подробности записаны в журнал.")

    result = st.session_state.get("mgts_result")
    if not result:
        st.info("Загрузите кадры или выберите папку с изображениями, затем запустите анализ.")
        return
    records, output_dir, summary = result["records"], Path(result["output"]), result["summary"]
    if result.get("backend") == "dummy":
        st.warning("Демо-режим использует готовую разметку; это не результат автоматического распознавания.")
        if summary["n_detections"] == 0:
            st.info("Рядом с кадрами не найдены файлы разметки labels/*.txt.")
    unknown_locations = sum(record.lat is None or record.lon is None for record in records)
    approximate_locations = sum(
        record.location_source == "synthetic" or record.location_source.endswith("_approx") for record in records
    )
    if unknown_locations:
        st.warning(f"Местоположение неизвестно у {unknown_locations} объектов; они останутся в списке, но не на карте.")
    if approximate_locations:
        st.warning(f"У {approximate_locations} объектов местоположение приблизительное или демонстрационное.")
    if not records:
        st.info("Объекты не найдены.")

    tabs = st.tabs(["Перечень", "Карта", "Кадры", "Сводка"])
    with tabs[0]:
        priorities = sorted({record.priority.value for record in records})
        selected_priorities = st.multiselect("Приоритет", priorities, default=priorities)
        classes = sorted({record.class_name for record in records})
        selected_classes = st.multiselect("Класс", classes, default=classes)
        review_only = st.checkbox("Требует проверки")
        visible = [record for record in records if record.priority.value in selected_priorities and record.class_name in selected_classes and (not review_only or record.needs_review)]
        rows = []
        for record in visible:
            row = record.to_row()
            if record.lat is None or record.lon is None:
                row["location_source"] = "Неизвестно"
            rows.append(row)
        st.dataframe(rows, use_container_width=True, hide_index=True)
        csv_path, geojson_path = output_dir / "objects.csv", output_dir / "objects.geojson"
        first, second = st.columns(2)
        first.download_button("Скачать CSV", csv_path.read_bytes(), "objects.csv", "text/csv")
        second.download_button("Скачать GeoJSON", geojson_path.read_bytes(), "objects.geojson", "application/geo+json")
    with tabs[1]:
        map_path = output_dir / "map.html"
        components.html(map_path.read_text(encoding="utf-8"), height=680, scrolling=True)
    with tabs[2]:
        annotated = sorted((output_dir / "annotated").glob("*.jpg")) if (output_dir / "annotated").exists() else []
        if annotated:
            st.image([str(path) for path in annotated], caption=[path.stem for path in annotated], use_container_width=True)
        if records:
            chosen_id = st.selectbox("Объект для объяснения", [record.object_id for record in records])
            chosen = next(record for record in records if record.object_id == chosen_id)
            left, right = st.columns(2)
            if chosen.crop_path and Path(chosen.crop_path).is_file():
                left.image(chosen.crop_path, caption=chosen.class_name)
            right.write(f"**{chosen.priority.value} · {chosen.severity.value}**")
            if chosen.needs_review:
                right.warning("Нужна ручная проверка")
            for finding in chosen.findings:
                right.write(f"- **{finding.code}** — {finding.evidence or 'признак обнаружен'}")
    with tabs[3]:
        first, second, third = st.columns(3)
        first.metric("Кадров", summary["n_frames"])
        second.metric("Объектов", summary["n_objects"])
        third.metric("Пропущено размытых", summary["n_skipped_blurry"])
        st.write("Приоритеты", summary["priority"])
        st.write("Ошибки обработки кадров", summary["n_errors"])
        st.write("Время, мс/кадр", summary["ms_per_frame"])


if __name__ == "__main__":
    main()
