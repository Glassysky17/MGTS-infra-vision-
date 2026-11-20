.PHONY: install test red green demo-data e2e app lint

install:
	pip install -r requirements.txt && pip install -e .

test:
	pytest

# TDD: заглушки считаются красными
red:
	MGTS_STRICT=1 pytest -x

# приёмка: всё должно быть зелёным без пропусков
green:
	MGTS_STRICT=1 pytest

# Синтетический мини-датасет с псевдо-GPS (для отладки без реальных данных)
demo-data:
	python scripts/make_demo_data.py --out data/demo --n 40

# Сквозной прогон: картинки -> перечень объектов -> CSV/GeoJSON/карта
e2e:
	python -m mgts.cli run --input data/demo/images --output outputs/e2e --track data/demo/frames.csv --backend dummy

app:
	streamlit run app/streamlit_app.py
