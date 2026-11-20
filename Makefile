.PHONY: install install-dev test green demo-data e2e app lint

install:
	uv sync --locked --all-extras

# Базовые зависимости для CLI, демо и тестов; без PyTorch, YOLO и Streamlit.
install-dev:
	uv sync --locked --extra dev

test:
	pytest

green: test

# Синтетический мини-датасет с псевдо-GPS (для отладки без реальных данных)
demo-data:
	python scripts/make_demo_data.py --out data/demo --n 40

# Сквозной прогон: картинки -> перечень объектов -> CSV/GeoJSON/карта
e2e:
	python -m mgts.cli run --input data/demo/images --output outputs/e2e --track data/demo/frames.csv --backend dummy

app:
	streamlit run app/streamlit_app.py
