.PHONY: install lint test eval-offline api ui sample-pdf

install:
	pip install -e ".[dev,hf]"

lint:
	ruff check . && ruff format --check .

test:
	pytest

eval-offline:
	python scripts/evaluate.py --offline --out-dir eval/results/offline_fake

sample-pdf:
	python scripts/make_sample_pdf.py

api:
	uvicorn documind.api.main:app --reload

ui:
	streamlit run app/streamlit_app.py
