.PHONY: install pipeline test lint serve docker

install:  ## everything needed to run the pipeline, charts and tests
	pip install -r requirements-dev.txt

pipeline:  ## raw CSVs in data/raw -> model, metrics and charts (about two minutes)
	python -m delivery_risk.pipeline --raw-dir data/raw

test:
	pytest -q

lint:
	ruff check .

serve:  ## demo page at http://localhost:7860, API docs at /docs
	uvicorn delivery_risk.serve:app --port 7860

docker:
	docker build -t late-delivery-risk . && docker run --rm -p 7860:7860 late-delivery-risk
