# Convenience targets. Nothing here is load-bearing — it just saves typing.
.PHONY: install ingest report test lint clean

install:        ## editable install with deep-learning + dev extras
	pip install -e ".[deep,dev]"

ingest:         ## pull the universe into the SQL store
	python -m quantlab.data.ingest

report:         ## run the full pipeline and regenerate reports/figures
	python scripts/generate_report.py

test:           ## run the test suite
	pytest -q

lint:           ## style check
	ruff check quantlab tests scripts

clean:          ## remove caches and generated artifacts
	rm -rf data/cache/*.db reports/figures/*.png reports/research_report.md
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
