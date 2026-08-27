PYTHON ?= python

.PHONY: install install-app test lint format-check verify train serve
install:
	$(PYTHON) -m pip install -e '.[dev]'
install-app:
	$(PYTHON) -m pip install -e '.[dev,app]'
test:
	$(PYTHON) -m pytest
lint:
	ruff check src tests train.py eval.py fortyguard_check.py plot.py summarize.py
format-check:
	ruff format --check src tests train.py eval.py fortyguard_check.py plot.py summarize.py
verify:
	$(PYTHON) -m compileall -q src train.py eval.py fortyguard_check.py plot.py summarize.py
	ruff check src tests train.py eval.py fortyguard_check.py plot.py summarize.py
	ruff format --check src tests train.py eval.py fortyguard_check.py plot.py summarize.py
	$(PYTHON) -m pytest
train:
	$(PYTHON) train.py --seed 0
serve:
	uvicorn coolworld.app:app --reload --host 127.0.0.1 --port 8000
