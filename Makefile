PYTHON ?= python
PY_SCRIPTS = train.py eval.py research.py promote.py provider_replay.py candra_fit.py fortyguard_check.py fortyguard_collect.py plot.py summarize.py

.PHONY: install install-app test lint format-check verify train research serve
install:
	$(PYTHON) -m pip install -e '.[dev]'
install-app:
	$(PYTHON) -m pip install -e '.[dev,app]'
test:
	$(PYTHON) -m pytest
lint:
	ruff check src tests $(PY_SCRIPTS)
format-check:
	ruff format --check src tests $(PY_SCRIPTS)
verify:
	$(PYTHON) -m compileall -q src $(PY_SCRIPTS)
	ruff check src tests $(PY_SCRIPTS)
	ruff format --check src tests $(PY_SCRIPTS)
	$(PYTHON) -m pytest
train:
	$(PYTHON) train.py --seed 17
research:
	$(PYTHON) research.py --stage all-pre-freeze
serve:
	uvicorn coolworld.app:app --reload --host 127.0.0.1 --port 8000
