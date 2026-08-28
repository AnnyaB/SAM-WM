PYTHON ?= python
PY_SCRIPTS = train.py eval.py research.py promote.py provider_replay.py candra_fit.py fortyguard_check.py fortyguard_collect.py plot.py summarize.py verify_runtime.py

.PHONY: install install-app test lint format-check verify verify-runtime verify-frontend verify-demo train research serve

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

verify-runtime:
	$(PYTHON) verify_runtime.py

verify-frontend:
	@if command -v node >/dev/null 2>&1; then \
		node --check static/app.js && \
		node --check static/bootstrap.js && \
		node --check static/product.js && \
		node --check static/interpretability.js; \
	else \
		echo "node not installed; frontend syntax check skipped locally"; \
	fi

verify-demo: verify verify-runtime verify-frontend
	@echo "SAM-WM CoolWorld source + runtime + frontend verification passed"

train:
	$(PYTHON) train.py --seed 17

research:
	$(PYTHON) research.py --out artifacts/research

serve:
	uvicorn coolworld.app:app --reload --host 127.0.0.1 --port 8000
