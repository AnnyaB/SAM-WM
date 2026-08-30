PYTHON ?= python

.PHONY: install install-app test lint format-check verify verify-runtime verify-frontend verify-demo train research paper-suite paper-figures serve

install:
	$(PYTHON) -m pip install -e '.[dev]'

install-app:
	$(PYTHON) -m pip install -e '.[dev,app]'

test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests scripts

format-check:
	ruff format --check src tests scripts

verify:
	$(PYTHON) -m compileall -q src scripts
	ruff check src tests scripts
	ruff format --check src tests scripts
	$(PYTHON) -m pytest

verify-runtime:
	$(PYTHON) scripts/verify_runtime.py

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
	$(PYTHON) scripts/train.py --seed 17

research:
	$(PYTHON) scripts/research.py --out artifacts/research

paper-suite:
	$(PYTHON) scripts/paper_suite.py --config config/paper.yaml --mode paper --fairurb-root "$(FAIRURB_ROOT)"

paper-figures:
	$(PYTHON) scripts/plot_results.py

serve:
	uvicorn coolworld.app:app --reload --host 127.0.0.1 --port 8000
