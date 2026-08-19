.PHONY: test lint serve

test:
	pytest -q

lint:
	ruff check src tests

serve:
	uvicorn coolworld.api:app --reload

check-js:
	node --check static/app.js
