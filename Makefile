PYTHON ?= python3.12

.PHONY: install test verify train
install:
	$(PYTHON) -m pip install -e '.[dev]'
test:
	$(PYTHON) -m pytest
verify:
	$(PYTHON) -m compileall -q src train.py eval.py fortyguard_check.py plot.py summarize.py
	$(PYTHON) -m pytest
train:
	$(PYTHON) train.py --seed 0
