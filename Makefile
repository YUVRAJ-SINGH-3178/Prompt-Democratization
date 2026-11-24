# Prompt Repository Lab Makefile
# Uses python from the active virtualenv or PATH by default.
# Override via: make PYTHON=.venv/Scripts/python.exe <target>

PYTHON ?= python
RUFF ?= ruff

.PHONY: run test reproduce measure lint format-check typecheck paper-artifacts extractor-report eval real-eval dev-eval frozen-eval frozen-eval-force

run:
	$(PYTHON) -m uvicorn app.main:app --reload

test:
	$(PYTHON) -m pytest -q

reproduce:
	$(PYTHON) -m app.cli reproduce all

measure:
	$(PYTHON) -m app.cli measure-break-even

paper-artifacts:
	$(PYTHON) -m app.cli paper-artifacts

extractor-report:
	$(PYTHON) -m app.cli extractor-report

eval:
	$(PYTHON) -m app.cli eval

real-eval:
	$(PYTHON) -m app.cli real-eval

dev-eval:
	$(PYTHON) -m app.cli dev-eval

frozen-eval:
	$(PYTHON) -m app.cli frozen-eval $(ARGS)

frozen-eval-force:
	$(PYTHON) -m app.cli frozen-eval --force

lint:
	$(RUFF) check app tests scripts

format-check:
	$(RUFF) format --check app tests scripts

typecheck:
	$(PYTHON) -m mypy
