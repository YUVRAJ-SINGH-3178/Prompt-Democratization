.PHONY: run test reproduce measure lint format-check typecheck paper-artifacts extractor-report eval real-eval dev-eval frozen-eval
run:
	.venv/Scripts/python.exe -m uvicorn app.main:app --reload
test:
	.venv/Scripts/python.exe -m pytest -q
reproduce:
	.venv/Scripts/python.exe -m app.cli reproduce all
measure:
	.venv/Scripts/python.exe -m app.cli measure-break-even
paper-artifacts:
	.venv/Scripts/python.exe -m app.cli paper-artifacts
extractor-report:
	.venv/Scripts/python.exe -m app.cli extractor-report
eval:
	.venv/Scripts/python.exe -m app.cli eval
real-eval:
	.venv/Scripts/python.exe -m app.cli real-eval
dev-eval:
	.venv/Scripts/python.exe -m app.cli dev-eval
frozen-eval:
	.venv/Scripts/python.exe -m app.cli frozen-eval
lint:
	.venv/Scripts/ruff.exe check app tests scripts
format-check:
	.venv/Scripts/ruff.exe format --check app tests scripts
typecheck:
	.venv/Scripts/python.exe -m mypy
