.PHONY: run test reproduce measure lint format-check typecheck paper-artifacts
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
lint:
	.venv/Scripts/ruff.exe check app tests scripts
format-check:
	.venv/Scripts/ruff.exe format --check app tests scripts
typecheck:
	.venv/Scripts/mypy.exe
