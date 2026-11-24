param(
    [Parameter(Position=0)]
    [string]$Target,
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$RemainingArgs
)

$Python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

switch ($Target) {
    "dev-eval" {
        & $Python -m app.cli dev-eval @RemainingArgs
    }
    "frozen-eval" {
        & $Python -m app.cli frozen-eval @RemainingArgs
    }
    "frozen-eval-force" {
        & $Python -m app.cli frozen-eval --force @RemainingArgs
    }
    "run" {
        & $Python -m uvicorn app.main:app --reload
    }
    "test" {
        & $Python -m pytest -q @RemainingArgs
    }
    "reproduce" {
        & $Python -m app.cli reproduce @RemainingArgs
    }
    "measure" {
        & $Python -m app.cli measure-break-even
    }
    "paper-artifacts" {
        & $Python -m app.cli paper-artifacts
    }
    "extractor-report" {
        & $Python -m app.cli extractor-report
    }
    "eval" {
        & $Python -m app.cli eval
    }
    "real-eval" {
        & $Python -m app.cli real-eval
    }
    "lint" {
        & ".venv\Scripts\ruff.exe" check app tests scripts
    }
    "format-check" {
        & ".venv\Scripts\ruff.exe" format --check app tests scripts
    }
    "typecheck" {
        & $Python -m mypy
    }
    default {
        Write-Host "Available targets: dev-eval, frozen-eval, frozen-eval-force, test, run, lint, typecheck, format-check, paper-artifacts"
    }
}
