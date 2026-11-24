@echo off
setlocal
set PYTHON=.venv\Scripts\python.exe
if not exist "%PYTHON%" (
    set PYTHON=python.exe
)

if "%1"=="" goto help
if "%1"=="dev-eval" goto dev_eval
if "%1"=="frozen-eval" goto frozen_eval
if "%1"=="frozen-eval-force" goto frozen_eval_force
if "%1"=="run" goto run
if "%1"=="test" goto test
if "%1"=="reproduce" goto reproduce
if "%1"=="measure" goto measure
if "%1"=="paper-artifacts" goto paper_artifacts
if "%1"=="extractor-report" goto extractor_report
if "%1"=="eval" goto eval
if "%1"=="real-eval" goto real_eval
if "%1"=="lint" goto lint
if "%1"=="format-check" goto format_check
if "%1"=="typecheck" goto typecheck

echo Unknown target: %1
exit /b 1

:dev_eval
shift
"%PYTHON%" -m app.cli dev-eval %*
exit /b %errorlevel%

:frozen_eval
shift
"%PYTHON%" -m app.cli frozen-eval %*
exit /b %errorlevel%

:frozen_eval_force
shift
"%PYTHON%" -m app.cli frozen-eval --force %*
exit /b %errorlevel%

:run
"%PYTHON%" -m uvicorn app.main:app --reload
exit /b %errorlevel%

:test
"%PYTHON%" -m pytest -q %2 %3 %4 %5
exit /b %errorlevel%

:reproduce
shift
"%PYTHON%" -m app.cli reproduce %*
exit /b %errorlevel%

:measure
"%PYTHON%" -m app.cli measure-break-even
exit /b %errorlevel%

:paper_artifacts
"%PYTHON%" -m app.cli paper-artifacts
exit /b %errorlevel%

:extractor_report
"%PYTHON%" -m app.cli extractor-report
exit /b %errorlevel%

:eval
"%PYTHON%" -m app.cli eval
exit /b %errorlevel%

:real_eval
"%PYTHON%" -m app.cli real-eval
exit /b %errorlevel%

:lint
.venv\Scripts\ruff.exe check app tests scripts
exit /b %errorlevel%

:format_check
.venv\Scripts\ruff.exe format --check app tests scripts
exit /b %errorlevel%

:typecheck
"%PYTHON%" -m mypy
exit /b %errorlevel%

:help
echo Available targets: dev-eval, frozen-eval, frozen-eval-force, test, run, lint, typecheck, format-check, paper-artifacts
exit /b 0
