# Prompt Repository Lab

A clean-room implementation of the supplied product brief. It starts with an empty SQLite database and no synthetic or invented university records. This repository does not contain earlier proprietary archives, authentication state, or real Faculty prompt text.

## Data Provenance and Inventory

Every data file in the repository carries an explicit provenance label. Data files are classified into three strict categories:

### 1. Real Data (n = 0)
- `data/gold/inbox/prompts.csv`: Inbox for real faculty prompts. Must contain rows with `consent=yes` and verified human authorship. Currently empty (n = 0).
- `data/gold/inbox/annotations_*.csv`: Independent human annotations for prompts in the inbox. Requires at least two separate annotators per prompt before evaluation. Currently empty (n = 0).

### 2. Authored by Builder (Development & Frozen Test Sets)
- `data/gold/dev_authored/dev_prompts.json` and `dev_prompts.csv`: 68 prompts across 9 voices (terse, verbose, non-native English, bulleted, numbered, negations, paraphrases, split conflicts, zero atoms) authored by the builder (`synthetic: 1`, `source: authored_by_builder`) for developing rule-based extractor logic.
- `data/gold/synthetic/authored_prompts_SYNTHETIC.csv`: 12 prompts authored by the builder (`synthetic: 1`, `source: authored_by_builder`) designated as the frozen test set.
- `data/gold/FROZEN_SHA256.txt`: SHA-256 hash (`e5bea36a5a7e0000e53fcc365e9db696ed8ddf6378b07fce27b245b9b98d2c71`) ensuring the frozen test set is immutable.
- `data/gold/atoms_gold.json`: Authored prompt fixtures from the project brief used exclusively as logic regression tests.

### 3. Synthetic Benchmark & Simulation Data
- `data/eval/extractor_frozen_report.json`: One-time frozen evaluation report on the 12 authored frozen prompts.
- `data/eval/SYNTHETIC_recovery.json` and `SYNTHETIC_recovery.md`: Synthetic recovery benchmark results (3000 synthetic prompts, 5 seeds, budgets 20 to 400, 5000 held-out pairs) evaluating linear and interaction preference recovery.
- `data/simulated/personas.sqlite`: Synthetic rater simulation database. Strictly isolated from repository feedback and paper tables.
- `data/eval/REAL_evaluation_report.json`: Target output for real evaluation. Generated only when real prompts and human comparisons exist.

## What the Numbers Do and Do Not Show

### Frozen Test Set (n = 12)
- **What it shows:** On 12 authored prompts that the extractor was not tuned on, the rule-based extractor achieved 1.000 precision (0 false positives), 0.348 recall (8/23 atoms found), and 57.41% slot score agreement (62/108 slots). It identified built-in conflicts in 2 prompts and missed conflicts in 2 prompts.
- **What it does not show:** Generalization to real faculty writing or real-world prompt distributions. Matching annotations on text authored by the builder is a regression test, not a claim of general accuracy.

### Synthetic Recovery Benchmark (n = 3000, 5 seeds, 5000 held-out pairs)
- **What it shows:** The Bradley-Terry estimator correctly recovers linear preference weights from simulated pairwise comparisons as sample size increases, moving from 80.96% (+/- 2.48%) accuracy at budget N = 20 to 94.79% (+/- 1.08%) accuracy at budget N = 400. Under non-linear interaction model misspecification, recovery reaches 94.65% (+/- 0.79%).
- **What it does not show:** Real faculty preference behavior, educational validity of prompts, or quality of LLM generation outputs. This checks the mathematical estimator under controlled simulation only.

### Real Data Status (n = 0)
- **What it shows:** The repository contains zero real faculty prompts and zero real comparison ratings.
- **What it does not show:** Any measured real-world accuracy figures. No placeholder numbers or estimates are displayed in the UI, reports, or CLI. Running `make real-eval` prints an honest missing prerequisites report and exits without numbers.

## Setup and Run

1. `python -m venv .venv`
2. `.venv\Scripts\python.exe -m pip install -r requirements.txt` (Linux/macOS: `.venv/bin/python -m pip install -r requirements.txt`)
3. Copy `.env.example` to `.env` and set optional values only when needed.
4. `.venv\Scripts\python.exe -m uvicorn app.main:app --reload` (Linux/macOS: `.venv/bin/python -m uvicorn app.main:app --reload`)

Open http://127.0.0.1:8000. The `/docs` API explorer requires a University account. The UI is local HTML/CSS/JS and makes no external network calls.

## Local Role Accounts

Create the demo University, Course Coordinator, and Faculty accounts with `.venv\Scripts\python.exe scripts\seed_demo_users.py`. The command prints randomly generated credentials once and writes them to the gitignored `credentials.local.txt`. Save the credentials, then remove that file when it is no longer needed. Re-running refuses to overwrite existing accounts; `--reset-demo-passwords` rotates the demo passwords.

Authentication uses Argon2id password hashes, opaque HttpOnly SameSite session cookies, CSRF tokens for changes, session idle and absolute expiry, login throttling, and audit events. University administrators manage local accounts, course grants, catalog records, and audit history. Coordinators see assigned courses. Faculty build prompts from approved templates and assigned topics, check prompts, rate templates, submit proposals, and view their own history.

Demo departments, courses, topics, accounts, memberships, and templates are synthetic and labeled in the UI.

## Command Reference and Verification

- `make test`: Run pytest suite.
- `make eval`: Run synthetic recovery benchmark across 5 seeds and 5 budgets.
- `make extractor-report`: Generate extractor report on gold and inbox data.
- `make real-eval`: Execute real-data evaluation pipeline (aborts honestly if real data is missing).
- `python -m scripts.make_real_data_checklist`: Print full readiness checklist of real prompts, annotators, and comparisons.
- `make reproduce`: Run routing experiments (all, curve, lebesgue, label_permutation).
- `make paper-artifacts`: Generate paper-ready tables and claims verification.
- `make lint`: Run Ruff linter.
- `make format-check`: Check code formatting with Ruff.
- `make typecheck`: Run Mypy static type checker.
- `node --check web/dashboard.js`: Verify front-end JavaScript syntax.
