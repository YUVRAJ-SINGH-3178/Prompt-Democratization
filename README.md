# Prompt Repository Lab
A clean-room implementation of the supplied product brief. It starts with an empty SQLite database and no synthetic or invented university records. Because the referenced archive was unavailable, this project does not contain the earlier repository's data, authentication, implementation-steps document, or exact Faculty prompt text.
## Setup and run
1. `python -m venv .venv`
2. `.venv\Scripts\python.exe -m pip install -r requirements.txt` (Linux/macOS: `.venv/bin/python -m pip install -r requirements.txt`)
3. Copy `.env.example` to `.env` and set optional values only when needed.
4. `.venv\Scripts\python.exe -m uvicorn app.main:app --reload` (Linux/macOS: `.venv/bin/python -m uvicorn app.main:app --reload`)

Open http://127.0.0.1:8000. The `/docs` API explorer requires a University account. The UI is local HTML/CSS/JS and makes no external network calls.

## Local role accounts

Create the demo University, Course Coordinator, and Faculty accounts with `.venv\Scripts\python.exe scripts\seed_demo_users.py`. The command prints randomly generated credentials once and writes them to the gitignored `credentials.local.txt`. Save the credentials, then remove that file when it is no longer needed. Re-running refuses to overwrite existing accounts; `--reset-demo-passwords` rotates the demo passwords.

Authentication uses Argon2id password hashes, opaque HttpOnly SameSite session cookies, CSRF tokens for changes, session idle and absolute expiry, login throttling, and audit events. University administrators manage local accounts, course grants, catalog records, and audit history. Coordinators see assigned courses. Faculty build prompts from approved templates and assigned topics, check prompts, rate templates, submit proposals, and view their own history.

Demo departments, courses, topics, accounts, memberships, and templates are synthetic and labeled in the UI. Six prompt library entries are authored synthetic substitutes. The reference annotation file contains metrics but not original Faculty prompt text, so the app does not claim source-text comparison. Accuracy remains “awaiting annotations” until 30 held-out prompts each have two distinct annotators; sample prompts are excluded.

## Experiments

Run `.venv\Scripts\python.exe -m app.cli reproduce lebesgue` or `reproduce curve`, `reproduce label_permutation`, or `reproduce all`. Each run writes a timestamped JSON file to `data/experiment_runs/` and a row to `experiment_runs` in SQLite. Default curve runs use seed 42 and 100 repetitions. To measure the routing cold-start threshold on available real ratings, run `.venv\Scripts\python.exe -m app.cli measure-break-even`. With no evidence, it reports `insufficient_real_feedback` and leaves the threshold unset. `ROUTER_MIN_RATINGS` may be set explicitly by an operator.

## Data and study

Templates, mapping records, ratings, lint checks, annotations, experiment metadata, and LLM study requests/results use SQLite. CSV imports report row errors. `synthetic=true` records are excluded from routing and clearly flagged. The optional generation runner requires `STUDY_API_KEY` or `OPENAI_API_KEY` and `STUDY_MODEL`; it is not run by setup. Its five conditions and automated output checks are exposed through `/api/study/run`. Results are not claims of improvement. Human scores are added through `/api/study/ratings`.

## Simulated persona pilot

Persona simulation data is isolated in `data/simulated/personas.sqlite`; it never enters the university repository feedback or paper metrics. Generate the seeded persona cards and print their table with `.venv\Scripts\python.exe scripts\generate_personas.py`. Preview the work with `.venv\Scripts\python.exe scripts\simulate_personas.py --plan-only`. To execute the 300 separate model requests, configure `STUDY_API_KEY`, `STUDY_MODEL`, and optional `STUDY_API_BASE` in a local `.env`, then run `.venv\Scripts\python.exe scripts\simulate_personas.py`. Calls use seed 731204 plus a deterministic call index and temperature 0.2; full request and response bodies are logged with credentials excluded. Refresh the analysis and provenance audit with `.venv\Scripts\python.exe scripts\analyze_simulation.py`.

The `/api/simulated` endpoint and **Simulated Study** page display a persistent `SIMULATED RATERS` banner. The simulator uses five explicitly synthetic study templates because the repository has no templates. The original five faculty prompt texts and source template were not present, so they are not fabricated or included as annotation items. T3 uses 30 anonymous common prompts drawn from T1. Annotation necessarily presents these prompt texts as blinded stimuli; persona cards, author identities, linter output, and other ratings are withheld. All model calls use the same configured model, so agreement is likely inflated and cannot establish real faculty behavior, factual correctness, educational effectiveness, or system accuracy.

The later comparison import accepts `faculty_id,item_id,metric,value` CSV rows at `/api/simulated/import-real-scores`. Use pseudonymous faculty IDs. Imported real comparison scores stay in their own table and are compared with simulated majorities only on matching item IDs; they do not enter routing or paper-rating tables.

Routing accepts category names rather than integer group labels. GP uses `ConstantKernel * RBF + WhiteKernel`, normalized target values, and a deterministic seed. Its minimum rating threshold stays unset until measured or configured. Legacy repository endpoints are restricted to the University role. No LLM generation study runs during setup or through the new dashboards.

## Verification

Install the extra developer checks with `.venv\Scripts\python.exe -m pip install -r requirements-dev.txt`. Run `.venv\Scripts\python.exe -m pytest -q --basetemp=artifacts\pytest-temp`, `.venv\Scripts\ruff.exe check app tests scripts`, `.venv\Scripts\ruff.exe format --check app tests scripts`, `.venv\Scripts\mypy.exe`, and `node --check web/dashboard.js`. Browser acceptance runs with `.venv\Scripts\python.exe scripts\browser_check.py`; it writes screenshots and a report under `artifacts/`.

