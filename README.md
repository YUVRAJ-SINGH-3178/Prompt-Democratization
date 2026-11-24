# Prompt Repository Lab

A clean-room implementation of the supplied product brief. It starts with an empty SQLite database and no synthetic or invented university records. This repository does not contain earlier proprietary archives, authentication state, or real Faculty prompt text.

## Data Provenance and Inventory

Every data file in the repository carries an explicit provenance label. Data files are classified into three strict categories:

### 1. Real Data (n = 0)
- `data/gold/inbox/prompts.csv`: Inbox for real faculty prompts. Must contain rows with `consent=yes` and verified human authorship. Currently empty (n = 0).
- `data/gold/inbox/annotations_*.csv`: Independent human annotations for prompts in the inbox. Requires at least two separate annotators per prompt before evaluation. Currently empty (n = 0).

### 2. Authored by Builder (Development & Frozen Test Sets)
- `data/gold/dev_authored/dev_prompts.json` and `dev_prompts.csv`: 68 prompts across 9 voices (terse, verbose, non-native English, bulleted, numbered, negations, paraphrases, split conflicts, zero atoms) authored by the builder (`synthetic: 1`, `source: authored_by_builder`) for developing rule-based extractor logic.
- `data/gold/dev_authored/authored_prompts_SYNTHETIC.csv`: 12 prompts authored by the builder (`synthetic: 1`, `source: authored_by_builder`) moved from the previous frozen test set into development.
- Total development dataset pool: 80 prompts (68 + 12).
- `data/gold/frozen_v2/`: Location for new frozen test set (evaluated once by `make frozen-eval`).
- `data/gold/FROZEN_SHA256.txt`: SHA-256 hash (line endings normalised CRLF to LF) for the legacy frozen set.
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

## What the Checker Can and Cannot See

### Recognised Atoms and Slots

The prompt requirement extractor recognizes exactly 9 specification atoms and 9 educational slots:

1. **Recognised Atoms (9):**
   - `very_easy`: Questions must be elementary or very easy.
   - `expert_level`: Questions must require advanced, expert, or research-level knowledge.
   - `no_basic`: Prohibits basic, introductory, or recall questions.
   - `ordered_easy_to_hard`: Questions must be arranged from easiest to hardest.
   - `hidden_answers`: Answers or answer key must be withheld or placed at the end.
   - `answer_after_each_q`: Explanations or answers must immediately follow each question.
   - `single_correct`: Exactly one option is correct.
   - `multi_interp`: Ambiguous wording or multiple interpretations (negative requirement).
   - `confusing_language`: Confusing, tricky, or misleading language (negative requirement).
   *(Plus system policy atom `clear_unambiguous` enforced by contradiction resolution).*

2. **Recognised Slots (9):**
   - `course`: Course name or subject area.
   - `topic`: Academic topic or syllabus module.
   - `cohort`: Student audience level (e.g. bachelor, postgraduate).
   - `count`: Number of questions requested.
   - `n_options`: Number of answer choices per question (e.g. 4 options).
   - `labeling`: Option label style (e.g. A to D).
   - `marking`: Correct answer indicator method.
   - `explanation`: Distractor rationale or explanation requirement.
   - `ordering`: Question sequence arrangement.

Everything outside these 9 atoms and 9 slots is unrecognised by the automated checker. If an author writes requirements regarding diagram formatting, grading penalties, LaTeX rendering, or code snippets, the automated checker cannot parse or verify them.

### Measured Recall and Precision (Development and Spent Frozen Sets)

We report measured recall and Wilson 95% confidence intervals alongside precision. No unqualified "100% precision" claims are made without recall context.

#### Combined Development Pool (n = 80 prompts: 68 authored dev + 12 legacy synthetic)
- **Rules Only:** Micro Precision 100.0% [95.4%, 100.0%], Micro Recall 84.2% [75.6%, 90.2%], Micro F1 91.4%, Slot Agreement 93.5% [91.4%, 95.1%], Review Rate 0.0%.
- **Reader Only (`openai/gpt-oss-20b`):** Micro Precision 87.1% [71.2%, 94.9%], Micro Recall 28.4% [20.3%, 38.2%], Micro F1 42.9%, Slot Agreement 54.6% [50.9%, 58.2%], Review Rate 0.0%.
- **Combined System:** Micro Precision 100.0% [95.2%, 100.0%], Micro Recall 80.0% [70.9%, 86.8%], Micro F1 88.9%, Slot Agreement 92.4% [90.2%, 94.1%], Review Rate 2.1%.

#### Spent Frozen v2 Benchmark (n = 30 unseen authored prompts)
- **Rules Only:** Micro Precision 100.0% [89.3%, 100.0%], Micro Recall 44.9% [31.9%, 58.7%], Micro F1 62.0%.
- **Reader Only (20B model):** Micro Precision 62.5% [30.6%, 86.3%], Micro Recall 10.2% [4.4%, 22.2%], Micro F1 17.5%.
- **Combined System:** Micro Precision 100.0% [89.3%, 100.0%], Micro Recall 44.9% [31.9%, 58.7%], Micro F1 62.0%, Review Rate 3.7%.

### Why Energy is a Lower Bound

Because the extractor recognises only 9 atoms and its measured recall on unseen prompts is bounded (44.9% on frozen v2), unrecognised requirements in user prompts can conceal undetected contradictions.

Therefore:
- Energy is strictly reported as a lower bound: `energy_is_lower_bound: true`.
- The UI and API display: "at least N conflicts found" or "[N, ...] conflicts".
- When N is 0, the display states: "0 conflicts found among recognised requirements; unrecognised requirements may add more".
- Single unqualified energy numbers are never presented while energy is a lower bound.

### Unrecognised-Cue Flags and Review Queue

To prevent silent misses where both rules and reader fail to recognize an intent, the system monitors lexical cues (`must`, `should`, `only`, `never`, `ensure`, `strictly`, etc.) grouped in `data/gold/cue_lexicon.json`.

- When a cue appears in a sentence with no accepted requirement atom, the system raises a `possible_unrecognised_requirement` flag attached to the sentence.
- The flag routes the sentence to the manual review queue for human confirmation.
- **Cost Analysis on Development Pool (n = 80):**
  - Missed gold requirements caught by cue flags: 68.0% (17 / 25).
  - False-flag rate on zero-gold prompts: 33.3% (6 / 18).
  - Review load: 0.38 flags per prompt (30 flags across 80 prompts).

### LLM Reader: Contributions, Quote Discards, and Failure Modes

- **Verbatim Evidence Requirement:** Every model claim must provide an exact verbatim substring quote from the prompt. Claims with missing or non-verbatim quotes are marked `unsupported` and discarded in code.
- **Strict vs Normalised Quote Comparison:** Normalised quote checks (ignoring whitespace differences, smart quotes, and dash variants) yielded identical accepted claims (194) and precision (87.1%) as strict substring checks on dev data, so strict verbatim matching is retained.
- **Discard Logging:** For development prompts only, non-verbatim discarded claims are logged to `data/eval/diagnostics/reader_discards_dev.jsonl`. No frozen prompt text is ever written to diagnostics.
- **Provider Failure Modes:** If an API key is missing, or if the model encounters rate limits (HTTP 429), permissions restrictions (HTTP 403), or malformed output, the system falls back to rules only and exposes the reason (`no_key`, `rate_limit`, `malformed_output`, `network_error`) in the UI.

### What is Required to Verify a Prompt

The prompt improvement gate enforces four conditions:
1. No accepted atom has `needs_review`.
2. No `possible_unrecognised_requirement` cue flag remains open on the prompt.
3. The LLM reader did not fail (it returned valid output, or the run was explicitly rules-only with that status visible).
4. `energy_is_lower_bound` is false.

Because `energy_is_lower_bound` remains true today (until real faculty prompt data with double-annotator agreement is evaluated), the gate returns `verified: "extractor_limited"`, displays an "EXTRACTOR LIMITED" badge in the UI, and presents the faculty notice:
> *"This check can only see requirements it recognises. Items marked for review may hide conflicts."*

Lifting the lower bound requires:
1. Real faculty prompts deposited with `consent=yes` in `data/gold/inbox/prompts.csv`.
2. Independent double-annotations from at least two annotators in `data/gold/inbox/annotations_*.csv`.
3. An evaluation demonstrating that unrecognised requirements do not hide active conflicts.

## Dual CLI Command Reference (Linux and Windows PowerShell)

The evaluation runner `scripts/run_eval.py` works on both Windows and Linux without requiring `make`.

| Task | Linux / GNU Make | Windows PowerShell |
|---|---|---|
| **Dev Evaluation** | `make dev-eval`<br>*or* `python scripts/run_eval.py dev-eval` | `python scripts/run_eval.py dev-eval` |
| **Model Sweep** | `python scripts/run_eval.py model-sweep --models openai/gpt-oss-20b,openai/gpt-oss-120b` | `python scripts/run_eval.py model-sweep --models openai/gpt-oss-20b,openai/gpt-oss-120b` |
| **Frozen Set v3** | `make frozen-eval`<br>*or* `python scripts/run_eval.py frozen-eval --set v3` | `python scripts/run_eval.py frozen-eval --set v3` |
| **Frozen Set v3 (Force Rerun)** | `make frozen-eval-force`<br>*or* `python scripts/run_eval.py frozen-eval --set v3 --force`<br>*(Note: GNU make cannot pass --force directly)* | `python scripts/run_eval.py frozen-eval --set v3 --force` |
| **Real Evaluation** | `make real-eval`<br>*or* `python scripts/run_eval.py real-eval` | `python scripts/run_eval.py real-eval` |
| **Run Full Test Suite** | `make test` | `python -m pytest` |
| **Run Linter** | `make lint` | `ruff check app tests scripts` |
| **Check Formatting** | `make format-check` | `ruff format --check app tests scripts` |
| **Static Type Check** | `make typecheck` | `mypy app` |
| **Dashboard JS Syntax** | `node --check web/dashboard.js` | `node --check web/dashboard.js` |
