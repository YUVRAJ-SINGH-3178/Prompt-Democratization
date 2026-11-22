from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.environ.get("PROMPT_LAB_DB", Path(__file__).resolve().parent.parent / "data" / "university_repo.db"))


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc, tb):
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, role TEXT NOT NULL, discipline TEXT NOT NULL, study_level TEXT NOT NULL, experience REAL NOT NULL CHECK(experience BETWEEN -1 AND 1), synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic IN(0,1)), created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS courses(id TEXT PRIMARY KEY, name TEXT NOT NULL, discipline TEXT NOT NULL, synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic IN(0,1)));
CREATE TABLE IF NOT EXISTS user_courses(user_id TEXT NOT NULL REFERENCES users(id), course_id TEXT NOT NULL REFERENCES courses(id), synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic IN(0,1)), PRIMARY KEY(user_id,course_id));
CREATE TABLE IF NOT EXISTS topics(id TEXT PRIMARY KEY, course_id TEXT NOT NULL REFERENCES courses(id), name TEXT NOT NULL, synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic IN(0,1)));
CREATE TABLE IF NOT EXISTS templates(id TEXT PRIMARY KEY, name TEXT NOT NULL, task_type TEXT NOT NULL, prompt TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'draft', source TEXT NOT NULL DEFAULT 'user', synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic IN(0,1)));
CREATE TABLE IF NOT EXISTS task_defaults(task_type TEXT PRIMARY KEY, template_id TEXT NOT NULL REFERENCES templates(id));
CREATE TABLE IF NOT EXISTS template_versions(id INTEGER PRIMARY KEY AUTOINCREMENT, template_id TEXT NOT NULL REFERENCES templates(id), version INTEGER NOT NULL, prompt TEXT NOT NULL, synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic IN(0,1)), created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE(template_id,version));
CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, template_id TEXT NOT NULL, feature_json TEXT NOT NULL, rating REAL NOT NULL CHECK(rating BETWEEN 1 AND 5), edited INTEGER NOT NULL DEFAULT 0, timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic IN(0,1)));
CREATE TABLE IF NOT EXISTS prompts_checked(id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT NOT NULL, atoms_json TEXT NOT NULL, completeness REAL NOT NULL, energy INTEGER NOT NULL, repair_json TEXT NOT NULL, synthetic INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS linter_extraction_runs(id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT NOT NULL, logged_prompt TEXT NOT NULL, input_prompt TEXT NOT NULL, output_json TEXT NOT NULL, synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic IN(0,1)), created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS experiment_runs(id TEXT PRIMARY KEY, name TEXT NOT NULL, result_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic IN(0,1)));
CREATE TABLE IF NOT EXISTS annotations(id INTEGER PRIMARY KEY AUTOINCREMENT, annotator_id TEXT NOT NULL, prompt_id TEXT NOT NULL, atoms_json TEXT NOT NULL, slots_json TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'import', synthetic INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS llm_study_runs(id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT NOT NULL, model TEXT NOT NULL, condition_name TEXT NOT NULL, request_json TEXT NOT NULL, response_json TEXT, checks_json TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, synthetic INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS llm_ratings(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL REFERENCES llm_study_runs(id), rater_id TEXT NOT NULL, correctness INTEGER CHECK(correctness BETWEEN 1 AND 5), distractor_quality INTEGER CHECK(distractor_quality BETWEEN 1 AND 5), bloom_level TEXT, synthetic INTEGER NOT NULL DEFAULT 0);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, factory=ClosingConnection)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    columns = {row[1] for row in db.execute("PRAGMA table_info(experiment_runs)")}
    if "synthetic" not in columns:
        db.execute("ALTER TABLE experiment_runs ADD COLUMN synthetic INTEGER NOT NULL DEFAULT 0")
    for table in (
        "users",
        "courses",
        "user_courses",
        "topics",
        "templates",
        "template_versions",
        "linter_extraction_runs",
    ):
        columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
        if "synthetic" not in columns:
            db.execute(f"ALTER TABLE {table} ADD COLUMN synthetic INTEGER NOT NULL DEFAULT 0")
    return db


def all_rows(query: str, args: tuple = ()) -> list[dict]:
    with connect() as db:
        return [dict(row) for row in db.execute(query, args).fetchall()]


def get_feedback() -> list[dict]:
    rows = all_rows("SELECT * FROM feedback WHERE synthetic=0")
    for r in rows:
        r["feature_json"] = json.loads(r["feature_json"])
    return rows
