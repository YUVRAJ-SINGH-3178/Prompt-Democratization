"""Storage for synthetic persona studies, isolated from university_repo.db."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("SIMULATED_DB_PATH", ROOT / "data" / "simulated" / "personas.sqlite"))

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS personas(
 persona_id TEXT PRIMARY KEY, card_json TEXT NOT NULL, synthetic INTEGER NOT NULL CHECK(synthetic=1),
 source TEXT NOT NULL CHECK(source='sim_persona'));
CREATE TABLE IF NOT EXISTS templates(
 template_id TEXT PRIMARY KEY, name TEXT NOT NULL, task_type TEXT NOT NULL, prompt TEXT NOT NULL,
 synthetic INTEGER NOT NULL CHECK(synthetic=1), source TEXT NOT NULL CHECK(source='sim_persona'),
 persona_id TEXT NOT NULL REFERENCES personas(persona_id));
CREATE TABLE IF NOT EXISTS prompts(
 prompt_id TEXT PRIMARY KEY, persona_id TEXT NOT NULL REFERENCES personas(persona_id), prompt TEXT NOT NULL,
 linter_json TEXT, synthetic INTEGER NOT NULL CHECK(synthetic=1), source TEXT NOT NULL CHECK(source='sim_persona'));
CREATE TABLE IF NOT EXISTS ratings(
 id INTEGER PRIMARY KEY, persona_id TEXT NOT NULL REFERENCES personas(persona_id), template_id TEXT NOT NULL REFERENCES templates(template_id),
 repetition INTEGER NOT NULL CHECK(repetition BETWEEN 1 AND 3), rating REAL NOT NULL CHECK(rating BETWEEN 1 AND 5), reason TEXT NOT NULL,
 synthetic INTEGER NOT NULL CHECK(synthetic=1), source TEXT NOT NULL CHECK(source='sim_persona'),
 UNIQUE(persona_id,template_id,repetition));
CREATE TABLE IF NOT EXISTS annotations(
 id INTEGER PRIMARY KEY, persona_id TEXT NOT NULL REFERENCES personas(persona_id), prompt_id TEXT NOT NULL,
 label TEXT NOT NULL CHECK(label IN('vague','contradictory','both','fine')), atoms_json TEXT NOT NULL, slots_json TEXT NOT NULL,
 synthetic INTEGER NOT NULL CHECK(synthetic=1), source TEXT NOT NULL CHECK(source='sim_persona'),
 UNIQUE(persona_id,prompt_id));
CREATE TABLE IF NOT EXISTS study_runs(
 run_id TEXT PRIMARY KEY, persona_id TEXT NOT NULL REFERENCES personas(persona_id), scorer_persona_id TEXT REFERENCES personas(persona_id),
 condition_name TEXT NOT NULL, request_json TEXT NOT NULL, response_json TEXT, checks_json TEXT, synthetic INTEGER NOT NULL CHECK(synthetic=1),
 source TEXT NOT NULL CHECK(source='sim_persona'));
CREATE TABLE IF NOT EXISTS question_scores(
 id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES study_runs(run_id), persona_id TEXT NOT NULL REFERENCES personas(persona_id),
 question_index INTEGER NOT NULL, distractor_quality INTEGER CHECK(distractor_quality BETWEEN 1 AND 5), bloom_level TEXT NOT NULL,
 correctness_judgement TEXT NOT NULL CHECK(correctness_judgement='unverified'), reason TEXT NOT NULL,
 synthetic INTEGER NOT NULL CHECK(synthetic=1), source TEXT NOT NULL CHECK(source='sim_persona'));
CREATE TABLE IF NOT EXISTS model_calls(
 call_id INTEGER PRIMARY KEY, persona_id TEXT NOT NULL REFERENCES personas(persona_id), task TEXT NOT NULL, seed INTEGER NOT NULL,
 temperature REAL NOT NULL, request_json TEXT NOT NULL, response_json TEXT, response_text TEXT, status TEXT NOT NULL,
 synthetic INTEGER NOT NULL CHECK(synthetic=1), source TEXT NOT NULL CHECK(source='sim_persona'), created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS real_faculty_scores(
 id INTEGER PRIMARY KEY, faculty_id TEXT NOT NULL, item_id TEXT NOT NULL, metric TEXT NOT NULL, value_json TEXT NOT NULL,
 source TEXT NOT NULL CHECK(source='real_faculty'), synthetic INTEGER NOT NULL CHECK(synthetic=0),
 imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE(faculty_id,item_id,metric));
CREATE INDEX IF NOT EXISTS idx_sim_annotations_item ON annotations(prompt_id);
CREATE INDEX IF NOT EXISTS idx_sim_ratings_persona ON ratings(persona_id,template_id,repetition);
"""


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc, tb):
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


_SCHEMA_INITIALIZED: set[str] = set()


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=30.0, factory=ClosingConnection)
    db.row_factory = sqlite3.Row
    path_key = str(DB_PATH.resolve())
    if path_key not in _SCHEMA_INITIALIZED:
        db.execute("PRAGMA journal_mode=WAL;")
        db.executescript(SCHEMA)
        _SCHEMA_INITIALIZED.add(path_key)
    return db


def audit(db: sqlite3.Connection) -> dict:
    simulated = (
        "personas",
        "templates",
        "prompts",
        "ratings",
        "annotations",
        "study_runs",
        "question_scores",
        "model_calls",
    )
    counts = {table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in simulated}
    failures = []
    for table in simulated:
        row = db.execute(f"SELECT COUNT(*) FROM {table} WHERE synthetic<>1 OR source<>'sim_persona'").fetchone()[0]
        if row:
            failures.append(f"{table} has {row} rows without synthetic=true/source=sim_persona")
        missing_id = db.execute(
            f"SELECT COUNT(*) FROM {table} WHERE persona_id IS NULL OR persona_id NOT GLOB 'SIM-P[0-9][0-9]'"
        ).fetchone()[0]
        if missing_id:
            failures.append(f"{table} has {missing_id} rows without a SIM-P01..SIM-P20 persona id")
    return {"counts": counts, "failures": failures, "all_simulated_rows_isolated": not failures}
