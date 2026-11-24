"""Local account, scoped access, audit, and accuracy storage."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("PROMPT_LAB_AUTH_DB", ROOT / "data" / "local_auth.sqlite"))


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc, tb):
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS auth_users(
 username TEXT PRIMARY KEY, role TEXT NOT NULL CHECK(role IN('University','Course Coordinator','Faculty')),
 display_name TEXT NOT NULL, password_hash TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN(0,1)),
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, deleted_at TEXT, synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic IN(0,1)), source TEXT NOT NULL DEFAULT 'user');
CREATE TABLE IF NOT EXISTS auth_sessions(
 token_hash TEXT PRIMARY KEY, username TEXT NOT NULL REFERENCES auth_users(username), csrf_token TEXT NOT NULL,
 created_at REAL NOT NULL, last_seen REAL NOT NULL, expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS login_state(
 identity_hash TEXT PRIMARY KEY, failures INTEGER NOT NULL, window_start REAL NOT NULL, locked_until REAL NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS audit_log(
 id INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT NOT NULL, role TEXT NOT NULL, action TEXT NOT NULL,
 target TEXT NOT NULL, details_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS managed_records(
 record_id TEXT PRIMARY KEY, entity_type TEXT NOT NULL, course_id TEXT, owner_username TEXT,
 data_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active', enabled INTEGER NOT NULL DEFAULT 1,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, deleted_at TEXT,
 synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic IN(0,1)), source TEXT NOT NULL DEFAULT 'user');
CREATE INDEX IF NOT EXISTS ix_records_type_course ON managed_records(entity_type,course_id,deleted_at);
CREATE TABLE IF NOT EXISTS course_memberships(
 username TEXT NOT NULL REFERENCES auth_users(username), course_id TEXT NOT NULL,
 access_role TEXT NOT NULL CHECK(access_role IN('coordinator','faculty')),
 synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic IN(0,1)), source TEXT NOT NULL DEFAULT 'user',
 PRIMARY KEY(username,course_id,access_role));
CREATE TABLE IF NOT EXISTS faculty_topics(
 username TEXT NOT NULL REFERENCES auth_users(username), topic_id TEXT NOT NULL,
 synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic IN(0,1)), source TEXT NOT NULL DEFAULT 'user',
 PRIMARY KEY(username,topic_id));
CREATE TABLE IF NOT EXISTS prompt_checks(
 id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, role TEXT NOT NULL, course_id TEXT,
 prompt TEXT NOT NULL, template_id TEXT, result_json TEXT NOT NULL,
 source TEXT NOT NULL DEFAULT 'user_prompt', sample_id TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS ix_prompt_owner ON prompt_checks(username,created_at);
CREATE TABLE IF NOT EXISTS faculty_ratings(
 id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, course_id TEXT NOT NULL,
 template_id TEXT NOT NULL, rating REAL NOT NULL CHECK(rating BETWEEN 1 AND 5),
 comment TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS accuracy_annotations(
 id INTEGER PRIMARY KEY AUTOINCREMENT, prompt_id TEXT NOT NULL, dataset TEXT NOT NULL CHECK(dataset IN('development','heldout')),
 annotator_id TEXT NOT NULL, atoms_json TEXT NOT NULL, slots_json TEXT NOT NULL,
 source TEXT NOT NULL DEFAULT 'real_annotator', synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic=0),
 UNIQUE(prompt_id,dataset,annotator_id));
CREATE TABLE IF NOT EXISTS accuracy_prompts(
 prompt_id TEXT PRIMARY KEY, prompt TEXT NOT NULL, dataset TEXT NOT NULL CHECK(dataset IN('development','heldout')),
 source TEXT NOT NULL, synthetic INTEGER NOT NULL DEFAULT 0 CHECK(synthetic=0));
CREATE TABLE IF NOT EXISTS linter_scoring_runs(
 id INTEGER PRIMARY KEY AUTOINCREMENT, extractor_version TEXT NOT NULL, dataset TEXT NOT NULL,
 prompt_count INTEGER NOT NULL, report_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS comparisons(
 id INTEGER PRIMARY KEY AUTOINCREMENT, rater_id TEXT NOT NULL, prompt_a_id TEXT NOT NULL,
 prompt_b_id TEXT NOT NULL, winner TEXT NOT NULL CHECK(winner IN('a','b','tie')),
 shown_order TEXT NOT NULL DEFAULT 'ab', features_a_json TEXT, features_b_json TEXT,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, duration_ms INTEGER);
CREATE INDEX IF NOT EXISTS ix_comparisons_rater ON comparisons(rater_id);
CREATE TABLE IF NOT EXISTS learner_fits(
 id INTEGER PRIMARY KEY AUTOINCREMENT, n_comparisons INTEGER NOT NULL, n_raters INTEGER NOT NULL,
 weights_json TEXT NOT NULL, intervals_json TEXT, versions_json TEXT NOT NULL,
 seed INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
"""


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
