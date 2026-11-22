"""Seed local demo roles once; passwords are emitted once and never logged."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.auth_db import connect  # noqa: E402
from app.security import hash_password  # noqa: E402

CREDENTIALS = ROOT / "credentials.local.txt"
USERS = [
    ("university", "University", "University Admin"),
    ("coordinator.a", "Course Coordinator", "Coordinator A"),
    ("coordinator.b", "Course Coordinator", "Coordinator B"),
    ("faculty.a", "Faculty", "Faculty A"),
    ("faculty.b", "Faculty", "Faculty B"),
    ("faculty.c", "Faculty", "Faculty C"),
    ("faculty.d", "Faculty", "Faculty D"),
]
DEPARTMENTS = [
    ("department-computing", "Computing", "Computing"),
    ("department-engineering", "Engineering", "Engineering"),
    ("department-business", "Business", "Business"),
]
COURSES = [
    ("course-ml", "department-computing", "Machine Learning", "science", "Bachelor"),
    ("course-dbs", "department-computing", "Database Management Systems", "science", "Master"),
    ("course-materials", "department-engineering", "Civil Engineering Materials", "engineering", "Bachelor"),
    ("course-marketing", "department-business", "Marketing Principles", "business", "Master"),
]
TOPICS = [
    ("topic-ml-supervised", "course-ml", "Supervised Learning"),
    ("topic-ml-evaluation", "course-ml", "Model Evaluation"),
    ("topic-dbs-normalization", "course-dbs", "Database Normalization"),
    ("topic-dbs-transactions", "course-dbs", "Transactions"),
    ("topic-materials-fatigue", "course-materials", "Fatigue"),
    ("topic-materials-corrosion", "course-materials", "Corrosion"),
    ("topic-marketing-segmentation", "course-marketing", "Segmentation"),
    ("topic-marketing-research", "course-marketing", "Market Research"),
]
COORDINATOR_COURSES = {
    "coordinator.a": ("course-ml", "course-dbs"),
    "coordinator.b": ("course-materials", "course-marketing"),
}
FACULTY_COURSES = {
    "faculty.a": ("course-ml",),
    "faculty.b": ("course-dbs",),
    "faculty.c": ("course-ml", "course-dbs"),
    "faculty.d": ("course-marketing",),
}


def seed(reset_passwords: bool = False, credentials_path: Path | None = None) -> list[tuple[str, str, str]]:
    with connect() as db:
        existing = {r[0] for r in db.execute("SELECT username FROM auth_users")}
        if existing and not reset_passwords:
            if all(username in existing for username, _, _ in USERS):
                raise RuntimeError(
                    "Demo users already seeded. Use --reset-demo-passwords to regenerate their passwords."
                )
            raise RuntimeError("Accounts already exist; refusing to overwrite them.")
        if reset_passwords and not existing:
            raise RuntimeError("Seed demo users before resetting their passwords.")

        credentials = []
        for username, role, display in USERS:
            password = secrets.token_urlsafe(24)
            if username in existing:
                db.execute(
                    "UPDATE auth_users SET password_hash=?,enabled=1,deleted_at=NULL WHERE username=?",
                    (hash_password(password), username),
                )
            else:
                db.execute(
                    "INSERT INTO auth_users(username,role,display_name,password_hash,synthetic,source) VALUES(?,?,?,?,1,'demo_seed')",
                    (username, role, display, hash_password(password)),
                )
            credentials.append((username, role, password))

        for record_id, name, discipline in DEPARTMENTS:
            db.execute(
                "INSERT OR IGNORE INTO managed_records(record_id,entity_type,data_json,synthetic,source) VALUES(?,'departments',?,1,'demo_seed')",
                (record_id, json.dumps({"name": name, "discipline": discipline})),
            )
        for course_id, department_id, name, routing_discipline, cohort in COURSES:
            db.execute(
                "INSERT OR IGNORE INTO managed_records(record_id,entity_type,data_json,synthetic,source) VALUES(?,'courses',?,1,'demo_seed')",
                (
                    course_id,
                    json.dumps(
                        {
                            "name": name,
                            "department_id": department_id,
                            "routing_discipline": routing_discipline,
                            "cohort": cohort,
                            "allowed_cohorts": ["Bachelor", "Master"],
                        }
                    ),
                ),
            )
        for topic_id, course_id, name in TOPICS:
            db.execute(
                "INSERT OR IGNORE INTO managed_records(record_id,entity_type,course_id,data_json,synthetic,source) VALUES(?,'topics',?,?,1,'demo_seed')",
                (topic_id, course_id, json.dumps({"name": name})),
            )
        sample_template = {
            "name": "SAMPLE — Complete assessment template (synthetic substitute)",
            "task_type": "quiz",
            "prompt": "Role: #role. For #cohort students in #course, write #question_count questions about #topic with four options and 4 choices A to D. Mark the correct answer, include an explanation, and order them by difficulty.",
            "sample": True,
        }
        db.execute(
            "INSERT OR IGNORE INTO managed_records(record_id,entity_type,data_json,status,synthetic,source) VALUES('sample-template-complete','templates',?,'approved',1,'sample_library')",
            (json.dumps(sample_template),),
        )
        for username, courses in COORDINATOR_COURSES.items():
            for course_id in courses:
                db.execute(
                    "INSERT OR IGNORE INTO course_memberships(username,course_id,access_role,synthetic,source) VALUES(?,?,'coordinator',1,'demo_seed')",
                    (username, course_id),
                )
        for username, courses in FACULTY_COURSES.items():
            for course_id in courses:
                db.execute(
                    "INSERT OR IGNORE INTO course_memberships(username,course_id,access_role,synthetic,source) VALUES(?,?,'faculty',1,'demo_seed')",
                    (username, course_id),
                )
                topic_ids = [topic_id for topic_id, topic_course, _ in TOPICS if topic_course == course_id]
                db.executemany(
                    "INSERT OR IGNORE INTO faculty_topics(username,topic_id,synthetic,source) VALUES(?,?,1,'demo_seed')",
                    [(username, topic_id) for topic_id in topic_ids],
                )
    target = credentials_path or CREDENTIALS
    target.write_text(
        "DEMO ONLY — synthetic local accounts. Keep this file private; delete it after saving the passwords.\n"
        + "username\trole\tpassword\n"
        + "".join(f"{username}\t{role}\t{password}\n" for username, role, password in credentials),
        encoding="utf-8",
    )
    return credentials


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-demo-passwords", action="store_true")
    args = parser.parse_args()
    credentials = seed(args.reset_demo_passwords)
    print("Demo credentials — shown once. Plaintext passwords are not stored in the database or audit log.")
    for username, role, password in credentials:
        print(f"{username}\t{role}\t{password}")
    print(f"Saved once to {CREDENTIALS.relative_to(ROOT)} (gitignored).")


if __name__ == "__main__":
    main()
