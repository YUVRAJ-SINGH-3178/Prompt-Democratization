"""Authenticated, role-scoped dashboards and operations."""

from __future__ import annotations

import csv
import io
import json
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import auth_db
from app.auth_db import connect
from app.linter.core import ATOMS, CONFLICTS, SLOTS, evaluate, extract_atoms
from app.linter.explain import evidence
from app.sample_prompts import SAMPLES
from app.security import audit, hash_password

router = APIRouter()
ADMIN_KINDS = {
    "departments",
    "courses",
    "topics",
    "templates",
    "mappings",
    "policy_atoms",
    "conflict_rules",
    "system_settings",
    "experiment_runs",
    "annotations",
}
SLOT_NAMES = set(SLOTS)


class LoginBody(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=256)


class RecordBody(BaseModel):
    record_id: str | None = None
    course_id: str | None = None
    owner_username: str | None = None
    data: dict[str, Any]
    status: str = "active"
    enabled: bool = True


class UserBody(BaseModel):
    username: str = Field(min_length=2, max_length=80)
    role: str
    display_name: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=16, max_length=256)


class GrantBody(BaseModel):
    username: str
    course_id: str
    access_role: str = "faculty"


class TopicGrantBody(BaseModel):
    username: str
    topic_ids: list[str]


class CoordinatorFacultyBody(UserBody):
    course_id: str


class TemplateDecision(BaseModel):
    decision: str


class CourseMappingBody(BaseModel):
    cohort: str


class PromptCheckBody(BaseModel):
    prompt: str | None = Field(default=None, max_length=20000)
    template_id: str | None = None
    course_id: str | None = None
    sample_id: str | None = None


class PromptImproveBody(BaseModel):
    prompt: str | None = Field(default=None, max_length=20000)
    sample_id: str | None = None
    template_id: str | None = None
    course_id: str | None = None
    topic_id: str | None = None
    cohort: str | None = None
    count: int | None = None
    user_choices: dict[str, str] | None = None
    use_llm: bool = False
    model: str | None = None


class BuilderBody(BaseModel):
    course_id: str
    topic_ids: list[str] = Field(min_length=1)
    cohort: str
    question_count: int = Field(ge=1, le=50)
    difficulty_mix: dict[str, int]
    question_type: str
    explanation_level: str
    extra_instruction: str = Field(default="", max_length=1000)


def who(request: Request) -> dict:
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(401, "Login required")
    return user


def _json_record(row) -> dict:
    result = dict(row)
    result["data"] = json.loads(result.pop("data_json"))
    result["synthetic"] = bool(result.get("synthetic", 0))
    return result


def _course_ids(username: str, access_role: str) -> list[str]:
    with connect() as db:
        return [
            r[0]
            for r in db.execute(
                "SELECT course_id FROM course_memberships WHERE username=? AND access_role=?", (username, access_role)
            )
        ]


def _deny_scope(request: Request, target: str, reason: str = "course outside assigned scope"):
    from app.security import deny

    user = who(request)
    deny(user, target, request.method, reason)
    raise HTTPException(403, "Course is outside your assigned scope")


def _require_course(request: Request, course_id: str, access_role: str = "coordinator") -> dict:
    user = who(request)
    if user["role"] != "University":
        if course_id not in _course_ids(user["username"], access_role):
            _deny_scope(request, f"course:{course_id}")
        with connect() as db:
            if not db.execute(
                "SELECT 1 FROM managed_records WHERE record_id=? AND entity_type='courses' AND deleted_at IS NULL",
                (course_id,),
            ).fetchone():
                _deny_scope(request, f"course:{course_id}", "course is deleted or unavailable")
    return user


def _get_record(record_id: str, kind: str | None = None, include_deleted: bool = False):
    query = "SELECT * FROM managed_records WHERE record_id=?"
    args: list[Any] = [record_id]
    if kind:
        query += " AND entity_type=?"
        args.append(kind)
    if not include_deleted:
        query += " AND deleted_at IS NULL"
    with connect() as db:
        return db.execute(query, args).fetchone()


def _audit_request(request: Request, action: str, target: str, details: dict | None = None):
    user = who(request)
    audit(user["username"], user["role"], action, target, details)


def _active_linter_policy(course_id: str | None = None, db=None):
    if db is not None:
        if course_id:
            policy_rows = db.execute(
                "SELECT data_json FROM managed_records WHERE entity_type='policy_atoms' AND (course_id=? OR course_id IS NULL) AND deleted_at IS NULL",
                (course_id,),
            ).fetchall()
            conflict_rows = db.execute(
                "SELECT data_json FROM managed_records WHERE entity_type='conflict_rules' AND (course_id=? OR course_id IS NULL) AND deleted_at IS NULL",
                (course_id,),
            ).fetchall()
        else:
            policy_rows = db.execute(
                "SELECT data_json FROM managed_records WHERE entity_type='policy_atoms' AND course_id IS NULL AND deleted_at IS NULL"
            ).fetchall()
            conflict_rows = db.execute(
                "SELECT data_json FROM managed_records WHERE entity_type='conflict_rules' AND course_id IS NULL AND deleted_at IS NULL"
            ).fetchall()
    else:
        with connect() as conn:
            return _active_linter_policy(course_id, conn)
    policy = tuple(
        sorted(
            {
                payload.get("atom")
                for row in policy_rows
                if (payload := json.loads(row[0])).get("enabled", True) and payload.get("atom") in ATOMS
            }
        )
    )
    if not policy_rows:
        policy = ("clear_unambiguous",)
    conflicts = []
    for row in conflict_rows:
        pair = json.loads(row[0]).get("atoms", [])
        if (
            json.loads(row[0]).get("enabled", True)
            and len(pair) == 2
            and all(atom in ATOMS for atom in pair)
            and pair[0] != pair[1]
        ):
            conflicts.append(tuple(pair))
    if not conflict_rows:
        conflicts = list(CONFLICTS)
    return tuple(conflicts), policy


@router.post("/api/auth/login")
def login(body: LoginBody, request: Request):
    from app.security import (
        SESSION_COOKIE,
        ABSOLUTE_SECONDS,
        clear_login_failures,
        create_session,
        failed_login,
        login_locked,
        verify_password,
    )
    from fastapi.responses import JSONResponse

    username = body.username.strip().casefold()
    ip = request.client.host if request.client else "unknown"
    if login_locked(username, ip):
        audit("anonymous", "anonymous", "login_locked", "login", {"ip_hash_only": True})
        raise HTTPException(429, "Login temporarily locked. Try again later.")
    with connect() as db:
        row = db.execute(
            "SELECT username,role,display_name,password_hash,enabled,deleted_at FROM auth_users WHERE username=?",
            (username,),
        ).fetchone()
    if not row or not row["enabled"] or row["deleted_at"] or not verify_password(row["password_hash"], body.password):
        failed_login(username, ip)
        audit("anonymous", "anonymous", "login_failed", "login", {"ip_hash_only": True})
        raise HTTPException(401, "Invalid username or password")
    clear_login_failures(username, ip)
    token, csrf = create_session(username)
    dashboard = {"University": "/admin", "Course Coordinator": "/coordinator", "Faculty": "/faculty"}[row["role"]]
    audit(username, row["role"], "login_success", "session")
    response = JSONResponse(
        {
            "username": username,
            "display_name": row["display_name"],
            "role": row["role"],
            "dashboard": dashboard,
            "csrf_token": csrf,
        }
    )
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        path="/",
        max_age=ABSOLUTE_SECONDS,
    )
    return response


@router.get("/api/auth/me")
def auth_me(request: Request):
    user = getattr(request.state, "user", None)
    if not user:
        return {"authenticated": False}
    return {
        "authenticated": True,
        **user,
        "csrf_token": getattr(request.state, "csrf_token", None),
        "dashboard": {"University": "/admin", "Course Coordinator": "/coordinator", "Faculty": "/faculty"}[
            user["role"]
        ],
    }


@router.post("/api/auth/logout")
def logout(request: Request):
    from app.security import SESSION_COOKIE, revoke_session
    from fastapi.responses import JSONResponse

    user = who(request)
    token = request.cookies.get(SESSION_COOKIE)
    revoke_session(token)
    audit(user["username"], user["role"], "logout", "session")
    response = JSONResponse({"ok": True})
    response.delete_cookie(
        SESSION_COOKIE, path="/", httponly=True, samesite="strict", secure=request.url.scheme == "https"
    )
    return response


@router.get("/api/admin/overview")
def admin_overview(request: Request):
    who(request)
    with connect() as db:
        users = db.execute("SELECT role,COUNT(*) n FROM auth_users WHERE deleted_at IS NULL GROUP BY role").fetchall()
        records = db.execute(
            "SELECT entity_type,COUNT(*) n FROM managed_records WHERE deleted_at IS NULL GROUP BY entity_type"
        ).fetchall()
        return {
            "users_by_role": {r[0]: r[1] for r in users},
            "records_by_type": {r[0]: r[1] for r in records},
            "pending_templates": db.execute(
                "SELECT COUNT(*) FROM managed_records WHERE entity_type='templates' AND status='pending' AND deleted_at IS NULL"
            ).fetchone()[0],
            "audit_rows": db.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0],
            "synthetic_demo_accounts": db.execute(
                "SELECT COUNT(*) FROM auth_users WHERE synthetic=1 AND deleted_at IS NULL"
            ).fetchone()[0],
        }


@router.get("/api/admin/users")
def admin_users(request: Request):
    who(request)
    with connect() as db:
        return {
            "items": [
                dict(r)
                for r in db.execute(
                    "SELECT username,role,display_name,enabled,created_at,deleted_at,synthetic,source FROM auth_users ORDER BY role,username"
                )
            ],
            "n": db.execute("SELECT COUNT(*) FROM auth_users").fetchone()[0],
        }


@router.post("/api/admin/users", status_code=201)
def admin_create_user(body: UserBody, request: Request):
    user = who(request)
    if body.role not in ("University", "Course Coordinator", "Faculty"):
        raise HTTPException(422, "Invalid role")
    username = body.username.strip().casefold()
    try:
        with connect() as db:
            db.execute(
                "INSERT INTO auth_users(username,role,display_name,password_hash) VALUES(?,?,?,?)",
                (username, body.role, body.display_name, hash_password(body.password)),
            )
    except Exception:
        raise HTTPException(409, "Username already exists")
    audit(user["username"], user["role"], "create_user", f"user:{username}", {"role": body.role})
    return {"username": username, "role": body.role, "password_stored_as": "argon2id hash"}


@router.post("/api/admin/users/{username}/reset-password")
def admin_reset_password(username: str, body: dict, request: Request):
    actor = who(request)
    password = str(body.get("password", ""))
    if len(password) < 16:
        raise HTTPException(422, "Password must be at least 16 characters")
    with connect() as db:
        cur = db.execute(
            "UPDATE auth_users SET password_hash=? WHERE username=? AND deleted_at IS NULL",
            (hash_password(password), username.casefold()),
        )
        if not cur.rowcount:
            raise HTTPException(404, "User not found")
    audit(actor["username"], actor["role"], "reset_password", f"user:{username}")
    return {"updated": True, "password_logged": False}


@router.patch("/api/admin/users/{username}/enabled")
def admin_enable_user(username: str, body: dict, request: Request):
    actor = who(request)
    enabled = bool(body.get("enabled"))
    with connect() as db:
        cur = db.execute(
            "UPDATE auth_users SET enabled=? WHERE username=? AND deleted_at IS NULL",
            (int(enabled), username.casefold()),
        )
        if not cur.rowcount:
            raise HTTPException(404, "User not found")
    audit(actor["username"], actor["role"], "enable_user" if enabled else "disable_user", f"user:{username}")
    return {"username": username, "enabled": enabled}


@router.delete("/api/admin/users/{username}")
def admin_delete_user(username: str, request: Request):
    actor = who(request)
    with connect() as db:
        cur = db.execute(
            "UPDATE auth_users SET deleted_at=CURRENT_TIMESTAMP,enabled=0 WHERE username=? AND deleted_at IS NULL",
            (username.casefold(),),
        )
        if not cur.rowcount:
            raise HTTPException(404, "User not found")
    audit(actor["username"], actor["role"], "soft_delete", f"user:{username}")
    return {"deleted": True, "undo": f"/api/admin/users/{username}/restore"}


@router.post("/api/admin/users/{username}/restore")
def admin_restore_user(username: str, request: Request):
    actor = who(request)
    with connect() as db:
        cur = db.execute(
            "UPDATE auth_users SET deleted_at=NULL,enabled=1 WHERE username=? AND deleted_at IS NOT NULL",
            (username.casefold(),),
        )
        if not cur.rowcount:
            raise HTTPException(404, "Deleted user not found")
    audit(actor["username"], actor["role"], "restore", f"user:{username}")
    return {"restored": True}


@router.get("/api/admin/records/{kind}")
def admin_records(kind: str, request: Request, include_deleted: bool = False):
    who(request)
    if kind not in ADMIN_KINDS:
        raise HTTPException(404, "Unknown record type")
    query = "SELECT * FROM managed_records WHERE entity_type=?"
    if not include_deleted:
        query += " AND deleted_at IS NULL"
    query += " ORDER BY created_at,record_id"
    with connect() as db:
        rows = db.execute(query, (kind,)).fetchall()
    return {"items": [_json_record(r) for r in rows], "n": len(rows)}


@router.post("/api/admin/records/{kind}", status_code=201)
def admin_add_record(kind: str, body: RecordBody, request: Request):
    actor = who(request)
    if kind not in ADMIN_KINDS:
        raise HTTPException(404, "Unknown record type")
    record_id = body.record_id or f"{kind}-{uuid.uuid4().hex[:12]}"
    with connect() as db:
        db.execute(
            "INSERT INTO managed_records(record_id,entity_type,course_id,owner_username,data_json,status,enabled) VALUES(?,?,?,?,?,?,?)",
            (
                record_id,
                kind,
                body.course_id,
                body.owner_username,
                json.dumps(body.data),
                body.status,
                int(body.enabled),
            ),
        )
    audit(actor["username"], actor["role"], "create", f"{kind}:{record_id}", {"course_id": body.course_id})
    return {"record_id": record_id, "entity_type": kind, "created": True}


@router.patch("/api/admin/records/{kind}/{record_id}")
def admin_update_record(kind: str, record_id: str, body: RecordBody, request: Request):
    actor = who(request)
    if kind not in ADMIN_KINDS:
        raise HTTPException(404, "Unknown record type")
    with connect() as db:
        cur = db.execute(
            "UPDATE managed_records SET course_id=?,owner_username=?,data_json=?,status=?,enabled=?,updated_at=CURRENT_TIMESTAMP WHERE record_id=? AND entity_type=? AND deleted_at IS NULL",
            (
                body.course_id,
                body.owner_username,
                json.dumps(body.data),
                body.status,
                int(body.enabled),
                record_id,
                kind,
            ),
        )
        if not cur.rowcount:
            raise HTTPException(404, "Record not found")
    audit(actor["username"], actor["role"], "edit", f"{kind}:{record_id}")
    return {"record_id": record_id, "updated": True}


@router.delete("/api/admin/records/{kind}/{record_id}")
def admin_delete_record(kind: str, record_id: str, request: Request):
    actor = who(request)
    with connect() as db:
        row = db.execute(
            "SELECT * FROM managed_records WHERE record_id=? AND entity_type=? AND deleted_at IS NULL",
            (record_id, kind),
        ).fetchone()
        if not row:
            raise HTTPException(404, "Record not found")
        snapshot = dict(row)
        db.execute("UPDATE managed_records SET deleted_at=CURRENT_TIMESTAMP WHERE record_id=?", (record_id,))
    audit(actor["username"], actor["role"], "soft_delete", f"{kind}:{record_id}", {"snapshot": snapshot})
    return {"deleted": True, "undo": f"/api/admin/records/{kind}/{record_id}/restore"}


@router.post("/api/admin/records/{kind}/{record_id}/restore")
def admin_restore_record(kind: str, record_id: str, request: Request):
    actor = who(request)
    with connect() as db:
        cur = db.execute(
            "UPDATE managed_records SET deleted_at=NULL,updated_at=CURRENT_TIMESTAMP WHERE record_id=? AND entity_type=? AND deleted_at IS NOT NULL",
            (record_id, kind),
        )
        if not cur.rowcount:
            raise HTTPException(404, "Deleted record not found")
    audit(actor["username"], actor["role"], "restore", f"{kind}:{record_id}")
    return {"restored": True}


@router.get("/api/admin/trash")
def admin_trash(request: Request):
    who(request)
    with connect() as db:
        records = [
            _json_record(r)
            for r in db.execute("SELECT * FROM managed_records WHERE deleted_at IS NOT NULL ORDER BY deleted_at DESC")
        ]
        users = [
            dict(r)
            for r in db.execute(
                "SELECT username,role,display_name,deleted_at FROM auth_users WHERE deleted_at IS NOT NULL"
            )
        ]
    return {"records": records, "users": users}


@router.get("/api/admin/audit")
def admin_audit(request: Request, limit: int = 500):
    who(request)
    with connect() as db:
        rows = db.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (max(1, min(limit, 5000)),)).fetchall()
    return {"items": [{**dict(r), "details": json.loads(r["details_json"])} for r in rows], "n": len(rows)}


@router.get("/api/admin/export")
def admin_export(request: Request):
    who(request)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for db_path in (auth_db.DB_PATH, Path(__file__).resolve().parent.parent / "data" / "university_repo.db"):
            if db_path.exists():
                archive.write(db_path, arcname=db_path.name)
    output.seek(0)
    audit(who(request)["username"], "University", "database_export", "database_bundle")
    return StreamingResponse(
        output,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=prompt-lab-export.zip"},
    )


@router.post("/api/admin/grants")
def admin_grant(body: GrantBody, request: Request):
    actor = who(request)
    if body.access_role not in ("coordinator", "faculty"):
        raise HTTPException(422, "Invalid access role")
    with connect() as db:
        row = db.execute(
            "SELECT role FROM auth_users WHERE username=? AND deleted_at IS NULL", (body.username.casefold(),)
        ).fetchone()
        if (
            not row
            or (body.access_role == "coordinator" and row[0] != "Course Coordinator")
            or (body.access_role == "faculty" and row[0] != "Faculty")
        ):
            raise HTTPException(422, "Account role does not match grant")
        db.execute(
            "INSERT OR IGNORE INTO course_memberships(username,course_id,access_role) VALUES(?,?,?)",
            (body.username.casefold(), body.course_id, body.access_role),
        )
    audit(actor["username"], actor["role"], "grant", f"{body.access_role}:{body.username}:{body.course_id}")
    return {"granted": True}


@router.get("/api/admin/grants")
def admin_grants(request: Request):
    who(request)
    with connect() as db:
        items = [
            dict(r)
            for r in db.execute(
                "SELECT username,course_id,access_role,synthetic,source FROM course_memberships ORDER BY access_role,course_id,username"
            )
        ]
    return {"items": items, "n": len(items)}


@router.delete("/api/admin/grants/{username}/{course_id}/{access_role}")
def admin_revoke(username: str, course_id: str, access_role: str, request: Request):
    actor = who(request)
    with connect() as db:
        cur = db.execute(
            "DELETE FROM course_memberships WHERE username=? AND course_id=? AND access_role=?",
            (username.casefold(), course_id, access_role),
        )
        if access_role == "faculty":
            db.execute(
                "DELETE FROM faculty_topics WHERE username=? AND topic_id IN (SELECT record_id FROM managed_records WHERE course_id=?)",
                (username.casefold(), course_id),
            )
    audit(actor["username"], actor["role"], "revoke", f"{access_role}:{username}:{course_id}")
    return {"revoked": bool(cur.rowcount)}


@router.get("/api/admin/accuracy")
def admin_accuracy(request: Request):
    who(request)
    return _accuracy_report()


@router.post("/api/admin/accuracy/annotations")
def admin_import_accuracy(body: dict, request: Request):
    actor = who(request)
    dataset, prompt_id, annotator = body.get("dataset"), body.get("prompt_id"), body.get("annotator_id")
    atoms, slots, prompt = body.get("atoms"), body.get("slots"), body.get("prompt")
    if dataset not in ("development", "heldout") or not prompt_id or not annotator:
        raise HTTPException(422, "dataset, prompt_id, and annotator_id are required")
    if not isinstance(atoms, list) or any(a not in ATOMS for a in atoms):
        raise HTTPException(422, "Unknown requirement atom")
    if not isinstance(slots, dict) or set(slots) != SLOT_NAMES or any(v not in (0, 0.5, 1) for v in slots.values()):
        raise HTTPException(422, "All nine slot scores are required as 0, 0.5, or 1")
    with connect() as db:
        if prompt:
            db.execute(
                "INSERT OR IGNORE INTO accuracy_prompts(prompt_id,prompt,dataset,source,synthetic) VALUES(?,?,?,'real_annotator',0)",
                (prompt_id, prompt, dataset),
            )
        if not db.execute(
            "SELECT 1 FROM accuracy_prompts WHERE prompt_id=? AND dataset=?", (prompt_id, dataset)
        ).fetchone():
            raise HTTPException(404, "Prompt text is not registered for this dataset")
        db.execute(
            "INSERT OR REPLACE INTO accuracy_annotations(prompt_id,dataset,annotator_id,atoms_json,slots_json,source,synthetic) VALUES(?,?,?,?,?,'real_annotator',0)",
            (prompt_id, dataset, annotator, json.dumps(atoms), json.dumps(slots)),
        )
    audit(
        actor["username"],
        actor["role"],
        "import_accuracy_annotation",
        f"{dataset}:{prompt_id}",
        {"annotator_id": annotator},
    )
    return _accuracy_report()


@router.post("/api/admin/accuracy/import-csv")
async def admin_import_accuracy_csv(request: Request, file: UploadFile = File(...)):
    actor = who(request)
    if actor["role"] != "University":
        raise HTTPException(403, "University role required")
    content = (await file.read()).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))
    errors = []
    imported = 0
    with connect() as db:
        for line, row in enumerate(reader, start=2):
            try:
                pid = row.get("prompt_id")
                dataset = row.get("dataset", "heldout")
                aid = row.get("annotator_id")
                if not pid or not aid:
                    raise ValueError("prompt_id and annotator_id are required")
                atoms_raw = row.get("atoms", "[]")
                atoms = (
                    json.loads(atoms_raw)
                    if isinstance(atoms_raw, str) and atoms_raw.startswith("[")
                    else [a.strip() for a in atoms_raw.split(",") if a.strip()]
                )
                if any(a not in ATOMS for a in atoms):
                    raise ValueError(f"Unknown atom in {atoms}")
                if "slots" in row and row["slots"]:
                    slots = json.loads(row["slots"])
                else:
                    slots = {s: float(row.get(s, 0.0)) for s in SLOTS}
                prompt_text = row.get("prompt")
                if prompt_text:
                    db.execute(
                        "INSERT OR IGNORE INTO accuracy_prompts(prompt_id,prompt,dataset,source,synthetic) VALUES(?,?,?,'real_annotator',0)",
                        (pid, prompt_text, dataset),
                    )
                db.execute(
                    "INSERT OR REPLACE INTO accuracy_annotations(prompt_id,dataset,annotator_id,atoms_json,slots_json,source,synthetic) VALUES(?,?,?,?,?,'real_annotator',0)",
                    (pid, dataset, aid, json.dumps(atoms), json.dumps(slots)),
                )
                imported += 1
            except Exception as e:
                errors.append({"row": line, "error": str(e)})
        db.commit()
    audit(actor["username"], actor["role"], "import_accuracy_csv", f"imported:{imported}:errors:{len(errors)}")
    return {"imported": imported, "errors": errors, "report": _accuracy_report()}


def _accuracy_report() -> dict:
    from app.linter.datasets import evaluate_heldout_dataset

    with connect() as db:
        return evaluate_heldout_dataset(db)


@router.get("/api/coordinator/overview")
def coordinator_overview(request: Request):
    user = who(request)
    course_ids = _course_ids(user["username"], "coordinator")
    if not course_ids:
        return {"courses": [], "course_ids": [], "faculty": 0, "prompt_checks": 0, "ratings": 0}
    placeholders = ",".join("?" for _ in course_ids)
    with connect() as db:
        courses = [
            _json_record(r)
            for r in db.execute(
                f"SELECT * FROM managed_records WHERE entity_type='courses' AND record_id IN ({placeholders}) AND deleted_at IS NULL",
                course_ids,
            )
        ]
        faculty = db.execute(
            f"SELECT COUNT(DISTINCT cm.username) FROM course_memberships cm JOIN auth_users u ON u.username=cm.username WHERE cm.access_role='faculty' AND cm.course_id IN ({placeholders}) AND u.deleted_at IS NULL",
            course_ids,
        ).fetchone()[0]
        checks = db.execute(
            f"SELECT COUNT(*) FROM prompt_checks WHERE course_id IN ({placeholders})", course_ids
        ).fetchone()[0]
        ratings = db.execute(
            f"SELECT COUNT(*) FROM faculty_ratings WHERE course_id IN ({placeholders})", course_ids
        ).fetchone()[0]
    return {
        "courses": courses,
        "course_ids": course_ids,
        "faculty": faculty,
        "prompt_checks": checks,
        "ratings": ratings,
    }


@router.get("/api/coordinator/records/{kind}")
def coordinator_records(kind: str, request: Request, course_id: str | None = None):
    user = who(request)
    allowed = _course_ids(user["username"], "coordinator")
    if course_id and course_id not in allowed:
        _deny_scope(request, f"{kind}:{course_id}")
    if kind not in {"courses", "topics", "templates", "mappings", "ratings", "annotations", "faculty"}:
        raise HTTPException(404, "Record type not available")
    ids = [course_id] if course_id else allowed
    if not ids:
        return {"items": [], "n": 0}
    placeholders = ",".join("?" for _ in ids)
    with connect() as db:
        if kind == "faculty":
            rows = db.execute(
                f"SELECT DISTINCT u.username,u.role,u.display_name,u.enabled,u.deleted_at,m.course_id FROM auth_users u JOIN course_memberships m ON m.username=u.username WHERE m.access_role='faculty' AND m.course_id IN ({placeholders})",
                ids,
            ).fetchall()
            items = [dict(row) for row in rows]
        elif kind == "ratings":
            items = [
                dict(row)
                for row in db.execute(
                    f"SELECT * FROM faculty_ratings WHERE course_id IN ({placeholders}) ORDER BY created_at DESC", ids
                )
            ]
        elif kind == "annotations":
            items = [
                dict(row)
                for row in db.execute(
                    f"SELECT pc.id,pc.username,pc.course_id,pc.result_json,pc.created_at FROM prompt_checks pc WHERE pc.course_id IN ({placeholders})",
                    ids,
                )
            ]
        else:
            rows = db.execute(
                f"SELECT * FROM managed_records WHERE entity_type=? AND course_id IN ({placeholders}) AND deleted_at IS NULL ORDER BY record_id",
                (kind, *ids),
            ).fetchall()
            items = [_json_record(row) for row in rows]
    return {"items": items, "n": len(items)}


@router.post("/api/coordinator/faculty", status_code=201)
def coordinator_create_faculty(body: CoordinatorFacultyBody, request: Request):
    actor = _require_course(request, body.course_id)
    if body.role != "Faculty":
        raise HTTPException(403, "Coordinators may create only faculty accounts")
    username = body.username.strip().casefold()
    with connect() as db:
        try:
            db.execute(
                "INSERT INTO auth_users(username,role,display_name,password_hash) VALUES(?, 'Faculty',?,?)",
                (username, body.display_name, hash_password(body.password)),
            )
            db.execute(
                "INSERT INTO course_memberships(username,course_id,access_role) VALUES(?,?,'faculty')",
                (username, body.course_id),
            )
        except Exception:
            raise HTTPException(409, "Username already exists or course grant failed")
    audit(actor["username"], actor["role"], "create_faculty", f"faculty:{username}:{body.course_id}")
    audit(actor["username"], actor["role"], "grant", f"faculty:{username}:{body.course_id}")
    return {"username": username, "role": "Faculty", "course_id": body.course_id}


@router.post("/api/coordinator/faculty/{username}/grant")
def coordinator_grant_faculty(username: str, body: dict, request: Request):
    course_id = str(body.get("course_id", ""))
    actor = _require_course(request, course_id)
    with connect() as db:
        target = db.execute(
            "SELECT role FROM auth_users WHERE username=? AND deleted_at IS NULL", (username.casefold(),)
        ).fetchone()
        if not target or target[0] != "Faculty":
            raise HTTPException(404, "Faculty account not found")
        db.execute(
            "INSERT OR IGNORE INTO course_memberships(username,course_id,access_role) VALUES(?,?,'faculty')",
            (username.casefold(), course_id),
        )
    audit(actor["username"], actor["role"], "grant", f"faculty:{username}:{course_id}")
    return {"granted": True}


@router.delete("/api/coordinator/faculty/{username}/grant/{course_id}")
def coordinator_revoke_faculty(username: str, course_id: str, request: Request):
    actor = _require_course(request, course_id)
    with connect() as db:
        cur = db.execute(
            "DELETE FROM course_memberships WHERE username=? AND course_id=? AND access_role='faculty'",
            (username.casefold(), course_id),
        )
        db.execute(
            "DELETE FROM faculty_topics WHERE username=? AND topic_id IN (SELECT record_id FROM managed_records WHERE course_id=?)",
            (username.casefold(), course_id),
        )
    audit(actor["username"], actor["role"], "revoke", f"faculty:{username}:{course_id}")
    return {"revoked": bool(cur.rowcount)}


@router.patch("/api/coordinator/faculty/{username}/enabled")
def coordinator_set_faculty_enabled(username: str, body: dict, request: Request):
    actor = who(request)
    course_id = str(body.get("course_id", ""))
    _require_course(request, course_id)
    enabled = bool(body.get("enabled"))
    with connect() as db:
        memberships = {
            r[0]
            for r in db.execute(
                "SELECT course_id FROM course_memberships WHERE username=? AND access_role='faculty'",
                (username.casefold(),),
            )
        }
        if course_id not in memberships:
            raise HTTPException(404, "Faculty course grant not found")
        scope = set(_course_ids(actor["username"], "coordinator"))
        if not memberships <= scope:
            _deny_scope(request, f"faculty:{username}", "faculty account also has courses outside coordinator scope")
        cur = db.execute(
            "UPDATE auth_users SET enabled=? WHERE username=? AND role='Faculty' AND deleted_at IS NULL",
            (int(enabled), username.casefold()),
        )
        if not cur.rowcount:
            raise HTTPException(404, "Faculty account not found")
    audit(actor["username"], actor["role"], "enable_faculty" if enabled else "disable_faculty", f"faculty:{username}")
    return {"username": username, "enabled": enabled}


@router.delete("/api/coordinator/faculty/{username}")
def coordinator_soft_delete_faculty(username: str, request: Request, course_id: str):
    actor = _require_course(request, course_id)
    with connect() as db:
        memberships = {
            r[0]
            for r in db.execute(
                "SELECT course_id FROM course_memberships WHERE username=? AND access_role='faculty'",
                (username.casefold(),),
            )
        }
        scope = set(_course_ids(actor["username"], "coordinator"))
        if not memberships or course_id not in memberships:
            raise HTTPException(404, "Faculty account not assigned to this course")
        if not memberships <= scope:
            _deny_scope(request, f"faculty:{username}", "faculty account also has courses outside coordinator scope")
        cur = db.execute(
            "UPDATE auth_users SET deleted_at=CURRENT_TIMESTAMP,enabled=0 WHERE username=? AND role='Faculty' AND deleted_at IS NULL",
            (username.casefold(),),
        )
        if not cur.rowcount:
            raise HTTPException(404, "Faculty account not found")
    audit(actor["username"], actor["role"], "soft_delete_faculty", f"faculty:{username}")
    return {"deleted": True}


@router.put("/api/coordinator/topics/{topic_id}")
def coordinator_edit_topic(topic_id: str, body: dict, request: Request):
    actor = who(request)
    row = _get_record(topic_id, "topics")
    if not row:
        raise HTTPException(404, "Topic not found")
    _require_course(request, row["course_id"])
    name = str(body.get("name", "")).strip()
    if not name:
        raise HTTPException(422, "Topic name is required")
    data = json.loads(row["data_json"])
    data["name"] = name
    with connect() as db:
        db.execute(
            "UPDATE managed_records SET data_json=?,updated_at=CURRENT_TIMESTAMP WHERE record_id=?",
            (json.dumps(data), topic_id),
        )
    audit(actor["username"], actor["role"], "edit_topic", f"topic:{topic_id}")
    return {"topic_id": topic_id, "name": name}


@router.put("/api/coordinator/faculty/{username}/topics")
def coordinator_assign_topics(username: str, body: TopicGrantBody, request: Request):
    actor = who(request)
    with connect() as db:
        faculty_courses = {
            r[0]
            for r in db.execute(
                "SELECT course_id FROM course_memberships WHERE username=? AND access_role='faculty'",
                (username.casefold(),),
            )
        }
        for topic_id in body.topic_ids:
            topic = db.execute(
                "SELECT course_id FROM managed_records WHERE record_id=? AND entity_type='topics' AND deleted_at IS NULL",
                (topic_id,),
            ).fetchone()
            if not topic or topic[0] not in faculty_courses:
                _deny_scope(request, f"topic:{topic_id}", "topic is not in faculty course scope")
            _require_course(request, topic[0])
        db.execute("DELETE FROM faculty_topics WHERE username=?", (username.casefold(),))
        db.executemany(
            "INSERT INTO faculty_topics(username,topic_id) VALUES(?,?)",
            [(username.casefold(), tid) for tid in body.topic_ids],
        )
    for tid in body.topic_ids:
        audit(actor["username"], actor["role"], "grant_topic", f"faculty:{username}:topic:{tid}")
    return {"assigned_topic_ids": body.topic_ids}


@router.get("/api/coordinator/faculty/{username}/topics")
def coordinator_faculty_topics(username: str, request: Request, course_id: str):
    _require_course(request, course_id)
    with connect() as db:
        if not db.execute(
            "SELECT 1 FROM course_memberships WHERE username=? AND course_id=? AND access_role='faculty'",
            (username.casefold(), course_id),
        ).fetchone():
            raise HTTPException(404, "Faculty account is not assigned to this course")
        available = db.execute(
            "SELECT record_id,data_json FROM managed_records WHERE entity_type='topics' AND course_id=? AND deleted_at IS NULL ORDER BY record_id",
            (course_id,),
        ).fetchall()
        assigned = {
            r[0] for r in db.execute("SELECT topic_id FROM faculty_topics WHERE username=?", (username.casefold(),))
        }
    return {
        "items": [
            {"topic_id": r[0], "name": json.loads(r[1]).get("name", r[0]), "assigned": r[0] in assigned}
            for r in available
        ]
    }


@router.put("/api/coordinator/courses/{course_id}/mapping")
def coordinator_course_mapping(course_id: str, body: CourseMappingBody, request: Request):
    actor = _require_course(request, course_id)
    if body.cohort not in ("Bachelor", "Master"):
        raise HTTPException(422, "Cohort must be Bachelor or Master")
    with connect() as db:
        row = db.execute(
            "SELECT data_json FROM managed_records WHERE record_id=? AND entity_type='courses' AND deleted_at IS NULL",
            (course_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, "Course not found")
        data = json.loads(row[0])
        data["cohort"] = body.cohort
        db.execute(
            "UPDATE managed_records SET data_json=?,updated_at=CURRENT_TIMESTAMP WHERE record_id=?",
            (json.dumps(data), course_id),
        )
    audit(actor["username"], actor["role"], "edit_cohort_mapping", f"course:{course_id}", {"cohort": body.cohort})
    return {"course_id": course_id, "cohort": body.cohort}


@router.post("/api/coordinator/templates/{template_id}/decision")
def coordinator_template_decision(template_id: str, body: TemplateDecision, request: Request):
    actor = who(request)
    row = _get_record(template_id, "templates")
    if not row:
        raise HTTPException(404, "Template proposal not found")
    _require_course(request, row["course_id"])
    if body.decision not in ("approved", "rejected"):
        raise HTTPException(422, "Decision must be approved or rejected")
    with connect() as db:
        db.execute(
            "UPDATE managed_records SET status=?,updated_at=CURRENT_TIMESTAMP WHERE record_id=?",
            (body.decision, template_id),
        )
    audit(actor["username"], actor["role"], "template_decision", f"template:{template_id}", {"decision": body.decision})
    return {"template_id": template_id, "status": body.decision}


@router.get("/api/coordinator/export.csv")
def coordinator_export(request: Request):
    user = who(request)
    ids = _course_ids(user["username"], "coordinator")
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["record_type", "course_id", "record_id", "data", "created_at"])
    with connect() as db:
        for cid in ids:
            for kind, query in (
                ("prompt_check", "SELECT id,prompt,result_json,created_at FROM prompt_checks WHERE course_id=?"),
                ("rating", "SELECT id,template_id,rating,created_at FROM faculty_ratings WHERE course_id=?"),
            ):
                for row in db.execute(query, (cid,)):
                    writer.writerow([kind, cid, row[0], json.dumps(list(row[1:-1])), row[-1]])
    audit(user["username"], user["role"], "course_export_csv", "coordinator_scope")
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=course-usage.csv"},
    )


@router.get("/api/faculty/dashboard")
def faculty_dashboard(request: Request):
    user = who(request)
    ids = _course_ids(user["username"], "faculty")
    if not ids:
        return {"courses": [], "topics": [], "templates": [], "message": "A coordinator must grant course access."}
    q = ",".join("?" for _ in ids)
    with connect() as db:
        courses = [
            _json_record(r)
            for r in db.execute(
                f"SELECT * FROM managed_records WHERE entity_type='courses' AND record_id IN ({q}) AND deleted_at IS NULL",
                ids,
            )
        ]
        topics = [
            _json_record(r)
            for r in db.execute(
                f"SELECT m.* FROM managed_records m JOIN faculty_topics ft ON ft.topic_id=m.record_id WHERE m.entity_type='topics' AND m.course_id IN ({q}) AND ft.username=? AND m.deleted_at IS NULL",
                (*ids, user["username"]),
            )
        ]
        templates = [
            _json_record(r)
            for r in db.execute(
                f"SELECT * FROM managed_records WHERE entity_type='templates' AND (course_id IN ({q}) OR course_id IS NULL) AND status='approved' AND deleted_at IS NULL",
                ids,
            )
        ]
        userrow = db.execute("SELECT synthetic,source FROM auth_users WHERE username=?", (user["username"],)).fetchone()
    return {
        "courses": courses,
        "topics": topics,
        "templates": templates,
        "demo_account": bool(userrow["synthetic"]),
        "message": None
        if templates
        else "No approved templates are available yet. Ask your coordinator to approve a proposal.",
    }


@router.post("/api/faculty/builder")
def faculty_builder(body: BuilderBody, request: Request):
    user = _require_course(request, body.course_id, "faculty")
    if body.cohort not in ("Bachelor", "Master"):
        raise HTTPException(422, "Cohort must be Bachelor or Master")
    if body.question_type not in ("multiple_choice", "scenario_mcq") or body.explanation_level not in (
        "brief",
        "detailed",
    ):
        raise HTTPException(422, "Choose a supported question type and explanation level")
    if (
        not body.difficulty_mix
        or any(not isinstance(v, int) or v < 0 for v in body.difficulty_mix.values())
        or sum(body.difficulty_mix.values()) != 100
    ):
        raise HTTPException(422, "Difficulty mix must total 100% using nonnegative whole percentages")
    with connect() as db:
        allowed_topics = {
            r[0] for r in db.execute("SELECT topic_id FROM faculty_topics WHERE username=?", (user["username"],))
        }
        topic_courses = (
            {
                r[0]: r[1]
                for r in db.execute(
                    "SELECT record_id,course_id FROM managed_records WHERE entity_type='topics' AND record_id IN ("
                    + ",".join("?" for _ in body.topic_ids)
                    + ") AND deleted_at IS NULL",
                    body.topic_ids,
                )
            }
            if body.topic_ids
            else {}
        )
        if not set(body.topic_ids) <= allowed_topics or any(
            topic_courses.get(tid) != body.course_id for tid in body.topic_ids
        ):
            _deny_scope(request, "faculty topics", "topic not granted to faculty")
        course = db.execute(
            "SELECT data_json FROM managed_records WHERE record_id=? AND entity_type='courses' AND deleted_at IS NULL",
            (body.course_id,),
        ).fetchone()
        templates = db.execute(
            "SELECT record_id,data_json FROM managed_records WHERE entity_type='templates' AND (course_id=? OR course_id IS NULL) AND status='approved' AND deleted_at IS NULL ORDER BY record_id",
            (body.course_id,),
        ).fetchall()
        if not course:
            raise HTTPException(404, "Course not found")
        if not templates:
            return {
                "status": "no_template",
                "message": "Ask a coordinator to approve a course template.",
                "prompt": None,
            }
        avg = {
            r[0]: r[1]
            for r in db.execute(
                "SELECT template_id,AVG(rating) FROM faculty_ratings WHERE course_id=? GROUP BY template_id",
                (body.course_id,),
            )
        }
    course_data = json.loads(course[0])
    template_ids = [r[0] for r in templates]
    defaults = {}
    for template_row in templates:
        task_type = str(json.loads(template_row[1]).get("task_type", "quiz"))
        current_id = defaults.get(task_type)
        if current_id is None or avg.get(template_row[0], 0) > avg.get(current_id, 0):
            defaults[task_type] = template_row[0]
    features = {
        "role": "faculty",
        "discipline": [course_data.get("routing_discipline", "other")],
        "study_level": body.cohort.casefold(),
        "experience": 0,
        "task_type": "quiz",
    }
    from app.database import get_feedback
    from app.routing.core import FeatureRouter, Rating, validate_features

    ratings = []
    for row in get_feedback():
        if row["template_id"] not in template_ids:
            continue
        try:
            ratings.append(
                Rating(
                    tuple(validate_features(row["feature_json"])),
                    row["template_id"],
                    float(row["rating"]),
                    bool(row["edited"]),
                    row["timestamp"],
                    False,
                )
            )
        except Exception:
            continue
    routed = FeatureRouter(min_ratings=None).recommend(features, ratings, template_ids, defaults)
    chosen = routed.get("template_id") or template_ids[0]
    template = next(json.loads(row[1]) for row in templates if row[0] == chosen)
    topics = []
    with connect() as db:
        for tid in body.topic_ids:
            topic = db.execute("SELECT data_json FROM managed_records WHERE record_id=?", (tid,)).fetchone()
            if topic:
                topics.append(json.loads(topic[0]).get("name", tid))
    replacements = {
        "#role": "faculty",
        "#course": course_data.get("name", body.course_id),
        "#question_count": str(body.question_count),
        "#topic": ", ".join(topics),
        "#cohort": body.cohort.lower(),
        "#audience": body.cohort.lower() + " students",
    }
    prompt = template.get("prompt", "")
    for key, val in replacements.items():
        prompt = prompt.replace(key, val)
    prompt += f"\nQuestion type: {body.question_type}. Difficulty mix: {json.dumps(body.difficulty_mix)}. Explanation level: {body.explanation_level}."
    if body.extra_instruction.strip():
        lint = evaluate(body.extra_instruction)
        if lint.energy:
            raise HTTPException(
                422, f"Extra instruction has conflicting requirements; remove: {', '.join(lint.repair)}"
            )
        prompt += "\nAdditional instruction: " + body.extra_instruction.strip()
    lint = evaluate(prompt)
    return {
        "status": "built",
        "template_id": chosen,
        "routing_mode": routed["mode"],
        "prompt": prompt,
        "completeness": lint.completeness,
        "slot_scores": lint.slot_scores,
        "energy": lint.energy,
        "repair": lint.repair,
    }


@router.get("/api/faculty/history")
def faculty_history(request: Request):
    user = who(request)
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM prompt_checks WHERE username=? ORDER BY id DESC LIMIT 100", (user["username"],)
        ).fetchall()
    return {"items": [{**dict(r), "result": json.loads(r["result_json"])} for r in rows], "n": len(rows)}


@router.post("/api/faculty/templates", status_code=201)
def faculty_propose_template(body: RecordBody, request: Request):
    user = who(request)
    _require_course(request, body.course_id or "", "faculty")
    record_id = body.record_id or f"proposal-{uuid.uuid4().hex[:12]}"
    with connect() as db:
        db.execute(
            "INSERT INTO managed_records(record_id,entity_type,course_id,owner_username,data_json,status) VALUES(?,'templates',?,?,?,'pending')",
            (record_id, body.course_id, user["username"], json.dumps(body.data)),
        )
    _audit_request(request, "propose_template", f"template:{record_id}")
    return {"record_id": record_id, "status": "pending"}


@router.post("/api/faculty/ratings")
def faculty_rate_template(body: dict, request: Request):
    user = who(request)
    course_id, template_id = str(body.get("course_id", "")), str(body.get("template_id", ""))
    _require_course(request, course_id, "faculty")
    try:
        rating = float(body.get("rating"))
    except (TypeError, ValueError):
        raise HTTPException(422, "Rating must be 1 to 5")
    if not 1 <= rating <= 5:
        raise HTTPException(422, "Rating must be 1 to 5")
    with connect() as db:
        if not db.execute(
            "SELECT 1 FROM managed_records WHERE record_id=? AND entity_type='templates' AND (course_id=? OR course_id IS NULL) AND status='approved' AND deleted_at IS NULL",
            (template_id, course_id),
        ).fetchone():
            raise HTTPException(404, "Approved template not found")
        db.execute(
            "INSERT INTO faculty_ratings(username,course_id,template_id,rating,comment) VALUES(?,?,?,?,?)",
            (user["username"], course_id, template_id, rating, str(body.get("comment", ""))[:500]),
        )
    return {"stored": True}


@router.get("/api/prompt-checker/templates")
def prompt_checker_templates(request: Request, course_id: str | None = None):
    user = who(request)
    if user["role"] == "Faculty":
        courses = _course_ids(user["username"], "faculty")
    elif user["role"] == "Course Coordinator":
        courses = _course_ids(user["username"], "coordinator")
    else:
        courses = []
    if course_id:
        if user["role"] != "University" and course_id not in courses:
            _deny_scope(request, f"templates:{course_id}")
        courses = [course_id]
    with connect() as db:
        if user["role"] == "University" and not course_id:
            rows = db.execute(
                "SELECT * FROM managed_records WHERE entity_type='templates' AND deleted_at IS NULL ORDER BY record_id"
            ).fetchall()
        elif courses:
            marks = ",".join("?" for _ in courses)
            rows = db.execute(
                f"SELECT * FROM managed_records WHERE entity_type='templates' AND (course_id IN ({marks}) OR course_id IS NULL) AND deleted_at IS NULL ORDER BY record_id",
                courses,
            ).fetchall()
        else:
            rows = []
    return {"items": [_json_record(r) for r in rows], "n": len(rows)}


def _checker_result(
    prompt: str, template_id: str | None, sample_id: str | None, user: dict, course_id: str | None, request: Request
) -> dict:
    conflicts, active_policy = _active_linter_policy(course_id)
    result = evaluate(prompt, conflicts=conflicts, policy_atoms=active_policy)
    notes = evidence(prompt, result.atoms, result.slot_scores)
    conflicting = []
    active = set(result.atoms) | set(active_policy)
    for a, b in conflicts:
        if a in active and b in active:
            conflicting.append([a, b])
    template_slots, template_atoms, policy_atoms = {}, set(), set(active_policy)
    with connect() as db:
        if template_id:
            row = db.execute(
                "SELECT data_json,course_id FROM managed_records WHERE record_id=? AND entity_type='templates' AND deleted_at IS NULL",
                (template_id,),
            ).fetchone()
            if not row:
                raise HTTPException(404, "Template not found")
            if row[1] and user["role"] != "University":
                allowed = _course_ids(user["username"], "faculty" if user["role"] == "Faculty" else "coordinator")
                if row[1] not in allowed:
                    _deny_scope(request, f"template:{template_id}", "template outside user's course scope")
                if (
                    user["role"] == "Faculty"
                    and db.execute("SELECT status FROM managed_records WHERE record_id=?", (template_id,)).fetchone()[0]
                    != "approved"
                ):
                    _deny_scope(request, f"template:{template_id}", "faculty may only check approved templates")
            if course_id and row[1] and row[1] != course_id:
                raise HTTPException(403, "Template is outside this course")
            td = json.loads(row[0])
            template_text = str(td.get("prompt", ""))
            template_slots = evaluate(template_text, conflicts=conflicts, policy_atoms=active_policy).slot_scores
            template_atoms = set(extract_atoms(template_text))
            if row[1]:
                records = db.execute(
                    "SELECT data_json FROM managed_records WHERE entity_type='policy_atoms' AND (course_id=? OR course_id IS NULL) AND deleted_at IS NULL",
                    (row[1],),
                ).fetchall()
            else:
                records = db.execute(
                    "SELECT data_json FROM managed_records WHERE entity_type='policy_atoms' AND course_id IS NULL AND deleted_at IS NULL"
                ).fetchall()
            policy_atoms = {
                payload.get("atom")
                for record in records
                if (payload := json.loads(record[0])).get("enabled", True) and payload.get("atom") in ATOMS
            }
    coverage = {}
    for slot, value in template_slots.items():
        score = result.slot_scores.get(slot, 0)
        coverage[slot] = {
            "template_requires": value > 0,
            "prompt_score": score,
            "covered": score >= value if value else True,
        }
    policy_conflicts = [
        [a, b]
        for a, b in conflicts
        if (a in result.atoms and b in policy_atoms) or (b in result.atoms and a in policy_atoms)
    ]
    output = {
        "atoms": result.atoms,
        "completeness": result.completeness,
        "completeness_formula": "sum of nine slot scores / 9",
        "slots": notes["slots"],
        "energy": result.energy,
        "conflict_count": len(conflicting),
        "conflicting_pairs": conflicting,
        "policy_conflicting_pairs": policy_conflicts,
        "repair": result.repair,
        "repair_diff": {"remove": result.repair, "keep": result.consistent_atoms},
        "atom_confidence": notes["atoms"],
        "highlight_spans": notes["spans"],
        "manual_review": notes["manual_review"] or not result.atoms,
        "template_coverage": coverage,
        "template_missing_slots": [s for s, row in coverage.items() if row["template_requires"] and not row["covered"]],
        "template_atoms": sorted(template_atoms),
        "template_policy_atoms": sorted(policy_atoms),
        "engine_version": "rule-linter-0.1.0",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "sample" if sample_id else "user_prompt",
        "sample_id": sample_id,
    }
    with connect() as db:
        db.execute(
            "INSERT INTO prompt_checks(username,role,course_id,prompt,template_id,result_json,source,sample_id) VALUES(?,?,?,?,?,?,?,?)",
            (
                user["username"],
                user["role"],
                course_id,
                prompt,
                template_id,
                json.dumps(output),
                output["source"],
                sample_id,
            ),
        )
    return output


@router.get("/api/real-prompts")
def real_prompt_list(request: Request):
    who(request)
    from app.linter.real_fixtures import REAL_TEST_PROMPTS

    return {
        "items": [
            {
                "id": p["id"],
                "name": p["name"],
                "text": p["text"],
                "source": p["source"],
                "reference": p["reference"],
            }
            for p in REAL_TEST_PROMPTS
        ],
        "n": len(REAL_TEST_PROMPTS),
        "source": "project_brief",
    }


@router.get("/api/samples")
def sample_list(request: Request):
    who(request)
    from app.linter.reference_annotations import reference_fixtures

    refs = {row["name"]: row for row in reference_fixtures()}
    items = []
    for sample in SAMPLES:
        conflicts, policy = _active_linter_policy()
        score = evaluate(sample["text"], conflicts=conflicts, policy_atoms=policy)
        ref = refs.get(sample["reference_name"], {})
        numeric_match = (
            round(score.completeness, 3) == ref.get("completeness")
            and score.energy == ref.get("energy")
            and score.repair == ref.get("repair")
        )
        items.append(
            {
                **sample,
                "label": "SAMPLE",
                "computed": {"completeness": score.completeness, "energy": score.energy, "repair": score.repair},
                "reference": ref,
                "numeric_match": numeric_match,
                "comparison_status": "numerical match; source text unavailable"
                if numeric_match
                else "numerical mismatch; source text unavailable",
            }
        )
    return {
        "items": items,
        "n": len(items),
        "provenance": "authored synthetic substitutes; reference metrics copied from app/linter/reference_annotations.json; original prompt text unavailable",
    }


@router.post("/api/prompt-checker/check")
def prompt_check(body: PromptCheckBody, request: Request):
    user = who(request)
    sample_id = body.sample_id
    prompt = body.prompt
    if sample_id:
        sample = next((s for s in SAMPLES if s["id"] == sample_id), None)
        if not sample:
            raise HTTPException(404, "Sample prompt not found")
        prompt = sample["text"]
    if not prompt:
        raise HTTPException(422, "Prompt text is required")
    course_id = body.course_id
    if user["role"] == "Faculty":
        allowed = _course_ids(user["username"], "faculty")
        if course_id and course_id not in allowed:
            _deny_scope(request, f"prompt-check:{course_id}")
    elif user["role"] == "Course Coordinator":
        allowed = _course_ids(user["username"], "coordinator")
        if course_id and course_id not in allowed:
            _deny_scope(request, f"prompt-check:{course_id}")
    return _checker_result(prompt, body.template_id, sample_id, user, course_id, request)


@router.post("/api/prompt-checker/improve")
def prompt_improve_endpoint(body: PromptImproveBody, request: Request):
    user = who(request)
    sample_id = body.sample_id
    prompt = body.prompt
    if sample_id:
        sample = next((s for s in SAMPLES if s["id"] == sample_id), None)
        if not sample:
            raise HTTPException(404, "Sample prompt not found")
        prompt = sample["text"]
    if not prompt:
        raise HTTPException(422, "Prompt text is required")
    course_id = body.course_id
    if user["role"] == "Faculty":
        allowed = _course_ids(user["username"], "faculty")
        if course_id and course_id not in allowed:
            _deny_scope(request, f"prompt-improve:{course_id}")
    elif user["role"] == "Course Coordinator":
        allowed = _course_ids(user["username"], "coordinator")
        if course_id and course_id not in allowed:
            _deny_scope(request, f"prompt-improve:{course_id}")

    from app.linter.improve import improve_prompt

    slot_values = {}
    if body.cohort:
        slot_values["cohort"] = body.cohort
    if body.count:
        slot_values["count"] = body.count
    if course_id:
        slot_values["course"] = course_id
    if body.topic_id:
        slot_values["topic"] = body.topic_id

    result = improve_prompt(
        prompt=prompt,
        user_choices=body.user_choices,
        slot_values=slot_values,
        use_llm=body.use_llm,
        model=body.model,
    )

    out = {
        "verified": result.verified,
        "rejection_reason": result.rejection_reason,
        "improved_prompt": result.improved_prompt,
        "before": result.before,
        "after": result.after,
        "delta_completeness": result.delta_completeness,
        "delta_energy": result.delta_energy,
        "resolutions": [
            {
                "conflict_pair": r.conflict_pair,
                "proposed_drop": r.proposed_drop,
                "retained_atom": r.retained_atom,
                "reason": r.reason,
                "policy_enforced": r.policy_enforced,
                "alternative_options": r.alternative_options,
            }
            for r in result.resolutions
        ],
        "dropped_atoms": result.dropped_atoms,
        "retained_atoms": result.retained_atoms,
        "filled_slots": result.filled_slots,
        "text_diff": result.text_diff,
        "mode": result.mode,
    }

    audit(
        user["username"],
        user["role"],
        "prompt_improve",
        f"verified:{result.verified}",
        {"dropped": result.dropped_atoms, "delta_comp": result.delta_completeness},
    )
    return out


@router.post("/api/admin/paper-artifacts")
def admin_generate_paper_artifacts(request: Request):
    actor = who(request)
    if actor["role"] != "University":
        raise HTTPException(403, "University role required")
    from app.paper_artifacts import generate_all_paper_artifacts

    result = generate_all_paper_artifacts()
    audit(actor["username"], actor["role"], "generate_paper_artifacts", "success")
    return result


@router.get("/api/admin/paper-artifacts")
def admin_get_paper_artifacts(request: Request):
    actor = who(request)
    if actor["role"] != "University":
        raise HTTPException(403, "University role required")
    manifest_path = Path(__file__).resolve().parent.parent / "artifacts" / "paper_ready" / "manifest.json"
    if not manifest_path.exists():
        return {"status": "not_generated", "message": "Paper artifacts have not been generated yet."}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@router.get("/api/faculty/samples")
def faculty_samples(request: Request):
    return sample_list(request)


def _accuracy_placeholder():
    return {"status": "awaiting annotations"}
