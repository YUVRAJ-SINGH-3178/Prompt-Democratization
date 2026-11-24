from __future__ import annotations

import csv
import io
import json
import uuid
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import numpy as np

from app.database import all_rows, connect, get_feedback
from app.linter.core import ATOMS, evaluate
from app.routing.core import FeatureRouter, Rating, validate_features, predict_group_baseline


@asynccontextmanager
async def lifespan(_: FastAPI):
    with connect():
        pass
    yield


app = FastAPI(
    title="Prompt Repository Lab",
    version="0.1.0",
    description="Governed prompt repository prototype. Metrics are computed from stored records; empty tables stay empty.",
    lifespan=lifespan,
)


@app.middleware("http")
async def access_control(request: Request, call_next):
    from app.security import SESSION_COOKIE, deny, resolve_session

    path = request.url.path
    dashboard_roles = {"/admin": "University", "/coordinator": "Course Coordinator", "/faculty": "Faculty"}
    user, csrf = resolve_session(request.cookies.get(SESSION_COOKIE))
    if path == "/dashboard.html":
        if not user:
            return RedirectResponse("/", status_code=303)
        target = {"University": "/admin", "Course Coordinator": "/coordinator", "Faculty": "/faculty"}[user["role"]]
        return RedirectResponse(target, status_code=303)
    if path == "/index.html":
        return RedirectResponse("/", status_code=303)
    if path == "/api/auth/me":
        if user:
            request.state.user = user
            request.state.csrf_token = csrf
        return await call_next(request)
    if path in ("/api/health", "/api/auth/login"):
        return await call_next(request)
    if path.startswith("/api/"):
        if not user:
            deny(None, path, request.method, "authentication required")
            return JSONResponse({"detail": "Login required"}, status_code=401)
        request.state.user = user
        request.state.csrf_token = csrf
        required = None
        if path.startswith("/api/admin/"):
            required = "University"
        elif path.startswith("/api/coordinator/"):
            required = "Course Coordinator"
        elif path.startswith("/api/faculty/"):
            required = "Faculty"
        elif path.startswith("/api/auth/"):
            required = "all"
        elif (
            path.startswith("/api/prompt-checker/")
            or path.startswith("/api/samples")
            or path.startswith("/api/real-prompts")
            or path.startswith("/api/prompts/")
            or path.startswith("/api/reviews")
        ):
            required = "all"
        else:
            required = "University"
        if required != "all" and user["role"] != required:
            deny(user, path, request.method, f"requires {required}")
            return JSONResponse({"detail": "Forbidden"}, status_code=403)
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            supplied = request.headers.get("x-csrf-token")
            if not csrf or not supplied or not secrets.compare_digest(csrf, supplied):
                deny(user, path, request.method, "CSRF token missing or invalid")
                return JSONResponse({"detail": "CSRF validation failed"}, status_code=403)
        return await call_next(request)
    required = dashboard_roles.get(path.rstrip("/"))
    if required:
        if not user:
            return RedirectResponse("/", status_code=303)
        if user["role"] != required:
            deny(user, path, request.method, f"dashboard requires {required}")
            return JSONResponse({"detail": "Forbidden"}, status_code=403)
        return await call_next(request)
    return await call_next(request)


WEB = Path(__file__).resolve().parent.parent / "web"
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "data" / "experiment_runs"


def latest_experiment_files():
    if not EXPERIMENT_DIR.exists():
        return []
    latest = {}
    for path in EXPERIMENT_DIR.glob("*.json"):
        kind = path.name.split("-202", 1)[0]
        if kind not in latest or path.stat().st_mtime > latest[kind].stat().st_mtime:
            latest[kind] = path
    return sorted(latest.values(), key=lambda p: p.name)


def experiment_file_count():
    return len(latest_experiment_files())


def routing_minimum():
    configured = os.environ.get("ROUTER_MIN_RATINGS")
    if configured and configured.isdigit() and int(configured) > 0:
        return int(configured), "environment override"
    report = Path(__file__).resolve().parent.parent / "data" / "break_even_report.json"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        value = data.get("break_even_rating_count")
        if data.get("measurement_status") == "measured" and isinstance(value, int) and value > 0:
            return value, "measured on real feedback"
    return None, "awaiting real feedback measurement"


def routing_std_threshold():
    try:
        value = float(os.environ.get("ROUTER_STD_THRESHOLD", "0.25"))
    except ValueError:
        value = 0.25
    return value if value > 0 else 0.25


class LintRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    extractor: str = "rule"
    synthetic: bool = False


class FeedbackRequest(BaseModel):
    template_id: str
    features: dict[str, Any]
    rating: float = Field(ge=1, le=5)
    edited: bool = False
    synthetic: bool = False


class RouteRequest(BaseModel):
    features: dict[str, Any]
    shortlist_size: int = Field(default=3, ge=1, le=20)
    method: str = "gaussian_process"


class BuildRequest(BaseModel):
    user_id: str
    course_id: str
    topic_id: str
    template_id: str
    question_count: int = Field(ge=1, le=100)


class TemplateRequest(BaseModel):
    id: str
    name: str
    task_type: str
    prompt: str
    synthetic: bool = False


class AnnotationRequest(BaseModel):
    annotator_id: str
    prompt_id: str
    atoms: list[str]
    slots: dict[str, float]
    synthetic: bool = False


class StudyRunRequest(BaseModel):
    faculty_prompt: str
    filled_template: str
    repaired_template: str
    lagrange_template: str
    gp_template: str
    model: str | None = None
    question_count: int = Field(default=5, ge=1, le=100)


class StudyRatingRequest(BaseModel):
    run_id: int
    rater_id: str
    correctness: int | None = Field(default=None, ge=1, le=5)
    distractor_quality: int | None = Field(default=None, ge=1, le=5)
    bloom_level: str | None = None


class MappingBootstrap(BaseModel):
    user_id: str
    role: str
    discipline: str
    study_level: str
    experience: float = Field(ge=-1, le=1)
    course_id: str
    course_name: str
    course_discipline: str
    topic_id: str
    topic_name: str
    synthetic: bool = False


@app.get("/api/health")
def health():
    return {"status": "ok", "data_source": "SQLite", "synthetic_rows_excluded_from_metrics": True}


@app.get("/api/overview")
def overview():
    names = ("templates", "feedback", "prompts_checked", "experiment_runs", "annotations", "users", "courses")
    counts = {t: all_rows(f"SELECT COUNT(*) AS n FROM {t}")[0]["n"] for t in names}
    counts["experiment_runs"] = experiment_file_count()
    ratings = all_rows("SELECT COUNT(*) AS n FROM feedback WHERE synthetic=0")[0]["n"]
    return {
        "counts": counts,
        "real_feedback_n": ratings,
        "synthetic_feedback_n": all_rows("SELECT COUNT(*) AS n FROM feedback WHERE synthetic=1")[0]["n"],
        "real_prompt_checks_n": all_rows("SELECT COUNT(*) AS n FROM prompts_checked WHERE synthetic=0")[0]["n"],
        "synthetic_prompt_checks_n": all_rows("SELECT COUNT(*) AS n FROM prompts_checked WHERE synthetic=1")[0]["n"],
        "synthetic_templates_n": all_rows("SELECT COUNT(*) AS n FROM templates WHERE synthetic=1")[0]["n"],
        "synthetic_mappings_n": sum(
            all_rows(f"SELECT COUNT(*) AS n FROM {table} WHERE synthetic=1")[0]["n"]
            for table in ("users", "courses", "user_courses", "topics")
        ),
        "provenance": "SQLite university_repo.db",
    }


@app.get("/api/templates")
def templates():
    return {
        "items": all_rows(
            "SELECT t.id,t.name,t.task_type,t.prompt,t.status,t.source,t.synthetic,CASE WHEN d.template_id IS NULL THEN 0 ELSE 1 END AS is_default FROM templates t LEFT JOIN task_defaults d ON d.template_id=t.id ORDER BY t.name"
        ),
        "n": all_rows("SELECT COUNT(*) AS n FROM templates")[0]["n"],
    }


@app.get("/api/linter")
def linter_status():
    return {
        "mode": "rule_based",
        "llm_extractor_enabled": bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("LLM_EXTRACTOR_MODEL")),
        "atoms": list(ATOMS),
        "synthetic": False,
    }


@app.get("/api/builder")
def builder_status():
    return {
        "approved_templates": all_rows("SELECT COUNT(*) AS n FROM templates WHERE status='approved'")[0]["n"],
        "users": all_rows("SELECT COUNT(*) AS n FROM users")[0]["n"],
        "course_mappings": all_rows("SELECT COUNT(*) AS n FROM user_courses")[0]["n"],
        "topics": all_rows("SELECT COUNT(*) AS n FROM topics")[0]["n"],
    }


@app.get("/api/routing")
def routing_status():
    minimum, _ = routing_minimum()
    return {
        "template_count": all_rows("SELECT COUNT(*) AS n FROM templates")[0]["n"],
        "real_feedback_count": all_rows("SELECT COUNT(*) AS n FROM feedback WHERE synthetic=0")[0]["n"],
        "synthetic_feedback_count": all_rows("SELECT COUNT(*) AS n FROM feedback WHERE synthetic=1")[0]["n"],
        "minimum_ratings": minimum,
        "task_defaults": all_rows("SELECT * FROM task_defaults"),
        "available_methods": [
            "gaussian_process",
            "lagrange_equispaced",
            "lagrange_chebyshev",
            "cubic_spline",
            "most_popular",
            "random",
        ],
    }


@app.post("/api/templates", status_code=201)
def add_template(body: TemplateRequest):
    with connect() as db:
        try:
            db.execute(
                "INSERT INTO templates(id,name,task_type,prompt,status,synthetic) VALUES(?,?,?,?, 'draft',?)",
                (body.id, body.name, body.task_type, body.prompt, int(body.synthetic)),
            )
            db.execute(
                "INSERT INTO template_versions(template_id,version,prompt,synthetic) VALUES(?,1,?,?)",
                (body.id, body.prompt, int(body.synthetic)),
            )
        except Exception as e:
            raise HTTPException(409, str(e))
    return {"id": body.id, "status": "draft", "synthetic": body.synthetic}


@app.post("/api/templates/{template_id}/approve")
def approve_template(template_id: str):
    with connect() as db:
        cur = db.execute("UPDATE templates SET status='approved' WHERE id=?", (template_id,))
        if not cur.rowcount:
            raise HTTPException(404, "Template not found")
    return {"id": template_id, "status": "approved"}


@app.post("/api/templates/{template_id}/default")
def set_task_default(template_id: str):
    rows = all_rows("SELECT task_type,status FROM templates WHERE id=?", (template_id,))
    if not rows:
        raise HTTPException(404, "Template not found")
    if rows[0]["status"] != "approved":
        raise HTTPException(409, "Only approved templates can be task defaults")
    with connect() as db:
        db.execute(
            "INSERT INTO task_defaults(task_type,template_id) VALUES(?,?) ON CONFLICT(task_type) DO UPDATE SET template_id=excluded.template_id",
            (rows[0]["task_type"], template_id),
        )
    return {"template_id": template_id, "task_type": rows[0]["task_type"], "default": True}


@app.post("/api/linter/check")
def lint(body: LintRequest):
    found = None
    extractor = "rule_based"
    model = None
    if body.extractor == "llm":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise HTTPException(
                422, "LLM extraction requires OPENAI_API_KEY; rule-based extraction remains available offline"
            )
        import urllib.request

        model = os.environ.get("LLM_EXTRACTOR_MODEL")
        if not model:
            raise HTTPException(422, "Set LLM_EXTRACTOR_MODEL to enable the optional extractor")
        logged_prompt = (
            "Return only JSON {atoms:[...]} using this allowed vocabulary: "
            + ", ".join(ATOMS)
            + ". Identify explicit requirements and respect negation.\nPROMPT:\n"
            + body.prompt
        )
        request = {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": logged_prompt}],
        }
        try:
            req = urllib.request.Request(
                os.environ.get("LLM_EXTRACTOR_API_BASE", "https://api.openai.com/v1/chat/completions"),
                data=json.dumps(request).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = json.loads(response.read())
            found = json.loads(raw["choices"][0]["message"]["content"]).get("atoms", [])
            if any(atom not in ATOMS for atom in found):
                raise ValueError("Model returned an unknown atom")
            with connect() as db:
                db.execute(
                    "INSERT INTO linter_extraction_runs(model,logged_prompt,input_prompt,output_json,synthetic) VALUES(?,?,?,?,?)",
                    (model, logged_prompt, body.prompt, json.dumps(found), int(body.synthetic)),
                )
            extractor = "llm"
        except Exception as e:
            raise HTTPException(502, f"Optional LLM extraction failed: {e}")
    elif body.extractor != "rule":
        raise HTTPException(422, "extractor must be rule or llm")
    r = evaluate(body.prompt, found)
    with connect() as db:
        cur = db.execute(
            "INSERT INTO prompts_checked(prompt,atoms_json,completeness,energy,repair_json,synthetic) VALUES(?,?,?,?,?,?)",
            (body.prompt, json.dumps(r.atoms), r.completeness, r.energy, json.dumps(r.repair), int(body.synthetic)),
        )
        prompt_id = str(cur.lastrowid)
    return {
        "prompt_id": prompt_id,
        "atoms": r.atoms,
        "completeness": r.completeness,
        "slot_scores": r.slot_scores,
        "energy": r.energy,
        "repair": r.repair,
        "consistent_atoms": r.consistent_atoms,
        "extractor": extractor,
        "model": model,
        "n": 1,
        "synthetic": body.synthetic,
    }


@app.post("/api/builder/fill")
def build_prompt(body: BuildRequest):
    rows = all_rows(
        "SELECT t.prompt,u.role,u.study_level,c.name AS course_name,top.name AS topic,(t.synthetic OR u.synthetic OR uc.synthetic OR c.synthetic OR top.synthetic) AS synthetic FROM templates t JOIN users u ON u.id=? JOIN user_courses uc ON uc.user_id=u.id JOIN courses c ON c.id=uc.course_id JOIN topics top ON top.course_id=c.id WHERE t.id=? AND c.id=? AND top.id=? AND t.status='approved'",
        (body.user_id, body.template_id, body.course_id, body.topic_id),
    )
    if not rows:
        raise HTTPException(404, "No approved template and matching user/course/topic mapping found")
    vals = rows[0]
    prompt = vals["prompt"]
    for key, value in {
        "#university_teacher": vals["role"],
        "#bachelor/master": vals["study_level"],
        "#course_name": vals["course_name"],
        "#topic": vals["topic"],
        "#question_count": str(body.question_count),
    }.items():
        prompt = prompt.replace(key, value)
    result = evaluate(prompt)
    return {
        "prompt": prompt,
        "completeness": result.completeness,
        "energy": result.energy,
        "atoms": result.atoms,
        "repair": result.repair,
        "slots_filled_by": "SQLite mapping tables and validated question_count",
        "synthetic": bool(vals["synthetic"]),
    }


def get_ratings():
    out = []
    for r in get_feedback():
        try:
            x = validate_features(r["feature_json"])
        except ValueError:
            continue
        out.append(
            Rating(tuple(x.tolist()), r["template_id"], float(r["rating"]), bool(r["edited"]), r["timestamp"], False)
        )
    return out


@app.post("/api/feedback", status_code=201)
def feedback(body: FeedbackRequest):
    try:
        x = validate_features(body.features)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if not all_rows("SELECT id FROM templates WHERE id=?", (body.template_id,)):
        raise HTTPException(404, "Template not found")
    with connect() as db:
        db.execute(
            "INSERT INTO feedback(template_id,feature_json,rating,edited,synthetic) VALUES(?,?,?,?,?)",
            (body.template_id, json.dumps(body.features), body.rating, int(body.edited), int(body.synthetic)),
        )
    return {"stored": True, "n": 1, "synthetic": body.synthetic, "validated_feature_dimension": len(x)}


@app.post("/api/routing/recommend")
def recommend(body: RouteRequest):
    try:
        validate_features(body.features)
    except ValueError as e:
        raise HTTPException(422, str(e))
    ts = all_rows("SELECT id,task_type FROM templates ORDER BY id")
    if not ts:
        return {
            "mode": "empty",
            "message": "Add a template to start routing",
            "predictions": [],
            "shortlist": [],
            "synthetic": False,
        }
    defaults = {r["task_type"]: r["template_id"] for r in all_rows("SELECT * FROM task_defaults")}
    if body.method == "gaussian_process":
        minimum, _ = routing_minimum()
        threshold = routing_std_threshold()
        return FeatureRouter(min_ratings=minimum, confidence_threshold=threshold).recommend(
            body.features, get_ratings(), [t["id"] for t in ts], defaults, body.shortlist_size
        )
    if body.method not in ("lagrange_equispaced", "lagrange_chebyshev", "cubic_spline", "most_popular", "random"):
        raise HTTPException(422, "Unknown routing method")
    feedbacks = [
        r
        for r in get_feedback()
        if not r.get("synthetic") and r["feature_json"].get("task_type") == body.features["task_type"]
    ]
    if not feedbacks:
        return {
            "mode": "empty",
            "message": "No real ratings yet",
            "predictions": [],
            "shortlist": [t["id"] for t in ts],
            "synthetic": False,
        }
    sums: dict[str, list[float]] = {t["id"]: [] for t in ts}
    for r in feedbacks:
        if r["template_id"] in sums:
            sums[r["template_id"]].append(float(r["rating"]))
    means = {k: (sum(v) / len(v) if v else None) for k, v in sums.items()}
    ranked = []
    if body.method == "random":
        import random

        chosen = random.Random(42).choice([t["id"] for t in ts])
    elif body.method == "most_popular":
        chosen = max(sums, key=lambda k: (len(sums[k]), k))
    else:
        roles = ("faculty", "ta", "student", "staff")
        query = roles.index(str(body.features["role"]))
        for t in ts:
            grouped: dict[int, list[float]] = {}
            for row in feedbacks:
                if row["template_id"] != t["id"]:
                    continue
                role = row["feature_json"].get("role")
                if role in roles:
                    grouped.setdefault(roles.index(role), []).append(float(row["rating"]))
            try:
                group_x = np.asarray(sorted(grouped), dtype=float)
                group_y = np.asarray([np.mean(grouped[i]) for i in sorted(grouped)], dtype=float)
                value = predict_group_baseline(body.method, group_x, group_y, query)
                ranked.append(
                    {
                        "template_id": t["id"],
                        "mean": value,
                        "n": len(sums[t["id"]]),
                        "groups_n": len(grouped),
                        "synthetic": False,
                    }
                )
            except ValueError:
                continue
        if not ranked:
            return {
                "mode": "empty",
                "method": body.method,
                "message": "Insufficient real ratings across named role groups for this interpolation method.",
                "predictions": [],
                "shortlist": [t["id"] for t in ts],
                "synthetic": False,
            }
        ranked.sort(key=lambda r: (-r["mean"], r["template_id"]))
        chosen = ranked[0]["template_id"]
    if body.method in ("most_popular", "random"):
        ranked = sorted(
            ({"template_id": t["id"], "mean": means[t["id"]], "n": len(sums[t["id"]]), "synthetic": False} for t in ts),
            key=lambda r: (-r["n"], -(r["mean"] if r["mean"] is not None else -float("inf")), r["template_id"]),
        )
    shortlist = [chosen] + [x["template_id"] for x in ranked if x["template_id"] != chosen]
    return {
        "mode": "baseline",
        "method": body.method,
        "template_id": chosen,
        "shortlist": shortlist[: body.shortlist_size],
        "predictions": ranked,
        "synthetic": False,
    }


@app.get("/api/data")
def data_summary():
    synthetic_tables = (
        "users",
        "courses",
        "user_courses",
        "topics",
        "templates",
        "template_versions",
        "feedback",
        "prompts_checked",
        "linter_extraction_runs",
        "experiment_runs",
        "annotations",
        "llm_study_runs",
        "llm_ratings",
    )
    synthetic_by_table = {
        table: all_rows(f"SELECT synthetic,COUNT(*) AS n FROM {table} GROUP BY synthetic") for table in synthetic_tables
    }
    table_counts = {
        table: all_rows(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"]
        for table in (
            "users",
            "courses",
            "user_courses",
            "topics",
            "templates",
            "task_defaults",
            "template_versions",
            "feedback",
            "prompts_checked",
            "linter_extraction_runs",
            "experiment_runs",
            "annotations",
            "llm_study_runs",
            "llm_ratings",
        )
    }
    table_counts["experiment_runs"] = experiment_file_count()
    return {
        "tables": table_counts,
        "provenance": "SQLite university_repo.db; experiment outputs also saved as JSON",
        "synthetic_by_table": synthetic_by_table,
        "synthetic_rows_labelled": True,
    }


@app.post("/api/data/bootstrap-mapping", status_code=201)
def bootstrap_mapping(body: MappingBootstrap):
    with connect() as db:
        db.execute(
            "INSERT INTO users(id,role,discipline,study_level,experience,synthetic) VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET role=excluded.role,discipline=excluded.discipline,study_level=excluded.study_level,experience=excluded.experience,synthetic=excluded.synthetic",
            (body.user_id, body.role, body.discipline, body.study_level, body.experience, int(body.synthetic)),
        )
        db.execute(
            "INSERT INTO courses(id,name,discipline,synthetic) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,discipline=excluded.discipline,synthetic=excluded.synthetic",
            (body.course_id, body.course_name, body.course_discipline, int(body.synthetic)),
        )
        db.execute(
            "INSERT INTO user_courses(user_id,course_id,synthetic) VALUES(?,?,?) ON CONFLICT(user_id,course_id) DO UPDATE SET synthetic=excluded.synthetic",
            (body.user_id, body.course_id, int(body.synthetic)),
        )
        db.execute(
            "INSERT INTO topics(id,course_id,name,synthetic) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET course_id=excluded.course_id,name=excluded.name,synthetic=excluded.synthetic",
            (body.topic_id, body.course_id, body.topic_name, int(body.synthetic)),
        )
    return {
        "created": True,
        "user_id": body.user_id,
        "course_id": body.course_id,
        "topic_id": body.topic_id,
        "synthetic": body.synthetic,
    }


@app.get("/api/experiments")
def experiments():
    files = latest_experiment_files()
    items = []
    for f in files:
        item = json.loads(f.read_text(encoding="utf-8"))
        item.setdefault("file", str(f.relative_to(Path(__file__).resolve().parent.parent)))
        items.append(item)
    return {
        "items": items,
        "n": len(items),
        "message": "Run `python -m app.cli reproduce all` to create reproducible reports" if not items else None,
    }


@app.get("/api/annotations")
def get_annotations():
    rows = all_rows("SELECT * FROM annotations WHERE source='held_out' AND synthetic=0 ORDER BY prompt_id,annotator_id")
    by_prompt = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_id"], []).append(row)
    complete = [(pid, rs) for pid, rs in by_prompt.items() if len({r["annotator_id"] for r in rs}) == 2]
    report = {"status": "awaiting annotations"}
    if len(complete) >= 30:
        from sklearn.metrics import cohen_kappa_score

        atom_report = {}
        slot_report = {}
        atom_predictions = {}
        for _, pair in complete:
            source = all_rows(
                "SELECT prompt FROM prompts_checked WHERE CAST(id AS TEXT)=? AND synthetic=0", (pair[0]["prompt_id"],)
            )
            if source:
                atom_predictions[pair[0]["prompt_id"]] = (
                    set(json.loads(pair[0]["atoms_json"])),
                    set(__import__("app.linter.core", fromlist=["extract_atoms"]).extract_atoms(source[0]["prompt"])),
                )
        for atom in ATOMS:
            tp = fp = fn = 0
            for gold, pred in atom_predictions.values():
                tp += atom in gold and atom in pred
                fp += atom not in gold and atom in pred
                fn += atom in gold and atom not in pred
            atom_report[atom] = {
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "n": len(atom_predictions),
            }
        for slot in (
            "role",
            "audience",
            "topic",
            "count",
            "n_options",
            "labeling",
            "marking",
            "explanation",
            "ordering",
        ):
            a = []
            b = []
            for _, pair in complete:
                av = json.loads(pair[0]["slots_json"])
                bv = json.loads(pair[1]["slots_json"])
                if slot in av and slot in bv:
                    a.append(av[slot])
                    b.append(bv[slot])
            slot_report[slot] = {"cohen_kappa": cohen_kappa_score(a, b) if len(a) > 1 else None, "n": len(a)}
        report = {
            "status": "measured",
            "n_prompts": len(complete),
            "extractor_reference_annotator": "lexicographically first annotator id",
            "atom_precision_recall": atom_report,
            "slot_agreement": slot_report,
        }
    return {
        "items": rows,
        "n": len(rows),
        "held_out_status": report["status"],
        "held_out_report": report,
        "manual_reference": "single annotator; original prompt text unavailable in source archive",
    }


@app.post("/api/annotations")
def add_annotation(body: AnnotationRequest):
    if any(a not in ATOMS for a in body.atoms):
        raise HTTPException(422, "Unknown atom")
    if any(not 0 <= v <= 1 for v in body.slots.values()):
        raise HTTPException(422, "Slot scores must be in [0,1]")
    with connect() as db:
        db.execute(
            "INSERT INTO annotations(annotator_id,prompt_id,atoms_json,slots_json,source,synthetic) VALUES(?,?,?,?, 'held_out',?)",
            (body.annotator_id, body.prompt_id, json.dumps(body.atoms), json.dumps(body.slots), int(body.synthetic)),
        )
    return {"stored": True, "n": 1, "synthetic": body.synthetic}


@app.post("/api/import/ratings")
async def import_ratings(file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(content))
    errors = []
    imported = 0
    with connect() as db:
        for i, row in enumerate(reader, start=2):
            try:
                template = row["template_id"]
                rating = float(row["rating"])
                features = json.loads(row["features"])
                if not 1 <= rating <= 5:
                    raise ValueError("rating must be in [1,5]")
                validate_features(features)
                db.execute(
                    "INSERT INTO feedback(template_id,feature_json,rating,edited,synthetic) VALUES(?,?,?,?,?)",
                    (
                        template,
                        json.dumps(features),
                        rating,
                        int(row.get("edited", "false").lower() in ("1", "true", "yes")),
                        int(row.get("synthetic", "false").lower() in ("1", "true", "yes")),
                    ),
                )
                imported += 1
            except Exception as e:
                errors.append({"row": i, "error": str(e)})
    return {"imported": imported, "errors": errors}


@app.post("/api/import/annotations")
async def import_annotations(file: UploadFile = File(...)):
    reader = csv.DictReader(io.StringIO((await file.read()).decode("utf-8-sig")))
    errors = []
    imported = 0
    with connect() as db:
        for i, row in enumerate(reader, start=2):
            try:
                atoms = json.loads(row["atoms"])
                slots = json.loads(row["slots"])
                if any(a not in ATOMS for a in atoms):
                    raise ValueError("unknown atom")
                if any(not 0 <= float(v) <= 1 for v in slots.values()):
                    raise ValueError("slot score outside [0,1]")
                synthetic = row.get("synthetic", "false").lower() in ("1", "true", "yes")
                db.execute(
                    "INSERT INTO annotations(annotator_id,prompt_id,atoms_json,slots_json,source,synthetic) VALUES(?,?,?,?, 'held_out',?)",
                    (row["annotator_id"], row["prompt_id"], json.dumps(atoms), json.dumps(slots), int(synthetic)),
                )
                imported += 1
            except Exception as e:
                errors.append({"row": i, "error": str(e)})
    return {"imported": imported, "errors": errors}


@app.get("/api/study")
def study_status():
    runs = all_rows("SELECT id,batch_id,model,condition_name,created_at,synthetic FROM llm_study_runs")
    ratings = all_rows(
        "SELECT r.*,s.batch_id,s.condition_name FROM llm_ratings r JOIN llm_study_runs s ON s.id=r.run_id WHERE r.synthetic=0 AND s.synthetic=0"
    )
    results = []
    agreement = {}
    if ratings:
        from itertools import combinations
        from scipy.stats import wilcoxon
        from sklearn.metrics import cohen_kappa_score

        by_key = {}
        for r in ratings:
            by_key[(r["batch_id"], r["rater_id"], r["condition_name"])] = r
        conditions = sorted({r["condition_name"] for r in runs if not r["synthetic"]})
        for metric in ("correctness", "distractor_quality"):
            pairs = []
            for left, right in combinations(conditions, 2):
                a = []
                b = []
                batches = {r["batch_id"] for r in runs if r["condition_name"] == left}
                raters = {r["rater_id"] for r in ratings}
                for batch in batches:
                    for rater in raters:
                        ra = by_key.get((batch, rater, left))
                        rb = by_key.get((batch, rater, right))
                        if ra and rb and ra[metric] is not None and rb[metric] is not None:
                            a.append(ra[metric])
                            b.append(rb[metric])
                if len(a) >= 2:
                    stat, p = wilcoxon(a, b, zero_method="wilcox") if any(x != y for x, y in zip(a, b)) else (0.0, 1.0)
                    pairs.append(
                        {
                            "metric": metric,
                            "condition_a": left,
                            "condition_b": right,
                            "n_pairs": len(a),
                            "statistic": float(stat),
                            "p_value": float(p),
                        }
                    )
            # Holm step-down adjustment, separately for each pre-specified metric family.
            ordered = sorted(range(len(pairs)), key=lambda i: pairs[i]["p_value"])
            adjusted = [0.0] * len(pairs)
            running = 0.0
            m = len(pairs)
            for rank, index in enumerate(ordered):
                running = max(running, min(1.0, (m - rank) * pairs[index]["p_value"]))
                adjusted[index] = running
            for index, pair in enumerate(pairs):
                pair["p_holm"] = adjusted[index]
            results.extend(pairs)
        for metric in ("correctness", "distractor_quality", "bloom_level"):
            values = {}
            for r in ratings:
                if r[metric] is not None:
                    values.setdefault(r["run_id"], []).append((r["rater_id"], r[metric]))
            a = []
            b = []
            for vals in values.values():
                if len(vals) == 2:
                    a.append(vals[0][1])
                    b.append(vals[1][1])
            agreement[metric] = {
                "cohen_kappa": float(cohen_kappa_score(a, b)) if len(a) >= 2 else None,
                "n_pairs": len(a),
            }
    return {
        "runs": runs,
        "ratings": ratings,
        "results": results,
        "agreement": agreement,
        "status": "not run" if not runs else ("awaiting real ratings" if not ratings else "computed on real ratings"),
        "statistics": "paired Wilcoxon signed-rank with Holm correction; Cohen's kappa for two-rater agreement",
    }


@app.post("/api/study/run", status_code=201)
def run_study(body: StudyRunRequest):
    from app.study import CONDITIONS, build_study_prompts, execute_condition

    prompts = {
        key: value + f"\n\nGenerate exactly {body.question_count} questions."
        for key, value in build_study_prompts(body.model_dump()).items()
    }
    model = body.model or os.environ.get("STUDY_MODEL")
    if not model:
        raise HTTPException(422, "Set STUDY_MODEL or provide model")
    batch = uuid.uuid4().hex
    completed = []
    try:
        with connect() as db:
            for index, condition in enumerate(CONDITIONS):
                result = execute_condition(prompts[condition], model, body.question_count, seed=4200 + index)
                cur = db.execute(
                    "INSERT INTO llm_study_runs(batch_id,model,condition_name,request_json,response_json,checks_json) VALUES(?,?,?,?,?,?)",
                    (
                        batch,
                        model,
                        condition,
                        json.dumps(result["request"]),
                        json.dumps(result["response"]),
                        json.dumps(result["automatic_checks"]),
                    ),
                )
                completed.append(
                    {
                        "run_id": cur.lastrowid,
                        "condition": condition,
                        "checks": result["automatic_checks"],
                        "synthetic": False,
                    }
                )
    except Exception as e:
        raise HTTPException(502, f"Study request failed; previously completed conditions are stored: {e}")
    return {"batch_id": batch, "model": model, "runs": completed, "status": "real run", "synthetic": False}


@app.post("/api/study/ratings", status_code=201)
def rate_study(body: StudyRatingRequest):
    if not all_rows("SELECT id FROM llm_study_runs WHERE id=?", (body.run_id,)):
        raise HTTPException(404, "Study run not found")
    with connect() as db:
        db.execute(
            "INSERT INTO llm_ratings(run_id,rater_id,correctness,distractor_quality,bloom_level) VALUES(?,?,?,?,?)",
            (body.run_id, body.rater_id, body.correctness, body.distractor_quality, body.bloom_level),
        )
    return {"stored": True, "synthetic": False}


@app.get("/api/settings")
def settings():
    threshold, source = routing_minimum()
    return {
        "theme": "dark",
        "llm_extractor_enabled": bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("LLM_EXTRACTOR_MODEL")),
        "llm_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        "auth": "not present in clean-room source",
        "default_min_ratings": threshold,
        "min_ratings_source": source,
        "std_threshold": routing_std_threshold(),
    }


@app.get("/api/simulated")
def simulated_summary():
    from app.simulated_analysis import analyze
    from app.simulated_database import audit, connect

    with connect() as db:
        report = analyze(db)
        return {
            "banner": "SIMULATED RATERS",
            "source": "sim_persona",
            "synthetic": True,
            "persona_ids": report["persona_ids"],
            "audit": audit(db),
            "report": report,
        }


@app.post("/api/simulated/import-real-scores")
async def import_real_simulation_comparison(file: UploadFile = File(...)):
    from app.simulated_analysis import analyze
    from app.simulated_database import connect

    reader = csv.DictReader(io.StringIO((await file.read()).decode("utf-8-sig")))
    required = {"faculty_id", "item_id", "metric", "value"}
    if not reader.fieldnames or not required <= set(reader.fieldnames):
        raise HTTPException(422, "CSV columns must include faculty_id,item_id,metric,value")
    errors = []
    imported = 0
    with connect() as db:
        for line, row in enumerate(reader, 2):
            try:
                if not row["faculty_id"].strip() or not row["item_id"].strip() or not row["metric"].strip():
                    raise ValueError("faculty_id, item_id, and metric are required")
                metric = row["metric"].strip()
                raw_value = row["value"].strip()
                if metric == "label":
                    if raw_value not in ("vague", "contradictory", "both", "fine"):
                        raise ValueError("label must be vague, contradictory, both, or fine")
                    value: Any = raw_value
                elif metric.startswith("atom:"):
                    atom = metric.split(":", 1)[1]
                    if atom not in ATOMS or raw_value.lower() not in ("true", "false", "1", "0"):
                        raise ValueError("atom metrics must use an allowed atom and a boolean value")
                    value = raw_value.lower() in ("true", "1")
                elif metric.startswith("slot:"):
                    slot = metric.split(":", 1)[1]
                    slot_names = {
                        "role",
                        "audience",
                        "topic",
                        "count",
                        "n_options",
                        "labeling",
                        "marking",
                        "explanation",
                        "ordering",
                    }
                    if slot not in slot_names or raw_value not in ("0", "0.0", "0.5", "1", "1.0"):
                        raise ValueError("slot metrics must use a named slot and value 0, 0.5, or 1")
                    value = float(raw_value)
                else:
                    raise ValueError("metric must be label, atom:<name>, or slot:<name>")
                db.execute(
                    "INSERT OR REPLACE INTO real_faculty_scores(faculty_id,item_id,metric,value_json,source,synthetic) VALUES(?,?,?,?, 'real_faculty',0)",
                    (row["faculty_id"].strip(), row["item_id"].strip(), metric, json.dumps(value)),
                )
                imported += 1
            except Exception as exc:
                errors.append({"line": line, "error": str(exc)})
        report = analyze(db)
    return {
        "imported": imported,
        "errors": errors,
        "source": "real_faculty",
        "synthetic": False,
        "comparison": report["real_faculty_comparison"],
    }


if WEB.exists():
    from app.role_api import router as role_router
    from app.comparison_api import router as comparison_router

    app.include_router(role_router)
    app.include_router(comparison_router)

    @app.get("/", include_in_schema=False)
    def login_page():
        return FileResponse(WEB / "login.html")

    @app.get("/admin", include_in_schema=False)
    def university_dashboard_page():
        return FileResponse(WEB / "dashboard.html")

    @app.get("/coordinator", include_in_schema=False)
    def coordinator_dashboard_page():
        return FileResponse(WEB / "dashboard.html")

    @app.get("/faculty", include_in_schema=False)
    def faculty_dashboard_page():
        return FileResponse(WEB / "dashboard.html")

    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
