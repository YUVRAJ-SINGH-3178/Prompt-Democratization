"""API endpoints for comparisons, learner, evaluation, and extractor reporting."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from app.auth_db import connect
from app.energy import feature_vector, feature_vector_from_text
from app.learner import (
    fit_bradley_terry,
    predict_preference,
)
from app.linter.core import evaluate, extract_atoms
from app.linter.lexicon import CONFLICTS
from app.linter.datasets import compute_cohen_kappa
from app.security import audit

router = APIRouter()


def who(request: Request) -> dict:
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(401, "Login required")
    return user


def _get_candidate_prompts() -> list[dict[str, str]]:
    from app.linter.real_fixtures import REAL_TEST_PROMPTS
    from app.sample_prompts import SAMPLES

    candidates: list[dict[str, str]] = []
    for p in REAL_TEST_PROMPTS:
        candidates.append({"id": str(p["id"]), "text": str(p["text"]), "name": str(p["name"])})
    for s in SAMPLES:
        candidates.append({"id": str(s["id"]), "text": str(s["text"]), "name": str(s["name"])})
    return candidates


# ------------------------------------------------------------------
# Comparison collection (Task 3)
# ------------------------------------------------------------------


class ComparisonBody(BaseModel):
    prompt_a_text: str = Field(min_length=1, max_length=20000)
    prompt_b_text: str = Field(min_length=1, max_length=20000)
    winner: str = Field(pattern=r"^(a|b|tie)$")
    duration_ms: int | None = None


@router.get("/api/faculty/comparison-pair")
def get_comparison_pair(request: Request, mode: str = "random"):
    user = who(request)
    if user["role"] != "Faculty":
        raise HTTPException(403, "Faculty role required")

    candidates = _get_candidate_prompts()
    if len(candidates) < 2:
        raise HTTPException(400, "Not enough candidate prompts available")

    # Check mode
    if mode == "uncertain":
        with connect() as db:
            fit = db.execute("SELECT weights_json FROM learner_fits ORDER BY id DESC LIMIT 1").fetchone()
        if fit and fit["weights_json"]:
            weights = json.loads(fit["weights_json"])
            best_pair = None
            min_diff = float("inf")
            pairs = random.sample(
                [(i, j) for i in range(len(candidates)) for j in range(len(candidates)) if i != j],
                min(20, len(candidates) * (len(candidates) - 1)),
            )
            for i, j in pairs:
                p1, p2 = candidates[i], candidates[j]
                fv1 = feature_vector_from_text(p1["text"])["values"]
                fv2 = feature_vector_from_text(p2["text"])["values"]
                pred = predict_preference(weights, fv1, fv2)
                diff = abs(pred["p_a_better"] - 0.5)
                if diff < min_diff:
                    min_diff = diff
                    best_pair = (p1, p2)
            c1, c2 = best_pair if best_pair else random.sample(candidates, 2)
            return {
                "prompt_a": c1["text"],
                "prompt_b": c2["text"],
                "prompt_a_id": c1["id"],
                "prompt_b_id": c2["id"],
                "selection_mode": "uncertain",
                "experimental": True,
                "note": "Experimental: uncertain pairs mode picks the pair closest to 50% predicted win probability. The synthetic study found no reliable gain over random selection.",
            }

    # Default random mode
    c1, c2 = random.sample(candidates, 2)
    return {
        "prompt_a": c1["text"],
        "prompt_b": c2["text"],
        "prompt_a_id": c1["id"],
        "prompt_b_id": c2["id"],
        "selection_mode": "random",
        "experimental": False,
        "note": "Random pair selection.",
    }


@router.get("/api/coordinator/comparison-progress")
def coordinator_comparison_progress(request: Request):
    user = who(request)
    if user["role"] not in ("Course Coordinator", "University"):
        raise HTTPException(403, "Coordinator or University role required")
    from app.role_api import _course_ids

    course_ids = _course_ids(user["username"], "coordinator") if user["role"] == "Course Coordinator" else []
    with connect() as db:
        if user["role"] == "University":
            rows = db.execute(
                "SELECT rater_id, COUNT(*) as n_total, SUM(CASE WHEN winner != 'tie' THEN 1 ELSE 0 END) as n_nontie FROM comparisons GROUP BY rater_id"
            ).fetchall()
        elif course_ids:
            q = ",".join("?" for _ in course_ids)
            faculty_users = [
                r[0]
                for r in db.execute(
                    f"SELECT DISTINCT username FROM course_memberships WHERE access_role='faculty' AND course_id IN ({q})",
                    course_ids,
                ).fetchall()
            ]
            if not faculty_users:
                return {
                    "items": [],
                    "total_comparisons": 0,
                    "nontie_comparisons": 0,
                    "required_comparisons": 50,
                    "required_raters": 2,
                }
            fq = ",".join("?" for _ in faculty_users)
            rows = db.execute(
                f"SELECT rater_id, COUNT(*) as n_total, SUM(CASE WHEN winner != 'tie' THEN 1 ELSE 0 END) as n_nontie FROM comparisons WHERE rater_id IN ({fq}) GROUP BY rater_id",
                faculty_users,
            ).fetchall()
        else:
            rows = []
    items = [{"rater_id": r["rater_id"], "comparisons": r["n_total"], "non_tie": r["n_nontie"] or 0} for r in rows]
    total = sum(i["comparisons"] for i in items)
    nontie = sum(i["non_tie"] for i in items)
    return {
        "items": items,
        "total_comparisons": total,
        "nontie_comparisons": nontie,
        "required_comparisons": 50,
        "required_raters": 2,
    }


@router.post("/api/faculty/comparisons", status_code=201)
def add_comparison(body: ComparisonBody, request: Request):
    user = who(request)
    if user["role"] != "Faculty":
        raise HTTPException(403, "Faculty role required")

    rater_id = user["username"]
    # Compute features
    fv_a = feature_vector_from_text(body.prompt_a_text)
    fv_b = feature_vector_from_text(body.prompt_b_text)

    # Randomize shown order
    shown_order = random.choice(["ab", "ba"])

    with connect() as db:
        db.execute(
            "INSERT INTO comparisons(rater_id, prompt_a_id, prompt_b_id, winner, shown_order, features_a_json, features_b_json, duration_ms) VALUES(?,?,?,?,?,?,?,?)",
            (
                rater_id,
                body.prompt_a_text[:100],
                body.prompt_b_text[:100],
                body.winner,
                shown_order,
                json.dumps(fv_a["values"]),
                json.dumps(fv_b["values"]),
                body.duration_ms,
            ),
        )
    return {"stored": True, "winner": body.winner, "shown_order": shown_order}


@router.get("/api/faculty/comparisons")
def get_my_comparisons(request: Request):
    user = who(request)
    with connect() as db:
        rows = db.execute(
            "SELECT id, prompt_a_id, prompt_b_id, winner, shown_order, duration_ms, created_at FROM comparisons WHERE rater_id=? ORDER BY id DESC LIMIT 200",
            (user["username"],),
        ).fetchall()
    return {"items": [dict(r) for r in rows], "n": len(rows)}


@router.get("/api/admin/comparisons")
def admin_comparisons(request: Request):
    user = who(request)
    if user["role"] != "University":
        raise HTTPException(403, "University role required")
    with connect() as db:
        rows = db.execute("SELECT * FROM comparisons ORDER BY id DESC LIMIT 1000").fetchall()
        total = db.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0]
        raters = db.execute("SELECT rater_id, COUNT(*) n FROM comparisons GROUP BY rater_id").fetchall()
    return {
        "items": [dict(r) for r in rows],
        "n": total,
        "raters": [dict(r) for r in raters],
    }


@router.get("/api/admin/comparisons/export.csv")
def admin_export_comparisons(request: Request):
    import csv
    import io
    from fastapi.responses import StreamingResponse

    user = who(request)
    if user["role"] != "University":
        raise HTTPException(403, "University role required")
    with connect() as db:
        rows = db.execute("SELECT * FROM comparisons ORDER BY id").fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "rater_id", "prompt_a_id", "prompt_b_id", "winner", "shown_order", "created_at", "duration_ms"])
    for r in rows:
        w.writerow(
            [
                r["id"],
                r["rater_id"],
                r["prompt_a_id"],
                r["prompt_b_id"],
                r["winner"],
                r["shown_order"],
                r["created_at"],
                r["duration_ms"],
            ]
        )
    audit(user["username"], user["role"], "export_comparisons", f"n={len(rows)}")
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=comparisons.csv"},
    )


@router.get("/api/admin/comparisons/kappa")
def admin_kappa(request: Request):
    user = who(request)
    if user["role"] != "University":
        raise HTTPException(403, "University role required")
    with connect() as db:
        rows = db.execute(
            "SELECT rater_id, prompt_a_id, prompt_b_id, winner FROM comparisons WHERE winner != 'tie'"
        ).fetchall()

    # Group by prompt pair -> rater -> winner
    pair_ratings: dict[tuple[str, str], dict[str, str]] = {}
    for r in rows:
        key = (r["prompt_a_id"], r["prompt_b_id"])
        pair_ratings.setdefault(key, {})[r["rater_id"]] = r["winner"]

    # Find rater pairs with shared items
    rater_ids = sorted({r["rater_id"] for r in rows})
    if len(rater_ids) < 2:
        return {"status": "not enough raters", "n_raters": len(rater_ids), "shared_pairs": 0}

    # Check all rater pairs
    best_kappa = None
    for i, r1 in enumerate(rater_ids):
        for r2 in rater_ids[i + 1 :]:
            shared = []
            for key, ratings in pair_ratings.items():
                if r1 in ratings and r2 in ratings:
                    shared.append((ratings[r1], ratings[r2]))
            if len(shared) >= 30:
                v1 = [s[0] for s in shared]
                v2 = [s[1] for s in shared]
                k = compute_cohen_kappa(v1, v2)
                if best_kappa is None or len(shared) > best_kappa.get("n", 0):
                    best_kappa = {
                        "rater_1": r1,
                        "rater_2": r2,
                        "cohen_kappa": k,
                        "n": len(shared),
                    }

    if best_kappa is None:
        return {
            "status": "not enough shared pairs",
            "n_raters": len(rater_ids),
            "shared_pairs": max(
                (
                    sum(1 for key, ratings in pair_ratings.items() if r1 in ratings and r2 in ratings)
                    for i, r1 in enumerate(rater_ids)
                    for r2 in rater_ids[i + 1 :]
                ),
                default=0,
            ),
            "required": 30,
        }

    return {"status": "computed", **best_kappa}


# ------------------------------------------------------------------
# Learner (Task 4)
# ------------------------------------------------------------------


@router.get("/api/admin/learner")
def admin_learner(request: Request):
    user = who(request)
    if user["role"] != "University":
        raise HTTPException(403, "University role required")

    with connect() as db:
        comparisons = db.execute(
            "SELECT rater_id, features_a_json, features_b_json, winner FROM comparisons WHERE winner != 'tie'"
        ).fetchall()

    data = []
    for r in comparisons:
        fa = json.loads(r["features_a_json"]) if r["features_a_json"] else None
        fb = json.loads(r["features_b_json"]) if r["features_b_json"] else None
        if fa and fb:
            data.append(
                {
                    "features_a": fa,
                    "features_b": fb,
                    "winner": r["winner"],
                    "rater_id": r["rater_id"],
                }
            )

    result = fit_bradley_terry(data)

    # Store fit if successful
    if result["model_status"] == "fitted":
        with connect() as db:
            db.execute(
                "INSERT INTO learner_fits(n_comparisons, n_raters, weights_json, intervals_json, versions_json, seed) VALUES(?,?,?,?,?,?)",
                (
                    result["n_comparisons"],
                    result["n_raters"],
                    json.dumps(result["weights"]),
                    json.dumps(result.get("weight_intervals_95")),
                    json.dumps(result.get("versions")),
                    result.get("bootstrap_seed", 42),
                ),
            )

    return result


@router.get("/api/admin/learner/history")
def admin_learner_history(request: Request):
    user = who(request)
    if user["role"] != "University":
        raise HTTPException(403, "University role required")
    with connect() as db:
        rows = db.execute("SELECT * FROM learner_fits ORDER BY id DESC LIMIT 20").fetchall()
    return {
        "items": [
            {
                **dict(r),
                "weights": json.loads(r["weights_json"]),
                "intervals": json.loads(r["intervals_json"]) if r["intervals_json"] else None,
                "versions": json.loads(r["versions_json"]),
            }
            for r in rows
        ]
    }


# ------------------------------------------------------------------
# Compare and Improve (Task 5)
# ------------------------------------------------------------------


class CompareRequest(BaseModel):
    prompt_a: str = Field(min_length=1, max_length=20000)
    prompt_b: str = Field(min_length=1, max_length=20000)


class ImproveRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    user_choices: dict[str, str] | None = None
    use_llm: bool = False


@router.post("/api/prompts/compare")
def compare_prompts(body: CompareRequest, request: Request):
    who(request)

    result_a = evaluate(body.prompt_a)
    result_b = evaluate(body.prompt_b)
    fv_a = feature_vector(result_a.atoms, result_a.slot_scores)
    fv_b = feature_vector(result_b.atoms, result_b.slot_scores)

    response: dict[str, Any] = {
        "prompt_a": {
            "atoms": result_a.atoms,
            "completeness": result_a.completeness,
            "slots": result_a.slot_scores,
            "energy": result_a.energy,
            "repair": result_a.repair,
            "features": fv_a["values"],
            "feature_contributions": fv_a["named"],
        },
        "prompt_b": {
            "atoms": result_b.atoms,
            "completeness": result_b.completeness,
            "slots": result_b.slot_scores,
            "energy": result_b.energy,
            "repair": result_b.repair,
            "features": fv_b["values"],
            "feature_contributions": fv_b["named"],
        },
        "rules_breakdown": {
            "conflict_energy_diff": result_a.energy - result_b.energy,
            "completeness_diff": round(result_a.completeness - result_b.completeness, 4),
            "a_conflicts": _conflicting_pairs(result_a.atoms),
            "b_conflicts": _conflicting_pairs(result_b.atoms),
            "a_missing_slots": [s for s, v in result_a.slot_scores.items() if v < 1.0],
            "b_missing_slots": [s for s, v in result_b.slot_scores.items() if v < 1.0],
        },
    }

    # Try to get fitted model
    with connect() as db:
        fit = db.execute("SELECT weights_json, intervals_json FROM learner_fits ORDER BY id DESC LIMIT 1").fetchone()

    if fit and fit["weights_json"]:
        weights = json.loads(fit["weights_json"])
        intervals = json.loads(fit["intervals_json"]) if fit["intervals_json"] else None
        pred = predict_preference(weights, fv_a["values"], fv_b["values"], intervals)
        response["model_prediction"] = pred
        response["model_status"] = "fitted"
    else:
        response["model_status"] = "insufficient_data"
        response["model_prediction"] = None

    return response


@router.post("/api/prompts/improve")
def improve_prompt_endpoint(body: ImproveRequest, request: Request):
    who(request)

    from app.linter.improve import resolve_conflicts

    result = evaluate(body.prompt)
    atoms = result.atoms

    # Find minimum drop set
    dropped_atoms, resolutions = resolve_conflicts(atoms, user_choices=body.user_choices)

    # Remove dropped requirements, fill missing slots, regenerate
    from app.linter.improve import build_improved_prompt_text, verify_improvement

    candidate, filled_slots = build_improved_prompt_text(body.prompt, dropped_atoms)
    verification = verify_improvement(body.prompt, candidate, dropped_atoms)

    # Verification gate: improved only if energy strictly lower or deficits strictly lower, and no conflicts remain
    improved_result = evaluate(candidate)
    improved_fv = feature_vector(improved_result.atoms, improved_result.slot_scores)
    original_fv = feature_vector(result.atoms, result.slot_scores)

    summed_deficit_before = sum(original_fv["values"][1:])
    summed_deficit_after = sum(improved_fv["values"][1:])

    gate_passed = (
        verification.verified
        and (improved_result.energy < result.energy or summed_deficit_after < summed_deficit_before)
        and improved_result.energy == 0
    )

    response: dict[str, Any] = {
        "gate_passed": gate_passed,
        "before": {
            "atoms": result.atoms,
            "completeness": result.completeness,
            "energy": result.energy,
            "features": original_fv["values"],
            "summed_deficits": summed_deficit_before,
        },
        "after": {
            "atoms": improved_result.atoms,
            "completeness": improved_result.completeness,
            "energy": improved_result.energy,
            "features": improved_fv["values"],
            "summed_deficits": summed_deficit_after,
        }
        if gate_passed
        else None,
        "improved_prompt": candidate if gate_passed else None,
        "original_prompt": body.prompt if not gate_passed else None,
        "dropped_atoms": dropped_atoms,
        "resolutions": [
            {
                "conflict_pair": r.conflict_pair,
                "proposed_drop": r.proposed_drop,
                "retained_atom": r.retained_atom,
                "reason": r.reason,
                "policy_enforced": r.policy_enforced,
                "alternative_options": r.alternative_options,
            }
            for r in resolutions
        ],
        "filled_slots": filled_slots if gate_passed else [],
        "changes": [],
        "verification_failed_reason": None
        if gate_passed
        else (verification.rejection_reason or "No verified improvement: energy and deficits did not decrease."),
    }

    if not gate_passed:
        response["original_prompt"] = body.prompt
        response["message"] = "No verified improvement was found. Returning the original."

    return response


def _conflicting_pairs(atoms: list[str]) -> list[list[str]]:
    active = set(atoms) | {"clear_unambiguous"}
    pairs = []
    for a, b in CONFLICTS:
        if a in active and b in active:
            pairs.append(sorted([a, b]))
    return pairs


# ------------------------------------------------------------------
# Extractor report (Task 1)
# ------------------------------------------------------------------


@router.get("/api/admin/extractor-report")
def extractor_report(request: Request):
    user = who(request)
    if user["role"] != "University":
        raise HTTPException(403, "University role required")
    return _extractor_report()


def _extractor_report() -> dict[str, Any]:
    """Evaluate the extractor against the gold file."""
    gold_path = Path(__file__).resolve().parent.parent / "data" / "gold" / "atoms_gold.json"
    if not gold_path.exists():
        return {"status": "no_gold_file", "message": "data/gold/atoms_gold.json not found"}

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    prompts = gold.get("prompts", [])

    if not prompts:
        return {"status": "empty_gold_file", "n_real": 0}

    # Split by tune/eval
    tune_prompts = [p for p in prompts if p.get("split") == "tune"]
    eval_prompts = [p for p in prompts if p.get("split") == "eval"]

    # Per-atom precision, recall, F1 on EVAL split
    target_prompts = eval_prompts if eval_prompts else prompts
    atom_metrics: dict[str, dict[str, Any]] = {}

    for atom in gold["atoms_vocabulary"]:
        tp = fp = fn = tn = 0
        for p in target_prompts:
            pred_atoms = extract_atoms(p["text"])
            gold_has = atom in p["atoms"]
            pred_has = atom in pred_atoms
            tp += int(gold_has and pred_has)
            fp += int(not gold_has and pred_has)
            fn += int(gold_has and not pred_has)
            tn += int(not gold_has and not pred_has)

        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        rec = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None

        atom_metrics[atom] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "n": len(target_prompts),
        }

    # Per-prompt detail
    prompt_results = []
    for p in prompts:
        pred = extract_atoms(p["text"])
        prompt_results.append(
            {
                "id": p["id"],
                "split": p.get("split", "unknown"),
                "gold_atoms": p["atoms"],
                "predicted_atoms": pred,
                "match": sorted(pred) == sorted(p["atoms"]),
                "extra": sorted(set(pred) - set(p["atoms"])),
                "missing": sorted(set(p["atoms"]) - set(pred)),
            }
        )

    return {
        "status": "computed",
        "n_real": gold.get("n_real", len(prompts)),
        "n_total": len(prompts),
        "tune_n": len(tune_prompts),
        "eval_n": len(eval_prompts),
        "eval_split": "eval" if eval_prompts else "all",
        "source_note": gold.get("source_note", ""),
        "atom_metrics": atom_metrics,
        "prompt_results": prompt_results,
    }


# ------------------------------------------------------------------
# Evaluation harness (Task 6)
# ------------------------------------------------------------------


@router.post("/api/admin/eval/synthetic")
def admin_run_synthetic_eval(request: Request):
    user = who(request)
    if user["role"] != "University":
        raise HTTPException(403, "University role required")
    from app.eval_harness import synthetic_recovery

    result = synthetic_recovery()
    audit(user["username"], user["role"], "run_synthetic_eval", f"SYNTHETIC n={len(result['results'])}")
    return {"status": "completed", "file": "data/eval/SYNTHETIC_recovery.json", "SYNTHETIC": True}


@router.get("/api/admin/eval")
def admin_eval_results(request: Request):
    user = who(request)
    if user["role"] != "University":
        raise HTTPException(403, "University role required")
    eval_dir = Path(__file__).resolve().parent.parent / "data" / "eval"
    files = []
    if eval_dir.exists():
        for f in sorted(eval_dir.glob("*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            files.append(
                {
                    "file": f.name,
                    "SYNTHETIC": data.get("SYNTHETIC", False),
                    "label": data.get("label", "unknown"),
                    "generated_at": data.get("generated_at"),
                }
            )
    return {"files": files, "n": len(files)}
