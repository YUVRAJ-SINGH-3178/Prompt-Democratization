"""Descriptive analyses for the isolated synthetic persona pilot."""

from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter, defaultdict

import numpy as np
from scipy.interpolate import BarycentricInterpolator, CubicSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from app.linter.core import ATOMS


def _kappa(a: list, b: list) -> float | None:
    if len(a) < 2:
        return None
    categories = set(a) | set(b)
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    pa = Counter(a)
    pb = Counter(b)
    expected = sum(pa[c] * pb[c] for c in categories) / (len(a) * len(b))
    return (observed - expected) / (1 - expected) if expected != 1 else (1.0 if observed == 1 else 0.0)


def _alpha_nominal(values: list[list[str | None]]) -> float | None:
    pairs = [
        (a, b) for row in values for i, a in enumerate(row) for b in row[i + 1 :] if a is not None and b is not None
    ]
    if not pairs:
        return None
    observed_disagreement = sum(a != b for a, b in pairs) / len(pairs)
    counts = Counter(x for row in values for x in row if x is not None)
    n = sum(counts.values())
    if n < 2:
        return None
    expected_disagreement = 1 - sum(v * (v - 1) for v in counts.values()) / (n * (n - 1))
    return 1 - observed_disagreement / expected_disagreement if expected_disagreement else 1.0


def _rating_analysis(db: sqlite3.Connection) -> dict:
    rows = db.execute(
        "SELECT persona_id,template_id,repetition,rating FROM ratings ORDER BY persona_id,template_id,repetition"
    ).fetchall()
    by_persona: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for p, t, rep, rating in rows:
        by_persona[p][t][rep] = rating
    retest = {}
    distributions = {}
    preferences = {}
    templates = sorted({r[1] for r in rows})
    for pid, per_template in by_persona.items():
        correlations = []
        for left, right in ((1, 2), (1, 3), (2, 3)):
            pairs = [(vals[left], vals[right]) for vals in per_template.values() if left in vals and right in vals]
            corr = float(np.corrcoef(np.array(pairs).T)[0, 1]) if len(pairs) > 1 and np.std(pairs) else None
            correlations.append({"repetitions": [left, right], "pearson_r": corr, "n": len(pairs)})
        all_scores = [v for vals in per_template.values() for v in vals.values()]
        averages = {t: float(np.mean(list(per_template[t].values()))) for t in per_template}
        best = max(averages.values(), default=None)
        retest[pid] = correlations
        distributions[pid] = {
            "n": len(all_scores),
            "mean": float(np.mean(all_scores)) if all_scores else None,
            "counts": {str(v): all_scores.count(v) for v in range(1, 6)},
        }
        preferences[pid] = {"preferred": [t for t, v in averages.items() if v == best], "means": averages}
    return {
        "n": len(rows),
        "personas": len(by_persona),
        "templates": templates,
        "test_retest": retest,
        "rating_distributions": distributions,
        "preferences": preferences,
    }


def _annotation_analysis(db: sqlite3.Connection) -> dict:
    rows = db.execute(
        "SELECT persona_id,prompt_id,label,atoms_json FROM annotations ORDER BY prompt_id,persona_id"
    ).fetchall()
    persons = sorted({r[0] for r in rows})
    items = sorted({r[1] for r in rows})
    lookup = {(p, item): (label, set(json.loads(atoms))) for p, item, label, atoms in rows}
    pairwise = {}
    metrics = ["label", *ATOMS]
    for i, left in enumerate(persons):
        for right in persons[i + 1 :]:
            for metric in metrics:
                a, b = [], []
                for item in items:
                    if (left, item) not in lookup or (right, item) not in lookup:
                        continue
                    la, aa = lookup[(left, item)]
                    lb, ab = lookup[(right, item)]
                    a.append(la if metric == "label" else metric in aa)
                    b.append(lb if metric == "label" else metric in ab)
                pairwise[f"{left}__{right}__{metric}"] = {"cohen_kappa": _kappa(a, b), "n": len(a)}
    alpha = {}
    for metric in metrics:
        matrix: list[list[str | None]] = []
        for item in items:
            row: list[str | None] = []
            for persona in persons:
                entry = lookup.get((persona, item))
                if not entry:
                    row.append(None)
                elif metric == "label":
                    row.append(entry[0])
                else:
                    row.append("present" if metric in entry[1] else "absent")
            matrix.append(row)
        alpha[metric] = _alpha_nominal(matrix)
    majority = {}
    for item in items:
        labels = [lookup[(p, item)][0] for p in persons if (p, item) in lookup]
        majority[item] = Counter(labels).most_common(1)[0][0]
    # This is agreement with the simulated majority label, never accuracy.
    linter = []
    prompt_by_id = {r[0]: r[1] for r in db.execute("SELECT prompt_id,prompt FROM prompts")}
    from app.linter.core import evaluate

    for item, label in majority.items():
        prompt = prompt_by_id.get(item)
        if prompt is None:
            continue
        lint = evaluate(prompt)
        vague = lint.completeness < 0.6
        contradictory = lint.energy > 0
        prediction = (
            "both" if vague and contradictory else "vague" if vague else "contradictory" if contradictory else "fine"
        )
        linter.append(prediction == label)
    return {
        "n_items": len(items),
        "n_personas": len(persons),
        "pairwise_cohen_kappa": pairwise,
        "krippendorff_alpha_nominal": alpha,
        "linter_vs_persona_majority_agreement": {
            "agreement": float(np.mean(linter)) if linter else None,
            "n": len(linter),
            "interpretation": "agreement only, not accuracy",
        },
    }


def _routing_analysis(db: sqlite3.Connection) -> dict:
    personas = [json.loads(r[0]) for r in db.execute("SELECT card_json FROM personas ORDER BY persona_id")]
    rating_rows = db.execute(
        "SELECT persona_id,template_id,AVG(rating) FROM ratings GROUP BY persona_id,template_id"
    ).fetchall()
    ratings = {(p, t): float(v) for p, t, v in rating_rows}
    templates = sorted({t for _, t in ratings})
    methods = ("gaussian_process", "lagrange_equispaced", "cubic_spline", "popularity", "random")
    rows: dict[str, list[dict[str, object]]] = {method: [] for method in methods}
    seed = 731204
    for test in personas:
        pid = test["id"]
        truth = {t: ratings[(pid, t)] for t in templates if (pid, t) in ratings}
        if not truth:
            continue
        training = [p for p in personas if p["id"] != pid and all((p["id"], t) in ratings for t in templates)]
        if len(training) < 4:
            continue
        xtrain = np.array([p["years_teaching"] for p in training], dtype=float)
        xtest = float(test["years_teaching"])
        predictions: dict[str, dict[str, float]] = {m: {} for m in methods}
        for template in templates:
            y = np.array([ratings[(p["id"], template)] for p in training], dtype=float)
            grouped_years: dict[float, list[float]] = defaultdict(list)
            for year, value in zip(xtrain, y):
                grouped_years[float(year)].append(float(value))
            interpolation_x = np.array(sorted(grouped_years), dtype=float)
            interpolation_y = np.array([np.mean(grouped_years[x]) for x in interpolation_x], dtype=float)
            if len(interpolation_x) >= 2:
                predictions["lagrange_equispaced"][template] = float(
                    BarycentricInterpolator(interpolation_x, interpolation_y)(xtest)
                )
            if len(interpolation_x) >= 3:
                predictions["cubic_spline"][template] = float(
                    CubicSpline(interpolation_x, interpolation_y, bc_type="natural", extrapolate=True)(xtest)
                )
            predictions["popularity"][template] = float(np.mean(y))
        # GP uses course experience plus template identity; every held-out persona is excluded.
        X, ygp = [], []
        for p in training:
            for ti, template in enumerate(templates):
                X.append(
                    [
                        p["years_teaching"] / 40,
                        ti / max(1, len(templates) - 1),
                        p["discipline"] == test["discipline"],
                        p["assessment_style"] == test["assessment_style"],
                    ]
                )
                ygp.append(ratings[(p["id"], template)])
        Xq = np.array([[xtest / 40, ti / max(1, len(templates) - 1), 1.0, 1.0] for ti in range(len(templates))])
        gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.3),
            normalize_y=True,
            random_state=seed,
            n_restarts_optimizer=0,
        ).fit(np.asarray(X), np.asarray(ygp))
        for t, prediction in zip(templates, gp.predict(Xq)):
            predictions["gaussian_process"][t] = float(prediction)
        rng = random.Random(seed + int(pid[-2:]))
        order = list(templates)
        rng.shuffle(order)
        predictions["random"] = {t: 1.0 if i == 0 else 0.0 for i, t in enumerate(order)}
        best = max(truth.values())
        actual_top = {t for t, v in truth.items() if v == best}
        for method in methods:
            pred = predictions[method]
            ranked = sorted(pred, key=lambda t: (-pred[t], t))
            rows[method].append(
                {
                    "persona_id": pid,
                    "top1": ranked[0] in actual_top,
                    "top3": bool(set(ranked[:3]) & actual_top),
                    "spread": float(max(pred.values()) - min(pred.values())),
                    "preferred_template": sorted(actual_top)[0],
                }
            )
    summary = {}
    for method, cases in rows.items():
        summary[method] = {
            "n_held_out_personas": len(cases),
            "top1_accuracy": float(np.mean([r["top1"] for r in cases])) if cases else None,
            "top3_accuracy": float(np.mean([r["top3"] for r in cases])) if cases else None,
            "mean_prediction_spread": float(np.mean([r["spread"] for r in cases])) if cases else None,
            "held_out": cases,
        }
    return {
        "evaluation": "leave-one-persona-out; simulated preference means as labels",
        "metrics_are_synthetic": True,
        "methods": summary,
    }


def real_comparison(db: sqlite3.Connection) -> dict:
    rows = db.execute("SELECT item_id,metric,value_json FROM real_faculty_scores").fetchall()
    if not rows:
        return {"status": "awaiting real faculty scores", "comparisons": {}}
    sim = db.execute("SELECT prompt_id,label,atoms_json,slots_json FROM annotations").fetchall()
    per_item: dict[str, list[tuple[str, set[str], dict[str, float]]]] = defaultdict(list)
    for item, label, atoms_json, slots_json in sim:
        per_item[item].append((label, set(json.loads(atoms_json)), json.loads(slots_json)))
    grouped: dict[str, list[tuple[object, object]]] = defaultdict(list)
    for item, metric, value in rows:
        if item not in per_item:
            continue
        votes = per_item[item]
        predicted: object
        if metric == "label":
            predicted = Counter(entry[0] for entry in votes).most_common(1)[0][0]
        elif metric.startswith("atom:"):
            atom = metric.split(":", 1)[1]
            predicted = Counter(atom in entry[1] for entry in votes).most_common(1)[0][0]
        elif metric.startswith("slot:"):
            slot = metric.split(":", 1)[1]
            values = [entry[2][slot] for entry in votes if slot in entry[2]]
            if not values:
                continue
            predicted = float(np.median(values))
        else:
            continue
        grouped[metric].append((predicted, json.loads(value)))

    return {
        "status": "measured" if grouped else "no_matching_item_ids",
        "comparisons": {
            metric: {
                "cohen_kappa_simulated_majority_vs_real": _kappa([x[0] for x in vals], [x[1] for x in vals]),
                "n": len(vals),
            }
            for metric, vals in grouped.items()
        },
    }


def analyze(db: sqlite3.Connection) -> dict:
    return {
        "label": "SIMULATED",
        "synthetic": True,
        "source": "sim_persona",
        "persona_ids": [r[0] for r in db.execute("SELECT persona_id FROM personas ORDER BY persona_id")],
        "status": "measured"
        if db.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
        else "awaiting model calls",
        "rating_analysis": _rating_analysis(db),
        "annotation_analysis": _annotation_analysis(db),
        "routing_comparison": _routing_analysis(db),
        "real_faculty_comparison": real_comparison(db),
        "validity_limits": [
            "All 20 raters use one shared underlying model; agreement is likely inflated.",
            "The simulated personas have no real-world faculty grounding.",
            "These observations cannot establish human agreement, factual correctness, educational effectiveness, or system accuracy.",
            "T3/T4 need anonymous prompt or question text from earlier calls as blinded stimuli; authorship and prior judgments are withheld, but this conflicts with a strict ban on seeing any other persona's raw text.",
        ],
        "comparison_plan": "Use the same items and rubric with 5 to 8 real faculty, import their item-level scores through /api/simulated/import-real-scores, and compute matched-item agreement.",
    }
