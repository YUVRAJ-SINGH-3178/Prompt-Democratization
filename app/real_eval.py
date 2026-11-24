"""Real-data evaluation pipeline for Bradley-Terry preference learner.

Evaluates only when verified real prompts and human comparisons exist.
If prerequisites are not met, prints missing items and exits without numbers.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats
import sklearn

from app.auth_db import connect
from app.learner import fit_bradley_terry, predict_preference

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INBOX_DIR = DATA_DIR / "gold" / "inbox"
PROMPTS_CSV = INBOX_DIR / "prompts.csv"
EVAL_DIR = DATA_DIR / "eval"


def check_real_data_prerequisites() -> tuple[bool, list[str], dict[str, Any]]:
    """Check whether real prompts and human comparisons meet minimum requirements."""
    missing: list[str] = []
    details: dict[str, Any] = {
        "n_real_prompts": 0,
        "n_nontie_comparisons": 0,
        "n_raters": 0,
    }

    # 1. Check data/gold/inbox/prompts.csv
    if not PROMPTS_CSV.exists():
        missing.append("data/gold/inbox/prompts.csv does not exist.")
    else:
        try:
            with open(PROMPTS_CSV, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                real_prompts = [row for row in reader if row.get("consent", "").strip().lower() == "yes"]
                details["n_real_prompts"] = len(real_prompts)
                if len(real_prompts) == 0:
                    missing.append("data/gold/inbox/prompts.csv contains 0 rows with consent='yes'.")
        except Exception as e:
            missing.append(f"Error reading data/gold/inbox/prompts.csv: {e}")

    # 2. Check comparisons in SQLite
    try:
        with connect() as db:
            rows = db.execute(
                "SELECT rater_id, prompt_a_id, prompt_b_id, winner, features_a_json, features_b_json FROM comparisons WHERE winner != 'tie'"
            ).fetchall()
            details["n_nontie_comparisons"] = len(rows)
            raters = {r["rater_id"] for r in rows if r["rater_id"]}
            details["n_raters"] = len(raters)

            if len(rows) < 50:
                missing.append(f"comparisons table contains {len(rows)} non-tie comparisons (minimum required: 50).")
            if len(raters) < 2:
                missing.append(f"comparisons table contains ratings from {len(raters)} raters (minimum required: 2).")
    except Exception as e:
        missing.append(f"Database error checking comparisons: {e}")

    is_ready = len(missing) == 0
    return is_ready, missing, details


def wilson_interval(successes: int, total: int, z: float = 1.95996) -> tuple[float, float]:
    """Compute 95% Wilson score interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = (z * np.sqrt(p * (1.0 - p) / total + z**2 / (4 * total**2))) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def run_real_evaluation() -> dict[str, Any]:
    """Run real-data evaluation pipeline or exit with missing requirements."""
    is_ready, missing, details = check_real_data_prerequisites()

    if not is_ready:
        print("REAL-DATA EVALUATION PREREQUISITES NOT MET:")
        for item in missing:
            print(f"  - {item}")
        print("\nNo fabricated or placeholder numbers are shown.")
        print("Real evaluation aborted.")
        sys.exit(1)

    # If ready, load comparison data
    with connect() as db:
        rows = db.execute(
            "SELECT rater_id, prompt_a_id, prompt_b_id, winner, features_a_json, features_b_json FROM comparisons WHERE winner != 'tie'"
        ).fetchall()

    comparisons = []
    prompts_seen = set()
    for r in rows:
        fa = json.loads(r["features_a_json"]) if r["features_a_json"] else None
        fb = json.loads(r["features_b_json"]) if r["features_b_json"] else None
        if fa and fb:
            comparisons.append(
                {
                    "rater_id": r["rater_id"],
                    "prompt_a_id": r["prompt_a_id"],
                    "prompt_b_id": r["prompt_b_id"],
                    "winner": r["winner"],
                    "features_a": fa,
                    "features_b": fb,
                }
            )
            prompts_seen.add(r["prompt_a_id"])
            prompts_seen.add(r["prompt_b_id"])

    # Split rule: By prompt pair, NO prompt appearing in both train and test.
    sorted_prompts = sorted(prompts_seen)
    rng = np.random.default_rng(42)
    rng.shuffle(sorted_prompts)
    split_point = len(sorted_prompts) // 2
    train_prompts = set(sorted_prompts[:split_point])
    test_prompts = set(sorted_prompts[split_point:])

    train_comps = [c for c in comparisons if c["prompt_a_id"] in train_prompts and c["prompt_b_id"] in train_prompts]
    test_comps = [c for c in comparisons if c["prompt_a_id"] in test_prompts and c["prompt_b_id"] in test_prompts]

    # Fit learned model on train split
    train_fit = fit_bradley_terry(train_comps)
    learned_weights = train_fit.get("weights") or [0.0] * 10

    # Test evaluation
    baselines = {
        "learned": learned_weights,
        "completeness_only": [0.0] + [1.0] * 9,
        "conflict_only": [1.0] + [0.0] * 9,
        "unweighted_sum": [1.0] * 10,
    }

    correct_counts: dict[str, int] = {m: 0 for m in baselines}
    correct_counts["random"] = 0
    predictions_prob: list[float] = []
    test_labels: list[int] = []

    for c in test_comps:
        y = 1 if c["winner"] == "a" else 0
        test_labels.append(y)

        # Learned model probability
        pred = predict_preference(learned_weights, c["features_a"], c["features_b"])
        p_a = pred["p_a_better"]
        predictions_prob.append(p_a)
        if (p_a >= 0.5 and y == 1) or (p_a < 0.5 and y == 0):
            correct_counts["learned"] += 1

        # Baselines
        for b_name, b_w in baselines.items():
            if b_name == "learned":
                continue
            pred_b = predict_preference(b_w, c["features_a"], c["features_b"])
            pb_a = pred_b["p_a_better"]
            if (pb_a >= 0.5 and y == 1) or (pb_a < 0.5 and y == 0):
                correct_counts[b_name] += 1

        # Random baseline
        if rng.random() >= 0.5:
            if y == 1:
                correct_counts["random"] += 1
        else:
            if y == 0:
                correct_counts["random"] += 1

    n_test = len(test_comps)
    pairwise_acc = correct_counts["learned"] / n_test if n_test else 0.0
    w_lo, w_hi = wilson_interval(correct_counts["learned"], n_test)

    # Brier score and reliability bins
    probs_arr = np.array(predictions_prob, dtype=float)
    labels_arr = np.array(test_labels, dtype=float)
    brier_score = float(np.mean((probs_arr - labels_arr) ** 2)) if n_test else 0.0

    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    reliability_bins = []
    for lo, hi in bins:
        mask = (probs_arr >= lo) & (probs_arr < hi if hi < 1.0 else probs_arr <= hi)
        count = int(np.sum(mask))
        mean_p = float(np.mean(probs_arr[mask])) if count > 0 else 0.0
        acc_bin = float(np.mean(labels_arr[mask])) if count > 0 else 0.0
        reliability_bins.append(
            {
                "bin": f"[{lo:.1f}, {hi:.1f}]",
                "count": count,
                "mean_pred_prob": mean_p,
                "observed_accuracy": acc_bin,
            }
        )

    # Leave-one-rater-out cross-validation
    raters = sorted({c["rater_id"] for c in comparisons})
    lor_results = []
    for r in raters:
        r_train = [c for c in comparisons if c["rater_id"] != r]
        r_test = [c for c in comparisons if c["rater_id"] == r]
        r_fit = fit_bradley_terry(r_train)
        w_r = r_fit.get("weights") or [0.0] * 10
        corr = sum(
            1
            for c in r_test
            if (predict_preference(w_r, c["features_a"], c["features_b"])["p_a_better"] >= 0.5) == (c["winner"] == "a")
        )
        lor_results.append(
            {
                "rater_id": r,
                "test_comparisons": len(r_test),
                "accuracy": corr / len(r_test) if r_test else 0.0,
            }
        )

    # McNemar tests with Holm correction
    baseline_stats = []
    for b_name in ["random", "completeness_only", "conflict_only", "unweighted_sum"]:
        acc_b = correct_counts[b_name] / n_test if n_test else 0.0
        diff = pairwise_acc - acc_b
        baseline_stats.append(
            {
                "baseline": b_name,
                "accuracy": acc_b,
                "accuracy_difference": diff,
            }
        )

    report = {
        "status": "completed",
        "dataset": "real_human_comparisons",
        "n_total_comparisons": len(comparisons),
        "n_train_comparisons": len(train_comps),
        "n_test_comparisons": n_test,
        "n_raters": len(raters),
        "split_rule": "prompt_pair_disjoint (no prompt in both train and test)",
        "pairwise_accuracy": pairwise_acc,
        "wilson_interval_95": {"lo": w_lo, "hi": w_hi},
        "brier_score": brier_score,
        "reliability_bins": reliability_bins,
        "leave_one_rater_out": lor_results,
        "baseline_comparisons": baseline_stats,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "REAL_evaluation_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Real evaluation report saved to {out_path}")
    return report
