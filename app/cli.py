from __future__ import annotations

import argparse
import json
import sqlite3
import platform
from datetime import datetime, timezone
from pathlib import Path

from app.database import DB_PATH
from app.experiments import write_run


def measure_break_even() -> dict:
    """Cross-validate GP against mean/popular over real feedback only; no default if evidence is absent."""
    if not DB_PATH.exists():
        rows = []
    else:
        with sqlite3.connect(DB_PATH) as db:
            rows = db.execute(
                "SELECT feature_json,template_id,rating FROM feedback WHERE synthetic=0 ORDER BY timestamp,id"
            ).fetchall()
    import numpy as np
    import scipy
    import sklearn

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "feedback table, synthetic=false",
        "seed": 42,
        "parameters": {
            "minimum_training_count": 3,
            "maximum_training_count": 40,
            "baseline": "per-template training mean",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "real_rating_count": len(rows),
        "break_even_rating_count": None,
        "measurement_status": "insufficient_real_feedback",
    }
    if len(rows) >= 8:
        # Per-template expanding-window validation compares GP with that template's training mean.
        import json as _json
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
        from app.routing.core import validate_features

        grouped: dict[str, list[tuple[object, float]]] = {}
        for feature_json, tid, rating in rows:
            try:
                x = validate_features(_json.loads(feature_json))
            except Exception:
                continue
            grouped.setdefault(tid, []).append((x, float(rating)))
        max_n = max((len(v) for v in grouped.values()), default=0)
        candidates = []
        for k in range(3, min(max_n - 1, 40) + 1):
            gp_sq = []
            base_sq = []
            evaluated_templates = 0
            for values in grouped.values():
                if len(values) <= k:
                    continue
                evaluated_templates += 1
                X = np.array([v[0] for v in values])
                y = np.array([v[1] for v in values])
                for i in range(k, len(values)):
                    model = GaussianProcessRegressor(
                        kernel=ConstantKernel(1) * RBF() + WhiteKernel(),
                        normalize_y=True,
                        n_restarts_optimizer=0,
                        random_state=42,
                    ).fit(X[i - k : i], y[i - k : i])
                    gp_sq.append(float(model.predict(X[i : i + 1])[0] - y[i]) ** 2)
                    base_sq.append(float(np.mean(y[i - k : i]) - y[i]) ** 2)
            if gp_sq:
                gp_rmse = float(np.sqrt(np.mean(gp_sq)))
                base_rmse = float(np.sqrt(np.mean(base_sq)))
                candidates.append(
                    {
                        "n_train_per_template": k,
                        "gp_rmse": gp_rmse,
                        "mean_baseline_rmse": base_rmse,
                        "n_test_predictions": len(gp_sq),
                        "n_templates": evaluated_templates,
                    }
                )
        crossing = next((r["n_train_per_template"] for r in candidates if r["gp_rmse"] < r["mean_baseline_rmse"]), None)
        result.update(
            {
                "measurement_status": "measured"
                if crossing
                else ("no_crossing_observed" if candidates else "insufficient_real_feedback"),
                "break_even_rating_count": crossing,
                "comparisons": candidates,
            }
        )
    out = Path(__file__).resolve().parent.parent / "data" / "break_even_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["file"] = str(out.relative_to(out.parents[1]))
    return result


def main():
    parser = argparse.ArgumentParser(
        prog="prompt-lab", description="Reproduce experiments and measure routing break-even from real data"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("reproduce")
    rep.add_argument("study", choices=["all", "lebesgue", "curve", "label_permutation"], default="all", nargs="?")
    rep.add_argument("--seed", type=int, default=42)
    rep.add_argument("--repetitions", type=int, default=100)
    sub.add_parser("measure-break-even")
    args = parser.parse_args()
    print(
        json.dumps(
            write_run(args.study, args.seed, args.repetitions) if args.command == "reproduce" else measure_break_even(),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
