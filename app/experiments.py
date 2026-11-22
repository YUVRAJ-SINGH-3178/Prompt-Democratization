from __future__ import annotations

import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.interpolate import BarycentricInterpolator, CubicSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "experiment_runs"


def suitability(x: np.ndarray) -> np.ndarray:
    return 0.15 + 0.7 * np.exp(-((x - 0.2) ** 2) / (2 * 0.35**2)) + 0.1 * np.tanh(3 * x)


def lebesgue(n: int, kind: str, grid_n: int = 20001) -> float:
    if kind == "equispaced":
        nodes = np.linspace(-1, 1, n)
    else:
        nodes = np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))
    grid = np.linspace(-1, 1, grid_n)
    # Compute cardinal basis in chunks to bound memory.
    weights = np.ones(n)
    for j in range(n):
        weights[j] = 1.0 / np.prod(nodes[j] - np.delete(nodes, j))
    maximum = 0.0
    for start in range(0, grid_n, 256):
        q = grid[start : start + 256]
        delta = q[:, None] - nodes[None, :]
        exact = np.isclose(delta, 0, atol=1e-14)
        numer = weights[None, :] / np.where(exact, 1.0, delta)
        basis = numer / numer.sum(axis=1, keepdims=True)
        for i in np.where(exact.any(axis=1))[0]:
            basis[i] = 0
            basis[i, np.argmax(exact[i])] = 1
        maximum = max(maximum, float(np.max(np.abs(basis).sum(axis=1))))
    return maximum


def lebesgue_report() -> dict:
    reference = {
        4: (1.63, 1.85),
        6: (3.11, 2.10),
        8: (6.93, 2.29),
        10: (17.85, 2.43),
        12: (51.21, 2.54),
        16: (512.35, 2.73),
        20: (5889.58, 2.87),
    }
    values = []
    for n in (4, 6, 8, 10, 12, 16, 20):
        eq = lebesgue(n, "equispaced")
        ch = lebesgue(n, "chebyshev")
        values.append(
            {
                "n": n,
                "equispaced": eq,
                "chebyshev": ch,
                "reference_equispaced": reference[n][0],
                "reference_chebyshev": reference[n][1],
                "matches_reference_2dp": round(eq, 2) == reference[n][0] and round(ch, 2) == reference[n][1],
            }
        )
    return {
        "grid_points": 20001,
        "values": values,
        "all_match_reference_2dp": all(v["matches_reference_2dp"] for v in values),
    }


def curve_report(seed: int = 42, repetitions: int = 100) -> dict:
    rng = np.random.default_rng(seed)
    q = np.linspace(-1, 1, 400)
    truth = suitability(q)
    results = []
    gp_convergence_warnings = 0
    for sigma in (0, 0.02, 0.05, 0.1, 0.2):
        for n in (4, 6, 8, 12):
            errs: dict[str, list[float]] = {
                m: [] for m in ("lagrange_equispaced", "lagrange_chebyshev", "cubic_spline", "gaussian_process")
            }
            for _ in range(repetitions):
                x = np.linspace(-1, 1, n)
                y = suitability(x) + rng.normal(0, sigma, n)
                errs["lagrange_equispaced"].append(
                    float(np.sqrt(np.mean((BarycentricInterpolator(x, y)(q) - truth) ** 2)))
                )
                xc = np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))
                yc = suitability(xc) + rng.normal(0, sigma, n)
                order = np.argsort(xc)
                xc, yc = xc[order], yc[order]
                errs["lagrange_chebyshev"].append(
                    float(np.sqrt(np.mean((BarycentricInterpolator(xc, yc)(q) - truth) ** 2)))
                )
                errs["cubic_spline"].append(
                    float(np.sqrt(np.mean((CubicSpline(x, y, bc_type="natural")(q) - truth) ** 2)))
                )
                kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(max(sigma * sigma, 1e-3))
                gp = GaussianProcessRegressor(
                    kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    gp.fit(x[:, None], y)
                    gp_convergence_warnings += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
                errs["gaussian_process"].append(float(np.sqrt(np.mean((gp.predict(q[:, None]) - truth) ** 2))))
            for method, arr in errs.items():
                results.append(
                    {
                        "sigma": sigma,
                        "n": n,
                        "method": method,
                        "rmse_mean": float(np.mean(arr)),
                        "rmse_sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                        "n_repetitions": len(arr),
                    }
                )
    checks = []
    for sigma in (0, 0.02, 0.05, 0.1, 0.2):
        at12 = [r for r in results if r["sigma"] == sigma and r["n"] == 12]
        gp = next(r["rmse_mean"] for r in at12 if r["method"] == "gaussian_process")
        checks.append({"check": f"GP lowest at n=12, sigma={sigma}", "passed": gp == min(r["rmse_mean"] for r in at12)})
    for sigma in (0.02, 0.05, 0.1, 0.2):
        at12 = [r for r in results if r["sigma"] == sigma and r["n"] == 12]
        eq = next(r["rmse_mean"] for r in at12 if r["method"] == "lagrange_equispaced")
        checks.append(
            {"check": f"equispaced worst at n=12, sigma={sigma}", "passed": eq == max(r["rmse_mean"] for r in at12)}
        )
    for n in (4, 6):
        if n in (4, 6):
            at = [r for r in results if r["sigma"] == 0.1 and r["n"] == n]
            gp = next(r["rmse_mean"] for r in at if r["method"] == "gaussian_process")
            checks.append(
                {"check": f"GP not lowest at n={n}, sigma=0.1", "passed": gp > min(r["rmse_mean"] for r in at)}
            )
    reference: dict[str, dict[int, float]] = {
        "lagrange_equispaced": {4: 0.159, 6: 0.112, 8: 0.140, 12: 0.516},
        "lagrange_chebyshev": {4: 0.138, 6: 0.097, 8: 0.095, 12: 0.098},
        "cubic_spline": {4: 0.159, 6: 0.092, 8: 0.094, 12: 0.098},
        "gaussian_process": {4: 0.254, 6: 0.187, 8: 0.129, 12: 0.070},
    }
    comparison = []
    for n in (4, 6, 8, 12):
        for method, values in reference.items():
            own = float(
                next(r["rmse_mean"] for r in results if r["sigma"] == 0.1 and r["n"] == n and r["method"] == method)
            )
            comparison.append(
                {
                    "sigma": 0.1,
                    "n": n,
                    "method": method,
                    "own_rmse": own,
                    "paper_reference_rmse": values[n],
                    "difference": own - values[n],
                }
            )
    return {
        "results": results,
        "gaussian_process": {
            "kernel": "ConstantKernel * RBF + WhiteKernel",
            "normalize_y": True,
            "n_restarts_optimizer": 2,
            "random_state": seed,
            "convergence_warning_count": gp_convergence_warnings,
        },
        "sigma_0_1_reference_comparison": comparison,
        "pattern_checks": checks,
    }


def label_permutation_report(seed: int = 42, template_count: int = 2000, n_groups: int = 4) -> dict:
    from itertools import permutations

    rng = np.random.default_rng(seed)
    all_spread = []
    all_error = []
    outside = 0
    labels = np.arange(n_groups, dtype=float)
    perms = list(permutations(range(n_groups)))
    for _ in range(template_count):
        vals = rng.uniform(0, 1, n_groups)
        truth = float((vals[0] + vals[1]) / 2)
        pred = []
        for perm in perms:
            # Each permutation reassigns integer coordinates to fixed groups; values stay with their group.
            xp = labels[list(perm)]
            yp = vals
            mixed_query = float((xp[0] + xp[1]) / 2)
            pred.append(float(BarycentricInterpolator(xp, yp)(mixed_query)))
        all_spread.append(max(pred) - min(pred))
        all_error.append(float(np.mean(np.abs(np.asarray(pred) - truth))))
        outside += sum(int(v < 0 or v > 1) for v in pred)
    return {
        "template_count": template_count,
        "labeling_count": len(perms),
        "mean_spread": float(np.mean(all_spread)),
        "mean_absolute_error": float(np.mean(all_error)),
        "outside_fraction": outside / (template_count * len(perms)),
        "paper_reference": {
            "template_count": 500,
            "mean_spread": 0.643,
            "mean_absolute_error": 0.193,
            "outside_fraction": 0.0375,
        },
    }


def write_run(name: str, seed: int = 42, repetitions: int = 100) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    params = {
        "seed": seed,
        "repetitions": repetitions,
        "lebesgue_grid_points": 20001,
        "curve_test_points": 400,
        "sigmas": [0, 0.02, 0.05, 0.1, 0.2],
        "n_values": [4, 6, 8, 12],
    }
    data = {
        "run_id": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": params,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "synthetic": True,
    }
    if name in ("all", "lebesgue"):
        data["lebesgue"] = lebesgue_report()
    if name in ("all", "curve"):
        data["curve"] = curve_report(seed, repetitions)
    if name in ("all", "label_permutation"):
        data["label_permutation"] = label_permutation_report(seed)
    path = OUT / f"{name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    data["run_id"] = path.stem
    data["file"] = str(path.relative_to(ROOT))
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    from app.database import connect

    with connect() as db:
        db.execute(
            "INSERT INTO experiment_runs(id,name,result_json,synthetic) VALUES(?,?,?,1)",
            (data["run_id"], name, json.dumps(data)),
        )
    return data
