"""Bradley-Terry learner on feature differences.

For a comparison where prompt i beat prompt j:
  x = features(j) - features(i), label = 1
Fit logistic loss without intercept.

Learned energy of a prompt = weight vector dot features.
P(A better than B) = sigmoid(energy(B) - energy(A)).
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from typing import Any

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression

from app.energy import FEATURE_NAMES

MINIMUM_COMPARISONS = 50
MINIMUM_RATERS = 2
BOOTSTRAP_SEED = 42
BOOTSTRAP_N = 200


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0))))


def fit_one_row(
    X: np.ndarray,
    C: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray | None:
    """Antisymmetric logistic loss minimization on single-row differences.

    Minimizes: sum_i log(1 + exp(-w^T x_i)) + (1 / (2*C)) * ||w||^2.
    Uses Newton-Raphson / IRLS with L2 regularization.
    Returns weight vector or None if singular.
    """
    if len(X) == 0:
        return None
    N, D = X.shape
    w = np.zeros(D, dtype=float)
    I_reg = np.eye(D) / C
    for _ in range(max_iter):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
        grad = X.T @ (p - 1.0) + w / C
        W = p * (1.0 - p)
        Hess = (X.T * W) @ X + I_reg
        try:
            step = np.linalg.solve(Hess, grad)
        except np.linalg.LinAlgError:
            return None
        w -= step
        if float(np.max(np.abs(step))) < tol:
            break
    return w


def fit_symmetric(X: np.ndarray, C: float = 1.0) -> np.ndarray:
    """Symmetric fit: adds the flipped row inside the fit and fits LogisticRegression.

    Uses C / 2 to match the objective scale of fit_one_row.
    """
    X_sym = np.vstack([X, -X])
    y_sym = np.array([1] * len(X) + [0] * len(X), dtype=int)
    model = LogisticRegression(fit_intercept=False, C=C / 2.0, max_iter=2000, tol=1e-8)
    model.fit(X_sym, y_sym)
    return model.coef_[0]


def fit_bradley_terry(
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fit a Bradley-Terry model on feature differences.

    Each comparison dict must have keys:
      features_a: list[float]  (length 10)
      features_b: list[float]  (length 10)
      winner: 'a' or 'b'  (ties excluded before calling)

    Returns model info dict with full fit point estimate and 200 bootstrap weight vectors.
    """
    # Filter out ties
    valid = [c for c in comparisons if c.get("winner") in ("a", "b")]
    raters = {c.get("rater_id") for c in valid if c.get("rater_id")}

    if len(valid) < MINIMUM_COMPARISONS or len(raters) < MINIMUM_RATERS:
        return {
            "model_status": "insufficient_data",
            "n_comparisons": len(valid),
            "n_raters": len(raters),
            "required_comparisons": MINIMUM_COMPARISONS,
            "required_raters": MINIMUM_RATERS,
            "weights": None,
            "feature_names": list(FEATURE_NAMES),
        }

    X, _ = _build_dataset(valid)
    weights_arr = fit_one_row(X, C=1.0)
    if weights_arr is None:
        weights_arr = fit_symmetric(X, C=1.0)
    weights = weights_arr.tolist()

    # Bootstrap uncertainty: resample whole comparisons with replacement
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_weights: list[list[float]] = []
    n_dropped = 0
    draws = 0
    max_extra = 20

    while len(boot_weights) < BOOTSTRAP_N and draws < (BOOTSTRAP_N + max_extra):
        draws += 1
        idx = rng.choice(len(valid), size=len(valid), replace=True)
        resample_comparisons = [valid[i] for i in idx]
        X_boot, _ = _build_dataset(resample_comparisons)
        w_boot = fit_one_row(X_boot, C=1.0)
        if w_boot is None or np.any(np.isnan(w_boot)):
            n_dropped += 1
            continue
        boot_weights.append(w_boot.tolist())

    interval_reliable = (n_dropped <= 10) and (len(boot_weights) == BOOTSTRAP_N)

    weight_intervals = []
    if boot_weights:
        bw = np.array(boot_weights)
        for i in range(len(weights)):
            lo = float(np.percentile(bw[:, i], 2.5))
            hi = float(np.percentile(bw[:, i], 97.5))
            weight_intervals.append({"lo": lo, "hi": hi})

    return {
        "model_status": "fitted",
        "n_comparisons": len(valid),
        "n_raters": len(raters),
        "weights": weights,
        "weight_intervals_95": weight_intervals,
        "bootstrap_weights": boot_weights,
        "dropped_bootstrap_count": n_dropped,
        "interval_reliable": interval_reliable,
        "reliable": interval_reliable,
        "feature_names": list(FEATURE_NAMES),
        "bootstrap_n": len(boot_weights),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "fitted_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_dataset(
    comparisons: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build X, y arrays for the BT model (one row per comparison).

    For winner='a': x = features(b) - features(a), label = 1
    For winner='b': x = features(a) - features(b), label = 1
    (Where x is loser - winner so that learned energy weights are positive
    when defects/penalties hurt preference).
    """
    X_rows = []
    y_rows = []
    for c in comparisons:
        fa = np.array(c["features_a"], dtype=float)
        fb = np.array(c["features_b"], dtype=float)
        if c["winner"] == "a":
            X_rows.append(fb - fa)
            y_rows.append(1)
        else:
            X_rows.append(fa - fb)
            y_rows.append(1)
    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=int)


def learned_energy(weights: list[float], features: list[float]) -> float:
    """Learned energy = weight vector dot features."""
    return float(np.dot(weights, features))


def predict_preference(
    weights: list[float],
    features_a: list[float],
    features_b: list[float],
    weight_intervals: list[dict[str, float]] | None = None,
    bootstrap_weights: list[list[float]] | None = None,
) -> dict[str, Any]:
    """P(A better than B) = sigmoid(energy(B) - energy(A)).

    Point estimate is computed from the full fit weights.
    95% uncertainty interval is computed from percentiles over bootstrap weights.
    """
    ea = learned_energy(weights, features_a)
    eb = learned_energy(weights, features_b)
    p_a_better = sigmoid(eb - ea)

    result: dict[str, Any] = {
        "energy_a": ea,
        "energy_b": eb,
        "p_a_better": p_a_better,
        "p_b_better": 1.0 - p_a_better,
    }

    if bootstrap_weights:
        diff = np.array(features_b, dtype=float) - np.array(features_a, dtype=float)
        p_values = [sigmoid(float(np.dot(w, diff))) for w in bootstrap_weights]
        result["p_a_better_interval_95"] = {
            "lo": float(np.percentile(p_values, 2.5)),
            "hi": float(np.percentile(p_values, 97.5)),
        }
    elif weight_intervals:
        p_values = []
        for combo in _corner_combos(weight_intervals):
            e_a = float(np.dot(combo, features_a))
            e_b = float(np.dot(combo, features_b))
            p_values.append(sigmoid(e_b - e_a))
        result["p_a_better_interval_95"] = {
            "lo": min(p_values),
            "hi": max(p_values),
        }

    return result


def _corner_combos(intervals: list[dict[str, float]]) -> list[list[float]]:
    """Sample corner combinations of weight intervals for uncertainty."""
    combos = []
    mid = [(iv["lo"] + iv["hi"]) / 2 for iv in intervals]
    combos.append(mid)
    combos.append([iv["lo"] for iv in intervals])
    combos.append([iv["hi"] for iv in intervals])
    return combos
