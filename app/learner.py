"""Bradley-Terry learner on feature differences.

For a comparison where prompt i beat prompt j:
  x = features(j) - features(i), label = 1
Fit LogisticRegression with fit_intercept=False, C=1.0, max_iter=2000.

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
    return 1.0 / (1.0 + np.exp(-x))


def fit_bradley_terry(
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fit a Bradley-Terry model on feature differences.

    Each comparison dict must have keys:
      features_a: list[float]  (length 10)
      features_b: list[float]  (length 10)
      winner: 'a' or 'b'  (ties excluded before calling)

    Returns model info dict.
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

    X, y = _build_dataset(valid)
    model = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000)
    model.fit(X, y)
    weights = model.coef_[0].tolist()

    # Bootstrap uncertainty
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_weights = []
    for _ in range(BOOTSTRAP_N):
        idx = rng.choice(len(X), size=len(X), replace=True)
        X_boot, y_boot = X[idx], y[idx]
        try:
            m = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000)
            m.fit(X_boot, y_boot)
            boot_weights.append(m.coef_[0].tolist())
        except Exception:
            continue

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
        "feature_names": list(FEATURE_NAMES),
        "bootstrap_n": BOOTSTRAP_N,
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
    """Build X, y arrays for the BT model.

    For winner='a': x = features(b) - features(a), label = 1
                    x_sym = features(a) - features(b), label = 0
    For winner='b': x = features(a) - features(b), label = 1
                    x_sym = features(b) - features(a), label = 0
    """
    X_rows = []
    y_rows = []
    for c in comparisons:
        fa = np.array(c["features_a"], dtype=float)
        fb = np.array(c["features_b"], dtype=float)
        if c["winner"] == "a":
            X_rows.append(fb - fa)
            y_rows.append(1)
            X_rows.append(fa - fb)
            y_rows.append(0)
        else:
            X_rows.append(fa - fb)
            y_rows.append(1)
            X_rows.append(fb - fa)
            y_rows.append(0)
    return np.array(X_rows), np.array(y_rows)


def learned_energy(weights: list[float], features: list[float]) -> float:
    """Learned energy = weight vector dot features."""
    return float(np.dot(weights, features))


def predict_preference(
    weights: list[float],
    features_a: list[float],
    features_b: list[float],
    weight_intervals: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """P(A better than B) = sigmoid(energy(B) - energy(A))."""
    ea = learned_energy(weights, features_a)
    eb = learned_energy(weights, features_b)
    p_a_better = sigmoid(eb - ea)

    result: dict[str, Any] = {
        "energy_a": ea,
        "energy_b": eb,
        "p_a_better": p_a_better,
        "p_b_better": 1.0 - p_a_better,
    }

    # Interval from bootstrap weight intervals
    if weight_intervals:
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
    # Use lo and hi endpoints plus the midpoint
    combos = []
    mid = [(iv["lo"] + iv["hi"]) / 2 for iv in intervals]
    combos.append(mid)
    combos.append([iv["lo"] for iv in intervals])
    combos.append([iv["hi"] for iv in intervals])
    return combos
