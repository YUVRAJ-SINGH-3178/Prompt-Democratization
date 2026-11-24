"""Tests for Bradley-Terry learner, uncertainty intervals, and coverage."""

from __future__ import annotations

import numpy as np

from app.learner import (
    BOOTSTRAP_N,
    fit_bradley_terry,
    fit_one_row,
    fit_symmetric,
    predict_preference,
    sigmoid,
)


def test_one_row_and_symmetric_fit_equivalence():
    """Verify that one-row fit and symmetric fit give the same weights up to numerical tolerance."""
    rng = np.random.default_rng(42)
    # 100 comparisons with 10 features
    X = rng.standard_normal((100, 10))

    w_one = fit_one_row(X, C=1.0)
    assert w_one is not None
    w_sym = fit_symmetric(X, C=1.0)

    # Weights match up to numerical tolerance
    diff = np.max(np.abs(w_one - w_sym))
    assert diff < 1e-3, f"Max difference between one-row and symmetric fit is {diff}"


def test_bootstrap_resampling_and_uncertainty_interval():
    """Verify bootstrap resampling, dropped count reporting, and interval computation."""
    rng = np.random.default_rng(101)
    comparisons = []
    for i in range(60):
        fa = rng.uniform(0, 1, 10).tolist()
        fb = rng.uniform(0, 1, 10).tolist()
        winner = "a" if i % 2 == 0 else "b"
        rater_id = "r1" if i < 30 else "r2"
        comparisons.append(
            {
                "features_a": fa,
                "features_b": fb,
                "winner": winner,
                "rater_id": rater_id,
            }
        )

    fit = fit_bradley_terry(comparisons)
    assert fit["model_status"] == "fitted"
    assert len(fit["weights"]) == 10
    assert len(fit["bootstrap_weights"]) == BOOTSTRAP_N
    assert isinstance(fit["dropped_bootstrap_count"], int)
    assert fit["dropped_bootstrap_count"] <= 10
    assert fit["interval_reliable"] is True

    fa_test = [0.1] * 10
    fb_test = [0.8] * 10
    pred = predict_preference(
        fit["weights"],
        fa_test,
        fb_test,
        bootstrap_weights=fit["bootstrap_weights"],
    )
    assert "p_a_better_interval_95" in pred
    iv = pred["p_a_better_interval_95"]
    assert 0.0 <= iv["lo"] <= iv["hi"] <= 1.0


def test_bootstrap_coverage():
    """Evaluate empirical coverage of the 95% bootstrap interval on 200 repeated synthetic datasets."""
    rng = np.random.default_rng(123)
    w_true = np.array([0.8, 0.4, 0.3, 0.5, 0.4, 0.2, 0.5, 0.3, 0.6, 0.2], dtype=float)

    # Fixed pair of test prompts
    fa_test = np.array([0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.2], dtype=float)
    fb_test = np.array([0.5, 0.3, 0.4, 0.3, 0.4, 0.2, 0.3, 0.4, 0.3, 0.3], dtype=float)
    ea_true = float(np.dot(w_true, fa_test))
    eb_true = float(np.dot(w_true, fb_test))
    p_true = sigmoid(eb_true - ea_true)

    n_datasets = 200
    n_comparisons = 150
    covered = 0

    diff_test = fb_test - fa_test

    for d in range(n_datasets):
        X_diffs = []
        for _ in range(n_comparisons):
            fa = rng.uniform(0, 1, 10)
            fb = rng.uniform(0, 1, 10)
            prob_a = sigmoid(float(np.dot(w_true, fb - fa)))
            if rng.random() < prob_a:
                X_diffs.append(fb - fa)
            else:
                X_diffs.append(fa - fb)
        X = np.array(X_diffs, dtype=float)

        boot_rng = np.random.default_rng(42 + d)
        p_boots = []
        draws = 0
        while len(p_boots) < 200 and draws < 220:
            draws += 1
            idx = boot_rng.choice(len(X), size=len(X), replace=True)
            w_boot = fit_one_row(X[idx], C=10.0)
            if w_boot is None:
                continue
            p_boots.append(sigmoid(float(np.dot(w_boot, diff_test))))

        lo = float(np.percentile(p_boots, 2.5))
        hi = float(np.percentile(p_boots, 97.5))

        if lo <= p_true <= hi:
            covered += 1

    coverage_rate = covered / n_datasets

    # Wilson 95% interval
    z = 1.95996
    p_hat = coverage_rate
    n = n_datasets
    denom = 1.0 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    wilson_lo = float(centre - half)
    wilson_hi = float(centre + half)

    # Must cover the true P(A > B) between 90% and 98% of the time
    assert 0.90 <= coverage_rate <= 0.98, (
        f"Coverage {coverage_rate * 100:.1f}% ({covered}/{n_datasets}) outside [90%, 98%], "
        f"Wilson 95% interval: [{wilson_lo:.4f}, {wilson_hi:.4f}]"
    )
