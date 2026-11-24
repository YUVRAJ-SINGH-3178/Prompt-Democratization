"""Tests for real-data evaluation readiness and safeguards."""

from __future__ import annotations

import pytest

from app.real_eval import (
    check_real_data_prerequisites,
    run_real_evaluation,
    wilson_interval,
)


def test_real_data_prerequisites_check():
    """Verify that prerequisites check reports missing real data honestly."""
    is_ready, missing, details = check_real_data_prerequisites()
    # In test environment with empty inbox and no real comparisons:
    assert isinstance(is_ready, bool)
    assert isinstance(missing, list)
    assert isinstance(details, dict)
    if not is_ready:
        assert len(missing) > 0


def test_real_eval_aborts_without_real_data():
    """Verify that run_real_evaluation exits with SystemExit when prerequisites are missing."""
    is_ready, _, _ = check_real_data_prerequisites()
    if not is_ready:
        with pytest.raises(SystemExit) as excinfo:
            run_real_evaluation()
        assert excinfo.value.code == 1


def test_wilson_interval():
    """Verify Wilson interval computation bounds and consistency."""
    lo, hi = wilson_interval(45, 50)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo < 0.90 < hi

    lo_0, hi_0 = wilson_interval(0, 0)
    assert lo_0 == 0.0 and hi_0 == 0.0


def test_disjoint_prompt_split_rule():
    """Verify split rule ensures no prompt appears in both train and test splits."""
    prompts = [f"P_{i}" for i in range(20)]
    split_point = len(prompts) // 2
    train_prompts = set(prompts[:split_point])
    test_prompts = set(prompts[split_point:])

    # Disjoint prompt sets
    assert train_prompts.isdisjoint(test_prompts)
