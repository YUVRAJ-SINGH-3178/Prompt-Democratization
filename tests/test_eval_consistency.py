"""Tests for synthetic recovery evaluation consistency between prose, tables, and setup."""

from __future__ import annotations

import json
import re
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"


def test_synthetic_recovery_consistency():
    json_path = EVAL_DIR / "SYNTHETIC_recovery.json"
    md_path = EVAL_DIR / "SYNTHETIC_recovery.md"

    assert json_path.exists(), "SYNTHETIC_recovery.json does not exist"
    assert md_path.exists(), "SYNTHETIC_recovery.md does not exist"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    md_text = md_path.read_text(encoding="utf-8")

    # 1. Label and synthetic attributes
    assert data.get("label") == "SYNTHETIC"
    assert data.get("SYNTHETIC") is True
    assert data.get("synthetic") == "not a real-data result"
    assert "LABEL: SYNTHETIC DATA" in md_text
    assert "not a real-data result" in md_text

    # 2. Setup block matches defaults
    assert data["n_prompts"] == 3000
    assert data["seeds"] == [42, 43, 44, 45, 46]
    assert data["budgets"] == [20, 50, 100, 200, 400]
    assert data["heldout_pairs"] == 5000
    assert data["beta"] == 1.0

    assert "- Prompts generated: 3000" in md_text
    assert "- Random seeds: [42, 43, 44, 45, 46]" in md_text
    assert "- Comparison budgets: [20, 50, 100, 200, 400]" in md_text
    assert "- Held-out evaluation pairs: 5000" in md_text

    # 3. Numbers in prose match table values
    # In table: | 20 | 80.96% (+/- 2.48%) | ...
    # In prose: At small budgets (N=20), the learned model achieves mean accuracy 80.96% (+/- 2.48%) across seeds.
    # Check that any percentage string in Observations appears in the linear table
    obs_section = md_text.split("## Observations")[1].split("## Model Misspecification")[0]
    percentages = re.findall(r"\d+\.\d+%\s*\(\+/-\s*\d+\.\d+%\)", obs_section)
    assert len(percentages) >= 5, "Expected observations to contain formatted percentages"

    linear_table = md_text.split("## Pairwise Accuracy on Noise-Free Ordering (Linear Case)")[1].split(
        "## Observations"
    )[0]
    for pct in percentages:
        assert pct in linear_table, f"Percentage {pct} in prose not found in linear table"

    # 4. Interaction case presence and disclaimer
    assert "## Model Misspecification: Non-linear Interaction Case" in md_text
    assert (
        "> **NOTE:** This checks the estimator under model misspecification and says nothing about real faculty."
        in md_text
    )

    # 5. Claims report consistency
    assert "claim_report" in data
    assert data["claim_report"]["status"] in ("win", "no_win")
    assert data["claim_report"]["synthetic"] == "not a real-data result"
    assert "## Claims Verification (eval/claims.yaml)" in md_text
