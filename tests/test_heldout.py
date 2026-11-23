from __future__ import annotations

import sqlite3

from app.linter.datasets import (
    CURRENT_EXTRACTOR_VERSION,
    compute_cohen_kappa,
    evaluate_heldout_dataset,
    get_development_prompts,
    get_heldout_prompts,
    seed_datasets,
)
from app.verification import wilson_interval


def test_development_and_heldout_set_sizes():
    dev = get_development_prompts()
    held = get_heldout_prompts()
    assert len(dev) >= 60, f"Development set must have >= 60 prompts, found {len(dev)}"
    assert len(held) >= 30, f"Held-out set must have >= 30 prompts, found {len(held)}"
    assert all(p["source"] in ("authored", "simulated_persona") for p in dev)
    assert all(p["dataset"] == "development" for p in dev)
    assert all(p["dataset"] == "heldout" for p in held)


def test_cohen_kappa_computation():
    # Perfect agreement
    assert compute_cohen_kappa([1, 0, 1, 1], [1, 0, 1, 1]) == 1.0
    # Complete disagreement
    assert compute_cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) < 0.0
    # Empty lists
    assert compute_cohen_kappa([], []) == 0.0


def test_wilson_interval_bounds():
    interval = wilson_interval(28, 30)
    assert interval is not None
    low, high = interval
    assert 0.0 <= low <= high <= 1.0
    assert low > 0.70


def test_awaiting_annotations_when_insufficient_annotators(tmp_path):
    test_db = tmp_path / "test_auth.sqlite"
    seed_datasets(test_db)
    with sqlite3.connect(test_db) as conn:
        conn.row_factory = sqlite3.Row
        report = evaluate_heldout_dataset(conn)
        assert report["status"] == "awaiting annotations"
        assert report["metrics"] == {}
        assert "manual_review_policy" in report


def test_heldout_scoring_with_double_annotators_logs_frozen_run(tmp_path):
    test_db = tmp_path / "test_auth.sqlite"
    seed_datasets(test_db)
    held = get_heldout_prompts()

    # Insert 30 double-annotated prompts into accuracy_annotations
    with sqlite3.connect(test_db) as conn:
        for p in held[:30]:
            pid = p["prompt_id"]
            # Annotator 1
            conn.execute(
                'INSERT INTO accuracy_annotations(prompt_id,dataset,annotator_id,atoms_json,slots_json,source,synthetic) VALUES(?,?,\'ann_1\',\'["single_correct"]\',\'{"role":1,"audience":1,"topic":1,"count":1,"n_options":1,"labeling":0,"marking":1,"explanation":1,"ordering":0}\',\'real\',0)',
                (pid, "heldout"),
            )
            # Annotator 2
            conn.execute(
                'INSERT INTO accuracy_annotations(prompt_id,dataset,annotator_id,atoms_json,slots_json,source,synthetic) VALUES(?,?,\'ann_2\',\'["single_correct"]\',\'{"role":1,"audience":1,"topic":1,"count":1,"n_options":1,"labeling":0,"marking":1,"explanation":1,"ordering":0}\',\'real\',0)',
                (pid, "heldout"),
            )
        conn.commit()

        report = evaluate_heldout_dataset(conn, freeze_version=CURRENT_EXTRACTOR_VERSION)
        assert report["status"] == "measured"
        assert report["n"] == 30
        assert "atom_metrics" in report
        assert "slot_agreement" in report
        assert "single_correct" in report["atom_metrics"]
        assert report["atom_metrics"]["single_correct"]["inter_annotator_cohen_kappa"] == 1.0

        # Verify run was logged in linter_scoring_runs
        run_count = conn.execute("SELECT COUNT(*) FROM linter_scoring_runs").fetchone()[0]
        assert run_count >= 1
