"""Unit tests for dev-eval and frozen-eval commands (Tasks 3 and 4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extractor.eval import (
    compute_normalized_file_sha256,
    run_dev_eval,
    run_frozen_eval,
    wilson_interval,
)


def test_wilson_interval() -> None:
    """Wilson interval returns valid bounds."""
    lo, hi = wilson_interval(95, 100)
    assert 0.85 < lo < 0.95
    assert 0.95 < hi <= 1.0

    lo0, hi0 = wilson_interval(0, 10)
    assert lo0 == 0.0
    assert hi0 > 0.0


def test_dev_eval_runs_and_writes_reports() -> None:
    """make dev-eval evaluates dev datasets and generates json and md reports."""
    res = run_dev_eval()
    assert res["provenance"]["total_prompts"] == 80
    assert res["provenance"]["dev_68_n"] == 68
    assert res["provenance"]["dev_12_n"] == 12

    json_path = Path("data/eval/dev_evaluation_report.json")
    md_path = Path("data/eval/dev_evaluation_report.md")

    assert json_path.exists()
    assert md_path.exists()

    md_text = md_path.read_text(encoding="utf-8")
    assert "Development Dataset Evaluation Report" in md_text
    assert "Split 1: Authored Development Prompts (n = 68)" in md_text
    assert "Split 2: Legacy Authored Prompts (n = 12)" in md_text
    # Check no em dashes in markdown
    assert "—" not in md_text


def test_frozen_eval_aborts_without_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """make frozen-eval does not run without the folder."""
    monkeypatch.setattr("app.extractor.eval.FROZEN_V2_DIR", tmp_path / "nonexistent_dir")
    with pytest.raises(FileNotFoundError, match="frozen_v2 directory not found"):
        run_frozen_eval()


def test_frozen_eval_hash_check_and_single_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """frozen-eval verifies hash, runs once, and refuses second run without force."""
    mock_frozen_dir = tmp_path / "frozen_v2"
    mock_frozen_dir.mkdir(parents=True)
    mock_file = mock_frozen_dir / "test_set.csv"
    mock_file.write_text(
        "prompt_id,prompt_text,very_easy,expert_level,no_basic,ordered_easy_to_hard,hidden_answers,answer_after_each_q,single_correct,multi_interp,confusing_language,role,audience,topic,count,n_options,labeling,marking,explanation,ordering\n"
        "F01,Create 5 MCQs on sorting algorithms.,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0\n",
        encoding="utf-8",
    )

    expected_hash = compute_normalized_file_sha256(mock_file)
    mock_hash_file = tmp_path / "FROZEN_V2_SHA256.txt"
    mock_hash_file.write_text(expected_hash, encoding="utf-8")

    mock_eval_dir = tmp_path / "eval"
    mock_eval_dir.mkdir(parents=True)

    monkeypatch.setattr("app.extractor.eval.FROZEN_V2_DIR", mock_frozen_dir)
    monkeypatch.setattr("app.extractor.eval.FROZEN_V2_HASH_FILE", mock_hash_file)
    monkeypatch.setattr("app.extractor.eval.EVAL_DIR", mock_eval_dir)

    # First run succeeds
    out1 = run_frozen_eval(force=False)
    assert out1["provenance"]["n_prompts"] == 1
    assert (mock_eval_dir / "frozen_v2_report.json").exists()
    assert (mock_eval_dir / "frozen_v2_report.md").exists()

    # Second run without force is refused!
    with pytest.raises(RuntimeError, match="Refusing to run a second time unless --force is passed"):
        run_frozen_eval(force=False)

    # Run with force=True succeeds and records forced run
    out_forced = run_frozen_eval(force=True)
    assert out_forced["execution"]["forced"] is True
    assert out_forced["execution"]["forced_run_count"] == 1
